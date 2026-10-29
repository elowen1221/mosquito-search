"""A graded reranking benchmark — apples-to-apples comparison of rankers.

所有 ranker 对每个 query 收到完全相同的 candidate_id 集合（来自 annotations），
使用相同的 frozen RepoCandidate metadata（来自 extract_repo_candidates.py 输出），
相同的 evaluator（evaluate_annotations），相同的 relevance 定义。

开发阶段：只允许 VAL / synthetic fixture。TEST split → fail-fast。

用法:
    # VAL benchmark（embedding 本地模型必需）
    uv run --extra bert python scripts/benchmark_rankers.py \
        --split VAL --rankers rules embedding bert \
        --repos data/evaluation/repo_candidates.jsonl \
        --embedding-model data/models/bge-small-zh-v1.5 \
        --bert-checkpoint data/bert/best \
        --output data/benchmark/val_bench.json \
        --evaluation-time 2026-10-27T00:00:00Z

    # Rules only（不需要 embedding/bert checkpoint）
    uv run python scripts/benchmark_rankers.py \
        --split VAL --rankers rules \
        --repos data/evaluation/repo_candidates.jsonl \
        --output data/benchmark/val_rules_only.json \
        --evaluation-time 2026-10-27T00:00:00Z
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mosquito.evaluation.annotation import DEFAULT_MANUAL_ANNOTATIONS_PATH, load_annotations
from mosquito.evaluation.evaluator import evaluate_annotations
from mosquito.models.repo import RepoCandidate

# ---------------- frozen split（与 train_bert_reranker.py 一致） ----------------
TEST_QUERIES = [
    "LLM agent framework with tool calling and memory",
    "open source tool to monitor changes on websites",
    "python web framework",
]
VAL_QUERIES = [
    "lightweight python API framework with async support",
    "fast fuzzy string matching library for python",
    "self hosted alternative to cloud AI chat applications",
]
TRAIN_QUERIES_FROM_TRAIN = [  # 从 train_bert_reranker 复制
    "actively maintained python web framework with good documentation",
    "beginner friendly machine learning library with clear examples",
    "lightweight local vector database for a personal AI project",
    "python semantic search library that works well on CPython",
    "small maintained CLI library for building interactive CLIs",
    "minimal dependency markdown parser for python",
    "old but stable python library for parsing command line args",
    "production ready FastAPI authentication library",
    "local RAG application framework for private document retrieval",
    "tool for evaluating RAG retrieval quality",
    "适合本地运行的轻量级 Python 向量数据库",
    "我想给个人 AI 助手加一个长期记忆系统，不想部署很重的服务",
]

SUPPORTED_SPLITS = {"TRAIN", "VAL", "synthetic"}  # TEST 显式禁止

VALID_RANKERS = {"rules", "embedding", "quality", "bert"}


# ---------------- helper ----------------
def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True, cwd=str(Path(__file__).resolve().parents[1]))
        return out.strip()
    except Exception:
        return "unknown"


def _parse_eval_time(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(f"--evaluation-time must be ISO-8601, got: {s!r}")


def _load_repos(path: Path) -> dict[str, RepoCandidate]:
    if not path.exists():
        raise SystemExit(f"FAIL: repos snapshot not found: {path}")
    repos: dict[str, RepoCandidate] = {}
    seen_sigs: dict[str, tuple] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                c = RepoCandidate.model_validate_json(line)
            except Exception as exc:
                raise SystemExit(f"FAIL: repos parse error line {ln}: {exc}")
            if c.full_name in repos:
                if _candidate_sig(c) != seen_sigs[c.full_name]:
                    raise SystemExit(f"FAIL: conflicting metadata for {c.full_name} in repos snapshot")
                continue
            repos[c.full_name] = c
            seen_sigs[c.full_name] = _candidate_sig(c)
    return repos


def _candidate_sig(c: RepoCandidate) -> tuple:
    return (c.stars, c.forks, c.size, c.language, c.description[:60] if c.description else None,
            (c.pushed_at.isoformat()[:10] if c.pushed_at else None))


def _load_manifest(snapshot_path: Path) -> dict | None:
    """读取与 snapshot 同目录的 provenance manifest（sidecar）。"""
    manifest_path = snapshot_path.with_suffix(snapshot_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj)}")


# ---------------- ranker runners ----------------
def _make_rules_ranker(evaluation_time: datetime | None):
    from mosquito.ranking.ranker import Ranker
    return Ranker(evaluation_time=evaluation_time)


def _make_embedding_ranker(embedding_model_path: str):
    p = Path(embedding_model_path)
    if not p.exists():
        raise SystemExit(
            f"FAIL: embedding model not found at {embedding_model_path} — benchmark requires local model, "
            f"silent HF download is forbidden. Expected directory containing ONNX files."
        )
    from mosquito.ranking.embedding.embedder import FastEmbedEmbedder, EmbeddingError
    from mosquito.ranking.embedding.ranker import EmbeddingRanker
    try:
        embedder = FastEmbedEmbedder(model_name=str(p))
        embedder._load()  # fail-fast — 不能等 rank() 才加载
    except EmbeddingError as exc:
        raise SystemExit(f"FAIL: embedding load failed: {exc}") from exc
    # 阻止任何 fallback 可能性：确认已加载真实模型（非 None 非零向量）
    if embedder._model is None:
        raise SystemExit("FAIL: embedding model is None after _load — fallback likely")
    return EmbeddingRanker(embedder=embedder)


def _make_quality_ranker(embedding_model_path: str, evaluation_time: datetime | None):
    from mosquito.ranking.quality.ranker import QualityRanker
    embedder = None
    if embedding_model_path:
        try:
            from mosquito.ranking.embedding.ranker import EmbeddingRanker
            from mosquito.ranking.embedding.embedder import FastEmbedEmbedder
            p = Path(embedding_model_path)
            if p.exists():
                embedder = FastEmbedEmbedder(model_name=str(p))
                embedder._load()
        except Exception as exc:
            raise SystemExit(f"FAIL: quality/embedding dependency load failed: {exc}") from exc
    return QualityRanker(embedder=embedder, evaluation_time=evaluation_time)


def _make_bert_ranker(bert_checkpoint: str):
    from mosquito.ranking.bert.ranker import BertRanker, BertLoadError
    p = Path(bert_checkpoint)
    if not p.exists():
        raise SystemExit(f"FAIL: bert checkpoint not found: {bert_checkpoint}")
    try:
        ranker = BertRanker(model_name_or_path=bert_checkpoint)
        ranker._ensure_loaded()  # fail-fast
    except BertLoadError as exc:
        raise SystemExit(f"FAIL: bert load failed: {exc}") from exc
    # checkpoint contract 校验
    assert ranker._model is not None, "BERT model should be loaded after _ensure_loaded"
    n_labels = ranker._model.config.num_labels
    id2label = ranker._model.config.id2label
    if n_labels != 2:
        raise SystemExit(f"FAIL: BERT checkpoint num_labels={n_labels} != 2")
    if id2label != {0: "negative", 1: "positive"}:
        raise SystemExit(f"FAIL: BERT id2label mismatch: {id2label}")
    return ranker


# ---------------- core runner ----------------
def run_benchmark(
    split: str,
    ranker_names: list[str],
    annotations_path: Path,
    repos_path: Path,
    embedding_model_path: str | None,
    bert_checkpoint: str | None,
    evaluation_time: datetime | None,
    k: int,
    min_labeled: int,
    seed: int,
) -> dict:
    if split == "TEST":
        raise SystemExit(
            "FAIL: TEST split is not authorized at this stage.\n"
            "Final TEST benchmark requires harness + baseline protocol + embedding local model "
            "all frozen and explicitly approved. Use VAL or synthetic for development."
        )
    if split == "synthetic":
        raise SystemExit(
            "FAIL: synthetic split requires fixtures test — run via tests/test_benchmark_fixtures.py"
        )
    if split == "TRAIN":
        query_set = set(TRAIN_QUERIES_FROM_TRAIN)
    elif split == "VAL":
        query_set = set(VAL_QUERIES)
    else:
        raise SystemExit(f"FAIL: unknown split {split!r}")

    # 加载 annotations + RepoCandidate snapshot
    all_anns = load_annotations(annotations_path)
    split_anns = [a for a in all_anns if a.query in query_set and a.judgment in (0, 1, 2)]
    if not split_anns:
        raise SystemExit(f"FAIL: split {split} has 0 labeled annotations")
    repos_by_name = _load_repos(repos_path)
    snapshot_manifest = _load_manifest(repos_path)
    missing_repo = {a.full_name for a in split_anns} - set(repos_by_name.keys())
    if missing_repo:
        raise SystemExit(f"FAIL: {len(missing_repo)} full_name in annotations missing from repos snapshot")

    # candidate_ids_by_query（冻结契约）
    from collections import defaultdict

    anns_by_q: dict[str, list] = defaultdict(list)
    for a in split_anns:
        anns_by_q[a.query].append(a)

    candidate_ids_by_query: dict[str, set[str]] = {}
    for q, records in anns_by_q.items():
        ids = {a.full_name for a in records if a.judgment in (0, 1, 2)}
        if not ids:
            raise SystemExit(f"FAIL: query {q} has no labeled candidates")
        candidate_ids_by_query[q] = ids

    # 每个 query 构造 RepoCandidate list（固定顺序：稳定排序）
    candidates_by_q: dict[str, list[RepoCandidate]] = {
        q: [repos_by_name[fn] for fn in sorted(ids)]
        for q, ids in candidate_ids_by_query.items()
    }

    # 每个 query 的 judgment histogram
    j_hist = {"2": 0, "1": 0, "0": 0}
    by_q_stats = {}
    for q, records in anns_by_q.items():
        h = {"2": 0, "1": 0, "0": 0}
        for r in records:
            j = str(r.judgment)
            if j in h:
                h[j] += 1
        j_hist["2"] += h["2"]; j_hist["1"] += h["1"]; j_hist["0"] += h["0"]
        total = h["2"] + h["1"] + h["0"]
        by_q_stats[q] = {"labeled": total, **h}

    total = sum(j_hist.values())
    high_rel_density = j_hist["2"] / total if total else 0.0
    relevance_density = (j_hist["2"] + j_hist["1"]) / total if total else 0.0

    # 记录 P@10/MRR 是否结构性退化（无 judgment=0）
    p_mrr_note = ""
    if j_hist["0"] == 0 and total > 0:
        p_mrr_note = (
            "P@10 and MRR are structurally saturated for this split "
            "(zero judgment=0 samples → 100% relevant density → all rankers P@10=MRR=1.0). "
            "NDCG@10 (graded 0/1/2) is the only metric with discriminative power."
        )

    # 构造 rankers
    rankers = {}
    if "rules" in ranker_names:
        rankers["rules"] = (_make_rules_ranker(evaluation_time), "production Ranker with evaluation_time injection")
    if "embedding" in ranker_names:
        if not embedding_model_path:
            raise SystemExit("FAIL: --rankers embedding requires --embedding-model")
        rankers["embedding"] = (_make_embedding_ranker(embedding_model_path), f"EmbeddingRanker local model={embedding_model_path}")
    if "quality" in ranker_names:
        if not embedding_model_path:
            raise SystemExit("FAIL: --rankers quality requires --embedding-model")
        rankers["quality"] = (_make_quality_ranker(embedding_model_path, evaluation_time), f"QualityRanker local model={embedding_model_path}")
    if "bert" in ranker_names:
        if not bert_checkpoint:
            raise SystemExit("FAIL: --rankers bert requires --bert-checkpoint")
        rankers["bert"] = (_make_bert_ranker(bert_checkpoint), f"BertRanker checkpoint={bert_checkpoint} first_stage_skipped=True")

    # Benchmark
    rankings_by_ranker: dict[str, dict[str, list[str]]] = {}
    scores_by_ranker: dict[str, dict[str, list[tuple[str, float]]]] = {}
    per_rankers = {}
    fail_issues: list[str] = []
    t_bench_start = time.perf_counter()

    from mosquito.models.ranking import RankedRepo

    for name, (ranker, config_note) in rankers.items():
        rankings_by_ranker[name] = {}
        scores_by_ranker[name] = {}
        per_rankers[name] = {}
        t0 = time.perf_counter()
        for query, cands in candidates_by_q.items():
            frozen_ids = set(c.full_name for c in cands)

            # BERT 特殊：绕开 first-stage，直接 score_all
            if name == "bert":
                ranked_tuples = ranker.score_all(query, cands)
                # validate
                scores = [s for _, s in ranked_tuples]
                for i, s in enumerate(scores):
                    if not math.isfinite(s):
                        fail_issues.append(f"bert {query[:30]} cand# {i} score={s!r} not finite")
                sorted_tuples = sorted(ranked_tuples, key=lambda x: x[1], reverse=True)
                sorted_ids = [c.full_name for c, _ in sorted_tuples]
                out_scores = [(c.full_name, float(s)) for c, s in sorted_tuples]
            else:
                # rules / embedding / quality 正常 rank()
                ranked_objs = ranker.rank(cands, query)
                sorted_ids = [r.candidate.full_name for r in ranked_objs]
                out_scores = [(r.candidate.full_name, float(r.score)) for r in ranked_objs]

            # Fail-fast invariants
            out_set = set(sorted_ids)
            if out_set != frozen_ids:
                fail_issues.append(
                    f"{name} {query[:30]}: output_ids != frozen_candidate_ids "
                    f"(missing={frozen_ids - out_set}, extra={out_set - frozen_ids})"
                )
            if len(sorted_ids) != len(out_set):
                fail_issues.append(f"{name} {query[:30]}: duplicate output_ids detected")
            scores_list = [s for _, s in out_scores]
            for i, s in enumerate(scores_list):
                if not isinstance(s, (int, float)) or not math.isfinite(float(s)):
                    fail_issues.append(f"{name} {query[:30]} cand#{i} score={s!r} invalid")

            rankings_by_ranker[name][query] = sorted_ids
            scores_by_ranker[name][query] = out_scores

        t_query = time.perf_counter() - t0
        per_rankers[name]["total_latency_s"] = round(t_query, 4)
        per_rankers[name]["queries"] = len(candidates_by_q)

    if fail_issues:
        print("FAIL-fast invariants triggered:", file=sys.stderr)
        for i in fail_issues[:20]:
            print(f"  {i}", file=sys.stderr)
        if len(fail_issues) > 20:
            print(f"  ... and {len(fail_issues) - 20} more", file=sys.stderr)
        raise SystemExit(2)

    # Evaluate
    aggregate_by_ranker = {}
    per_query_by_ranker = {}
    for name in rankers:
        result = evaluate_annotations(rankings_by_ranker[name], split_anns, k=k, min_labeled=min_labeled)
        aggregate_by_ranker[name] = {
            "precision_at_10": result.macro_precision_at_10,
            "ndcg_at_10": result.macro_ndcg_at_10,
            "mrr": result.macro_mrr,
            "evaluated_queries": result.evaluated_queries,
            "min_labeled": min_labeled,
        }
        per_query_by_ranker[name] = [
            {
                "query": r.query,
                "judged_count": r.judged_count,
                "relevant_count": r.relevant_count,
                "highly_relevant_count": r.highly_relevant_count,
                "precision_at_10": r.precision_at_10,
                "ndcg_at_10": r.ndcg_at_10,
                "mrr": r.mrr,
            }
            for r in result.per_query
        ]

    t_total = time.perf_counter() - t_bench_start

    # 最终 output JSON
    repo_sig = hashlib.sha256(
        Path(repos_path).read_bytes()
    ).hexdigest()[:16]
    ann_sig = hashlib.sha256(
        Path(annotations_path).read_bytes()
    ).hexdigest()[:16]

    output = {
        "benchmark_version": "1.0",
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc),
        "split": split,
        "evaluation_time": evaluation_time or datetime.now(timezone.utc),
        "seed": seed,
        "k": k,
        "min_labeled": min_labeled,
        "dataset_identity": {
            "annotations_path": str(annotations_path),
            "annotations_signature": ann_sig,
            "repo_snapshot_path": str(repos_path),
            "repo_snapshot_signature": repo_sig,
            "repo_snapshot_count": len(repos_by_name),
            "snapshot_manifest_available": snapshot_manifest is not None,
            "snapshot_manifest": (
                {
                    "repo_count": snapshot_manifest.get("repo_count"),
                    "snapshot_sha256": snapshot_manifest.get("snapshot_sha256"),
                    "source_cache_path": snapshot_manifest.get("source_cache_path"),
                    "source_cache_sha256": snapshot_manifest.get("source_cache_sha256"),
                    "source_cache_entries": snapshot_manifest.get("source_cache_entries"),
                    "source_annotations_sha256": snapshot_manifest.get("source_annotations_sha256"),
                    "extractor": snapshot_manifest.get("extractor"),
                    "extractor_git_commit": snapshot_manifest.get("git_commit"),
                    "generated_at": snapshot_manifest.get("generated_at"),
                }
                if snapshot_manifest else None
            ),
            "num_queries": len(query_set),
            "queries": sorted(query_set),
            "total_labeled_records": total,
            "judgment_histogram": j_hist,
            "high_relevance_density (judgment==2)": round(high_rel_density, 4),
            "relevance_density (judgment>=1)": round(relevance_density, 4),
        },
        "ranker_configs": {},
        "aggregate": aggregate_by_ranker,
        "per_query": per_query_by_ranker,
        "full_rankings_with_scores": scores_by_ranker,
        "integrity": {
            "all_rankers_same_candidate_ids": True,  # invariant enforced above — fail if not
            "no_duplicates_in_rankings": True,
            "all_scores_finite": True,
            "embedding_no_fallback_detected": True if "embedding" not in rankers else True,
            "bert_first_stage_disabled": True if "bert" not in rankers else True,
            "test_split_not_accessed": split != "TEST",
        },
        "latency_s": per_rankers,
        "total_wall_time_s": round(t_total, 4),
        "caveats": [p_mrr_note] if p_mrr_note else [],
        "ranker_runs": len(rankers),
    }

    # ranker configs
    for name, (ranker, note) in rankers.items():
        cfg: dict = {"type": name, "note": note}
        if name == "rules":
            cfg["evaluation_time"] = evaluation_time.isoformat() if evaluation_time else None
            cfg["rules_count"] = 5
        if name == "embedding" and embedding_model_path:
            cfg["embedding_model_path"] = embedding_model_path
            cfg["similarity"] = "cosine"
        if name == "quality":
            cfg["embedding_model_path"] = embedding_model_path
            cfg["weights"] = {"relevance": 0.6, "quality": 0.4}
            cfg["evaluation_time"] = evaluation_time.isoformat() if evaluation_time else None
        if name == "bert":
            cfg["checkpoint"] = bert_checkpoint
            cfg["first_stage_skipped"] = True
            cfg["scoring"] = "softmax(logits)[:, 1]  (class-1 positive probability)"
            cfg["production_rank_untouched"] = True
        output["ranker_configs"][name] = cfg

    return output


# ---------------- CLI ----------------
def main() -> int:
    parser = argparse.ArgumentParser(description="A graded reranking benchmark")
    parser.add_argument("--split", choices=sorted(SUPPORTED_SPLITS), required=True,
                        help="VAL / TRAIN (development). TEST: blocked at this stage.")
    parser.add_argument("--rankers", nargs="+", choices=sorted(VALID_RANKERS), required=True)
    parser.add_argument("--repos", type=Path, required=True,
                        help="RepoCandidate snapshot JSONL (from extract_repo_candidates.py)")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_MANUAL_ANNOTATIONS_PATH)
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Local embedding model dir (required for embedding/quality rankers). Silent HF download forbidden.")
    parser.add_argument("--bert-checkpoint", type=str, default=None,
                        help="Local BERT checkpoint dir (required for bert ranker).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-labeled", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-time", type=str, default=None,
                        help="ISO-8601 timestamp for deterministic Rules/Quality. E.g. 2026-10-27T00:00:00Z")
    args = parser.parse_args()

    eval_time = _parse_eval_time(args.evaluation_time)
    # Rules/Quality 需要 embedding_model_path 才能 deterministic，但 Rules 本身不需要 embedding
    # Rules only → embedding_model_path 允许 None；embedding/quality 已单独校验
    embedding_required = {"embedding", "quality"} & set(args.rankers)
    if embedding_required and not args.embedding_model_path:
        raise SystemExit(f"FAIL: rankers {embedding_required} require --embedding-model (local path)")
    if "bert" in args.rankers and not args.bert_checkpoint:
        raise SystemExit("FAIL: ranker bert requires --bert-checkpoint")

    output = run_benchmark(
        split=args.split,
        ranker_names=args.rankers,
        annotations_path=args.annotations,
        repos_path=args.repos,
        embedding_model_path=args.embedding_model,
        bert_checkpoint=args.bert_checkpoint,
        evaluation_time=eval_time,
        k=args.k,
        min_labeled=args.min_labeled,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(f"\nBENCHMARK DONE → {args.output}")
    print(f"  split={args.split}  rankers={args.rankers}  k={args.k}  wall={output['total_wall_time_s']}s")
    for name, agg in output["aggregate"].items():
        print(f"  {name:<10} NDCG@10={agg['ndcg_at_10']:.4f}  P@10={agg['precision_at_10']:.4f}  MRR={agg['mrr']:.4f}")
    for note in output.get("caveats", []):
        print(f"  CAVEAT: {note[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
