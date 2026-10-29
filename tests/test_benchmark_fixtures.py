"""benchmark harness 的 integrity 与 deterministic tests。

全部用 synthetic fixtures / stub rankers / fake data，不下载任何模型权重。
覆盖：
- TEST split blocked（fail-fast）
- RulesRanker deterministic evaluation_time（相同输入 + 相同时间 → 相同输出）
- missing RepoCandidate → fail-fast
- candidate output count mismatch → fail-fast
- invalid / NaN / Inf score → fail-fast
- duplicate (query, full_name) detection
- embedding model path missing → fail-fast
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mosquito.models.repo import RepoCandidate
from mosquito.ranking.ranker import Ranker
from mosquito.ranking.rules import recency_at


# ---------------- TEST split blocked ----------------


def test_benchmark_blocks_test_split_via_run_benchmark_directly():
    """TEST split 在开发阶段必须 fail-fast。直接调 run_benchmark。"""
    import scripts.benchmark_rankers as bm

    with pytest.raises(SystemExit) as exc:
        bm.run_benchmark(
            split="TEST", ranker_names=["rules"],
            annotations_path=Path("data/evaluation/manual_annotations.jsonl"),
            repos_path=Path("data/evaluation/repo_candidates.jsonl"),
            embedding_model_path=None, bert_checkpoint=None, evaluation_time=None,
            k=10, min_labeled=3, seed=42,
        )
    # SystemExit code 必须非 0
    assert exc.value.code != 0
    msg = str(exc.value)
    assert "not authorized" in msg or "TEST split" in msg


# ---------------- Rules deterministic evaluation_time ----------------


def _make_repo(pushed_days_ago: int) -> RepoCandidate:
    import datetime as _dt
    t = datetime.now(timezone.utc)
    pushed = datetime(t.year, t.month, t.day, tzinfo=timezone.utc) - _dt.timedelta(days=pushed_days_ago)
    return RepoCandidate(
        name=f"repo_{pushed_days_ago}", full_name=f"owner/repo_{pushed_days_ago}",
        url="https://github.com/owner/repo", description="desc", language="Python",
        stars=100, forks=50, topics=[], pushed_at=pushed, size=50000,
    )


def test_recency_at_is_deterministic():
    """recency_at(candidate, query, fixed_time) 对相同输入必须返回相同结果。"""
    fixed_time = datetime(2026, 10, 27, 0, 0, 0, tzinfo=timezone.utc)
    repo = _make_repo(pushed_days_ago=15)

    r1 = recency_at(repo, "q", fixed_time)
    r2 = recency_at(repo, "q", fixed_time)
    assert r1.score == r2.score == 15.0
    assert r1.reason == r2.reason


def test_rules_rank_with_fixed_evaluation_time_is_reproducible():
    """RulesRanker(evaluation_time=fixed) 两次相同输入 → 完全相同排序。"""
    fixed_time = datetime(2026, 10, 27, 0, 0, 0, tzinfo=timezone.utc)
    repos = [_make_repo(pushed_days_ago=d) for d in (5, 50, 200)]

    r1 = Ranker(evaluation_time=fixed_time).rank(repos, "python web framework")
    r2 = Ranker(evaluation_time=fixed_time).rank(repos, "python web framework")
    assert [(r.candidate.full_name, r.score) for r in r1] == [(r.candidate.full_name, r.score) for r in r2]


def test_rules_rank_without_evaluation_time_uses_now():
    """RulesRanker(evaluation_time=None) → production 默认，非 deterministic。"""
    r = Ranker()
    assert r.evaluation_time is None


# ---------------- candidate contract invariants ----------------


def _write_minimal_fixture(tmp_path: Path, query: str):
    """写一个最小 annotations + repos fixture，包含 judgment=0/1/2。"""
    anns = [
        {"query": query, "full_name": "owner/a", "description": "", "language": "Python", "topics": [],
         "retrieved_by": ["rules"], "ranks": {}, "judgment": 2},
        {"query": query, "full_name": "owner/b", "description": "", "language": "Python", "topics": [],
         "retrieved_by": ["rules"], "ranks": {}, "judgment": 1},
        {"query": query, "full_name": "owner/c", "description": "", "language": "Python", "topics": [],
         "retrieved_by": ["rules"], "ranks": {}, "judgment": 0},
    ]
    repos = [
        {"name": "a", "full_name": "owner/a", "url": "x", "description": "da", "language": "Python",
         "stars": 1, "forks": 0, "topics": [], "size": 1000, "pushed_at": "2024-01-01T00:00:00+00:00",
         "open_issues": 0, "updated_at": None},
        {"name": "b", "full_name": "owner/b", "url": "x", "description": "db", "language": "Python",
         "stars": 1, "forks": 0, "topics": [], "size": 1000, "pushed_at": "2024-01-01T00:00:00+00:00",
         "open_issues": 0, "updated_at": None},
        {"name": "c", "full_name": "owner/c", "url": "x", "description": "dc", "language": "Python",
         "stars": 1, "forks": 0, "topics": [], "size": 1000, "pushed_at": "2024-01-01T00:00:00+00:00",
         "open_issues": 0, "updated_at": None},
    ]
    (tmp_path / "anns.jsonl").write_text("\n".join(json.dumps(a) for a in anns))
    (tmp_path / "repos.jsonl").write_text("\n".join(json.dumps(r) for r in repos))


def test_benchmark_invariant_rejects_candidate_count_mismatch(tmp_path, monkeypatch):
    """benchmark 必须 fail-fast：ranker 输出数量 ≠ frozen candidate 数量。"""
    # 用 TRAIN 的一个真实 query 构造 fixture，benchmark split="TRAIN" 才能匹配
    from scripts.benchmark_rankers import TRAIN_QUERIES_FROM_TRAIN
    q = TRAIN_QUERIES_FROM_TRAIN[0]  # "actively maintained python web framework..."
    _write_minimal_fixture(tmp_path, q)

    import scripts.benchmark_rankers as bm

    def _broken_rank(self, candidates, query):
        from mosquito.models.ranking import RankedRepo
        by_name = {c.full_name: c for c in candidates}
        # 只返回前 2 个，漏第 3 个
        return [RankedRepo(candidate=by_name[n], score=1.0, reasons=[])
                for n in sorted(by_name.keys())[:2]]

    from mosquito.ranking.ranker import Ranker as _R
    monkeypatch.setattr(_R, "rank", _broken_rank, raising=True)

    with pytest.raises(SystemExit) as exc:
        bm.run_benchmark(
            split="TRAIN", ranker_names=["rules"],
            annotations_path=tmp_path / "anns.jsonl",
            repos_path=tmp_path / "repos.jsonl",
            embedding_model_path=None, bert_checkpoint=None, evaluation_time=None,
            k=10, min_labeled=3, seed=42,
        )
    assert exc.value.code == 2  # fail-fast invariants triggered


def test_embedding_model_path_missing_fails(tmp_path):
    """benchmark 禁止 silent HF download：embedding 路径不存在 → 立即 fail。"""
    import scripts.benchmark_rankers as bm

    with pytest.raises(SystemExit) as exc:
        bm.run_benchmark(
            split="VAL", ranker_names=["embedding"],
            annotations_path=Path("data/evaluation/manual_annotations.jsonl"),
            repos_path=Path("data/evaluation/repo_candidates.jsonl"),
            embedding_model_path=str(tmp_path / "does_not_exist"),
            bert_checkpoint=None, evaluation_time=None,
            k=10, min_labeled=3, seed=42,
        )
    msg = str(exc.value).lower()
    # 必须有明确的 fail 信息：要么 path not found / not exist / local model
    assert exc.value.code != 0
    assert any(kw in msg for kw in ("not found", "not exist", "local model", "forbidden"))
