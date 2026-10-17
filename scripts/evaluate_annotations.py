"""人工标注评测 CLI：基于 manual_annotations.jsonl 计算每查询指标。

用法:
    uv run python scripts/evaluate_annotations.py [--annotations PATH] [--top-k 10] [--min-labeled 5]

流程：加载人工标注 -> 对每个有标注的 query 执行 GitHub 发现（缓存命中）
      -> 三种 ranker 排序 -> 仅用已标注记录计算 P@10 / NDCG@10 / MRR
      -> 打印每查询报告与宏平均（仅统计标注充足的 query）。
"""

import argparse
import asyncio
import logging
from pathlib import Path

from mosquito.evaluation.annotation import (
    DEFAULT_MANUAL_ANNOTATIONS_PATH,
    load_annotations,
    validate_annotations,
)
from mosquito.evaluation.discovery import (
    DEFAULT_CACHE_PATH,
    DiscoveryCache,
    DiscoveryService,
    DiscoveryStatus,
    aggregate_query_outcome,
)
from mosquito.evaluation.evaluator import AnnotationEvaluationResult, evaluate_annotations
from mosquito.query_planning.planner import QueryPlanner
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.ranker import QualityRanker
from mosquito.ranking.ranker import Ranker
from mosquito.sources.github import GitHubSource

CANDIDATE_POOL = 50


def _build_rankers() -> dict[str, object]:
    return {"rules": Ranker(), "embedding": EmbeddingRanker(), "quality": QualityRanker()}


async def _rankings_for_query(discovery, planner, rankers, query: str) -> dict[str, list[str]] | None:
    planned = planner.plan(query)
    outcomes = [await discovery.discover(q, limit=CANDIDATE_POOL) for q in planned]
    outcome = aggregate_query_outcome(outcomes)
    if outcome.status != DiscoveryStatus.SUCCESS:
        return None
    return {
        name: [r.candidate.full_name for r in ranker.rank(outcome.candidates, query)]
        for name, ranker in rankers.items()
    }


def _print_report(ranker_name: str, result: AnnotationEvaluationResult, top_k: int) -> None:
    print(f"\n=== {ranker_name} 人工标注评测 (k={top_k}, min_labeled={result.min_labeled}) ===")
    print(
        f"{'Query':<48}{'Judged':>7}{'Rel':>5}{'HR':>5}{'Cov':>7}"
        f"{'P@10':>8}{'NDCG@10':>9}{'MRR':>7}"
    )
    for r in result.per_query:
        print(
            f"{r.query[:46]:<48}{r.judged_count:>7}{r.relevant_count:>5}"
            f"{r.highly_relevant_count:>5}{r.coverage:>7.2f}"
            f"{r.precision_at_10:>8.3f}{r.ndcg_at_10:>9.3f}{r.mrr:>7.3f}"
        )
    print(
        f"宏平均（{result.evaluated_queries} 个 query）: "
        f"P@10={result.macro_precision_at_10:.3f} "
        f"NDCG@10={result.macro_ndcg_at_10:.3f} MRR={result.macro_mrr:.3f}"
    )
    if result.excluded_queries:
        print(f"因标注不足被排除的 query（< {result.min_labeled} 条标注）: {len(result.excluded_queries)} 个")
        for q in result.excluded_queries:
            print(f"  - {q}")


async def main(annotations_path: Path, top_k: int, min_labeled: int, cache_path: Path) -> int:
    annotations = load_annotations(annotations_path)
    if not annotations:
        print(f"标注文件为空或不存在: {annotations_path}")
        return 1
    issues = validate_annotations(annotations)
    if issues:
        print(f"警告: 发现 {len(issues)} 个校验问题")
        for issue in issues[:10]:
            print(f"  [{issue.issue_type}] 第 {issue.index + 1} 条: {issue.message}")

    queries = sorted({a.query for a in annotations})
    source = GitHubSource()
    cache = DiscoveryCache(cache_path)
    discovery = DiscoveryService(source, cache=cache)
    rankers = _build_rankers()
    planner = QueryPlanner()

    rankings_by_ranker = {name: {} for name in rankers}
    for query in queries:
        rankings = await _rankings_for_query(discovery, planner, rankers, query)
        if rankings is None:
            logging.warning("query %r 检索失败，跳过评测", query)
            continue
        for name, ranked in rankings.items():
            rankings_by_ranker[name][query] = ranked

    for name, rankings in rankings_by_ranker.items():
        result = evaluate_annotations(rankings, annotations, k=top_k, min_labeled=min_labeled)
        _print_report(name, result, top_k)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人工标注评测：rules / embedding / quality")
    parser.add_argument(
        "--annotations", type=Path, default=DEFAULT_MANUAL_ANNOTATIONS_PATH, help="标注文件路径（默认 data/evaluation/manual_annotations.jsonl）"
    )
    parser.add_argument("--top-k", type=int, default=10, help="评测截断 K（默认 10）")
    parser.add_argument("--min-labeled", type=int, default=5, help="宏平均所需最少标注数（默认 5）")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH, help="发现结果缓存路径")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(main(args.annotations, args.top_k, args.min_labeled, args.cache))
    )