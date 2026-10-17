"""构建评测标注候选池：多 ranker 候选并集 + 来源记录，供人工标注。

用法:
    uv run python scripts/build_judgment_pool.py [--queries PATH] [--output PATH]
        [--pool-depth 10] [--seed 42] [--cache PATH] [--max-attempts N] [--base-delay SECONDS]

流程：加载查询 -> QueryPlanner 将每个 query 规划为 1-4 个关键词查询
      -> DiscoveryService 对每个规划查询执行带缓存与有界重试的 GitHub 发现
      -> 按 full_name 并集去重 -> 同一候选集上运行 rules / embedding / quality
      -> 取各自 Top-K 并集去重 -> 记录来源与排名（不含分数）
      -> 固定 seed 确定性打乱 -> 输出标注 JSONL + 覆盖率报告。
"""

import argparse
import asyncio
import logging
from pathlib import Path

from mosquito.evaluation.discovery import (
    DEFAULT_CACHE_PATH,
    DiscoveryCache,
    DiscoveryService,
    DiscoveryStatus,
    aggregate_query_outcome,
    summarize_coverage,
)
from mosquito.evaluation.pool import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_QUERIES_PATH,
    DEFAULT_SEED,
    PoolError,
    build_pool,
    load_queries,
    to_annotation_records,
    write_annotation_file,
)
from mosquito.query_planning.planner import QueryPlanner
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.ranker import QualityRanker
from mosquito.ranking.ranker import Ranker
from mosquito.sources.github import GitHubSource

CANDIDATE_POOL = 50


def _build_rankers() -> dict[str, object]:
    return {"rules": Ranker(), "embedding": EmbeddingRanker(), "quality": QualityRanker()}


def _print_coverage(queries: list[str], outcomes: dict, unique_repos: int) -> None:
    summary = summarize_coverage(queries, outcomes)
    print("\n=== 覆盖率报告 ===")
    print(f"总评测查询: {summary.total}")
    print(f"成功评估的查询: {summary.evaluated}")
    print(f"有候选的查询: {summary.with_candidates}")
    print(f"真空查询: {summary.empty}")
    print(f"失败查询: {summary.failed}")
    if summary.failure_reasons:
        print("--- 失败原因 ---")
        for status, qs in summary.failure_reasons.items():
            print(f"  {status} ({len(qs)}): {', '.join(qs)}")
    print(f"总唯一仓库: {unique_repos}")
    print(f"平均候选数（成功评估的查询）: {summary.avg_candidates:.1f}")
    print("--- 每个查询的候选数 ---")
    for q in queries:
        outcome = outcomes.get(q)
        if outcome is None:
            print(f"    -  {q}")
        elif outcome.status == DiscoveryStatus.SUCCESS:
            print(f"  {len(outcome.candidates):>3}  {q}")
        elif outcome.status == DiscoveryStatus.EMPTY:
            print(f"    0  {q} (真空)")
        else:
            print(f"    !  {q} ({outcome.status.value})")


async def main(
    queries_path: Path,
    output_path: Path,
    pool_depth: int,
    seed: int,
    cache_path: Path,
    max_attempts: int,
    base_delay: float,
) -> int:
    try:
        queries = load_queries(queries_path)
    except PoolError as exc:
        print(f"查询文件加载失败: {exc}")
        return 1
    if not queries:
        print("查询文件为空")
        return 1

    source = GitHubSource()
    cache = DiscoveryCache(cache_path)
    discovery = DiscoveryService(source, cache=cache, max_attempts=max_attempts, base_delay=base_delay)
    rankers = _build_rankers()
    planner = QueryPlanner()
    annotations = []
    outcomes: dict[str, object] = {}
    for query in queries:
        planned = planner.plan(query)
        planned_outcomes = [await discovery.discover(q, limit=CANDIDATE_POOL) for q in planned]
        outcome = aggregate_query_outcome(planned_outcomes)
        outcomes[query] = outcome
        if outcome.status == DiscoveryStatus.SUCCESS:
            # 三种 ranker 复用同一候选集，ranking 仍使用原始用户 query
            pooled = build_pool(outcome.candidates, rankers, query, pool_depth)
            annotations.extend(to_annotation_records(query, pooled, seed=seed))
        elif outcome.status == DiscoveryStatus.EMPTY:
            logging.info("query %r 为真空（GitHub 返回 0 结果）", query)
        else:
            logging.warning("query %r 检索失败: %s (%s)", query, outcome.status.value, outcome.error)

    cache.save()
    write_annotation_file(output_path, annotations)
    print(f"已生成 {len(annotations)} 条标注记录 -> {output_path}")
    unique_repos = len({a.full_name for a in annotations})
    _print_coverage(queries, outcomes, unique_repos)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建评测标注候选池")
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="查询 JSONL 路径（默认 data/evaluation/queries.jsonl）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="标注文件输出路径（默认 data/evaluation/pool_annotations.jsonl）",
    )
    parser.add_argument("--pool-depth", type=int, default=10, help="每个 ranker 的 Top-K（默认 10）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="打乱随机种子（默认 42）")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="发现结果缓存路径（默认 data/evaluation/discovery_cache.json）",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="每次检索最大尝试次数（默认 3）")
    parser.add_argument("--base-delay", type=float, default=1.0, help="重试基础延迟秒数（默认 1.0）")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                args.queries,
                args.output,
                args.pool_depth,
                args.seed,
                args.cache,
                args.max_attempts,
                args.base_delay,
            )
        )
    )