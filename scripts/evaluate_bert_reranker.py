"""评测：在 held-out test queries 上对比 rules / embedding / quality / bert 四套 ranker 的 retrieval 指标。

用法：
    # 用训练好的 bert 模型评测
    uv run --extra bert python scripts/evaluate_bert_reranker.py

    # 指定 bert 模型路径
    uv run --extra bert python scripts/evaluate_bert_reranker.py --bert-model data/bert/best

测试集在最终评测前不得参与阈值或超参数选择。此脚本只计算指标，不做任何调参。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from mosquito.evaluation.annotation import (
    DEFAULT_MANUAL_ANNOTATIONS_PATH,
    load_annotations,
)
from mosquito.evaluation.discovery import (
    DEFAULT_CACHE_PATH,
    DiscoveryCache,
    DiscoveryService,
    DiscoveryStatus,
    aggregate_query_outcome,
)
from mosquito.evaluation.evaluator import evaluate_annotations
from mosquito.models.discover import RankingMode
from mosquito.query_planning.planner import QueryPlanner
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.ranker import QualityRanker
from mosquito.ranking.ranker import Ranker
from mosquito.sources.github import GitHubSource

CANDIDATE_POOL = 50

# 与训练脚本一致的冻结切分
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _rankings_for_query(discovery, planner, rankers, query: str) -> dict[str, list[str]] | None:
    planned = planner.plan(query)
    outcomes = [await discovery.discover(q, limit=CANDIDATE_POOL) for q in planned]
    outcome = aggregate_query_outcome(outcomes)
    if outcome.status != DiscoveryStatus.SUCCESS:
        return None
    result = {}
    for name, ranker in rankers.items():
        start = time.perf_counter()
        ranked = ranker.rank(outcome.candidates, query)
        elapsed = time.perf_counter() - start
        logger.info("  %-10s query=%-55s lat=%.3fs", name, query[:55], elapsed)
        result[name] = [r.candidate.full_name for r in ranked]
    return result


async def main(bert_model: str | None, annotations_path: Path, cache_path: Path, top_k: int) -> int:
    annotations = load_annotations(annotations_path)
    test_queries = TEST_QUERIES
    annotations_by_query = {}
    for a in annotations:
        annotations_by_query.setdefault(a.query, []).append(a)
    for q in test_queries:
        if q not in annotations_by_query:
            logger.warning("test query %r 无标注记录，跳过", q)
    test_annotations = [r for r in annotations if r.query in test_queries]
    if not test_annotations:
        logger.error("测试集无可用标注")
        return 1

    source = GitHubSource()
    cache = DiscoveryCache(cache_path)
    discovery = DiscoveryService(source, cache=cache)
    planner = QueryPlanner()

    rankers: dict[str, object] = {
        "rules": Ranker(),
        "embedding": EmbeddingRanker(),
        "quality": QualityRanker(),
    }

    if bert_model:
        try:
            from mosquito.ranking.bert.ranker import BertRanker

            bert = BertRanker(model_name_or_path=bert_model)
            rankers["bert"] = bert
            logger.info("bert ranker 已启用: %s", bert_model)
        except Exception as exc:
            logger.error("bert ranker 加载失败: %s，继续评测其他 ranker", exc)

    # 评测缓存里有 val+test 所有 query，但只在 test_queries 上对比
    rankings_by_ranker: dict[str, dict[str, list[str]]] = {name: {} for name in rankers}
    for query in test_queries:
        rankings = await _rankings_for_query(discovery, planner, rankers, query)
        if rankings is None:
            logger.warning("query %r 检索失败，跳过", query)
            continue
        for name, ranked in rankings.items():
            rankings_by_ranker[name][query] = ranked

    # 打印对比表
    print(f"\n=== held-out test ({len(test_queries)} queries) top_k={top_k} ===")
    header = f"{'Ranker':<12} {'P@10':>8} {'NDCG@10':>8} {'MRR':>8} {'Queries':>8}"
    print(header)
    print("-" * len(header))

    for ranker_name, rankings in rankings_by_ranker.items():
        result = evaluate_annotations(rankings, test_annotations, k=top_k, min_labeled=3)
        print(
            f"{ranker_name:<12} "
            f"{result.macro_precision_at_10:>8.4f} "
            f"{result.macro_ndcg_at_10:>8.4f} "
            f"{result.macro_mrr:>8.4f} "
            f"{result.evaluated_queries:>8}"
        )

    # per-query 详情
    print("\n=== per-query 详情 ===")
    for ranker_name, rankings in rankings_by_ranker.items():
        result = evaluate_annotations(rankings, test_annotations, k=top_k, min_labeled=3)
        print(f"\n{ranker_name}:")
        for r in result.per_query:
            print(
                f"  {r.query[:50]:<50}  P@10={r.precision_at_10:.3f}  "
                f"NDCG@10={r.ndcg_at_10:.3f}  MRR={r.mrr:.3f}  "
                f"judged={r.judged_count}  rel={r.relevant_count}"
            )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="held-out test 评测：rules / embedding / quality / bert")
    parser.add_argument("--bert-model", default=None, help="BERT 模型路径（训练后保存的目录）")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_MANUAL_ANNOTATIONS_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.bert_model, args.annotations, args.cache, args.top_k)))