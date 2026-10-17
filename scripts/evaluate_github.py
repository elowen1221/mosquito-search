"""离线评测 CLI：对 rules / embedding / quality 三种排序在标注数据集上评测。

用法:
    uv run python scripts/evaluate_github.py [--dataset PATH] [--top-k 10]

流程：加载标注数据集 -> 对每个 query 执行 GitHub 发现 -> 三种 ranker 排序
      -> 提取 Top-K full_name -> 与标注对比 -> 打印对比表。
"""

import argparse
import asyncio
import logging
from pathlib import Path

from mosquito.evaluation.dataset import DEFAULT_DATASET_PATH, DatasetError, load_judgments
from mosquito.evaluation.evaluator import evaluate_rankings
from mosquito.evaluation.models import RankerEvaluationResult
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.ranker import QualityRanker
from mosquito.ranking.ranker import Ranker
from mosquito.sources.github import GitHubError, GitHubSource

CANDIDATE_POOL = 50


def _build_rankers() -> dict[str, object]:
    return {
        "rules": Ranker(),
        "embedding": EmbeddingRanker(),
        "quality": QualityRanker(),
    }


async def _rank_query(source: GitHubSource, ranker, query: str, top_k: int) -> list[str]:
    candidates = await source.search(query, limit=CANDIDATE_POOL)
    ranked = ranker.rank(candidates, query)
    return [r.candidate.full_name for r in ranked[:top_k]]


async def _evaluate_ranker(
    source: GitHubSource, name: str, ranker, judgments, top_k: int
) -> RankerEvaluationResult:
    rankings: dict[str, list[str]] = {}
    for judgment in judgments:
        try:
            rankings[judgment.query] = await _rank_query(source, ranker, judgment.query, top_k)
        except GitHubError as exc:
            logging.warning("query %r 评测失败，已跳过: %s", judgment.query, exc)
    return evaluate_rankings(name, rankings, judgments, k=top_k)


def _print_table(results: list[RankerEvaluationResult], top_k: int) -> None:
    print(f"\n=== 离线评测结果 (k={top_k}) ===")
    print(f"{'Ranker':<12}{'P@10':>10}{'NDCG@10':>10}{'MRR':>10}{'Queries':>10}")
    for r in results:
        print(
            f"{r.ranker_name:<12}{r.precision_at_10:>10.4f}"
            f"{r.ndcg_at_10:>10.4f}{r.mrr:>10.4f}{r.evaluated_queries:>10}"
        )


async def main(dataset_path: Path, top_k: int) -> int:
    try:
        judgments = load_judgments(dataset_path)
    except DatasetError as exc:
        print(f"数据集加载失败: {exc}")
        return 1
    if not judgments:
        print("数据集为空，无可评测 query")
        return 1

    source = GitHubSource()
    results = []
    for name, ranker in _build_rankers().items():
        results.append(await _evaluate_ranker(source, name, ranker, judgments, top_k))
    _print_table(results, top_k)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mosquito 离线评测：rules / embedding / quality")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="标注数据集 JSONL 路径（默认 data/evaluation/github_queries.jsonl）",
    )
    parser.add_argument("--top-k", type=int, default=10, help="评测截断 K（默认 10）")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.dataset, args.top_k)))