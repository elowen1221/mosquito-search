"""评测器单元测试：排序结果与标注对比。"""

import math

import pytest

from mosquito.evaluation.evaluator import evaluate_rankings
from mosquito.evaluation.models import QueryJudgment


def _judgments() -> list[QueryJudgment]:
    return [
        QueryJudgment(query="q1", judgments={"a/b": 2, "c/d": 1, "e/f": 0}),
        QueryJudgment(query="q2", judgments={"x/y": 2}),
    ]


def test_perfect_ranking():
    # 完美排序：返回的仓库全部相关（judgment >= 1）
    rankings = {"q1": ["a/b", "c/d"], "q2": ["x/y"]}
    result = evaluate_rankings("rules", rankings, _judgments())
    assert result.ranker_name == "rules"
    assert result.precision_at_10 == pytest.approx(1.0)
    assert result.ndcg_at_10 == pytest.approx(1.0)
    assert result.mrr == pytest.approx(1.0)
    assert result.evaluated_queries == 2


def test_reversed_ranking():
    rankings = {"q1": ["e/f", "c/d", "a/b"], "q2": ["z/w", "x/y"]}
    result = evaluate_rankings("rules", rankings, _judgments())
    # q1: 2/3，q2: 1/2 → P@10 = (2/3 + 1/2) / 2
    assert result.precision_at_10 == pytest.approx((2 / 3 + 0.5) / 2)
    assert 0.0 < result.ndcg_at_10 < 1.0
    assert result.mrr == pytest.approx(0.5)


def test_irrelevant_results():
    rankings = {"q1": ["z/z", "y/y"], "q2": ["w/w"]}
    result = evaluate_rankings("rules", rankings, _judgments())
    assert result.precision_at_10 == 0.0
    assert result.ndcg_at_10 == 0.0
    assert result.mrr == 0.0


def test_missing_unjudged_repos():
    # 排序结果包含未标注仓库，应被忽略（贡献 0）
    rankings = {"q1": ["unknown/a", "a/b", "unknown/b"]}
    result = evaluate_rankings("rules", rankings, _judgments())
    assert result.precision_at_10 == pytest.approx(1 / 3)
    # dcg = 2/log2(3)，idcg = 2/log2(2) + 1/log2(3)
    expected_ndcg = (2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert result.ndcg_at_10 == pytest.approx(expected_ndcg)


def test_empty_rankings():
    result = evaluate_rankings("rules", {}, _judgments())
    assert result.evaluated_queries == 0
    assert result.precision_at_10 == 0.0
    assert result.ndcg_at_10 == 0.0
    assert result.mrr == 0.0


def test_no_relevant_documents():
    judgments = [QueryJudgment(query="q1", judgments={"a/b": 0, "c/d": 0})]
    rankings = {"q1": ["a/b", "c/d"]}
    result = evaluate_rankings("rules", rankings, judgments)
    assert result.precision_at_10 == 0.0
    assert result.ndcg_at_10 == 0.0
    assert result.mrr == 0.0


def test_k_larger_than_result_count():
    rankings = {"q1": ["a/b", "c/d"]}
    result = evaluate_rankings("rules", rankings, _judgments(), k=10)
    assert result.precision_at_10 == pytest.approx(1.0)
    assert result.ndcg_at_10 == pytest.approx(1.0)


def test_unjudged_query_skipped():
    rankings = {"q1": ["a/b"], "unjudged_query": ["x/y"]}
    result = evaluate_rankings("rules", rankings, _judgments())
    assert result.evaluated_queries == 1
    assert result.precision_at_10 == pytest.approx(1.0)