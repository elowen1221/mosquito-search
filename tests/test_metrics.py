"""评测指标纯函数单元测试。"""

import math

import pytest

from mosquito.evaluation.metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_partial(self):
        assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5

    def test_irrelevant(self):
        assert precision_at_k(["a", "b"], {"x", "y"}, k=2) == 0.0

    def test_empty_ranking(self):
        assert precision_at_k([], {"a"}, k=10) == 0.0

    def test_no_relevant(self):
        assert precision_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_larger_than_results(self):
        # 只有 3 个结果，k=10 → 分母取 min(10, 3)=3
        assert precision_at_k(["a", "b", "c"], {"a", "b"}, k=10) == pytest.approx(2 / 3)

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0


class TestDcgAtK:
    def test_graded(self):
        dcg = dcg_at_k(["a", "b", "c"], {"a": 2, "b": 1, "c": 0}, k=3)
        assert dcg == pytest.approx(2 / math.log2(2) + 1 / math.log2(3))

    def test_unjudged_ignored(self):
        assert dcg_at_k(["a", "x"], {"a": 2}, k=2) == pytest.approx(2 / math.log2(2))

    def test_empty(self):
        assert dcg_at_k([], {"a": 2}, k=10) == 0.0


class TestNdcgAtK:
    def test_perfect(self):
        assert ndcg_at_k(["a", "b", "c"], {"a": 2, "b": 1, "c": 0}, k=3) == pytest.approx(1.0)

    def test_reversed(self):
        ndcg = ndcg_at_k(["c", "b", "a"], {"a": 2, "b": 1, "c": 0}, k=3)
        assert 0.0 < ndcg < 1.0

    def test_no_relevant(self):
        assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, k=2) == 0.0

    def test_empty(self):
        assert ndcg_at_k([], {"a": 2}, k=10) == 0.0

    def test_k_larger_than_results(self):
        assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, k=10) == pytest.approx(1.0)


class TestReciprocalRank:
    def test_first(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_none(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty(self):
        assert reciprocal_rank([], {"a"}) == 0.0


class TestMeanReciprocalRank:
    def test_mean(self):
        rankings = [["a", "b"], ["x", "a"]]
        relevant = [{"a"}, {"a"}]
        assert mean_reciprocal_rank(rankings, relevant) == pytest.approx((1.0 + 0.5) / 2)

    def test_empty(self):
        assert mean_reciprocal_rank([], []) == 0.0