"""QualityScorer 单元测试：各特征归一化边界与权重组合。"""

import math

import pytest

from mosquito.ranking.quality.scorer import W_POPULARITY, QualityBreakdown, QualityScorer


class TestPopularity:
    def test_zero(self, make_candidate):
        assert QualityScorer()._popularity(make_candidate("a", stars=0, forks=0)) == 0.0

    def test_cap(self, make_candidate):
        assert QualityScorer()._popularity(make_candidate("a", stars=100_000, forks=100_000)) == 1.0

    def test_log_scale(self, make_candidate):
        # 1000 stars → stars 分量 log10(1001)/log10(100001) ≈ 0.60，popularity = 0.7·0.60 ≈ 0.42
        p = QualityScorer()._popularity(make_candidate("a", stars=1000, forks=0))
        assert 0.3 < p < 0.5


class TestMaintenance:
    def test_recent(self, make_candidate):
        assert QualityScorer()._maintenance(make_candidate("a", pushed_days_ago=0)) == pytest.approx(1.0)

    def test_decay_at_tau(self, make_candidate):
        assert QualityScorer()._maintenance(make_candidate("a", pushed_days_ago=180)) == pytest.approx(
            math.exp(-1.0), abs=0.01
        )

    def test_missing_pushed_at(self, make_candidate):
        repo = make_candidate("a")
        repo.pushed_at = None
        assert QualityScorer()._maintenance(repo) == 0.0


class TestScale:
    def test_moderate(self, make_candidate):
        assert QualityScorer()._scale(make_candidate("a", size=5000)) == 1.0

    def test_too_small(self, make_candidate):
        assert QualityScorer()._scale(make_candidate("a", size=50)) == 0.5

    def test_too_large(self, make_candidate):
        assert QualityScorer()._scale(make_candidate("a", size=400_000)) == 0.5


class TestMetadata:
    def test_full(self, make_candidate):
        repo = make_candidate("a", description="desc", topics=["a", "b", "c", "d", "e"])
        assert QualityScorer()._metadata(repo) == 1.0

    def test_empty(self, make_candidate):
        repo = make_candidate("a", description=None, topics=[])
        assert QualityScorer()._metadata(repo) == 0.0


class TestTotal:
    def test_perfect_repo(self, make_candidate):
        repo = make_candidate(
            "a",
            stars=100_000,
            forks=100_000,
            pushed_days_ago=0,
            size=5000,
            description="d",
            topics=["a", "b", "c", "d", "e"],
        )
        breakdown = QualityScorer().score(repo)
        assert isinstance(breakdown, QualityBreakdown)
        assert breakdown.total == pytest.approx(1.0)

    def test_bounded(self, make_candidate):
        repo = make_candidate(
            "a", stars=0, forks=0, pushed_days_ago=10_000, size=0, description=None, topics=[]
        )
        breakdown = QualityScorer().score(repo)
        assert 0.0 <= breakdown.total <= 1.0

    def test_stars_contribution_capped(self, make_candidate):
        # 即使 stars 满分，popularity 对 total 的贡献上限也只有 W_POPULARITY=0.3
        repo = make_candidate("a", stars=100_000, forks=100_000)
        breakdown = QualityScorer().score(repo)
        assert breakdown.popularity == 1.0
        assert W_POPULARITY * breakdown.popularity == pytest.approx(0.3)