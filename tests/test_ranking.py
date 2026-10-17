"""ranking 规则与 Ranker 的纯单元测试（不依赖网络）。"""

from datetime import datetime, timedelta, timezone

import pytest

from mosquito.models.repo import RepoCandidate
from mosquito.ranking.ranker import Ranker
from mosquito.ranking.rules import language_match, project_size, recency, stars


def make_candidate(**overrides) -> RepoCandidate:
    defaults = {
        "name": "demo",
        "full_name": "owner/demo",
        "url": "https://github.com/owner/demo",
        "language": "Python",
        "stars": 100,
        "forks": 10,
        "size": 5000,
        "pushed_at": datetime.now(timezone.utc) - timedelta(days=10),
    }
    defaults.update(overrides)
    return RepoCandidate(**defaults)


class TestLanguageMatch:
    def test_match(self):
        result = language_match(make_candidate(), "python web framework")
        assert result.score == 20.0
        assert "语言匹配" in result.reason

    def test_no_match(self):
        result = language_match(make_candidate(language="Rust"), "python web framework")
        assert result.score == 0.0
        assert result.reason == ""

    def test_unknown_language_in_query(self):
        result = language_match(make_candidate(), "awesome cli tool")
        assert result.score == 0.0


class TestRecency:
    def test_recent(self):
        result = recency(make_candidate(pushed_at=datetime.now(timezone.utc) - timedelta(days=5)), "")
        assert result.score == 15.0

    def test_medium(self):
        result = recency(make_candidate(pushed_at=datetime.now(timezone.utc) - timedelta(days=60)), "")
        assert result.score == 10.0

    def test_old(self):
        result = recency(make_candidate(pushed_at=datetime.now(timezone.utc) - timedelta(days=800)), "")
        assert result.score == 0.0

    def test_missing_pushed_at(self):
        result = recency(make_candidate(pushed_at=None), "")
        assert result.score == 0.0


class TestStars:
    def test_zero(self):
        assert stars(make_candidate(stars=0), "").score == 0.0

    def test_log_scale(self):
        result = stars(make_candidate(stars=1000), "")
        assert result.score == pytest.approx(15.0, abs=0.1)
        assert "stars" in result.reason


class TestProjectSize:
    def test_moderate(self):
        assert project_size(make_candidate(size=5000), "").score == 5.0

    def test_too_small(self):
        assert project_size(make_candidate(size=10), "").score == 0.0

    def test_too_large(self):
        assert project_size(make_candidate(size=10_000_000), "").score == 0.0


class TestRanker:
    def test_sorts_by_score_desc(self):
        hot = make_candidate(name="hot", stars=100_000, pushed_at=datetime.now(timezone.utc) - timedelta(days=1))
        cold = make_candidate(name="cold", stars=0, pushed_at=datetime.now(timezone.utc) - timedelta(days=1000))
        ranked = Ranker().rank([cold, hot], "python")
        assert [r.candidate.name for r in ranked] == ["hot", "cold"]
        assert ranked[0].score > ranked[1].score

    def test_reasons_present(self):
        ranked = Ranker().rank([make_candidate()], "python")
        assert ranked[0].reasons
        assert all(r.score >= 0 for r in ranked)

    def test_empty_candidates(self):
        assert Ranker().rank([], "python") == []