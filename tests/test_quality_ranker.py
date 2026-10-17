"""QualityRanker 单元测试：relevance 与 quality 显式拆分、final 组合与排序。"""

import numpy as np
import pytest

from mosquito.models.ranking import QualityRankedRepo
from mosquito.ranking.quality.ranker import W_QUALITY, W_RELEVANCE, QualityRanker


def test_final_score_combination(make_candidate, controlled_embedder):
    embedder = controlled_embedder
    embedder.repo_vectors["owner/a"] = np.array([1.0, 0.0])
    embedder.query_vectors["python"] = np.array([1.0, 0.0])
    repo = make_candidate(
        "a",
        stars=100_000,
        forks=100_000,
        pushed_days_ago=0,
        size=5000,
        description="d",
        topics=["a", "b", "c", "d", "e"],
    )

    ranked = QualityRanker(embedder=embedder).rank([repo], "python")
    r = ranked[0]
    assert isinstance(r, QualityRankedRepo)
    assert r.relevance_score == pytest.approx(1.0)
    assert r.quality_score == pytest.approx(1.0)
    assert r.final_score == pytest.approx(W_RELEVANCE * 1.0 + W_QUALITY * 1.0)
    assert r.score == r.final_score


def test_sorts_by_final_score(make_candidate, controlled_embedder):
    embedder = controlled_embedder
    embedder.repo_vectors["owner/a"] = np.array([1.0, 0.0])
    embedder.repo_vectors["owner/b"] = np.array([0.0, 1.0])
    embedder.query_vectors["python"] = np.array([1.0, 0.0])

    # a：高 relevance、低 quality；b：低 relevance、高 quality
    a = make_candidate("a", stars=0, forks=0, pushed_days_ago=10_000, size=0, description=None, topics=[])
    b = make_candidate(
        "b",
        stars=100_000,
        forks=100_000,
        pushed_days_ago=0,
        size=5000,
        description="d",
        topics=["a", "b", "c", "d", "e"],
    )

    ranked = QualityRanker(embedder=embedder).rank([a, b], "python")
    assert [r.candidate.name for r in ranked] == ["a", "b"]
    assert ranked[0].final_score == pytest.approx(W_RELEVANCE * 1.0 + W_QUALITY * 0.0)
    assert ranked[1].final_score == pytest.approx(W_RELEVANCE * 0.0 + W_QUALITY * 1.0)


def test_reasons_include_scores(make_candidate, controlled_embedder):
    ranked = QualityRanker(embedder=controlled_embedder).rank([make_candidate("a")], "python")
    reasons = ranked[0].reasons
    assert any(r.startswith("relevance:") for r in reasons)
    assert any(r.startswith("quality:") for r in reasons)


def test_empty_candidates(controlled_embedder):
    assert QualityRanker(embedder=controlled_embedder).rank([], "python") == []