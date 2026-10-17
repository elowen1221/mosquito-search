"""EmbeddingRanker 单元测试：使用可控向量验证余弦相似度排序。"""

import numpy as np
import pytest

from mosquito.models.ranking import RankedRepo
from mosquito.ranking.embedding.ranker import EmbeddingRanker, cosine_similarity


def test_cosine_similarity():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_ranks_by_cosine_similarity(make_candidate, controlled_embedder):
    embedder = controlled_embedder
    embedder.repo_vectors["owner/a"] = np.array([1.0, 0.0])
    embedder.repo_vectors["owner/b"] = np.array([0.0, 1.0])
    embedder.repo_vectors["owner/c"] = np.array([0.7071, 0.7071])
    embedder.query_vectors["python"] = np.array([1.0, 0.0])

    candidates = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
    ranked = EmbeddingRanker(embedder).rank(candidates, "python")

    assert [r.candidate.name for r in ranked] == ["a", "c", "b"]
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[1].score == pytest.approx(0.7071, abs=1e-3)
    assert ranked[2].score == pytest.approx(0.0)


def test_output_is_ranked_repo_list(make_candidate, controlled_embedder):
    ranked = EmbeddingRanker(controlled_embedder).rank([make_candidate("a")], "python")
    assert isinstance(ranked, list)
    assert isinstance(ranked[0], RankedRepo)
    assert ranked[0].reasons == [f"embedding 相似度: {ranked[0].score:.4f}"]


def test_score_in_range(make_candidate, controlled_embedder):
    ranked = EmbeddingRanker(controlled_embedder).rank([make_candidate("a")], "python")
    assert 0.0 <= ranked[0].score <= 1.0


def test_empty_candidates(controlled_embedder):
    assert EmbeddingRanker(controlled_embedder).rank([], "python") == []