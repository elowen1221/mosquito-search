"""两种 Ranker 接口一致性测试：输出结构相同，可并存、独立调用、直接比较。"""

from mosquito.models.ranking import RankedRepo
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.ranker import Ranker


def test_both_rankers_produce_ranked_repo_list(make_candidate, controlled_embedder):
    candidates = [make_candidate("a", stars=500), make_candidate("b", stars=10)]
    rankers = [Ranker(), EmbeddingRanker(controlled_embedder)]

    for ranker in rankers:
        result = ranker.rank(candidates, "python")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, RankedRepo) for r in result)
        assert all(r.score >= 0 for r in result)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)


def test_both_rankers_accept_same_input(make_candidate, controlled_embedder):
    candidates = [make_candidate("a")]
    query = "python web framework"
    rule_result = Ranker().rank(candidates, query)
    embedding_result = EmbeddingRanker(controlled_embedder).rank(candidates, query)
    assert rule_result[0].candidate == embedding_result[0].candidate