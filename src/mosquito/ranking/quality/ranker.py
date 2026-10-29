"""QualityRanker：组合 semantic relevance（embedding 余弦）与 repository quality。

只负责组合与排序，不实现任何特征归一化逻辑（归一化在 QualityScorer）。
"""

from mosquito.models.ranking import QualityRankedRepo
from mosquito.models.repo import RepoCandidate
from mosquito.ranking.embedding.embedder import Embedder
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.scorer import QualityScorer

W_RELEVANCE = 0.6
W_QUALITY = 0.4


class QualityRanker:
    """final_score = 0.6·relevance + 0.4·quality，按 final_score 降序。

    evaluation_time=None 时 maintenance 用当前时间；传入固定时间则 benchmark 可 deterministic。
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        scorer: QualityScorer | None = None,
        evaluation_time=None,
    ) -> None:
        self.relevance_ranker = EmbeddingRanker(embedder=embedder)
        self.scorer = scorer or QualityScorer(evaluation_time=evaluation_time)

    def rank(self, candidates: list[RepoCandidate], query: str) -> list[QualityRankedRepo]:
        if not candidates:
            return []

        # relevance 复用 EmbeddingRanker 的余弦相似度
        relevance_map = {
            r.candidate.full_name: r.score for r in self.relevance_ranker.rank(candidates, query)
        }

        results: list[QualityRankedRepo] = []
        for repo in candidates:
            relevance = relevance_map[repo.full_name]
            quality = self.scorer.score(repo)
            final = round(W_RELEVANCE * relevance + W_QUALITY * quality.total, 4)
            results.append(
                QualityRankedRepo(
                    candidate=repo,
                    score=final,
                    relevance_score=relevance,
                    quality_score=quality.total,
                    final_score=final,
                    reasons=[
                        f"relevance: {relevance:.4f}",
                        (
                            f"quality: {quality.total:.4f} "
                            f"(流行度 {quality.popularity:.2f}, 维护 {quality.maintenance:.2f}, "
                            f"规模 {quality.scale:.2f}, 元数据 {quality.metadata:.2f})"
                        ),
                    ],
                )
            )
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results