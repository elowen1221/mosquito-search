"""EmbeddingRanker：基于 query 与仓库文本的余弦相似度排序。

与规则 Ranker 实现同一协议（ranking/protocol.py），输出同为 list[RankedRepo]。
"""

import numpy as np

from mosquito.models.ranking import RankedRepo
from mosquito.models.repo import RepoCandidate
from mosquito.ranking.embedding.embedder import Embedder, FastEmbedEmbedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingRanker:
    """对候选仓库按 embedding 余弦相似度排序。"""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or FastEmbedEmbedder.get_instance()

    def rank(self, candidates: list[RepoCandidate], query: str) -> list[RankedRepo]:
        if not candidates:
            return []
        query_vec = self.embedder.embed_query(query)
        repo_vecs = self.embedder.embed_repos(candidates)

        ranked: list[RankedRepo] = []
        for repo, vec in zip(candidates, repo_vecs):
            sim = max(0.0, cosine_similarity(query_vec, vec))
            score = round(sim, 4)
            ranked.append(
                RankedRepo(candidate=repo, score=score, reasons=[f"embedding 相似度: {score:.4f}"])
            )
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked