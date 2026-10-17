"""共享测试 fixtures。"""

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from mosquito.models.repo import RepoCandidate
from mosquito.ranking.embedding.embedder import FastEmbedEmbedder


@pytest.fixture
def make_candidate():
    def _make(
        name: str,
        stars: int = 100,
        forks: int = 10,
        language: str = "Python",
        description: str | None = None,
        topics: list[str] | None = None,
        pushed_days_ago: int = 5,
        size: int = 5000,
    ) -> RepoCandidate:
        return RepoCandidate(
            name=name,
            full_name=f"owner/{name}",
            url=f"https://github.com/owner/{name}",
            description=description,
            language=language,
            stars=stars,
            forks=forks,
            topics=topics or [],
            pushed_at=datetime.now(timezone.utc) - timedelta(days=pushed_days_ago),
            size=size,
        )

    return _make


class ControlledEmbedder:
    """确定性向量编码器：未指定时按文本哈希生成，可手动覆盖以控制相似度。"""

    def __init__(self) -> None:
        self.repo_vectors: dict[str, np.ndarray] = {}
        self.query_vectors: dict[str, np.ndarray] = {}
        self.repo_batch_calls = 0
        self.query_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        digest = hashlib.md5(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
        v = rng.random(8)
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def embed_repos(self, repos):
        missing = [r for r in repos if r.full_name not in self.repo_vectors]
        if missing:
            self.repo_batch_calls += 1
            for r in missing:
                self.repo_vectors[r.full_name] = self._vec(r.full_name)
        return [self.repo_vectors[r.full_name] for r in repos]

    def embed_query(self, query: str) -> np.ndarray:
        if query not in self.query_vectors:
            self.query_calls += 1
            self.query_vectors[query] = self._vec(query)
        return self.query_vectors[query]


@pytest.fixture
def controlled_embedder() -> ControlledEmbedder:
    return ControlledEmbedder()


class CountingEmbedder(FastEmbedEmbedder):
    """统计 _encode 调用次数，用于测试 FastEmbedEmbedder 的缓存逻辑。"""

    def __init__(self) -> None:
        super().__init__()
        self.encode_calls = 0

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        self.encode_calls += 1
        return [np.array([1.0, 0.0]) for _ in texts]


@pytest.fixture
def counting_embedder() -> CountingEmbedder:
    return CountingEmbedder()