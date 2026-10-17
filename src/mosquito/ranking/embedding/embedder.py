"""Embedding 编码器：懒加载模型 + 内存缓存（text hash key + LRU）+ 批量编码。"""

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from mosquito.config import settings
from mosquito.models.repo import RepoCandidate
from mosquito.ranking.embedding.text import build_repo_text, normalize, text_hash

DEFAULT_CACHE_SIZE = 1000


class EmbeddingError(Exception):
    """Embedding 模型加载或编码失败。"""


class Embedder(ABC):
    """Embedding 编码抽象，便于测试注入假实现。"""

    @abstractmethod
    def embed_repos(self, repos: list[RepoCandidate]) -> list[np.ndarray]:
        """批量编码仓库文本，按文本哈希缓存。"""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """编码查询文本，按 query 字符串缓存。"""


class FastEmbedEmbedder(Embedder):
    """基于 fastembed（ONNX）的编码器，模型首次使用时懒加载。"""

    _instance: "FastEmbedEmbedder | None" = None

    def __init__(self, model_name: str | None = None, cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._repo_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    @classmethod
    def get_instance(cls) -> "FastEmbedEmbedder":
        """进程内单例，避免重复加载模型。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        except Exception as exc:
            raise EmbeddingError(f"加载 embedding 模型失败（{self.model_name}）: {exc}") from exc

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        self._load()
        try:
            return list(self._model.embed(texts))
        except Exception as exc:
            raise EmbeddingError(f"embedding 编码失败: {exc}") from exc

    @staticmethod
    def _get(cache: OrderedDict, key: str) -> np.ndarray | None:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _put(self, cache: OrderedDict, key: str, value: np.ndarray) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._cache_size:
            cache.popitem(last=False)

    def embed_repos(self, repos: list[RepoCandidate]) -> list[np.ndarray]:
        keys = [text_hash(build_repo_text(r)) for r in repos]
        missing_indices = [i for i, key in enumerate(keys) if key not in self._repo_cache]
        if missing_indices:
            missing_texts = [build_repo_text(repos[i]) for i in missing_indices]
            vectors = self._encode(missing_texts)
            for i, vec in zip(missing_indices, vectors):
                self._put(self._repo_cache, keys[i], vec)
        return [self._get(self._repo_cache, key) for key in keys]

    def embed_query(self, query: str) -> np.ndarray:
        key = normalize(query)
        cached = self._get(self._query_cache, key)
        if cached is None:
            cached = self._encode([key])[0]
            self._put(self._query_cache, key, cached)
        return cached