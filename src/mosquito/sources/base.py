"""数据源抽象基类。"""

from abc import ABC, abstractmethod

from mosquito.models.repo import RepoCandidate


class Source(ABC):
    """数据源最小抽象：仅定义异步 search() 接口，不引入 registry/plugin 等扩展机制。"""

    name: str = "unknown"

    @abstractmethod
    async def search(self, query: str, limit: int = 50) -> list[RepoCandidate]:
        """根据自然语言 query 搜索候选仓库。"""
        raise NotImplementedError