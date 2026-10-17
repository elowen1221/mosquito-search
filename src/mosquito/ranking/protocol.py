"""Ranker 协议：规则排序与 embedding 排序共用同一接口，可并存、独立调用、直接比较。"""

from typing import Protocol

from mosquito.models.ranking import RankedRepo
from mosquito.models.repo import RepoCandidate


class Ranker(Protocol):
    def rank(self, candidates: list[RepoCandidate], query: str) -> list[RankedRepo]:
        """对候选仓库排序，返回按 score 降序的 RankedRepo 列表。"""
        ...