"""Ranker：聚合规则计算总分并排序。"""

from mosquito.models.ranking import RankedRepo
from mosquito.models.repo import RepoCandidate
from mosquito.ranking.rules import DEFAULT_RULES, Rule


class Ranker:
    """对候选仓库应用规则评分，返回按 score 降序的 RankedRepo 列表。"""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules if rules is not None else DEFAULT_RULES

    def rank(self, candidates: list[RepoCandidate], query: str) -> list[RankedRepo]:
        ranked: list[RankedRepo] = []
        for candidate in candidates:
            total = 0.0
            reasons: list[str] = []
            for rule in self.rules:
                result = rule(candidate, query)
                total += result.score
                if result.score > 0:
                    reasons.append(result.reason)
            ranked.append(RankedRepo(candidate=candidate, score=round(total, 2), reasons=reasons))
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked