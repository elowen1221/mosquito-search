"""Ranker：聚合规则计算总分并排序。"""

from datetime import datetime, timezone

from mosquito.models.ranking import RankedRepo
from mosquito.models.repo import RepoCandidate
from mosquito.ranking.rules import make_default_rules


class Ranker:
    """对候选仓库应用规则评分，返回按 score 降序的 RankedRepo 列表。

    evaluation_time=None 时 recency 用 datetime.now(timezone.utc)（production 默认），
    传入固定时间则 benchmark 可 deterministic。

    确定性实现：__init__ 时通过 make_default_rules() 把 recency 预先 partial 绑定到
    固定 evaluation_time，rank() 内不再做任何 callable identity check。
    """

    def __init__(self, rules=None, evaluation_time: datetime | None = None) -> None:
        if rules is None:
            self.rules = make_default_rules(evaluation_time)
        else:
            self.rules = rules
        self.evaluation_time = evaluation_time

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
