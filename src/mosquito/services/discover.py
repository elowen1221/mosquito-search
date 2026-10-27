"""DiscoverService：编排 数据源搜索 → 排序（rules / embedding / quality / bert，可对比）。"""

from fastapi import HTTPException

from mosquito.evaluation.discovery import (
    DiscoveryService as RetrievalService,
    DiscoveryStatus,
    aggregate_query_outcome,
)
from mosquito.models.discover import ComparisonResult, DiscoverResponse, RankingMode
from mosquito.query_planning.planner import QueryPlanner
from mosquito.ranking.protocol import Ranker
from mosquito.sources.base import Source

# 第一版固定获取约 50 个候选，交由本地 Ranker 排序
CANDIDATE_POOL_SIZE = 50


class DiscoverService:
    def __init__(
        self,
        source: Source,
        rule_ranker: Ranker,
        embedding_ranker: Ranker,
        quality_ranker: Ranker,
        bert_ranker: Ranker | None = None,
        planner: QueryPlanner | None = None,
        retrieval: RetrievalService | None = None,
    ) -> None:
        self.source = source
        self.rule_ranker = rule_ranker
        self.embedding_ranker = embedding_ranker
        self.quality_ranker = quality_ranker
        self.bert_ranker = bert_ranker
        self.planner = planner or QueryPlanner()
        self.retrieval = retrieval or RetrievalService(source)

    def _select(self, ranking: RankingMode) -> Ranker:
        if ranking == "rules":
            return self.rule_ranker
        if ranking == "embedding":
            return self.embedding_ranker
        if ranking == "bert":
            if self.bert_ranker is None:
                raise HTTPException(status_code=503, detail="bert ranker 未启用")
            return self.bert_ranker
        return self.quality_ranker

    async def discover(
        self,
        query: str,
        limit: int = 10,
        ranking: RankingMode = "rules",
        compare: bool = False,
    ) -> DiscoverResponse:
        planned = self.planner.plan(query)
        outcomes = [
            await self.retrieval.discover(q, limit=CANDIDATE_POOL_SIZE) for q in planned
        ]
        outcome = aggregate_query_outcome(outcomes)
        if not outcome.is_success:
            status = 429 if outcome.status == DiscoveryStatus.RATE_LIMITED else 502
            raise HTTPException(status_code=status, detail=outcome.error or outcome.status.value)
        candidates = outcome.candidates

        if compare:
            rules = self.rule_ranker.rank(candidates, query)[:limit]
            embedding = self.embedding_ranker.rank(candidates, query)[:limit]
            quality = self.quality_ranker.rank(candidates, query)[:limit]
            bert_ranked = self.bert_ranker.rank(candidates, query)[:limit] if self.bert_ranker else None
            chosen = {
                "rules": rules, "embedding": embedding, "quality": quality,
                "bert": bert_ranked or rules,
            }[ranking]
            return DiscoverResponse(
                query=query,
                total=len(candidates),
                results=chosen,
                comparison=ComparisonResult(rules=rules, embedding=embedding, quality=quality),
            )

        ranked = self._select(ranking).rank(candidates, query)
        return DiscoverResponse(query=query, total=len(ranked), results=ranked[:limit])
