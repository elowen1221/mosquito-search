"""RecommendService：检索 → 现有 ranker 相关性 → profile fit → 组合排序。"""

from __future__ import annotations

from fastapi import HTTPException

from mosquito.evaluation.discovery import (
    DiscoveryService as RetrievalService,
    DiscoveryStatus,
    aggregate_query_outcome,
)
from mosquito.models.profile import Profile
from mosquito.models.recommend import RecommendedRepo, RecommendResponse
from mosquito.query_planning.planner import QueryPlanner
from mosquito.ranking.protocol import Ranker
from mosquito.recommendation.fit import assess
from mosquito.sources.base import Source

CANDIDATE_POOL_SIZE = 50
W_BASE = 0.6
W_FIT = 0.4


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """把任意量纲的相关性分数安全归一到 0–1。

    已在 0–1 内则原样返回；否则按最大值缩放（保留候选间的相对差距，
    避免 min-max 把差距拉伸成全 0/全 1）。
    """
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if 0.0 <= low and high <= 1.0:
        return dict(scores)
    shifted = {key: value - low for key, value in scores.items()} if low < 0 else dict(scores)
    peak = max(shifted.values())
    if peak <= 0:
        return {key: 0.0 for key in scores}
    return {key: round(value / peak, 4) for key, value in shifted.items()}


class RecommendService:
    def __init__(
        self,
        source: Source,
        base_ranker: Ranker,
        planner: QueryPlanner | None = None,
        retrieval: RetrievalService | None = None,
    ) -> None:
        self.source = source
        self.base_ranker = base_ranker
        self.planner = planner or QueryPlanner()
        self.retrieval = retrieval or RetrievalService(source)

    async def recommend(
        self, query: str, profile: Profile, limit: int = 10
    ) -> RecommendResponse:
        planned = self.planner.plan(query, profile)
        outcomes = [
            await self.retrieval.discover(q, limit=CANDIDATE_POOL_SIZE) for q in planned
        ]
        outcome = aggregate_query_outcome(outcomes)
        if not outcome.is_success:
            status = 429 if outcome.status == DiscoveryStatus.RATE_LIMITED else 502
            raise HTTPException(status_code=status, detail=outcome.error or outcome.status.value)
        candidates = outcome.candidates

        base_scores = normalize_scores(
            {r.candidate.full_name: r.score for r in self.base_ranker.rank(candidates, query)}
        )

        results: list[RecommendedRepo] = []
        for repo in candidates:
            fit = assess(repo, profile)
            base = base_scores.get(repo.full_name, 0.0)
            results.append(
                RecommendedRepo(
                    repo=repo,
                    base_score=base,
                    fit_score=fit.fit_score,
                    final_score=round(W_BASE * base + W_FIT * fit.fit_score, 4),
                    fit_breakdown={
                        "topic_fit": fit.topic_fit,
                        "stack_fit": fit.stack_fit,
                        "difficulty_fit": fit.difficulty_fit,
                        "learning_value": fit.learning_value,
                    },
                    difficulty=fit.difficulty,
                    recommended_for=fit.recommended_for,
                    why_recommended=fit.why_recommended,
                )
            )
        results.sort(key=lambda r: r.final_score, reverse=True)
        return RecommendResponse(
            query=query, profile=profile, total=len(candidates), results=results[:limit]
        )
