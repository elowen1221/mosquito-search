"""mosquito FastAPI 入口。"""

from fastapi import Depends, FastAPI

from mosquito.models.discover import DiscoverRequest, DiscoverResponse
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.ranker import QualityRanker
from mosquito.ranking.ranker import Ranker
from mosquito.services.discover import DiscoverService
from mosquito.sources.github import GitHubSource

app = FastAPI(title="mosquito", version="0.1.0", description="GitHub 仓库发现 MVP")


def get_discover_service() -> DiscoverService:
    # 排序逻辑在各自 Ranker 内，route 只做编排调用
    return DiscoverService(
        source=GitHubSource(),
        rule_ranker=Ranker(),
        embedding_ranker=EmbeddingRanker(),
        quality_ranker=QualityRanker(),
    )


@app.post("/discover", response_model=DiscoverResponse)
async def discover(
    request: DiscoverRequest,
    service: DiscoverService = Depends(get_discover_service),
) -> DiscoverResponse:
    return await service.discover(request.query, request.limit, request.ranking, request.compare)