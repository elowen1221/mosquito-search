"""/discover 接口的请求与响应模型。"""

from typing import Literal

from pydantic import BaseModel, Field

from mosquito.models.ranking import QualityRankedRepo, RankedRepo

RankingMode = Literal["rules", "embedding", "quality", "bert"]


class DiscoverRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="自然语言搜索描述")
    limit: int = Field(default=10, ge=1, le=50, description="返回的候选数量上限")
    ranking: RankingMode = Field(
        default="rules", description="排序方式：rules 规则评分 / embedding 向量相似度 / quality 质量感知"
    )
    compare: bool = Field(default=False, description="同时返回所有排序结果用于对比")


class ComparisonResult(BaseModel):
    rules: list[RankedRepo]
    embedding: list[RankedRepo]
    quality: list[QualityRankedRepo]


class DiscoverResponse(BaseModel):
    query: str
    total: int
    # QualityRankedRepo 在前，保证 quality 模式的扩展字段在序列化时保留
    results: list[QualityRankedRepo | RankedRepo]
    comparison: ComparisonResult | None = None