"""排序结果模型。"""

from pydantic import BaseModel, Field

from mosquito.models.repo import RepoCandidate


class RankedRepo(BaseModel):
    """携带评分信息的候选仓库，与 RepoCandidate 分离。"""

    candidate: RepoCandidate
    score: float = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)


class QualityRankedRepo(RankedRepo):
    """Quality-aware 排序结果：显式拆分 semantic relevance 与 repository quality。

    继承 RankedRepo 以兼容 Ranker 协议与响应模型，score 即 final_score。
    """

    relevance_score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    final_score: float = Field(ge=0, le=1)