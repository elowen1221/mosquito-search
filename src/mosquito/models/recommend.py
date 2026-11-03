"""/recommend 接口的请求与响应模型。"""

from pydantic import BaseModel, Field

from mosquito.models.profile import Profile
from mosquito.models.repo import RepoCandidate


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="自然语言学习需求")
    limit: int = Field(default=10, ge=1, le=50, description="返回的推荐数量上限")
    profile: Profile


class RecommendedRepo(BaseModel):
    """单个推荐结果：仓库 + 相关性 + profile fit + 可读解释。"""

    repo: RepoCandidate
    base_score: float = Field(ge=0, le=1, description="现有 ranker 的相关性分数（已归一到 0–1）")
    fit_score: float = Field(ge=0, le=1, description="profile fit 综合分")
    final_score: float = Field(ge=0, description="0.6 × base_score + 0.4 × fit_score")
    fit_breakdown: dict[str, float] = Field(default_factory=dict, description="各维度分数")
    difficulty: str = Field(description="easy / medium / hard")
    recommended_for: list[str] = Field(default_factory=list, description="推荐用途")
    why_recommended: list[str] = Field(default_factory=list, description="模板化推荐理由")


class RecommendResponse(BaseModel):
    query: str
    profile: Profile
    total: int
    results: list[RecommendedRepo]
