"""仓库标准化数据模型。"""

from datetime import datetime

from pydantic import BaseModel, Field


class RepoCandidate(BaseModel):
    """标准化后的仓库事实数据，不含任何评分信息。"""

    name: str
    full_name: str
    url: str
    description: str | None = None
    language: str | None = None
    stars: int = Field(default=0, ge=0)
    forks: int = Field(default=0, ge=0)
    open_issues: int = Field(default=0, ge=0)
    topics: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    size: int = Field(default=0, ge=0)