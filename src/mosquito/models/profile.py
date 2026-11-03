"""用户画像模型：只描述当前学习情境，不做长期记忆或行为建模。"""

from typing import Literal

from pydantic import BaseModel, Field

SkillLevel = Literal["beginner", "intermediate", "advanced"]
TimeBudget = Literal["weekend", "few_weeks"]


class Profile(BaseModel):
    identity: str = Field(..., min_length=1, max_length=50, description="身份，例如 student / engineer")
    learning_goal: str = Field(..., min_length=1, max_length=100, description="当前学习目标，例如 learn RAG")
    tech_stack: list[str] = Field(
        default_factory=list, max_length=20, description='当前技术栈，例如 ["Python"]'
    )
    skill_level: SkillLevel = Field(
        default="beginner", description="熟练度：beginner / intermediate / advanced"
    )
    time_budget: TimeBudget | None = Field(
        default=None, description="可选时间预算：weekend / few_weeks"
    )
