"""离线评测数据模型。"""

from pydantic import BaseModel, Field, field_validator

VALID_JUDGMENTS = (0, 1, 2)


class QueryJudgment(BaseModel):
    """单个 query 的人工标注：full_name -> 相关度（0 无关 / 1 相关 / 2 高度相关）。"""

    query: str
    judgments: dict[str, int]

    @field_validator("judgments")
    @classmethod
    def _validate_judgment_values(cls, value: dict[str, int]) -> dict[str, int]:
        invalid = {name: rel for name, rel in value.items() if rel not in VALID_JUDGMENTS}
        if invalid:
            raise ValueError(f"judgment 值必须为 0/1/2，发现非法值: {invalid}")
        return value


class RankerEvaluationResult(BaseModel):
    """单个 ranker 在标注数据集上的评测结果。"""

    ranker_name: str
    precision_at_10: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    evaluated_queries: int = Field(ge=0)