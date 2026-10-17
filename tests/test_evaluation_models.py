"""评测数据模型单元测试。"""

import pytest
from pydantic import ValidationError

from mosquito.evaluation.models import QueryJudgment, RankerEvaluationResult


def test_query_judgment_valid():
    j = QueryJudgment(query="python", judgments={"a/b": 2, "c/d": 1, "e/f": 0})
    assert j.query == "python"
    assert j.judgments["a/b"] == 2


def test_invalid_judgment_value_above():
    with pytest.raises(ValidationError):
        QueryJudgment(query="python", judgments={"a/b": 3})


def test_invalid_judgment_value_negative():
    with pytest.raises(ValidationError):
        QueryJudgment(query="python", judgments={"a/b": -1})


def test_ranker_evaluation_result():
    r = RankerEvaluationResult(
        ranker_name="rules", precision_at_10=0.5, ndcg_at_10=0.6, mrr=0.7, evaluated_queries=5
    )
    assert r.ranker_name == "rules"
    assert r.evaluated_queries == 5