"""人工标注工作流测试：标签解析 / unsure / 恢复 / 确定性顺序 / 重复检测 / 指标集成。"""

import pytest

from mosquito.evaluation.annotation import (
    ManualAnnotation,
    load_annotations,
    parse_label,
    prepare_annotations,
    review_order,
    review_tier,
    save_annotations,
    unlabeled_count,
    validate_annotations,
)
from mosquito.evaluation.evaluator import evaluate_annotations


def make_annotation(query, full_name, judgment=None, retrieved_by=None, ranks=None, note=None):
    return ManualAnnotation(
        query=query,
        full_name=full_name,
        description=f"{full_name} desc",
        language="Python",
        topics=["python"],
        retrieved_by=retrieved_by or ["rules"],
        ranks=ranks or {"rules": 1},
        judgment=judgment,
        note=note,
    )


# ---- 标签解析 ----

def test_parse_label_valid():
    assert parse_label("2") == 2
    assert parse_label("1") == 1
    assert parse_label("0") == 0
    assert parse_label("u") == "unsure"
    assert parse_label("U") == "unsure"
    assert parse_label("unsure") == "unsure"
    assert parse_label("s") is None
    assert parse_label("skip") is None


def test_parse_label_invalid():
    with pytest.raises(ValueError):
        parse_label("x")
    with pytest.raises(ValueError):
        parse_label("3")


# ---- unsure 与未标注行为 ----

def test_unsure_excluded_from_metrics():
    annotations = [
        make_annotation("q1", "a", judgment=2),
        make_annotation("q1", "b", judgment="unsure"),
        make_annotation("q1", "c", judgment=None),
    ]
    rankings = {"q1": ["a", "b", "c"]}
    result = evaluate_annotations(rankings, annotations, min_labeled=1)
    report = result.per_query[0]
    assert report.judged_count == 1  # 只有 a 计入
    assert report.relevant_count == 1
    assert report.precision_at_10 == 1.0


def test_unjudged_not_irrelevant():
    # 未标注仓库排在前面，不应拉低 P@10
    annotations = [
        make_annotation("q1", "a", judgment=2),
        make_annotation("q1", "b", judgment=0),
    ]
    rankings = {"q1": ["unjudged_x", "unjudged_y", "a", "b"]}
    result = evaluate_annotations(rankings, annotations, min_labeled=1)
    report = result.per_query[0]
    assert report.precision_at_10 == 0.5  # 基于过滤后的 [a, b]
    assert report.retrieved_count == 4
    assert report.coverage == 0.5  # 2/4 已标注


def test_unlabeled_count_includes_unsure():
    records = [
        make_annotation("q1", "a", judgment=2),
        make_annotation("q1", "b"),
        make_annotation("q1", "c", judgment="unsure"),
    ]
    assert unlabeled_count(records) == 2


# ---- 持久化与恢复 ----

def test_resume_behavior(tmp_path):
    path = tmp_path / "manual.jsonl"
    records = [
        make_annotation("q1", "a", judgment=2, note="good"),
        make_annotation("q1", "b"),
    ]
    save_annotations(path, records)
    loaded = load_annotations(path)
    assert len(loaded) == 2
    assert loaded[0].judgment == 2
    assert loaded[0].note == "good"
    assert loaded[1].judgment is None


def test_load_missing_file_returns_empty(tmp_path):
    assert load_annotations(tmp_path / "nope.jsonl") == []


# ---- 确定性审阅顺序 ----

def test_deterministic_ordering():
    records = [make_annotation("q1", f"repo{i}", ranks={"rules": i}) for i in range(1, 6)]
    assert review_order(records, 3) == review_order(records, 3)


def test_review_order_prioritizes_top_rank():
    records = [
        make_annotation("q1", "tail", ranks={"rules": 9}),
        make_annotation("q1", "top", ranks={"rules": 1}),
    ]
    order = review_order(records, 2)
    assert records[order[0]].full_name == "top"


def test_review_order_query_coverage():
    records = [
        make_annotation("q1", "a", ranks={"rules": 1}),
        make_annotation("q1", "b", ranks={"rules": 2}),
        make_annotation("q2", "c", ranks={"rules": 1}),
    ]
    order = review_order(records, 2)
    assert {records[i].query for i in order[:2]} == {"q1", "q2"}


def test_review_tier_classification():
    top = make_annotation("q1", "a", ranks={"rules": 1})
    assert review_tier(top) == "top_rank"
    disagree = make_annotation(
        "q1", "b", retrieved_by=["rules", "embedding"], ranks={"rules": 4, "embedding": 9}
    )
    assert review_tier(disagree) == "disagreement"
    multi = make_annotation(
        "q1", "c", retrieved_by=["rules", "embedding"], ranks={"rules": 4, "embedding": 5}
    )
    assert review_tier(multi) == "multi_variant"
    tail = make_annotation("q1", "d", ranks={"rules": 9})
    assert review_tier(tail) == "tail"


# ---- 校验 ----

def test_duplicate_and_conflicting_detection():
    records = [
        make_annotation("q1", "a", judgment=2),
        make_annotation("q1", "a", judgment=2),  # 重复
        make_annotation("q1", "b", judgment=1),
        make_annotation("q1", "b", judgment=0),  # 冲突
    ]
    issues = validate_annotations(records)
    types = [i.issue_type for i in issues]
    assert "duplicate" in types
    assert "conflicting" in types


def test_invalid_label_detection():
    records = [make_annotation("q1", "a", judgment=5)]
    issues = validate_annotations(records)
    assert any(i.issue_type == "invalid_label" for i in issues)


def test_missing_field_detection():
    records = [ManualAnnotation(query="", full_name="")]
    issues = validate_annotations(records)
    assert sum(1 for i in issues if i.issue_type == "missing_field") == 2


# ---- 指标集成 ----

def test_per_query_and_macro_reporting():
    annotations = [
        make_annotation("q1", "a", judgment=2),
        make_annotation("q1", "b", judgment=1),
        make_annotation("q1", "c", judgment=0),
        make_annotation("q2", "d", judgment=2),
        make_annotation("q2", "e", judgment=0),
        make_annotation("q3", "f", judgment=1),  # 只有 1 条，低于 min_labeled
    ]
    rankings = {"q1": ["a", "b", "c"], "q2": ["d", "e"], "q3": ["f"]}
    result = evaluate_annotations(rankings, annotations, min_labeled=2)
    assert result.evaluated_queries == 2  # q1, q2
    assert result.excluded_queries == ["q3"]
    q1 = next(r for r in result.per_query if r.query == "q1")
    assert q1.judged_count == 3
    assert q1.relevant_count == 2
    assert q1.highly_relevant_count == 1
    assert q1.coverage == 1.0
    assert q1.precision_at_10 == pytest.approx(2 / 3)


def test_prepare_annotations_from_pool():
    from mosquito.evaluation.pool import PoolAnnotation

    pool = [
        PoolAnnotation(
            query="q1",
            full_name="owner/a",
            description="desc",
            language="Python",
            topics=["x"],
            retrieved_by=["rules"],
            ranks={"rules": 1},
        )
    ]
    annotations = prepare_annotations(pool)
    assert len(annotations) == 1
    assert annotations[0].judgment is None
    assert annotations[0].note is None
    # 不含任何分数字段
    assert "score" not in annotations[0].model_dump()