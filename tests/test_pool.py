"""候选池生成单元测试：并集/去重/provenance/确定性打乱。"""

import json

import pytest

from mosquito.evaluation.pool import (
    PoolError,
    build_pool,
    load_queries,
    to_annotation_records,
    union_candidates,
    write_annotation_file,
)
from mosquito.models.ranking import RankedRepo


class RecordingRanker:
    """记录收到的候选集，按预设顺序返回排序结果。"""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.received: list | None = None

    def rank(self, candidates, query):
        self.received = candidates
        by_name = {c.full_name: c for c in candidates}
        return [RankedRepo(candidate=by_name[n], score=1.0) for n in self.order if n in by_name]


def test_union_and_deduplication_with_provenance(make_candidate):
    candidates = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
    rankers = {
        "rules": RecordingRanker(["owner/a", "owner/b"]),
        "embedding": RecordingRanker(["owner/b", "owner/c"]),
    }
    pooled = build_pool(candidates, rankers, "q", pool_depth=10)
    names = [p.candidate.full_name for p in pooled]
    # 并集去重，按 full_name 排序
    assert names == ["owner/a", "owner/b", "owner/c"]
    assert len(names) == len(set(names))
    # 重复仓库保留两个来源与各自排名
    b = next(p for p in pooled if p.candidate.full_name == "owner/b")
    assert b.ranks == {"rules": 2, "embedding": 1}


def test_provenance_ranks(make_candidate):
    candidates = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
    rankers = {
        "rules": RecordingRanker(["owner/a", "owner/b", "owner/c"]),
        "embedding": RecordingRanker(["owner/c", "owner/a"]),
    }
    pooled = build_pool(candidates, rankers, "q", pool_depth=10)
    by_name = {p.candidate.full_name: p for p in pooled}
    assert by_name["owner/a"].ranks == {"rules": 1, "embedding": 2}
    assert by_name["owner/c"].ranks == {"rules": 3, "embedding": 1}


def test_same_candidate_set_reused_across_rankers(make_candidate):
    candidates = [make_candidate("a"), make_candidate("b")]
    r1 = RecordingRanker(["owner/a"])
    r2 = RecordingRanker(["owner/b"])
    build_pool(candidates, {"rules": r1, "embedding": r2}, "q", pool_depth=10)
    # 所有 ranker 收到同一候选集对象，而非各自独立获取
    assert r1.received is candidates
    assert r2.received is candidates


def test_deterministic_shuffle(make_candidate):
    candidates = [make_candidate(f"repo-{i}") for i in range(10)]
    rankers = {"rules": RecordingRanker([f"owner/repo-{i}" for i in range(10)])}
    first = to_annotation_records("q", build_pool(candidates, rankers, "q", 10), seed=42)
    second = to_annotation_records("q", build_pool(candidates, rankers, "q", 10), seed=42)
    assert [r.full_name for r in first] == [r.full_name for r in second]


def test_shuffle_breaks_ranker_order(make_candidate):
    # 打乱后不应保持 ranker 原始顺序（降低标注偏差）
    candidates = [make_candidate(f"repo-{i}") for i in range(10)]
    rankers = {"rules": RecordingRanker([f"owner/repo-{i}" for i in range(10)])}
    records = to_annotation_records("q", build_pool(candidates, rankers, "q", 10), seed=42)
    names = [r.full_name for r in records]
    assert names != sorted(names)


def test_annotation_records_no_scores(make_candidate):
    candidates = [make_candidate("a", description="desc", language="Python", topics=["cli"])]
    rankers = {"rules": RecordingRanker(["owner/a"])}
    records = to_annotation_records("q", build_pool(candidates, rankers, "q", 10))
    assert len(records) == 1
    rec = records[0]
    # 只允许 query / 仓库事实 / provenance / judgment 字段，无任何分数
    assert set(rec.model_dump().keys()) == {
        "query",
        "full_name",
        "description",
        "language",
        "topics",
        "retrieved_by",
        "ranks",
        "judgment",
    }
    assert rec.judgment is None
    assert rec.retrieved_by == ["rules"]
    assert rec.ranks == {"rules": 1}


def test_union_candidates_deduplicates_by_full_name(make_candidate):
    a1 = make_candidate("a")
    a2 = make_candidate("a")  # 同 full_name，不同对象
    b = make_candidate("b")
    merged = union_candidates([[a1, b], [a2]])
    assert [c.full_name for c in merged] == ["owner/a", "owner/b"]
    assert merged[0] is a1  # 保留首次出现的候选


def test_union_candidates_empty(make_candidate):
    assert union_candidates([]) == []
    assert union_candidates([[make_candidate("a")], []]) == [make_candidate("a")]


def test_original_query_preserved_in_annotations(make_candidate):
    # 候选来自规划查询，但标注记录必须保留原始用户 query
    candidates = [make_candidate("a")]
    rankers = {"rules": RecordingRanker(["owner/a"])}
    records = to_annotation_records("原始查询", build_pool(candidates, rankers, "原始查询", 10))
    assert all(r.query == "原始查询" for r in records)


def test_pool_depth_behavior(make_candidate):
    candidates = [make_candidate(f"repo-{i}") for i in range(10)]
    rankers = {
        "rules": RecordingRanker([f"owner/repo-{i}" for i in range(5)]),
        "embedding": RecordingRanker([f"owner/repo-{i}" for i in range(5, 10)]),
    }
    pooled = build_pool(candidates, rankers, "q", pool_depth=2)
    # 每个 ranker 只取 Top-2，无重叠 → 并集 4
    assert len(pooled) == 4


def test_pool_depth_zero(make_candidate):
    candidates = [make_candidate("a")]
    pooled = build_pool(candidates, {"rules": RecordingRanker(["owner/a"])}, "q", pool_depth=0)
    assert pooled == []


def test_load_queries(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"query": "python"}\n{"query": "rust"}\n', encoding="utf-8")
    assert load_queries(path) == ["python", "rust"]


def test_load_queries_missing_query_field(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"foo": "bar"}\n', encoding="utf-8")
    with pytest.raises(PoolError, match="query"):
        load_queries(path)


def test_load_queries_missing_file(tmp_path):
    with pytest.raises(PoolError, match="不存在"):
        load_queries(tmp_path / "nope.jsonl")


def test_write_annotation_file(tmp_path, make_candidate):
    out = tmp_path / "pool.jsonl"
    candidates = [make_candidate("a")]
    records = to_annotation_records("q", build_pool(candidates, {"rules": RecordingRanker(["owner/a"])}, "q", 10))
    write_annotation_file(out, records)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["query"] == "q"
    assert data["judgment"] is None
    assert data["retrieved_by"] == ["rules"]