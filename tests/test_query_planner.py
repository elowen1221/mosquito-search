"""QueryPlanner 单元测试：确定性 / 英文简化 / 中文规划 / 数量与去重约束。"""

import pytest

from mosquito.query_planning.planner import QueryPlanner


@pytest.fixture
def planner() -> QueryPlanner:
    return QueryPlanner()


def test_deterministic_output(planner):
    query = "lightweight python API framework with async support"
    assert planner.plan(query) == planner.plan(query)


def test_english_long_query_simplification(planner):
    planned = planner.plan("lightweight python API framework with async support")
    assert planned
    # 填充词被移除
    for q in planned:
        assert "with" not in q.split()
        assert "support" not in q.split()
    # 技术术语保留
    assert any("python" in q for q in planned)
    assert any("async" in q for q in planned)
    assert any("framework" in q for q in planned)


def test_english_short_query_unchanged(planner):
    assert planner.plan("python web framework") == ["python web framework"]


def test_chinese_query_planning(planner):
    planned = planner.plan("适合本地运行的轻量级 Python 向量数据库")
    assert planned
    assert any("vector database" in q for q in planned)
    assert any("python" in q for q in planned)


def test_chinese_no_overlapping_term_duplication(planner):
    # "向量数据库" 匹配后，"向量"/"数据库" 不应再重复匹配
    planned = planner.plan("适合本地运行的轻量级 Python 向量数据库")
    for q in planned:
        assert q.count("vector database") <= 1
        assert "vector database vector" not in q
        assert "vector database database" not in q


def test_chinese_memory_query(planner):
    planned = planner.plan("我想给个人 AI 助手加一个长期记忆系统，不想部署很重的服务")
    assert planned
    assert any("memory" in q for q in planned)
    assert any("ai" in q for q in planned)


def test_max_four_queries(planner):
    for q in [
        "lightweight python API framework with async support",
        "适合本地运行的轻量级 Python 向量数据库",
        "我想给个人 AI 助手加一个长期记忆系统，不想部署很重的服务",
        "python web framework",
    ]:
        assert 1 <= len(planner.plan(q)) <= 4


def test_no_duplicate_planned_queries(planner):
    for q in [
        "lightweight python API framework with async support",
        "适合本地运行的轻量级 Python 向量数据库",
        "我想给个人 AI 助手加一个长期记忆系统，不想部署很重的服务",
    ]:
        planned = planner.plan(q)
        assert len(planned) == len(set(planned))


def test_empty_query(planner):
    assert planner.plan("") == []
    assert planner.plan("   ") == []