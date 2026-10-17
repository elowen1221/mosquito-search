"""DiscoveryService 测试：缓存命中 / 空结果缓存 / 失败分类 / 有界重试 / 覆盖率汇总。"""

import pytest

from mosquito.evaluation.discovery import (
    DiscoveryCache,
    DiscoveryOutcome,
    DiscoveryService,
    DiscoveryStatus,
    aggregate_query_outcome,
    summarize_coverage,
)
from mosquito.sources.github import (
    GitHubAuthError,
    GitHubError,
    GitHubNetworkError,
    GitHubRateLimitError,
)


class FakeSource:
    """可编程假数据源：按脚本依次返回候选或抛出异常，记录调用次数。"""

    name = "fake"

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls = 0

    async def search(self, query: str, limit: int = 50):
        self.calls += 1
        step = self.script.pop(0) if self.script else []
        if isinstance(step, Exception):
            raise step
        return step


def make_service(source, cache_path, max_attempts=3, base_delay=0.0) -> DiscoveryService:
    return DiscoveryService(
        source, cache=DiscoveryCache(cache_path), max_attempts=max_attempts, base_delay=base_delay
    )


async def test_cache_hit_avoids_source_call(tmp_path, make_candidate):
    source = FakeSource([[make_candidate("a")]])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=2)
    first = await svc.discover("python", limit=50)
    assert first.status == DiscoveryStatus.SUCCESS
    assert source.calls == 1
    second = await svc.discover("python", limit=50)
    assert second.status == DiscoveryStatus.SUCCESS
    assert source.calls == 1  # 缓存命中，不再调用 GitHub


async def test_empty_success_is_cached(tmp_path):
    source = FakeSource([[]])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=2)
    first = await svc.discover("nothing", limit=50)
    assert first.status == DiscoveryStatus.EMPTY
    assert source.calls == 1
    second = await svc.discover("nothing", limit=50)
    assert second.status == DiscoveryStatus.EMPTY
    assert source.calls == 1  # 真空结果也被缓存


async def test_cache_key_includes_query_and_limit(tmp_path, make_candidate):
    source = FakeSource([[make_candidate("a")], [make_candidate("b")]])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=2)
    await svc.discover("python", limit=50)
    await svc.discover("python", limit=10)  # limit 不同 → 缓存未命中
    await svc.discover("rust", limit=50)  # query 不同 → 缓存未命中
    assert source.calls == 3


async def test_cache_persists_across_instances(tmp_path, make_candidate):
    path = tmp_path / "cache.json"
    source = FakeSource([[make_candidate("a")]])
    svc = make_service(source, path, max_attempts=2)
    await svc.discover("python", limit=50)
    svc.cache.save()
    # 新实例加载磁盘缓存
    source2 = FakeSource([])
    svc2 = make_service(source2, path, max_attempts=2)
    outcome = await svc2.discover("python", limit=50)
    assert outcome.status == DiscoveryStatus.SUCCESS
    assert source2.calls == 0  # 命中磁盘缓存


async def test_failure_not_cached(tmp_path):
    source = FakeSource([GitHubRateLimitError("rate limited")])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=1)
    await svc.discover("python", limit=50)
    assert svc.cache.get("fake", "python", 50) is None


async def test_network_failure_not_counted_as_empty(tmp_path):
    source = FakeSource([GitHubNetworkError("connection refused")])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=1)
    outcome = await svc.discover("python", limit=50)
    assert outcome.status == DiscoveryStatus.NETWORK_ERROR
    assert outcome.candidates == []


async def test_rate_limit_failure_not_counted_as_empty(tmp_path):
    source = FakeSource([GitHubRateLimitError("rate limited")])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=1)
    outcome = await svc.discover("python", limit=50)
    assert outcome.status == DiscoveryStatus.RATE_LIMITED


async def test_retry_stops_after_configured_attempts(tmp_path):
    source = FakeSource(
        [GitHubRateLimitError("rl"), GitHubRateLimitError("rl"), GitHubRateLimitError("rl")]
    )
    svc = make_service(source, tmp_path / "cache.json", max_attempts=3)
    outcome = await svc.discover("python", limit=50)
    assert outcome.status == DiscoveryStatus.RATE_LIMITED
    assert source.calls == 3  # 恰好重试 3 次后停止


async def test_retry_succeeds_after_transient_failure(tmp_path, make_candidate):
    source = FakeSource([GitHubRateLimitError("rl"), [make_candidate("a")]])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=3)
    outcome = await svc.discover("python", limit=50)
    assert outcome.status == DiscoveryStatus.SUCCESS
    assert source.calls == 2


async def test_5xx_retried_then_succeeds(tmp_path, make_candidate):
    source = FakeSource([GitHubError("boom", status_code=500), [make_candidate("a")]])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=3)
    outcome = await svc.discover("python", limit=50)
    assert outcome.status == DiscoveryStatus.SUCCESS
    assert source.calls == 2


async def test_4xx_not_retried(tmp_path):
    source = FakeSource([GitHubError("bad request", status_code=422)])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=3)
    outcome = await svc.discover("python", limit=50)
    assert outcome.status == DiscoveryStatus.API_ERROR
    assert source.calls == 1  # 永久客户端错误不重试


async def test_auth_error_not_retried(tmp_path):
    source = FakeSource([GitHubAuthError("bad token")])
    svc = make_service(source, tmp_path / "cache.json", max_attempts=3)
    outcome = await svc.discover("python", limit=50)
    assert outcome.status == DiscoveryStatus.API_ERROR
    assert source.calls == 1


def test_aggregate_query_outcome_union(make_candidate):
    outcomes = [
        DiscoveryOutcome(status=DiscoveryStatus.SUCCESS, candidates=[make_candidate("a")]),
        DiscoveryOutcome(
            status=DiscoveryStatus.SUCCESS,
            candidates=[make_candidate("a"), make_candidate("b")],
        ),
    ]
    agg = aggregate_query_outcome(outcomes)
    assert agg.status == DiscoveryStatus.SUCCESS
    assert [c.full_name for c in agg.candidates] == ["owner/a", "owner/b"]


def test_aggregate_query_outcome_empty_vs_failed():
    empty = aggregate_query_outcome([DiscoveryOutcome(status=DiscoveryStatus.EMPTY)])
    assert empty.status == DiscoveryStatus.EMPTY
    failed = aggregate_query_outcome(
        [DiscoveryOutcome(status=DiscoveryStatus.RATE_LIMITED, error="rl")]
    )
    assert failed.status == DiscoveryStatus.RATE_LIMITED


def test_summarize_coverage_separates_empty_and_failed(make_candidate):
    queries = ["q1", "q2", "q3", "q4"]
    outcomes = {
        "q1": DiscoveryOutcome(status=DiscoveryStatus.SUCCESS, candidates=[make_candidate("a")]),
        "q2": DiscoveryOutcome(status=DiscoveryStatus.EMPTY),
        "q3": DiscoveryOutcome(status=DiscoveryStatus.RATE_LIMITED, error="rl"),
        "q4": DiscoveryOutcome(status=DiscoveryStatus.NETWORK_ERROR, error="net"),
    }
    summary = summarize_coverage(queries, outcomes)
    assert summary.total == 4
    assert summary.evaluated == 2
    assert summary.with_candidates == 1
    assert summary.empty == 1
    assert summary.failed == 2
    assert summary.failure_reasons == {"rate_limited": ["q3"], "network_error": ["q4"]}
    assert summary.avg_candidates == 0.5  # 1 个候选 / 2 个成功评估的查询