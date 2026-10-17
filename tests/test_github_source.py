"""GitHubSource 测试：使用 httpx.MockTransport 模拟 GitHub API。"""

import logging

import httpx
import pytest

from mosquito.sources.github import GitHubAuthError, GitHubError, GitHubRateLimitError, GitHubSource

SAMPLE_ITEM = {
    "name": "demo",
    "full_name": "owner/demo",
    "html_url": "https://github.com/owner/demo",
    "description": "A demo repo",
    "language": "Python",
    "stargazers_count": 1234,
    "forks_count": 56,
    "open_issues_count": 7,
    "topics": ["python", "cli"],
    "updated_at": "2026-08-01T10:00:00Z",
    "pushed_at": "2026-08-15T10:00:00Z",
    "size": 2048,
}


def make_source(handler, token: str | None = "test-token") -> GitHubSource:
    return GitHubSource(token=token, transport=httpx.MockTransport(handler))


async def test_search_normalizes_items():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.url.params["q"] == "python cli"
        return httpx.Response(200, json={"items": [SAMPLE_ITEM]})

    candidates = await make_source(handler).search("python cli")
    assert len(candidates) == 1
    repo = candidates[0]
    assert repo.name == "demo"
    assert repo.full_name == "owner/demo"
    assert repo.url == "https://github.com/owner/demo"
    assert repo.description == "A demo repo"
    assert repo.language == "Python"
    assert repo.stars == 1234
    assert repo.forks == 56
    assert repo.open_issues == 7
    assert repo.topics == ["python", "cli"]
    assert repo.size == 2048
    assert repo.pushed_at is not None


async def test_search_empty_items():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    assert await make_source(handler).search("nothing") == []


async def test_search_rate_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "rate limit exceeded"})

    with pytest.raises(GitHubRateLimitError):
        await make_source(handler).search("python")


async def test_search_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "bad credentials"})

    with pytest.raises(GitHubAuthError):
        await make_source(handler).search("python")


async def test_search_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    with pytest.raises(GitHubError):
        await make_source(handler).search("python")


async def test_search_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(GitHubError):
        await make_source(handler).search("python")


async def test_anonymous_no_auth_header_and_warning(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"items": []})

    with caplog.at_level(logging.WARNING):
        source = make_source(handler, token=None)
        assert "GITHUB_TOKEN 未设置" in caplog.text
        await source.search("python")