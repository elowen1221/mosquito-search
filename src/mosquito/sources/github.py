"""GitHub 数据源：封装 GitHub Search API。"""

import logging
from datetime import datetime

import httpx

from mosquito.config import settings
from mosquito.models.repo import RepoCandidate
from mosquito.sources.base import Source

logger = logging.getLogger(__name__)

SEARCH_REPOS_URL = "https://api.github.com/search/repositories"


class GitHubError(Exception):
    """GitHub API 调用失败。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubRateLimitError(GitHubError):
    """GitHub API 限流（403/429）。"""


class GitHubAuthError(GitHubError):
    """GitHub API 认证失败（401）。"""


class GitHubNetworkError(GitHubError):
    """GitHub API 网络错误（连接失败/超时等）。"""


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("无法解析 GitHub 时间字段: %s", value)
        return None


class GitHubSource(Source):
    """使用 GitHub Search API 搜索公开仓库，将响应标准化为 RepoCandidate。

    GITHUB_TOKEN 可选：存在时携带 Authorization 头，缺失时匿名访问并输出 warning。
    """

    name = "github"

    def __init__(
        self,
        token: str | None = None,
        per_page: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.token = token if token is not None else settings.github_token
        self.per_page = per_page or settings.github_search_per_page
        self._transport = transport
        if not self.token:
            logger.warning("GITHUB_TOKEN 未设置，将以匿名方式访问 GitHub API（存在速率限制）")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def search(self, query: str, limit: int = 50) -> list[RepoCandidate]:
        params = {"q": query, "per_page": min(limit, self.per_page)}
        try:
            async with httpx.AsyncClient(timeout=15, transport=self._transport) as client:
                resp = await client.get(SEARCH_REPOS_URL, params=params, headers=self._headers())
        except httpx.HTTPError as exc:
            raise GitHubNetworkError(f"GitHub API 请求失败: {exc}") from exc

        if resp.status_code == 401:
            raise GitHubAuthError("GitHub API 认证失败，请检查 GITHUB_TOKEN", status_code=resp.status_code)
        if resp.status_code in (403, 429):
            raise GitHubRateLimitError(
                f"GitHub API 限流（HTTP {resp.status_code}），请稍后重试或设置 GITHUB_TOKEN",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            raise GitHubError(
                f"GitHub API 返回错误（HTTP {resp.status_code}）: {resp.text[:200]}",
                status_code=resp.status_code,
            )

        data = resp.json()
        return [self._to_candidate(item) for item in data.get("items", [])]

    @staticmethod
    def _to_candidate(item: dict) -> RepoCandidate:
        return RepoCandidate(
            name=item["name"],
            full_name=item["full_name"],
            url=item["html_url"],
            description=item.get("description"),
            language=item.get("language"),
            stars=item.get("stargazers_count", 0),
            forks=item.get("forks_count", 0),
            open_issues=item.get("open_issues_count", 0),
            topics=item.get("topics", []),
            updated_at=_parse_datetime(item.get("updated_at")),
            pushed_at=_parse_datetime(item.get("pushed_at")),
            size=item.get("size", 0),
        )