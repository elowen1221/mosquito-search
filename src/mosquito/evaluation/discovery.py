"""评测检索：结构化发现结果 + 本地缓存 + 有界重试。

- DiscoveryOutcome 区分 success / empty / rate_limited / network_error / api_error，
  网络与 API 失败不会被当作零结果查询；
- DiscoveryCache 以 (source, query, limit) 为键做本地确定性缓存，成功与真空结果均缓存；
- DiscoveryService 在缓存未命中时执行有界指数退避重试（限流 / 5xx / 网络错误），
  永久性客户端错误不重试。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

from mosquito.evaluation.pool import union_candidates
from mosquito.models.repo import RepoCandidate
from mosquito.sources.base import Source
from mosquito.sources.github import (
    GitHubAuthError,
    GitHubError,
    GitHubNetworkError,
    GitHubRateLimitError,
)

DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "evaluation" / "discovery_cache.json"
)


class DiscoveryStatus(str, Enum):
    """检索结果状态：成功 / 真空 / 限流 / 网络错误 / API 错误。"""

    SUCCESS = "success"
    EMPTY = "empty"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"


@dataclass
class DiscoveryOutcome:
    """单次检索的结构化结果。"""

    status: DiscoveryStatus
    candidates: list[RepoCandidate] = field(default_factory=list)
    error: str | None = None

    @property
    def is_success(self) -> bool:
        """成功或真空均视为成功评估（可缓存）。"""
        return self.status in (DiscoveryStatus.SUCCESS, DiscoveryStatus.EMPTY)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "candidates": [c.model_dump(mode="json") for c in self.candidates],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiscoveryOutcome":
        return cls(
            status=DiscoveryStatus(data["status"]),
            candidates=[RepoCandidate(**c) for c in data.get("candidates", [])],
            error=data.get("error"),
        )


class DiscoveryCache:
    """本地确定性缓存：key = sha256(source, query, limit)。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, dict] = {}
        if self.path and self.path.exists():
            self._load()

    @staticmethod
    def _key(source: str, query: str, limit: int) -> str:
        raw = f"{source}\0{query}\0{limit}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, source: str, query: str, limit: int) -> DiscoveryOutcome | None:
        entry = self._data.get(self._key(source, query, limit))
        return DiscoveryOutcome.from_dict(entry) if entry else None

    def put(self, source: str, query: str, limit: int, outcome: DiscoveryOutcome) -> None:
        self._data[self._key(source, query, limit)] = outcome.to_dict()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {}


class DiscoveryService:
    """带缓存与有界重试的检索服务，返回结构化结果。"""

    def __init__(
        self,
        source: Source,
        cache: DiscoveryCache | None = None,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.source = source
        self.cache = cache
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def discover(self, query: str, limit: int = 50) -> DiscoveryOutcome:
        """缓存命中直接返回；否则检索并缓存成功/真空结果。"""
        if self.cache is not None:
            cached = self.cache.get(self.source.name, query, limit)
            if cached is not None:
                return cached
        outcome = await self._search_with_retry(query, limit)
        if self.cache is not None and outcome.is_success:
            self.cache.put(self.source.name, query, limit, outcome)
        return outcome

    async def _search_with_retry(self, query: str, limit: int) -> DiscoveryOutcome:
        """有界指数退避重试：限流 / 5xx / 网络错误可重试，永久错误不重试。"""
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                candidates = await self.source.search(query, limit=limit)
                status = DiscoveryStatus.SUCCESS if candidates else DiscoveryStatus.EMPTY
                return DiscoveryOutcome(status=status, candidates=candidates)
            except GitHubRateLimitError as exc:
                last = exc
            except GitHubNetworkError as exc:
                last = exc
            except GitHubAuthError as exc:
                return DiscoveryOutcome(status=DiscoveryStatus.API_ERROR, error=str(exc))
            except GitHubError as exc:
                if exc.status_code is not None and 500 <= exc.status_code < 600:
                    last = exc  # 5xx 可重试
                else:
                    return DiscoveryOutcome(status=DiscoveryStatus.API_ERROR, error=str(exc))
            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.base_delay * (2**attempt))

        if isinstance(last, GitHubRateLimitError):
            return DiscoveryOutcome(status=DiscoveryStatus.RATE_LIMITED, error=str(last))
        if isinstance(last, GitHubNetworkError):
            return DiscoveryOutcome(status=DiscoveryStatus.NETWORK_ERROR, error=str(last))
        return DiscoveryOutcome(
            status=DiscoveryStatus.API_ERROR, error=str(last) if last else "unknown error"
        )


def aggregate_query_outcome(outcomes: list[DiscoveryOutcome]) -> DiscoveryOutcome:
    """将多个规划查询的结果聚合为查询级结果。

    - 任一规划查询成功（有候选）→ SUCCESS（候选并集去重）
    - 否则任一规划查询为真空 → EMPTY
    - 否则全部失败 → 取第一个失败状态
    """
    if not outcomes:
        return DiscoveryOutcome(status=DiscoveryStatus.API_ERROR, error="无规划查询")
    if any(o.status == DiscoveryStatus.SUCCESS for o in outcomes):
        candidates = union_candidates(
            [o.candidates for o in outcomes if o.status == DiscoveryStatus.SUCCESS]
        )
        return DiscoveryOutcome(status=DiscoveryStatus.SUCCESS, candidates=candidates)
    if any(o.status == DiscoveryStatus.EMPTY for o in outcomes):
        return DiscoveryOutcome(status=DiscoveryStatus.EMPTY)
    status = outcomes[0].status
    errors = [o.error for o in outcomes if o.error]
    return DiscoveryOutcome(status=status, error="; ".join(errors) if errors else None)


@dataclass
class CoverageSummary:
    """覆盖率汇总：区分成功评估（含真空）与失败。"""

    total: int
    evaluated: int
    with_candidates: int
    empty: int
    failed: int
    failure_reasons: dict[str, list[str]]
    avg_candidates: float


def summarize_coverage(
    queries: list[str], outcomes: dict[str, DiscoveryOutcome]
) -> CoverageSummary:
    """汇总覆盖率：失败查询与真空查询分开统计。"""
    total = len(queries)
    with_candidates = sum(
        1 for o in outcomes.values() if o.status == DiscoveryStatus.SUCCESS
    )
    empty = sum(1 for o in outcomes.values() if o.status == DiscoveryStatus.EMPTY)
    evaluated = with_candidates + empty
    failed = total - evaluated

    failure_reasons: dict[str, list[str]] = {}
    for q, o in outcomes.items():
        if o.status in (
            DiscoveryStatus.RATE_LIMITED,
            DiscoveryStatus.NETWORK_ERROR,
            DiscoveryStatus.API_ERROR,
        ):
            failure_reasons.setdefault(o.status.value, []).append(q)

    total_candidates = sum(
        len(o.candidates)
        for o in outcomes.values()
        if o.status == DiscoveryStatus.SUCCESS
    )
    avg = total_candidates / evaluated if evaluated else 0.0
    return CoverageSummary(
        total=total,
        evaluated=evaluated,
        with_candidates=with_candidates,
        empty=empty,
        failed=failed,
        failure_reasons=failure_reasons,
        avg_candidates=avg,
    )