"""baseline 评分规则：每条规则返回 (score, reason)。"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from mosquito.models.repo import RepoCandidate

# 常见语言关键词 → 用于从 query 中识别语言意图
LANGUAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "python": ("python", "py"),
    "javascript": ("javascript", "js", "node", "nodejs"),
    "typescript": ("typescript", "ts"),
    "rust": ("rust",),
    "go": ("go", "golang"),
    "java": ("java",),
    "c++": ("c++", "cpp"),
    "c": ("c语言", "c lang"),
    "ruby": ("ruby",),
    "php": ("php",),
    "kotlin": ("kotlin",),
    "swift": ("swift",),
    "dart": ("dart",),
    "scala": ("scala",),
    "r": ("r语言", "r lang"),
    "lua": ("lua",),
    "zig": ("zig",),
    "solidity": ("solidity",),
}

# 语言匹配加分
LANGUAGE_MATCH_SCORE = 20.0
# 更新时间分层加分
RECENCY_BUCKETS: tuple[tuple[int, float], ...] = ((30, 15.0), (90, 10.0), (365, 5.0))
# 项目规模（KB）适中区间加分
SIZE_MIN_KB = 100
SIZE_MAX_KB = 200_000
SIZE_SCORE = 5.0


@dataclass
class RuleResult:
    score: float
    reason: str


Rule = Callable[[RepoCandidate, str], RuleResult]


def _detect_language(query: str) -> str | None:
    lowered = query.lower()
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return lang
    return None


def language_match(candidate: RepoCandidate, query: str) -> RuleResult:
    """query 中出现语言关键词且与仓库语言一致时加分。"""
    lang = _detect_language(query)
    if lang and candidate.language and candidate.language.lower() == lang:
        return RuleResult(LANGUAGE_MATCH_SCORE, f"语言匹配: {candidate.language}")
    return RuleResult(0.0, "")


def recency(candidate: RepoCandidate, query: str) -> RuleResult:
    """pushed_at 越新分越高。"""
    if candidate.pushed_at is None:
        return RuleResult(0.0, "")
    days = (datetime.now(timezone.utc) - candidate.pushed_at).days
    for threshold, score in RECENCY_BUCKETS:
        if days <= threshold:
            return RuleResult(score, f"最近更新于 {days} 天前")
    return RuleResult(0.0, "")


def stars(candidate: RepoCandidate, query: str) -> RuleResult:
    """stars 对数缩放，上限 20 分。"""
    score = min(20.0, round(math.log10(candidate.stars + 1) * 5, 2))
    if score > 0:
        return RuleResult(score, f"stars: {candidate.stars}")
    return RuleResult(0.0, "")


def forks(candidate: RepoCandidate, query: str) -> RuleResult:
    """forks 对数缩放，上限 10 分。"""
    score = min(10.0, round(math.log10(candidate.forks + 1) * 3, 2))
    if score > 0:
        return RuleResult(score, f"forks: {candidate.forks}")
    return RuleResult(0.0, "")


def project_size(candidate: RepoCandidate, query: str) -> RuleResult:
    """项目规模（KB）适中加分，过大过小不加分。"""
    if SIZE_MIN_KB <= candidate.size <= SIZE_MAX_KB:
        return RuleResult(SIZE_SCORE, f"项目规模适中: {candidate.size} KB")
    return RuleResult(0.0, "")


DEFAULT_RULES: list[Rule] = [language_match, recency, stars, forks, project_size]