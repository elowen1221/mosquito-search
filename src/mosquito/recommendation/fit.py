"""Profile-aware fit scoring：轻量、可解释的启发式打分。

四个维度（均为 0–1）：
- topic_fit      学习目标关键词是否出现在仓库文本中
- stack_fit      仓库语言 / topics 是否匹配技术栈
- difficulty_fit 由仓库体积估计难度，再与 skill_level 对齐
- learning_value 是否像 tutorial / example / starter，而不是纯库代码

fit_score = 0.35·topic + 0.30·stack + 0.20·difficulty + 0.15·learning
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mosquito.models.profile import Profile
from mosquito.models.repo import RepoCandidate

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 学习目标里的低信息词："learn RAG" → ["rag"]
_GOAL_STOPWORDS = frozenset(
    {
        "learn", "learning", "study", "studying", "to", "a", "an", "the",
        "and", "or", "for", "with", "my", "i", "want", "need", "how", "about",
        "use", "using", "practice", "project", "projects", "repo", "repos",
        "repository", "repositories", "some", "good", "best", "build",
    }
)

# 常见缩写展开，用于扩大 topic 匹配（确定性、可解释）
_GOAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "rag": ("retrieval", "augmented", "generation"),
    "llm": ("large", "language", "model"),
    "ml": ("machine", "learning"),
}

_TUTORIAL_KEYWORDS = (
    "tutorial", "tutorials", "example", "examples", "demo", "demos",
    "starter", "template", "cookbook", "playground", "hands-on",
    "sample", "samples", "walkthrough", "教程", "示例", "入门",
)
_GUIDE_KEYWORDS = (
    "guide", "learning", "course", "docs", "documentation", "awesome",
    "getting started", "笔记", "指南",
)
_COMPLEXITY_KEYWORDS = (
    "framework", "platform", "enterprise", "distributed", "microservice",
    "microservices", "kubernetes", "cluster",
)

FIT_WEIGHTS = {"topic": 0.35, "stack": 0.30, "difficulty": 0.20, "learning": 0.15}

# topic 证据权重：证据来源越"专一"越强
_TOPIC_NAME = 1.0          # name 直接命中核心目标
_TOPIC_TOPICS = 1.0        # topics 命中，且仓库主题聚焦
_TOPIC_TOPICS_BROAD = 0.6  # topics 命中，但仓库是多主题合集（单个 topic 不代表主线）
_TOPIC_DESCRIPTION = 0.65  # 仅 description 命中（中强证据）
_BROAD_TOPIC_COUNT = 12    # topics ≥ 12 视为多主题合集

_DIFFICULTY_FIT: dict[str, dict[str, float]] = {
    "beginner": {"easy": 1.0, "medium": 0.65, "hard": 0.25},
    "intermediate": {"easy": 0.9, "medium": 1.0, "hard": 0.55},
    "advanced": {"easy": 0.75, "medium": 0.9, "hard": 1.0},
}

_TIERS = ("easy", "medium", "hard")


@dataclass
class FitAssessment:
    fit_score: float
    topic_fit: float
    stack_fit: float
    difficulty_fit: float
    learning_value: float
    difficulty: str
    recommended_for: list[str] = field(default_factory=list)
    why_recommended: list[str] = field(default_factory=list)


def _repo_text(repo: RepoCandidate) -> str:
    return " ".join(
        [
            repo.name,
            repo.full_name,
            repo.description or "",
            repo.language or "",
            " ".join(repo.topics),
        ]
    ).lower()


def _goal_terms(goal: str) -> list[str]:
    terms = [t for t in _TOKEN_RE.findall(goal.lower()) if t not in _GOAL_STOPWORDS]
    return list(dict.fromkeys(terms))


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _topic_fit(
    terms: list[str],
    name_tokens: set[str],
    topic_tokens: set[str],
    desc_tokens: set[str],
    broad_collection: bool,
) -> float:
    """学习目标匹配度（0–1）。

    证据优先级：name > topics > description；多主题合集的 topic 只算弱证据。
    """
    if not terms:
        return 0.5
    strong_pool = name_tokens | topic_tokens
    all_pool = strong_pool | desc_tokens
    topics_weight = _TOPIC_TOPICS_BROAD if broad_collection else _TOPIC_TOPICS

    total = 0.0
    for term in terms:
        expanded = _GOAL_EXPANSIONS.get(term, ())
        concept_in_name = bool(expanded) and all(word in name_tokens for word in expanded)
        concept_in_strong = bool(expanded) and all(word in strong_pool for word in expanded)
        concept_in_any = bool(expanded) and all(word in all_pool for word in expanded)

        if term in name_tokens or concept_in_name:
            total += _TOPIC_NAME
        elif term in topic_tokens or concept_in_strong:
            total += topics_weight
        elif term in desc_tokens or concept_in_any:
            total += _TOPIC_DESCRIPTION
    return round(total / len(terms), 4)


def _stack_fit(repo: RepoCandidate, tech_stack: list[str], tokens: set[str]) -> float:
    stack = {s.strip().lower() for s in tech_stack if s.strip()}
    if not stack:
        return 0.5
    language = (repo.language or "").lower()
    if language and language in stack:
        return 1.0
    if any(term in tokens for term in stack):
        return 0.75
    return 0.3


def _difficulty(repo: RepoCandidate, text: str) -> str:
    if repo.size <= 1500:
        tier = 0
    elif repo.size <= 15000:
        tier = 1
    else:
        tier = 2
    # 描述里出现框架/平台/分布式等词时，按更难一档处理
    if any(keyword in text for keyword in _COMPLEXITY_KEYWORDS):
        tier = min(2, tier + 1)
    return _TIERS[tier]


def _learning_value(text: str) -> tuple[float, bool]:
    if any(keyword in text for keyword in _TUTORIAL_KEYWORDS):
        return 1.0, True
    if any(keyword in text for keyword in _GUIDE_KEYWORDS):
        return 0.7, False
    return 0.35, False


def _recommended_for(
    difficulty: str, learning_value: float, time_budget: str | None
) -> list[str]:
    uses: list[str] = []
    if learning_value >= 0.7:
        uses.append("learning")
    if learning_value >= 0.7 and difficulty != "hard":
        uses.append("course_project")
    if difficulty == "hard" or learning_value < 0.5:
        uses.append("source_reading")
    if time_budget == "weekend" and difficulty == "easy":
        uses.append("weekend_project")
    return uses


def _reasons(
    repo: RepoCandidate,
    profile: Profile,
    topic_fit: float,
    difficulty: str,
    is_tutorial: bool,
) -> list[str]:
    reasons: list[str] = []
    language = (repo.language or "").strip()
    if language and language.lower() in {s.strip().lower() for s in profile.tech_stack}:
        reasons.append(f"主要使用 {language}，与你的技术栈匹配")
    if topic_fit >= 0.6:
        reasons.append(f"仓库主题与学习目标「{profile.learning_goal}」匹配")
    if difficulty == "easy":
        reasons.append("项目规模较小，适合初学者阅读和修改")
    elif difficulty == "medium":
        reasons.append("项目规模适中，需要一些耐心阅读")
    else:
        reasons.append("项目规模较大，建议先读文档再上手")
    if is_tutorial:
        reasons.append("包含教程/示例性质内容，适合动手练习")
    return reasons[:4]


def assess(repo: RepoCandidate, profile: Profile) -> FitAssessment:
    """对单个候选仓库计算 profile fit 与可读解释。"""
    text = _repo_text(repo)
    name_tokens = _tokens(repo.name)
    topic_tokens = _tokens(" ".join(repo.topics))
    desc_tokens = _tokens(repo.description or "")

    topic_fit = _topic_fit(
        _goal_terms(profile.learning_goal),
        name_tokens,
        topic_tokens,
        desc_tokens,
        broad_collection=len(repo.topics) >= _BROAD_TOPIC_COUNT,
    )
    stack_fit = _stack_fit(repo, profile.tech_stack, _tokens(text))
    difficulty = _difficulty(repo, text)
    difficulty_fit = _DIFFICULTY_FIT[profile.skill_level][difficulty]
    learning_value, is_tutorial = _learning_value(text)

    fit_score = round(
        FIT_WEIGHTS["topic"] * topic_fit
        + FIT_WEIGHTS["stack"] * stack_fit
        + FIT_WEIGHTS["difficulty"] * difficulty_fit
        + FIT_WEIGHTS["learning"] * learning_value,
        4,
    )

    return FitAssessment(
        fit_score=fit_score,
        topic_fit=topic_fit,
        stack_fit=stack_fit,
        difficulty_fit=difficulty_fit,
        learning_value=learning_value,
        difficulty=difficulty,
        recommended_for=_recommended_for(difficulty, learning_value, profile.time_budget),
        why_recommended=_reasons(repo, profile, topic_fit, difficulty, is_tutorial),
    )
