"""QueryPlanner：将自然语言查询改写为 GitHub 搜索友好的关键词查询。

确定性启发式规则，无 LLM 调用：
- 英文查询：归一化空白/大小写，去除低信息填充词，生成关键词变体；
- 中文查询：通过小型技术术语映射字典，将有用概念翻译为英文关键词。

每个查询返回 1-4 个去重后的关键词查询，供 GitHub Search API 使用。
"""

from __future__ import annotations

import re

from mosquito.models.profile import Profile

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 学习目标缩写展开，用于生成更完整的检索变体
_GOAL_EXPANSIONS: dict[str, str] = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "ml": "machine learning",
}

# 学习目标里的额外低信息词（"learn RAG" → ["rag"]）
_GOAL_FILLER = frozenset({"learn", "learning", "study", "studying", "practice", "practicing"})

# 低信息填充词：停用词 + 口语化表达（"I want" / "looking for" / "works well" 等）
_FILLER_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
        "with", "that", "this", "these", "those", "is", "are", "be", "it",
        "i", "me", "my", "you", "your", "we", "they", "want", "wants",
        "looking", "need", "needs", "using", "use", "used", "build",
        "building", "create", "creating", "make", "making", "run", "running",
        "runs", "support", "supports", "supporting", "based", "works",
        "well", "suitable", "can", "help", "add", "adding", "give", "get",
        "gets", "like", "would", "should", "could", "does", "do", "has",
        "have", "had", "not", "no", "but", "also", "very", "really",
        "just", "only", "more", "most", "some", "any", "all", "each",
        "other", "another", "such", "than", "then", "there", "here",
        "about", "into", "from", "over", "under", "between", "through",
        "during", "before", "after", "up", "down", "out", "off", "again",
        "once", "too", "own", "same", "so", "if", "then", "because",
        "while", "when", "where", "which", "who", "whom", "whose",
    }
)

# 可丢弃的修饰/泛化词：保留在完整变体中，丢弃后扩大召回
_OPTIONAL_WORDS = frozenset(
    {
        "lightweight", "minimal", "small", "fast", "quick", "simple", "easy",
        "good", "great", "best", "better", "nice", "stable", "maintained",
        "active", "actively", "production", "ready", "old", "beginner",
        "friendly", "clear", "personal", "private", "local", "self",
        "hosted", "open", "source", "cloud", "alternative", "library",
        "libraries", "tool", "tools", "package", "packages", "app", "apps",
        "application", "applications", "project", "projects", "system",
        "systems", "service", "services", "module", "modules", "plugin",
        "plugins", "wrapper", "wrappers", "client", "clients", "engine",
        "engines", "platform", "platforms", "deploy", "assistant",
    }
)

# 同义词替换：生成覆盖更广的变体
_SYNONYMS: dict[str, str] = {
    "api": "web",
    "async": "asynchronous",
    "cli": "terminal",
    "library": "framework",
    "vector": "embedding",
    "database": "store",
    "vector database": "embedded vector store",
    "long term memory": "memory system",
}

# 中文技术术语映射：长短语在前，避免短词先匹配吃掉长词的一部分
_ZH_TERMS: list[tuple[str, str]] = [
    ("向量数据库", "vector database"),
    ("长期记忆", "long term memory"),
    ("记忆系统", "memory system"),
    ("机器学习", "machine learning"),
    ("命令行", "command line"),
    ("向量", "vector"),
    ("数据库", "database"),
    ("本地", "local"),
    ("轻量级", "lightweight"),
    ("个人", "personal"),
    ("助手", "assistant"),
    ("检索", "retrieval"),
    ("语义", "semantic"),
    ("搜索", "search"),
    ("模糊", "fuzzy"),
    ("匹配", "matching"),
    ("字符串", "string"),
    ("网页", "web"),
    ("文章", "article"),
    ("文本", "text"),
    ("监控", "monitor"),
    ("网站", "website"),
    ("终端", "terminal"),
    ("参数", "argument"),
    ("解析", "parse"),
    ("认证", "authentication"),
    ("框架", "framework"),
    ("异步", "async"),
    ("接口", "api"),
    ("依赖", "dependency"),
    ("重试", "retry"),
    ("请求", "request"),
    ("失败", "failed"),
    ("提取", "extract"),
    ("存储", "store"),
    ("嵌入式", "embedded"),
    ("记忆", "memory"),
    ("长期", "long term"),
]


class QueryPlanner:
    """确定性启发式查询规划器。"""

    def plan(self, query: str, profile: Profile | None = None) -> list[str]:
        """返回 1-4 个 GitHub 搜索友好的关键词查询。

        传入 profile 时额外生成 profile-aware 变体（目标关键词 + 技术栈 + 学习场景），
        不传则行为与原来完全一致。
        """
        query = query.strip()
        if not query:
            return []
        base = self._plan_chinese(query) if _CJK_RE.search(query) else self._plan_english(query)
        if profile is None:
            return base
        return self._dedupe(base + self._profile_variants(profile))[:4]

    def _profile_variants(self, profile: Profile) -> list[str]:
        """由 profile 生成 "rag python tutorial" 这类面向学习的检索变体。"""
        terms = [t for t in _LATIN_TOKEN_RE.findall(profile.learning_goal.lower())]
        terms = [
            t for t in dict.fromkeys(terms) if t not in _FILLER_WORDS and t not in _GOAL_FILLER
        ]
        if not terms:
            return []
        stack = next((s.strip().lower() for s in profile.tech_stack if s.strip()), "")
        goal = " ".join(terms)
        stem = f"{goal} {stack}".strip()
        variants = [f"{stem} tutorial", f"{stem} example"]
        if len(terms) == 1 and terms[0] in _GOAL_EXPANSIONS:
            expanded = _GOAL_EXPANSIONS[terms[0]]
            variants.append(f"{expanded} {stack}".strip())
        return variants

    # ---- 英文 ----

    def _plan_english(self, query: str) -> list[str]:
        tokens = _LATIN_TOKEN_RE.findall(query.lower())
        keywords = [t for t in tokens if t not in _FILLER_WORDS]
        keywords = list(dict.fromkeys(keywords))
        if not keywords:
            return [query]
        variants = [" ".join(keywords)]
        core = [k for k in keywords if k not in _OPTIONAL_WORDS]
        if core and core != keywords:
            variants.append(" ".join(core))
        syn = self._synonym_variant(keywords)
        if syn:
            variants.append(syn)
        return self._dedupe(variants)[:4]

    # ---- 中文 ----

    def _plan_chinese(self, query: str) -> list[str]:
        latin = _LATIN_TOKEN_RE.findall(query.lower())
        zh = self._match_zh_terms(query)
        terms = list(dict.fromkeys(latin + zh))
        if not terms:
            return [query]
        variants = []
        core = [t for t in terms if t not in _OPTIONAL_WORDS]
        if core:
            variants.append(" ".join(core))
        variants.append(" ".join(terms))
        syn = self._synonym_variant(terms)
        if syn:
            variants.append(syn)
        return self._dedupe(variants)[:4]

    def _match_zh_terms(self, query: str) -> list[str]:
        """按查询中出现顺序返回匹配到的英文术语，避免重叠匹配。"""
        spans: list[tuple[int, int, str]] = []
        for zh, en in sorted(_ZH_TERMS, key=lambda p: len(p[0]), reverse=True):
            for m in re.finditer(re.escape(zh), query):
                start, end = m.span()
                if any(start < e and s < end for s, e, _ in spans):
                    continue
                spans.append((start, end, en))
        spans.sort(key=lambda t: t[0])
        return [en for _, _, en in spans]

    def _synonym_variant(self, terms: list[str]) -> str | None:
        for i, term in enumerate(terms):
            if term in _SYNONYMS:
                replaced = list(terms)
                replaced[i] = _SYNONYMS[term]
                return " ".join(replaced)
        return None

    @staticmethod
    def _dedupe(variants: list[str]) -> list[str]:
        return list(dict.fromkeys(variants))