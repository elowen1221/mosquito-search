"""Repo 检索文本构造与归一化。

文本构造是 BERT reranker 的单一来源：EmbeddingRanker（build_repo_text）
与 train_bert_reranker（build_pairs）都通过 build_repo_text_from_fields 拼接，
保证 train / inference / evaluation 路径格式严格一致，永不漂移。
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from mosquito.models.repo import RepoCandidate

# Canonical 标签：中文（与 test_text.py 既有断言一致，且 embedding ranker 已线上使用）
_LANGUAGE_TAG = "语言"
_TOPICS_TAG = "主题"


def normalize(text: str) -> str:
    """折叠连续空白并转为小写。"""
    return " ".join(text.split()).lower()


def text_hash(text: str) -> str:
    """文本的 sha256 哈希，用作 embedding 缓存 key。

    文本变化（description/topics/language 等语义内容）时哈希变化，缓存自动失效；
    stars 等非语义字段不在文本中，不影响缓存。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_repo_text_from_fields(
    full_name: str,
    description: str | None = None,
    language: str | None = None,
    topics: Iterable[str] | None = None,
) -> str:
    """仓库事实 → canonical 检索文本。

    字段权重：full_name（含 owner）> description > topics > language；空字段跳过。
    所有 BERT reranker 输入（训练 / 推理 / 离线评测）都必须调用本函数。
    """
    parts = [full_name]
    if description:
        parts.append(description)
    if language:
        parts.append(f"[{_LANGUAGE_TAG}: {language}]")
    if topics:
        topic_list = [t for t in topics if t]
        if topic_list:
            parts.append(f"[{_TOPICS_TAG}: {', '.join(topic_list)}]")
    return normalize(" ".join(parts))


def build_repo_text(repo: RepoCandidate) -> str:
    """将 RepoCandidate 数据拼接为 canonical 检索文本（委托 build_repo_text_from_fields）。"""
    return build_repo_text_from_fields(
        full_name=repo.full_name,
        description=repo.description,
        language=repo.language,
        topics=repo.topics,
    )