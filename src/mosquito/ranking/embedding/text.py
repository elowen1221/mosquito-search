"""Repo 检索文本构造与归一化。"""

import hashlib

from mosquito.models.repo import RepoCandidate


def normalize(text: str) -> str:
    """折叠连续空白并转为小写。"""
    return " ".join(text.split()).lower()


def text_hash(text: str) -> str:
    """文本的 sha256 哈希，用作 embedding 缓存 key。

    文本变化（description/topics/language 等语义内容）时哈希变化，缓存自动失效；
    stars 等非语义字段不在文本中，不影响缓存。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_repo_text(repo: RepoCandidate) -> str:
    """将仓库事实数据拼接为用于 embedding 的检索文本。

    字段权重：full_name（含 owner）> description > topics > language。
    空字段跳过。
    """
    parts = [repo.full_name]
    if repo.description:
        parts.append(repo.description)
    if repo.language:
        parts.append(f"[语言: {repo.language}]")
    if repo.topics:
        parts.append(f"[主题: {', '.join(repo.topics)}]")
    return normalize(" ".join(parts))