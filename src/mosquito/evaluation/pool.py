"""评测候选池生成：多 ranker 候选并集 + 来源记录，供人工标注。

- 保留 provenance（哪个 ranker 检索到、各自排名），但不暴露 ranker 分数；
- 标注记录按固定 seed 确定性打乱显示顺序，降低标注偏差；
- 与评测指标（metrics.py）分离。
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from mosquito.models.repo import RepoCandidate

DEFAULT_QUERIES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "evaluation" / "queries.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "evaluation" / "pool_annotations.jsonl"
)
DEFAULT_SEED = 42


class PoolError(Exception):
    """候选池生成失败。"""


class PoolAnnotation(BaseModel):
    """单条待标注记录：query + 仓库事实 + 来源信息 + judgment（初始为 null）。

    只含 provenance（retrieved_by / ranks），不含任何 ranker 分数。
    """

    query: str
    full_name: str
    description: str | None = None
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    retrieved_by: list[str] = Field(default_factory=list)
    ranks: dict[str, int] = Field(default_factory=dict)
    judgment: int | None = None


@dataclass
class PooledRepo:
    """候选仓库及其来源信息（ranker 名 -> 1-based 排名）。"""

    candidate: RepoCandidate
    ranks: dict[str, int]


def union_candidates(candidate_lists: list[list[RepoCandidate]]) -> list[RepoCandidate]:
    """按 full_name 并集去重，保留首次出现的候选。"""
    by_name: dict[str, RepoCandidate] = {}
    for candidates in candidate_lists:
        for candidate in candidates:
            by_name.setdefault(candidate.full_name, candidate)
    return list(by_name.values())


def build_pool(
    candidates: list[RepoCandidate],
    rankers: dict[str, object],
    query: str,
    pool_depth: int = 10,
) -> list[PooledRepo]:
    """对同一候选集运行所有 ranker，记录每个仓库的来源与排名。

    - 只记录排名，不记录分数；
    - 返回按 full_name 排序，保证后续打乱的输入确定性。
    """
    if pool_depth <= 0:
        return []
    by_name = {c.full_name: c for c in candidates}
    provenance: dict[str, dict[str, int]] = {}
    for name, ranker in rankers.items():
        ranked = ranker.rank(candidates, query)
        for position, item in enumerate(ranked[:pool_depth], start=1):
            provenance.setdefault(item.candidate.full_name, {})[name] = position
    return [
        PooledRepo(candidate=by_name[full_name], ranks=provenance[full_name])
        for full_name in sorted(provenance)
    ]


def to_annotation_records(
    query: str, pooled: list[PooledRepo], seed: int = DEFAULT_SEED
) -> list[PoolAnnotation]:
    """转换为标注记录，并按固定 seed 确定性打乱显示顺序。"""
    records = [
        PoolAnnotation(
            query=query,
            full_name=p.candidate.full_name,
            description=p.candidate.description,
            language=p.candidate.language,
            topics=p.candidate.topics,
            retrieved_by=sorted(p.ranks.keys()),
            ranks=p.ranks,
        )
        for p in pooled
    ]
    random.Random(seed).shuffle(records)
    return records


def load_queries(path: str | Path = DEFAULT_QUERIES_PATH) -> list[str]:
    """加载查询 JSONL，每行需包含 query 字段。"""
    path = Path(path)
    if not path.exists():
        raise PoolError(f"查询文件不存在: {path}")

    queries: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PoolError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc
            query = record.get("query")
            if not query:
                raise PoolError(f"第 {line_no} 行缺少 query 字段")
            queries.append(query)
    return queries


def write_annotation_file(path: str | Path, annotations: list[PoolAnnotation]) -> None:
    """写出标注 JSONL，每行一条 PoolAnnotation。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(ann.model_dump_json() + "\n")