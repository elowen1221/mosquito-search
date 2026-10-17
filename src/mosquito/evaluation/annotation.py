"""人工相关性标注工作流：数据准备 / 确定性审阅顺序 / 校验 / 持久化。

标注方案：
- 2 = 高度相关：直接满足原始用户意图，是值得推荐的强候选；
- 1 = 部分相关：可能有用或相关，但不完整、间接、过窄/过宽或存在明显偏差；
- 0 = 不相关：不能有意义地满足原始意图；
- unsure = 审阅者无法根据现有元数据确定。

设计要点：
- 审阅顺序确定性优先：top-rank > 分歧 > 多来源 > 查询覆盖 > 尾部；
- 同一仓库在不同 query 下需要独立标注，不跨 query 合并；
- 文件格式支持第二审阅者：另存为 manual_annotations_reviewer2.jsonl 即可，
  审阅者身份在文件层面分离（暂不实现一致性指标）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from mosquito.evaluation.pool import PoolAnnotation

VALID_LABELS: tuple = (2, 1, 0, "unsure")

DEFAULT_POOL_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "evaluation" / "pool_annotations.jsonl"
)
DEFAULT_MANUAL_ANNOTATIONS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "evaluation" / "manual_annotations.jsonl"
)
DEFAULT_INITIAL_TARGET = 150


class ManualAnnotation(BaseModel):
    """单条人工标注记录：仅含审阅者有用字段，不含任何 ranker 分数。"""

    query: str
    full_name: str
    description: str | None = None
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    retrieved_by: list[str] = Field(default_factory=list)
    ranks: dict[str, int] = Field(default_factory=dict)
    judgment: int | str | None = None  # 2/1/0/unsure/None(未标注)
    note: str | None = None


def parse_label(text: str) -> int | str | None:
    """解析标注输入：2/1/0/u/s。返回标签或 None（跳过）。"""
    text = text.strip().lower()
    if text == "2":
        return 2
    if text == "1":
        return 1
    if text == "0":
        return 0
    if text in ("u", "unsure"):
        return "unsure"
    if text in ("s", "skip"):
        return None
    raise ValueError(f"无效标注: {text!r}（可选 2/1/0/u/s）")


def prepare_annotations(pool_records: list[PoolAnnotation]) -> list[ManualAnnotation]:
    """将候选池记录转换为人工标注记录（judgment 初始为 None）。"""
    return [
        ManualAnnotation(
            query=r.query,
            full_name=r.full_name,
            description=r.description,
            language=r.language,
            topics=r.topics,
            retrieved_by=r.retrieved_by,
            ranks=r.ranks,
        )
        for r in pool_records
    ]


def _rank_spread(ranks: dict[str, int]) -> int:
    if not ranks:
        return 0
    return max(ranks.values()) - min(ranks.values())


def review_tier(record: ManualAnnotation) -> str:
    """审阅优先级分类：top_rank / disagreement / multi_variant / tail。"""
    ranks = record.ranks
    if any(r <= 3 for r in ranks.values()):
        return "top_rank"
    if len(record.retrieved_by) >= 2 and _rank_spread(ranks) >= 5:
        return "disagreement"
    if len(record.retrieved_by) >= 2:
        return "multi_variant"
    return "tail"


_TIER_ORDER = {"top_rank": 0, "disagreement": 1, "multi_variant": 2, "tail": 4}


def review_order(
    records: list[ManualAnnotation], initial_target: int | None = None
) -> list[int]:
    """返回记录索引的确定性审阅顺序。

    优先级：top-rank > 分歧 > 多来源 > 查询覆盖 > 尾部。
    查询覆盖作为同层内的 tiebreaker：每个 query 至少一条尽早出现。
    """
    n = len(records)

    def key(i: int) -> tuple:
        rec = records[i]
        tier = _TIER_ORDER[review_tier(rec)]
        min_rank = min(rec.ranks.values(), default=999)
        return (tier, min_rank, rec.full_name)

    ordered = sorted(range(n), key=key)
    # 查询覆盖 pass：每个 query 的第一条记录提前
    covered: set[str] = set()
    result: list[int] = []
    for i in ordered:
        if records[i].query not in covered:
            covered.add(records[i].query)
            result.append(i)
    for i in ordered:
        if i not in result:
            result.append(i)
    if initial_target is not None:
        return result[:initial_target]
    return result


@dataclass
class ValidationIssue:
    """标注校验问题。"""

    index: int
    issue_type: str  # invalid_label / duplicate / missing_field / conflicting
    message: str


def validate_annotations(records: list[ManualAnnotation]) -> list[ValidationIssue]:
    """检测无效标签 / 重复记录 / 缺失字段 / 冲突标注。"""
    issues: list[ValidationIssue] = []
    seen: dict[tuple[str, str], int] = {}
    for i, rec in enumerate(records):
        if not rec.query:
            issues.append(ValidationIssue(i, "missing_field", "缺少 query"))
        if not rec.full_name:
            issues.append(ValidationIssue(i, "missing_field", "缺少 full_name"))
        if rec.judgment is not None and rec.judgment not in VALID_LABELS:
            issues.append(ValidationIssue(i, "invalid_label", f"非法标签: {rec.judgment!r}"))
        key = (rec.query, rec.full_name)
        if key in seen:
            prev = seen[key]
            prev_rec = records[prev]
            if (
                prev_rec.judgment is not None
                and rec.judgment is not None
                and prev_rec.judgment != rec.judgment
            ):
                issues.append(ValidationIssue(i, "conflicting", f"与第 {prev + 1} 条标注冲突"))
            else:
                issues.append(ValidationIssue(i, "duplicate", f"与第 {prev + 1} 条重复"))
        else:
            seen[key] = i
    return issues


def unlabeled_count(records: list[ManualAnnotation]) -> int:
    """未标注记录数：judgment 为 None 或 unsure 均视为未标注。"""
    return sum(1 for r in records if r.judgment not in (0, 1, 2))


def load_annotations(
    path: str | Path = DEFAULT_MANUAL_ANNOTATIONS_PATH,
) -> list[ManualAnnotation]:
    """加载标注 JSONL；文件不存在时返回空列表（便于恢复流程）。"""
    path = Path(path)
    if not path.exists():
        return []
    records: list[ManualAnnotation] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ManualAnnotation.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"第 {line_no} 行解析失败: {exc}") from exc
    return records


def save_annotations(path: str | Path, records: list[ManualAnnotation]) -> None:
    """写出标注 JSONL，每行一条 ManualAnnotation。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")