"""评测数据集加载：JSONL 格式，每行一个 QueryJudgment。

文件 I/O 与指标计算分离，本模块只负责加载与校验。
"""

import json
from pathlib import Path

from pydantic import ValidationError

from mosquito.evaluation.models import QueryJudgment

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "evaluation" / "github_queries.jsonl"
)


class DatasetError(Exception):
    """评测数据集加载或校验失败。"""


def load_judgments(path: str | Path = DEFAULT_DATASET_PATH) -> list[QueryJudgment]:
    """加载 JSONL 标注数据集，逐行校验，出错时给出明确行号。"""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"评测数据集不存在: {path}")

    judgments: list[QueryJudgment] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc
            try:
                judgments.append(QueryJudgment.model_validate(record))
            except ValidationError as exc:
                raise DatasetError(f"第 {line_no} 行校验失败: {exc}") from exc
    return judgments