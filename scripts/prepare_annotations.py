"""准备人工标注文件：从候选池生成标注记录并确定确定性审阅顺序。

用法:
    uv run python scripts/prepare_annotations.py [--pool PATH] [--output PATH] [--initial-target 150]

输出 manual_annotations.jsonl，每行一条 ManualAnnotation（judgment 初始为 null）。
审阅顺序优先：top-rank > 分歧 > 多来源 > 查询覆盖 > 尾部。
"""

import argparse
from pathlib import Path

from mosquito.evaluation.annotation import (
    DEFAULT_INITIAL_TARGET,
    DEFAULT_MANUAL_ANNOTATIONS_PATH,
    DEFAULT_POOL_PATH,
    prepare_annotations,
    review_order,
    review_tier,
    save_annotations,
)
from mosquito.evaluation.pool import PoolAnnotation, PoolError


def _load_pool(path: Path) -> list[PoolAnnotation]:
    if not path.exists():
        raise PoolError(f"候选池文件不存在: {path}")
    records: list[PoolAnnotation] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(PoolAnnotation.model_validate_json(line))
            except Exception as exc:
                raise PoolError(f"第 {line_no} 行解析失败: {exc}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="准备人工标注文件")
    parser.add_argument(
        "--pool", type=Path, default=DEFAULT_POOL_PATH, help="候选池 JSONL 路径（默认 data/evaluation/pool_annotations.jsonl）"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_MANUAL_ANNOTATIONS_PATH, help="标注文件输出路径（默认 data/evaluation/manual_annotations.jsonl）"
    )
    parser.add_argument(
        "--initial-target", type=int, default=DEFAULT_INITIAL_TARGET, help="推荐初始审阅数（默认 150）"
    )
    args = parser.parse_args()

    try:
        pool = _load_pool(args.pool)
    except PoolError as exc:
        print(f"候选池加载失败: {exc}")
        return 1
    if not pool:
        print("候选池为空")
        return 1

    annotations = prepare_annotations(pool)
    order = review_order(annotations, args.initial_target)
    initial_set = [annotations[i] for i in order]
    save_annotations(args.output, annotations)

    queries = len({a.query for a in annotations})
    initial_queries = len({a.query for a in initial_set})
    tiers: dict[str, int] = {}
    for a in initial_set:
        t = review_tier(a)
        tiers[t] = tiers.get(t, 0) + 1

    print(f"已生成 {len(annotations)} 条标注记录 -> {args.output}")
    print(f"覆盖查询数: {queries}")
    print(f"推荐初始审阅数: {len(initial_set)}（覆盖 {initial_queries} 个查询）")
    print(f"初始集构成: {tiers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())