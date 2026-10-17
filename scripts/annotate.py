"""人工标注 CLI：逐条展示候选并接受 2/1/0/u/s/q。

用法:
    uv run python scripts/annotate.py [--annotations PATH] [--initial-target 150] [--show-provenance]

- 每条标注后立即保存，进度不丢失；
- 自动从已有文件恢复（跳过已标注记录）；
- q 退出并保存；s 跳过（不保存）；
- 第二审阅者：--annotations data/evaluation/manual_annotations_reviewer2.jsonl。
"""

import argparse
from pathlib import Path

from mosquito.evaluation.annotation import (
    DEFAULT_INITIAL_TARGET,
    DEFAULT_MANUAL_ANNOTATIONS_PATH,
    load_annotations,
    parse_label,
    review_order,
    save_annotations,
    validate_annotations,
)


def _show(rec, index: int, total: int, show_provenance: bool) -> None:
    print(f"\n=== {index}/{total} ===")
    print(f"查询: {rec.query}")
    print(f"仓库: {rec.full_name}")
    if rec.description:
        print(f"描述: {rec.description}")
    if rec.language:
        print(f"语言: {rec.language}")
    if rec.topics:
        print(f"主题: {', '.join(rec.topics)}")
    if show_provenance and rec.retrieved_by:
        ranks = ", ".join(f"{r}({rec.ranks.get(r)})" for r in rec.retrieved_by)
        print(f"来源: {ranks}")


def main() -> int:
    parser = argparse.ArgumentParser(description="人工相关性标注 CLI")
    parser.add_argument(
        "--annotations", type=Path, default=DEFAULT_MANUAL_ANNOTATIONS_PATH, help="标注文件路径（默认 data/evaluation/manual_annotations.jsonl）"
    )
    parser.add_argument("--initial-target", type=int, default=DEFAULT_INITIAL_TARGET, help="本轮审阅上限（默认 150）")
    parser.add_argument("--show-provenance", action="store_true", help="显示 retrieved_by 与 ranks")
    args = parser.parse_args()

    records = load_annotations(args.annotations)
    if not records:
        print(f"标注文件为空或不存在: {args.annotations}")
        return 1

    issues = validate_annotations(records)
    if issues:
        print(f"警告: 发现 {len(issues)} 个校验问题（继续前请检查）")
        for issue in issues[:10]:
            print(f"  [{issue.issue_type}] 第 {issue.index + 1} 条: {issue.message}")

    order = review_order(records, args.initial_target)
    pending = [i for i in order if records[i].judgment is None]
    done = len(records) - len([r for r in records if r.judgment is None])
    print(f"待标注: {len(pending)} 条（已标注 {done} 条，自动恢复）")

    for pos, i in enumerate(pending, start=1):
        rec = records[i]
        _show(rec, pos, len(pending), args.show_provenance)
        while True:
            try:
                raw = input("标注 (2=高度相关, 1=部分相关, 0=不相关, u=不确定, s=跳过, q=退出): ").strip()
            except EOFError:
                save_annotations(args.annotations, records)
                print("\n已保存，退出")
                return 0
            if raw.lower() == "q":
                save_annotations(args.annotations, records)
                print("已保存，退出")
                return 0
            try:
                label = parse_label(raw)
                break
            except ValueError as exc:
                print(exc)
        if label is None:
            continue  # 跳过，不保存
        rec.judgment = label
        note = input("备注 (可选，直接回车跳过): ").strip()
        if note:
            rec.note = note
        save_annotations(args.annotations, records)  # 每条保存，进度不丢失

    save_annotations(args.annotations, records)
    print("本轮审阅完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())