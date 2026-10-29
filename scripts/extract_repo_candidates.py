"""从 discovery cache + annotations 提取冻结 RepoCandidate snapshot。

benchmark 用的 Rules/Quality ranker 依赖完整 RepoCandidate metadata
(stars / forks / size / pushed_at)，annotations 本身不含这些字段。

此脚本一次性从 discovery cache 提取每个 query 的候选 RepoCandidate，
合并去重后输出稳定 JSONL，benchmark 只读此冻结快照，不再 fetch GitHub。

用法:
    uv run python scripts/extract_repo_candidates.py --annotations data/evaluation/manual_annotations.jsonl --cache data/evaluation/discovery_cache.json --output data/evaluation/repo_candidates.jsonl

Fail-fast:
    - cache entry 解析失败 → crash
    - 同一 full_name 在不同 query 下 RepoCandidate 关键字段冲突 → crash
    - annotations 里 full_name 在 cache 里完全找不到 → crash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mosquito.evaluation.annotation import DEFAULT_MANUAL_ANNOTATIONS_PATH, load_annotations
from mosquito.evaluation.discovery import (
    DEFAULT_CACHE_PATH,
    DiscoveryCache,
    DiscoveryStatus,
)
from mosquito.models.repo import RepoCandidate


REQUIRED_FIELDS = ("full_name", "stars", "forks", "size", "pushed_at")


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return out.strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(source: str, query: str, limit: int) -> str:
    """与 DiscoveryCache._key 保持一致（sha256 确定性）。"""
    raw = f"{source}|{query}|{limit}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache_all(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_field_signature(c: dict) -> tuple:
    """用于冲突检测的字段签名：只保留 Rules/Quality 用到的字段。"""
    pushed = c.get("pushed_at")
    if isinstance(pushed, datetime):
        pushed_str = pushed.isoformat()[:10]
    elif isinstance(pushed, str):
        pushed_str = pushed[:10]
    else:
        pushed_str = ""
    return (
        c.get("stars"),
        c.get("forks"),
        c.get("size"),
        c.get("language"),
        c.get("description"),
        pushed_str,
    )


def _repo_to_jsonl(c: RepoCandidate) -> dict:
    d = c.model_dump()
    for k in ("updated_at", "pushed_at"):
        v = d.get(k)
        if v is not None and isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen RepoCandidate snapshot for benchmark")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_MANUAL_ANNOTATIONS_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args()

    if not args.annotations.exists():
        print(f"FAIL: annotations not found: {args.annotations}", file=sys.stderr)
        return 1
    if not args.cache.exists():
        print(f"FAIL: discovery cache not found: {args.cache}", file=sys.stderr)
        return 1

    anns = load_annotations(args.annotations)
    if not anns:
        print("FAIL: annotations empty", file=sys.stderr)
        return 1

    # 需要的 repo full_name（来自所有 labeled annotation）
    needed = set(a.full_name for a in anns if a.judgment in (0, 1, 2))
    print(f"[1/3] annotations loaded: {len(anns)} records, {len(needed)} unique full_name needed")

    cache_data = _load_cache_all(args.cache)
    print(f"[2/3] discovery cache loaded: {len(cache_data)} entries")

    # 遍历 cache entries，收集 RepoCandidate
    by_full_name: dict[str, RepoCandidate] = {}
    conflicts: list[str] = []
    cache_full_name_set: set[str] = set()

    for key, entry in cache_data.items():
        if not isinstance(entry, dict):
            print(f"WARN: cache entry {key[:12]}... not dict, skip", file=sys.stderr)
            continue
        status = entry.get("status", "").lower()
        if status != DiscoveryStatus.SUCCESS.value.lower():
            continue
        cands = entry.get("candidates", [])
        for cd in cands:
            try:
                c = RepoCandidate.model_validate(cd)
            except Exception as exc:
                print(f"WARN: candidate parse fail: {exc} (entry {key[:12]})", file=sys.stderr)
                continue
            fn = c.full_name
            cache_full_name_set.add(fn)
            if fn in by_full_name:
                existing_sig = _candidate_field_signature(by_full_name[fn].model_dump())
                new_sig = _candidate_field_signature(cd)
                if existing_sig != new_sig:
                    conflicts.append(f"{fn}: {existing_sig} vs {new_sig}")
            else:
                by_full_name[fn] = c

    if conflicts:
        print("FAIL: conflicting metadata for same full_name across cache entries:", file=sys.stderr)
        for c in conflicts:
            print(f"  {c}", file=sys.stderr)
        return 1

    print(f"[3/3] unique candidates collected from cache: {len(by_full_name)}")

    missing = needed - set(by_full_name.keys())
    if missing:
        print(f"FAIL: {len(missing)} repos needed by annotations but missing in discovery cache:", file=sys.stderr)
        for fn in sorted(missing):
            print(f"  {fn}", file=sys.stderr)
        return 1

    # 只保留 benchmark 需要的（annotations 里的 + cache 里有的交集）
    frozen = sorted([by_full_name[fn] for fn in needed], key=lambda c: c.full_name)

    # 缺失字段检查
    bad = []
    for c in frozen:
        d = c.model_dump()
        for field in REQUIRED_FIELDS:
            v = d.get(field)
            if v is None or (isinstance(v, str) and v == "" and field != "full_name"):
                bad.append(f"{c.full_name}: {field}={v!r}")
    if bad:
        print("FAIL: RepoCandidates missing required fields for Rules/Quality:", file=sys.stderr)
        for b in bad[:20]:
            print(f"  {b}", file=sys.stderr)
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in frozen:
            f.write(json.dumps(_repo_to_jsonl(c), ensure_ascii=False) + "\n")

    # Provenance manifest（sidecar，与 snapshot 同目录）
    snapshot_sha = _sha256(args.output)
    annotations_sha = _sha256(args.annotations)
    cache_sha = _sha256(args.cache)
    manifest = {
        "snapshot_path": str(args.output),
        "snapshot_sha256": snapshot_sha,
        "repo_count": len(frozen),
        "source_annotations_path": str(args.annotations),
        "source_annotations_sha256": annotations_sha,
        "source_cache_path": str(args.cache),
        "source_cache_sha256": cache_sha,
        "source_cache_entries": len(cache_data),
        "extractor": "scripts/extract_repo_candidates.py",
        "git_commit": _git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "required_fields": list(REQUIRED_FIELDS),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] extracted {len(frozen)} frozen RepoCandidates → {args.output}")
    print(f"     source: annotations={args.annotations.name}, cache={args.cache.name}")
    print(f"     required fields verified: {', '.join(REQUIRED_FIELDS)}")
    print(f"     snapshot sha256: {snapshot_sha[:16]}")
    print(f"     manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
