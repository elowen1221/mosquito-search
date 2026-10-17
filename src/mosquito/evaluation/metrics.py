"""评测指标纯函数：P@K / DCG / NDCG / RR / MRR。

不依赖任何外部评测库，全部为确定性纯函数。
约定：
- ranked 为按分数降序的 full_name 列表；
- relevant 为相关（judgment >= 1）的 full_name 集合；
- judgments 为 full_name -> 0/1/2 的原始分级标注；
- 未标注（不在 judgments 中）的仓库贡献 0；
- P@K 分母取 min(k, len(ranked))，避免返回结果不足 k 时被不公平惩罚。
"""

import math


def precision_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """前 k 个结果中相关仓库占比。"""
    if not ranked or k <= 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for name in top if name in relevant)
    return hits / min(k, len(ranked))


def dcg_at_k(ranked: list[str], judgments: dict[str, int], k: int = 10) -> float:
    """分级 DCG：rel_i / log2(i+2)，直接使用原始 0/1/2 分级。"""
    if not ranked or k <= 0:
        return 0.0
    dcg = 0.0
    for i, name in enumerate(ranked[:k]):
        rel = judgments.get(name, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked: list[str], judgments: dict[str, int], k: int = 10) -> float:
    """NDCG@K：DCG 除以理想排序（按分级降序）的 DCG。"""
    dcg = dcg_at_k(ranked, judgments, k)
    ideal_names = [name for name, _ in sorted(judgments.items(), key=lambda kv: kv[1], reverse=True)]
    idcg = dcg_at_k(ideal_names, judgments, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """第一个相关仓库排名的倒数，无相关返回 0。"""
    for i, name in enumerate(ranked):
        if name in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(rankings: list[list[str]], relevant_sets: list[set[str]]) -> float:
    """多个 query 的 MRR，空输入返回 0。"""
    if not rankings:
        return 0.0
    total = sum(reciprocal_rank(r, rel) for r, rel in zip(rankings, relevant_sets))
    return total / len(rankings)