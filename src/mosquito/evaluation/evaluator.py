"""评测器：将排序结果（full_name 列表）与标注数据对比计算指标。

核心评测器不调用 GitHub API，也不依赖任何 ranking 实现；
只接收 query -> 已排序 full_name 列表 与标注数据。
"""

from dataclasses import dataclass, field

from mosquito.evaluation.annotation import ManualAnnotation
from mosquito.evaluation.metrics import ndcg_at_k, precision_at_k, reciprocal_rank
from mosquito.evaluation.models import QueryJudgment, RankerEvaluationResult


def evaluate_rankings(
    ranker_name: str,
    rankings: dict[str, list[str]],
    judgments: list[QueryJudgment],
    k: int = 10,
) -> RankerEvaluationResult:
    """对每个 query 的排序结果计算 P@K / NDCG@K / MRR 并取平均。

    - relevance >= 1 视为相关（用于 P@K 与 MRR）；
    - NDCG 使用原始 0/1/2 分级；
    - 未标注的 query 跳过，不计入 evaluated_queries。
    """
    judgment_map = {j.query: j for j in judgments}
    precisions: list[float] = []
    ndcgs: list[float] = []
    rrs: list[float] = []

    for query, ranked in rankings.items():
        judgment = judgment_map.get(query)
        if judgment is None:
            continue
        relevant = {name for name, rel in judgment.judgments.items() if rel >= 1}
        precisions.append(precision_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, judgment.judgments, k))
        rrs.append(reciprocal_rank(ranked, relevant))

    n = len(precisions)
    if n == 0:
        return RankerEvaluationResult(
            ranker_name=ranker_name, precision_at_10=0.0, ndcg_at_10=0.0, mrr=0.0, evaluated_queries=0
        )
    return RankerEvaluationResult(
        ranker_name=ranker_name,
        precision_at_10=sum(precisions) / n,
        ndcg_at_10=sum(ndcgs) / n,
        mrr=sum(rrs) / n,
        evaluated_queries=n,
    )


@dataclass
class QueryAnnotationReport:
    """单个 query 的人工标注评测报告。"""

    query: str
    judged_count: int
    relevant_count: int
    highly_relevant_count: int
    retrieved_count: int
    coverage: float
    precision_at_10: float
    ndcg_at_10: float
    mrr: float


@dataclass
class AnnotationEvaluationResult:
    """人工标注评测结果：每查询报告 + 宏平均。"""

    per_query: list[QueryAnnotationReport] = field(default_factory=list)
    macro_precision_at_10: float = 0.0
    macro_ndcg_at_10: float = 0.0
    macro_mrr: float = 0.0
    evaluated_queries: int = 0
    excluded_queries: list[str] = field(default_factory=list)
    min_labeled: int = 5


def evaluate_annotations(
    rankings: dict[str, list[str]],
    annotations: list[ManualAnnotation],
    k: int = 10,
    min_labeled: int = 5,
) -> AnnotationEvaluationResult:
    """基于人工标注计算每查询指标，仅使用已标注记录。

    - unsure 与未标注（judgment 为 None）视为未标注，不参与指标计算；
    - 未标注仓库不会被当作不相关（从排序列表中过滤后再计算指标）；
    - 宏平均仅统计已标注数 >= min_labeled 的 query，其余在 excluded_queries 中列出。
    """
    by_query: dict[str, list[ManualAnnotation]] = {}
    for ann in annotations:
        by_query.setdefault(ann.query, []).append(ann)

    per_query: list[QueryAnnotationReport] = []
    for query, ranked in rankings.items():
        records = by_query.get(query, [])
        labeled = {r.full_name: r.judgment for r in records if r.judgment in (0, 1, 2)}
        if not labeled:
            continue
        ranked_labeled = [name for name in ranked if name in labeled]
        relevant = {name for name, rel in labeled.items() if rel >= 1}
        retrieved_count = len(ranked)
        judged_count = len(labeled)
        per_query.append(
            QueryAnnotationReport(
                query=query,
                judged_count=judged_count,
                relevant_count=len(relevant),
                highly_relevant_count=sum(1 for rel in labeled.values() if rel == 2),
                retrieved_count=retrieved_count,
                coverage=judged_count / retrieved_count if retrieved_count else 0.0,
                precision_at_10=precision_at_k(ranked_labeled, relevant, k),
                ndcg_at_10=ndcg_at_k(ranked_labeled, labeled, k),
                mrr=reciprocal_rank(ranked_labeled, relevant),
            )
        )

    eligible = [r for r in per_query if r.judged_count >= min_labeled]
    excluded = [r.query for r in per_query if r.judged_count < min_labeled]
    if eligible:
        macro_p = sum(r.precision_at_10 for r in eligible) / len(eligible)
        macro_n = sum(r.ndcg_at_10 for r in eligible) / len(eligible)
        macro_m = sum(r.mrr for r in eligible) / len(eligible)
    else:
        macro_p = macro_n = macro_m = 0.0
    return AnnotationEvaluationResult(
        per_query=per_query,
        macro_precision_at_10=macro_p,
        macro_ndcg_at_10=macro_n,
        macro_mrr=macro_m,
        evaluated_queries=len(eligible),
        excluded_queries=excluded,
        min_labeled=min_labeled,
    )