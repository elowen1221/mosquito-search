"""QualityScorer：基于 GitHub metadata 的仓库质量评分，与 query 无关。

特征归一化原则：GitHub 指标呈幂律分布，采用对数 + cap 有界归一化，
避免 min-max 对离群值敏感；权重预算限制单一特征（尤其 stars）的贡献上限。

时间冻结：__init__ 接受可选 evaluation_time，用于 deterministic benchmark。
不传入时 production 默认行为不变（用 datetime.now(timezone.utc)）。
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from mosquito.models.repo import RepoCandidate

# 权重预算：stars 只占 popularity 的一部分，总贡献上限 0.3
W_POPULARITY = 0.3
W_MAINTENANCE = 0.3
W_SCALE = 0.2
W_METADATA = 0.2

# 归一化参数
STARS_CAP = 100_000
FORKS_CAP = 100_000
RECENCY_TAU_DAYS = 180.0
SIZE_MIN_KB = 100
SIZE_MAX_KB = 200_000
TOPICS_CAP = 5


@dataclass
class QualityBreakdown:
    """质量评分的分项与总分，便于独立单测与 reasons 输出。"""

    popularity: float
    maintenance: float
    scale: float
    metadata: float
    total: float


class QualityScorer:
    """对单个 RepoCandidate 计算质量分（[0,1]），不依赖 query 与 embedding。

    evaluation_time=None 时 maintenance 用 datetime.now(timezone.utc)，
    传入固定时间则 production 行为保持不变但 benchmark 可 deterministic。
    """

    def __init__(self, evaluation_time: datetime | None = None) -> None:
        self._eval_time = evaluation_time

    def _now(self) -> datetime:
        return self._eval_time if self._eval_time is not None else datetime.now(timezone.utc)

    @staticmethod
    def _log_norm(value: int, cap: int) -> float:
        """对数归一化：log10(x+1)/log10(cap+1)，有界 [0,1]，压缩幂律长尾。"""
        return min(1.0, math.log10(value + 1) / math.log10(cap + 1))

    def _popularity(self, repo: RepoCandidate) -> float:
        """流行度：stars 为主、forks 为辅，均对数归一。"""
        stars = self._log_norm(repo.stars, STARS_CAP)
        forks = self._log_norm(repo.forks, FORKS_CAP)
        return round(0.7 * stars + 0.3 * forks, 4)

    def _maintenance(self, repo: RepoCandidate) -> float:
        """维护新鲜度：pushed_at 距 evaluation_time 指数衰减，τ=180 天。"""
        if repo.pushed_at is None:
            return 0.0
        days = max(0, (self._now() - repo.pushed_at).days)
        return round(math.exp(-days / RECENCY_TAU_DAYS), 4)

    def _scale(self, repo: RepoCandidate) -> float:
        """项目规模：适中区间 [100KB, 200MB] 满分，越界按比例衰减。"""
        if SIZE_MIN_KB <= repo.size <= SIZE_MAX_KB:
            return 1.0
        if repo.size < SIZE_MIN_KB:
            return round(repo.size / SIZE_MIN_KB, 4)
        return round(SIZE_MAX_KB / repo.size, 4)

    def _metadata(self, repo: RepoCandidate) -> float:
        """元数据完整度：description 存在性 + topics 数量（饱和到 5）。"""
        desc = 1.0 if repo.description else 0.0
        topics = min(len(repo.topics), TOPICS_CAP) / TOPICS_CAP
        return round(0.5 * desc + 0.5 * topics, 4)

    def score(self, repo: RepoCandidate) -> QualityBreakdown:
        popularity = self._popularity(repo)
        maintenance = self._maintenance(repo)
        scale = self._scale(repo)
        metadata = self._metadata(repo)
        total = round(
            W_POPULARITY * popularity
            + W_MAINTENANCE * maintenance
            + W_SCALE * scale
            + W_METADATA * metadata,
            4,
        )
        return QualityBreakdown(
            popularity=popularity,
            maintenance=maintenance,
            scale=scale,
            metadata=metadata,
            total=total,
        )