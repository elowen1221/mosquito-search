"""BertRanker：基于 BERT Cross-Encoder 的二阶段重排序。

架构：候选召回 → 第一阶段快速排序（复用 quality ranker） → 取 Top-N 截断
     → BERT Cross-Encoder 批量 rerank → Top-K 返回。

模型 lazy load：只有显式调用 rank() 时才加载 torch/transformers 权重，
未选择 bert ranking 时不消耗内存。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mosquito.models.ranking import RankedRepo
from mosquito.models.repo import RepoCandidate
from mosquito.ranking.embedding.text import build_repo_text

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# 模型懒加载：未安装 transformers 或未选中 bert 模式时不导入
try:
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover - 无 transformers 环境
    _HF_AVAILABLE = False
    logger.warning("torch/transformers 未安装，BertRanker 不可用。安装：pip install 'mosquito[bert]'")


DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_N = 20  # 第一阶段截断上限，控制 BERT 推理量
DEFAULT_BATCH_SIZE = 16  # 推理吞吐参数，与训练 batch_size 解耦
DEFAULT_MAX_LENGTH = 256  # 与 Formal Training 冻结协议一致（train / eval / inference 单一默认）


class BertLoadError(Exception):
    """BERT 模型加载失败，不允许悄悄退化。"""


def _check_hf() -> None:
    if not _HF_AVAILABLE:
        raise BertLoadError(
            "torch/transformers 未安装。安装 bert 依赖：uv sync --extra bert"
        )


def _build_pair(query: str, repo_text: str) -> str:
    """构造 cross-encoder 输入对：query + repo_text。"""
    return (query.strip(), repo_text.strip())


class BertRanker:
    """Cross-Encoder 重排序器：对第一阶段 Top-N 做批量二分类打分。

    第一阶段默认复用 quality ranker（0.6·relevance + 0.4·quality），
    因为它综合了语义与质量信号，截断后 BERT 只负责精细精排。
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        top_n: int = DEFAULT_TOP_N,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        first_stage_ranker=None,
        device: str | None = None,
        _tokenizer=None,
        _model=None,
    ) -> None:
        self.model_name = model_name_or_path
        self.top_n = max(1, top_n)
        self.batch_size = max(1, batch_size)
        self.max_length = max(1, max_length)
        self.first_stage_ranker = first_stage_ranker
        self.device = device
        self._tokenizer = _tokenizer
        self._model = _model
        self._loaded = _tokenizer is not None and _model is not None

    def _ensure_loaded(self) -> None:
        """lazy load：首次 rank() 时才加载权重。"""
        if self._loaded:
            return
        _check_hf()
        try:
            logger.info("加载 BERT 模型 %s ...", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info("BERT 模型加载完成，device=%s", self.device)
        except Exception as exc:
            raise BertLoadError(f"BERT 模型加载失败 ({self.model_name}): {exc}") from exc

    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """批量 cross-encoder 推理，返回每个 pair 的相关性分数。"""
        self._ensure_loaded()
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                queries = [p[0] for p in batch]
                texts = [p[1] for p in batch]
                enc = self._tokenizer(
                    queries,
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                # RankedRepo.score 约束 >= 0：所有输出压到 (0,1)（对排序是单调变换，不改变相对顺序）
                if logits.shape[-1] == 1:
                    # 单输出回归（如 bge-reranker）：sigmoid
                    batch_scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
                else:
                    # 多分类：取正类 softmax 概率
                    batch_scores = torch.softmax(logits, dim=-1)[:, -1].cpu().tolist()
                scores.extend([float(s) for s in batch_scores])
        return scores

    def rank(self, candidates: list[RepoCandidate], query: str) -> list[RankedRepo]:
        if not candidates:
            return []

        # 第一阶段：快速排序取 Top-N
        first_stage = self.first_stage_ranker
        if first_stage is None:
            from mosquito.ranking.quality.ranker import QualityRanker

            first_stage = QualityRanker()

        try:
            self._ensure_loaded()
        except BertLoadError:
            raise  # 模型加载失败，明确报错，不退化

        first_ranked = first_stage.rank(candidates, query)
        truncated = first_ranked[: self.top_n]

        if not truncated:
            return [
                RankedRepo(
                    candidate=r.candidate,
                    score=r.score,
                    reasons=r.reasons + ["bert: 无候选"],
                )
                for r in first_ranked
            ]

        pairs = [_build_pair(query, build_repo_text(r.candidate)) for r in truncated]
        scores = self._score_batch(pairs)

        reranked: list[RankedRepo] = []
        for rank, score in zip(truncated, scores):
            reranked.append(
                RankedRepo(
                    candidate=rank.candidate,
                    score=round(score, 4),
                    reasons=[f"bert cross-encoder: {score:.4f}"] + rank.reasons[:2],
                )
            )
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked

    def score_all(self, query: str, candidates: list[RepoCandidate]) -> list[tuple[RepoCandidate, float]]:
        """对传入全部 candidates 打分，不调用 production first-stage，不 top_n 截断。

        benchmark harness 专用：返回 (candidate, score) 的原始列表，
        score 与 _score_batch 语义完全一致（softmax 正类概率）。
        production rank() 行为完全不受影响。
        """
        self._ensure_loaded()
        pairs = [_build_pair(query, build_repo_text(c)) for c in candidates]
        scores = self._score_batch(pairs)
        return list(zip(candidates, scores))