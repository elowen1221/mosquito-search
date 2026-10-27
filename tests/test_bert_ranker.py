"""BertRanker 单元测试：使用 stub scorer，不依赖真实 torch/transformers。

测试覆盖：
1. 无候选返回空列表
2. Top-N 截断逻辑正确
3. 分数降序排列
4. first_stage_ranker 注入生效
5. _HF_AVAILABLE=False 时 rank() 抛 BertLoadError
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mosquito.models.repo import RepoCandidate
from mosquito.ranking.bert.ranker import DEFAULT_MAX_LENGTH, BertRanker, BertLoadError


def _candidates(n: int = 5) -> list[RepoCandidate]:
    return [
        RepoCandidate(
            name=f"repo{i}",
            full_name=f"owner/repo{i}",
            url=f"https://github.com/owner/repo{i}",
            description=f"desc for repo {i}",
            language="Python",
            topics=["ai", f"topic{i}"],
            stars=i * 10,
            forks=i,
            size=1024,
        )
        for i in range(n)
    ]


class _StubRanker:
    """第一阶段 stub：按顺序给每个候选分配递增分数。"""

    def rank(self, candidates, query):
        from mosquito.models.ranking import RankedRepo
        return [RankedRepo(candidate=c, score=float(i), reasons=[f"stub score={i}"]) for i, c in enumerate(candidates)]


class _StubBertRanker(BertRanker):
    """注入 fake _score_batch，返回可预测分数，跳过 _ensure_loaded。"""

    def __init__(self, scores, **kwargs):
        kwargs.setdefault("first_stage_ranker", _StubRanker())
        super().__init__(**kwargs)
        self._scores = scores
        self._loaded = True
        self._tokenizer = object()
        self._model = object()

    def _score_batch(self, pairs):
        assert len(pairs) == len(self._scores), f"pairs={len(pairs)} scores={len(self._scores)}"
        return list(self._scores)


def test_empty_candidates():
    ranker = _StubBertRanker(scores=[])
    assert ranker.rank([], "q") == []


def test_top_n_truncation():
    """Top-N 截断生效：即使有 50 个候选也只 rerank 前 3。"""
    ranker = _StubBertRanker(scores=[10.0, 20.0, 30.0], top_n=3)
    result = ranker.rank(_candidates(50), "q")
    assert len(result) == 3
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_score_descending():
    """rerank 后按 BERT 分数降序排列。"""
    ranker = _StubBertRanker(scores=[0.5, 0.9, 0.1, 0.7], top_n=4)
    result = ranker.rank(_candidates(4), "q")
    scores = [r.score for r in result]
    assert scores == [0.9, 0.7, 0.5, 0.1]


def test_first_stage_ranker_injection():
    """注入的 first_stage_ranker 被正确调用。"""
    called_with_candidates = None

    class _TrackingRanker:
        def rank(self, candidates, query):
            nonlocal called_with_candidates
            called_with_candidates = candidates
            return _StubRanker().rank(candidates, query)

    ranker = _StubBertRanker(scores=[1.0, 2.0], top_n=2, first_stage_ranker=_TrackingRanker())
    cands = _candidates(5)
    ranker.rank(cands, "q")
    assert called_with_candidates is cands


def test_hf_not_available_raises():
    """torch/transformers 不可用时 rank() 抛 BertLoadError。"""
    import mosquito.ranking.bert.ranker as mod

    original = mod._HF_AVAILABLE
    mod._HF_AVAILABLE = False
    try:
        ranker = BertRanker()
        try:
            ranker.rank(_candidates(3), "q")
        except BertLoadError:
            pass
        else:
            raise AssertionError("应该抛 BertLoadError")
    finally:
        mod._HF_AVAILABLE = original


# --- max_length contract：train / eval / inference 必须一致（冻结协议 256） ---


def test_default_max_length_matches_frozen_training_protocol():
    """BertRanker 默认 inference max_length 必须等于 Formal Training 冻结值 256。"""
    assert DEFAULT_MAX_LENGTH == 256
    assert BertRanker().max_length == 256


def test_tokenizer_called_with_max_length_256_and_truncation():
    """锁定 tokenizer 调用契约：max_length=256、truncation=True、padding=True。"""
    torch = pytest.importorskip("torch")

    captured: dict = {}

    class _RecordingTokenizer:
        def __call__(self, queries, texts, **kwargs):
            captured.update(kwargs)
            captured["n_pairs"] = len(queries)
            return {"input_ids": torch.zeros(len(queries), 4, dtype=torch.long)}

    class _FixedModel:
        def __call__(self, **enc):
            n = enc["input_ids"].shape[0]
            return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]] * n))

    ranker = BertRanker(_tokenizer=_RecordingTokenizer(), _model=_FixedModel(), device="cpu")
    scores = ranker._score_batch([("query", "owner/repo text")])

    assert captured["max_length"] == 256
    assert captured["truncation"] is True
    assert captured["padding"] is True
    assert captured["n_pairs"] == 1
    assert len(scores) == 1


def test_max_length_is_explicit_configurable_not_magic_number():
    """max_length 应为可配置参数，显式覆盖生效。"""
    ranker = BertRanker(max_length=128)
    assert ranker.max_length == 128


def test_inference_batch_size_remains_independent_of_training_batch_size():
    """推理 batch_size 可保持 16，不要求与训练 batch_size=8 一致。"""
    assert BertRanker().batch_size == 16