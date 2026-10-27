"""BERT 训练脚本核心逻辑测试（不下载真实权重）。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from scripts.train_bert_reranker import CrossEncoderDataset, build_pairs


# --- build_pairs regression：走 canonical builder ---

def test_build_pairs_uses_canonical_text_format():
    """确认 build_pairs 输出已与 inference 路径共享同一 builder。"""
    # 构造两条最小 annotation-like 记录
    ann1 = SimpleNamespace(
        query="q1", full_name="fastapi/fastapi",
        description="Modern web framework", language="Python",
        topics=["api", "async"], judgment=2,
    )
    ann2 = SimpleNamespace(
        query="q1", full_name="some/lib",
        description="Other thing", language=None,
        topics=[], judgment=0,
    )
    pairs = build_pairs(["q1"], [ann1, ann2])
    assert len(pairs) == 2
    # ann1 应该被标记为 positive、repo_text 包含 canonical 中文标签
    p_pos = [p for p in pairs if p.label == 1][0]
    p_neg = [p for p in pairs if p.label == 0][0]
    assert "[语言: python]" in p_pos.repo_text
    assert "[主题: api, async]" in p_pos.repo_text
    # ann2 无 language/topics，不应出现任何 bracket tag
    assert "[语言:" not in p_neg.repo_text
    assert p_neg.repo_text.startswith("some/lib other thing")


# --- CrossEncoderDataset / token_type_ids ---

class _FakeBatchEncoding(dict):
    """伪装成 BatchEncoding：dict-like、有 .items() 迭代、每个 value 是带 squeeze 的 0-d tensor。"""

    def __init__(self, **tensors):
        super().__init__(tensors)


class _FakeTokenizer:
    def __init__(self, *, with_token_type_ids=True):
        self._with_ttid = with_token_type_ids

    def save_pretrained(self, *a, **k):
        return None

    def __call__(self, query, repo_text, **kwargs):
        # 模拟真实 tokenizer 输出：3 个 key（含或不含 token_type_ids），shape 都 [1, max_len]
        L = kwargs.get("max_length", 512)
        tok_id = torch.arange(L).unsqueeze(0)
        mask = torch.ones(1, L, dtype=torch.int64)
        data = {
            "input_ids": tok_id,
            "attention_mask": mask,
        }
        if self._with_ttid:
            # 前一半 segment=0，后一半 segment=1 → 保证两 segment 可区分
            ttid = torch.zeros(1, L, dtype=torch.int64)
            ttid[:, L // 2 :] = 1
            data["token_type_ids"] = ttid
        return _FakeBatchEncoding(**data)


def test_dataset_returns_token_type_ids_when_available():
    ds = CrossEncoderDataset(
        pairs=[SimpleNamespace(query="q", repo_text="r", label=1)],
        tokenizer=_FakeTokenizer(with_token_type_ids=True),
        max_length=512,
    )
    item = ds[0]
    assert set(item.keys()) >= {"input_ids", "attention_mask", "token_type_ids", "label"}
    assert item["input_ids"].ndim == 1
    assert item["token_type_ids"].ndim == 1
    # 两个 segment 都有
    ttid = item["token_type_ids"]
    assert (ttid == 0).any().item(), "segment 0 必须存在"
    assert (ttid == 1).any().item(), "segment 1 必须存在"


def test_dataset_graceful_when_tokenizer_has_no_token_type_ids():
    """某些 tokenizer（如 GPT 系）不返回 token_type_ids — dataset 不应强制塞入。"""
    ds = CrossEncoderDataset(
        pairs=[SimpleNamespace(query="q", repo_text="r", label=0)],
        tokenizer=_FakeTokenizer(with_token_type_ids=False),
        max_length=512,
    )
    item = ds[0]
    assert "token_type_ids" not in item
    assert set(item.keys()) == {"input_ids", "attention_mask", "label"}


def test_train_epoch_kwargs_forward_receives_token_type_ids():
    """验证 train_epoch 里 model(**model_kwargs, labels=labels) 确实收到 token_type_ids。"""
    from scripts.train_bert_reranker import train_epoch

    # 一个 fake 模型：只记录 forward kwargs
    received = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kwargs):
            received.update(kwargs)
            return MagicMock(loss=torch.tensor(0.42, requires_grad=True))

    ds = CrossEncoderDataset(
        pairs=[SimpleNamespace(query="q", repo_text="r", label=1)],
        tokenizer=_FakeTokenizer(with_token_type_ids=True),
        max_length=512,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    model = FakeModel()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_epoch(model, loader, optim, "cpu")

    assert "token_type_ids" in received, "forward 必须收到 token_type_ids"
    assert "input_ids" in received
    assert "attention_mask" in received
    # label 通过 labels= 关键字单独传入，forward 会收到
    assert received.get("labels") is not None


# --- --skip-test-eval TEST isolation ---

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_parser_accepts_skip_test_eval_flag():
    """argparse 正确接受 --skip-test-eval，默认为 False。"""
    from scripts.train_bert_reranker import TrainConfig

    # 间接测 argparse：TrainConfig 字段存在且默认值正确
    cfg_default = TrainConfig(
        model_name="m", train_epochs=1, batch_size=1, lr=1e-5,
        warmup_ratio=0.1, max_length=256, smoke_test=False,
    )
    assert cfg_default.skip_test_eval is False

    cfg_flagged = TrainConfig(
        model_name="m", train_epochs=1, batch_size=1, lr=1e-5,
        warmup_ratio=0.1, max_length=256, smoke_test=False,
        skip_test_eval=True,
    )
    assert cfg_flagged.skip_test_eval is True


def _fake_annotation(full_name="owner/x", judgment=2):
    return SimpleNamespace(
        query="any_q", full_name=full_name, description=None,
        language=None, topics=[], judgment=judgment,
    )


def _run_main_with(skip_test_eval: bool):
    """用 mock 包装 main 所有外部依赖，只验证 TEST control flow。"""
    import scripts.train_bert_reranker as tbr
    from scripts.train_bert_reranker import SplitMeta  # SplitMeta 是 dataclass，asdict 需要

    out_dir = Path("/tmp/mock_bert_ckpt")

    fake_split = SplitMeta(
        train_queries=["q_train_1"],
        val_queries=["q_val_1"],
        test_queries=["q_test_1"],
    )
    fake_anns = [
        SimpleNamespace(query="q_train_1", full_name="owner/train1", description=None,
                        language=None, topics=[], judgment=2),
        SimpleNamespace(query="q_val_1", full_name="owner/val1", description=None,
                        language=None, topics=[], judgment=0),
    ]
    fake_pairs_train = [SimpleNamespace(query="q_train_1", repo_text="t", label=1)]
    fake_pairs_val = [SimpleNamespace(query="q_val_1", repo_text="v", label=0)]
    fake_pairs_test = [SimpleNamespace(query="q_test_1", repo_text="tt", label=1)]

    # 记录 build_pairs 被哪些 query 列表调用
    called_with_query_lists = []
    called_evaluate = []

    def _fake_build_pairs(queries, annotations):
        called_with_query_lists.append(list(queries))
        if queries == fake_split.train_queries:
            return fake_pairs_train
        if queries == fake_split.val_queries:
            return fake_pairs_val
        if queries == fake_split.test_queries:
            return fake_pairs_test
        return []

    def _fake_evaluate(model, loader, device):
        called_evaluate.append(True)
        return 0.7

    def _fake_save(*args, **kwargs):
        return None

    def _fake_optim(*args, **kwargs):
        m = SimpleNamespace(zero_grad=lambda: None, step=lambda: None, __class__=torch.optim.AdamW)
        return m

    def _fake_sched(*args, **kwargs):
        return SimpleNamespace(step=lambda: None)

    # 一个 fake model：train_epoch 需要 forward+backward；main history 需要纯 float loss
    import torch
    class _FakeOut:
        """模拟 ModelOutput，带真实 tensor 的 loss/logits，可 backward。"""
        def __init__(self, batch_size=1):
            self.loss = torch.tensor(0.5, requires_grad=True)
            self.logits = torch.zeros(batch_size, 2)  # 2-class classification

    class FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def train(self): return self
        def eval(self): return self
        def to(self, d): return self
        def state_dict(self): return {}
        def load_state_dict(self, *a, **k): return None
        def save_pretrained(self, *a, **k): return None
        def zero_grad(self): return None
        def parameters(self): return [self.p]
        def __call__(self, **kwargs):
            bs = kwargs.get("input_ids", torch.zeros(1)).shape[0] if "input_ids" in kwargs else 1
            return _FakeOut(batch_size=bs)

    fake_model = FakeModule()
    fake_tok = _FakeTokenizer(with_token_type_ids=True)

    # mock tokenizer/model 必须像 HF 类：有 from_pretrained 静态方法
    class _TokCls:
        @staticmethod
        def from_pretrained(*a, **k): return fake_tok
    class _ModelCls:
        @staticmethod
        def from_pretrained(*a, **k): return fake_model

    cfg = tbr.TrainConfig(
        model_name="any", train_epochs=1, batch_size=1, lr=1e-5,
        warmup_ratio=0.1, max_length=256, smoke_test=False,
        skip_test_eval=skip_test_eval,
    )

    # 外部依赖全部 mock：避免下载 HF 权重、读真实标注、创建 DataLoader 等
    with patch.object(tbr, "load_and_split", return_value=fake_split), \
         patch.object(tbr, "load_annotations", return_value=fake_anns), \
         patch.object(tbr, "build_pairs", side_effect=_fake_build_pairs), \
         patch.object(tbr, "evaluate", side_effect=_fake_evaluate), \
         patch.object(tbr, "AutoTokenizer", _TokCls), \
         patch.object(tbr, "AutoModelForSequenceClassification", _ModelCls), \
         patch.object(tbr.torch.optim, "AdamW", side_effect=_fake_optim), \
         patch.object(tbr, "get_linear_schedule_with_warmup", side_effect=_fake_sched):
        rc = tbr.main(cfg, out_dir)

    return {
        "rc": rc,
        "build_pairs_calls": called_with_query_lists,
        "evaluate_calls": called_evaluate,
        "train_queries": fake_split.train_queries,
        "val_queries": fake_split.val_queries,
        "test_queries": fake_split.test_queries,
    }


def test_skip_test_eval_true_does_not_touch_test_pairs():
    """--skip-test-eval 开启时 build_pairs 绝不被 test_queries 调用。"""
    r = _run_main_with(skip_test_eval=True)
    called = r["build_pairs_calls"]
    # TRAIN / VAL 正常调用
    assert r["train_queries"] in called, f"TRAIN 未被 build_pairs 调用: {called}"
    assert r["val_queries"] in called, f"VAL 未被 build_pairs 调用: {called}"
    # TEST 必须完全不在
    assert r["test_queries"] not in called, f"TEST 被 build_pairs 调用了: {called}"


def test_skip_test_eval_true_does_not_evaluate_test():
    """--skip-test-eval 开启时 evaluate 不调用 TEST loader。"""
    r = _run_main_with(skip_test_eval=True)
    # 只要验证一次 evaluate 被调用（VAL）且没对 TEST loader 做任何事
    # 更细粒度：在 _run_main_with 里 build_pairs_calls 已排除 test
    assert r["rc"] == 0
    assert len(r["evaluate_calls"]) == 1, f"evaluate 应只对 VAL 调用 1 次，实际 {len(r['evaluate_calls'])}"


def test_skip_test_eval_false_preserves_original_behavior():
    """--skip-test-eval 未开启时，TEST 路径仍完整可用（向后兼容）。"""
    r = _run_main_with(skip_test_eval=False)
    called = r["build_pairs_calls"]
    assert r["train_queries"] in called
    assert r["val_queries"] in called
    assert r["test_queries"] in called, f"默认 TEST 应被 build_pairs 调用: {called}"
    assert r["rc"] == 0


# =====================================================================
# Checkpoint selection policy: VAL NDCG@10 (strict >, not >=)
# 新增 7 个 targeted regression tests
# =====================================================================

import json
import tempfile


def _run_main_with_selection_policy(
    *,
    skip_test_eval: bool,
    epochs: int,
    ndcg_values: list[float],
    p10_values: list[float],
    mrr_values: list[float],
    acc_values: list[float] | None = None,
) -> dict:
    """扩展版 helper：同时 mock evaluate (分类 accuracy) 和 rerank_val (ranking metrics)，
    捕获 model.save_pretrained 的调用记录、history 内容、metadata.json 内容。

    返回 dict 包含:
      save_calls: list[str] — save_pretrained 被哪些相对路径调用
      rerank_calls_val_queries: list[list[str]] — rerank_val 每次传入的 val_queries
      rerank_calls_test_queries_seen: bool — rerank_val 是否有任何一次收到 test_queries
      metadata: dict — 最终写入的 metadata.json
      history: list[dict] — history 条目列表
      best_epoch: int — metadata["best_epoch"]
      best_val_ndcg10: float — metadata["best_val_ndcg10"]
    """
    import scripts.train_bert_reranker as tbr
    from scripts.train_bert_reranker import SplitMeta

    out_dir = Path(tempfile.mkdtemp(prefix="mock_bert_"))

    fake_split = SplitMeta(
        train_queries=["q_train_1"],
        val_queries=["q_val_1", "q_val_2"],  # 至少 2 个 VAL query，让 rerank_val 有意义
        test_queries=["q_test_1"],
    )
    fake_anns = [
        SimpleNamespace(query="q_train_1", full_name="owner/train1", description=None,
                        language=None, topics=[], judgment=2),
        SimpleNamespace(query="q_val_1", full_name="owner/val1", description=None,
                        language=None, topics=[], judgment=0),
        SimpleNamespace(query="q_val_2", full_name="owner/val2", description=None,
                        language=None, topics=[], judgment=2),
    ]
    fake_pairs_train = [SimpleNamespace(query="q_train_1", repo_text="t", label=1)]
    fake_pairs_val = [
        SimpleNamespace(query="q_val_1", repo_text="v1", label=0),
        SimpleNamespace(query="q_val_2", repo_text="v2", label=1),
    ]
    fake_pairs_test = [SimpleNamespace(query="q_test_1", repo_text="tt", label=1)]

    called_with_query_lists = []
    called_evaluate = []
    rerank_call_args = []   # 每次 rerank_val 被调用时完整参数列表
    save_calls = []

    def _fake_build_pairs(queries, annotations):
        called_with_query_lists.append(list(queries))
        if queries == fake_split.train_queries:
            return fake_pairs_train
        if queries == fake_split.val_queries:
            return fake_pairs_val
        if queries == fake_split.test_queries:
            return fake_pairs_test
        return []

    acc_iter = iter(acc_values or [0.5] * epochs)
    def _fake_evaluate(model, loader, device):
        called_evaluate.append(True)
        return next(acc_iter)

    # 记录并返回预设 reranking metrics
    rr_idx = [0]
    def _fake_rerank_val(model, tokenizer, device, val_queries, annotations, max_length, batch_size):
        rerank_call_args.append({
            "val_queries": list(val_queries),
            "test_queries_in_split": list(fake_split.test_queries),
        })
        i = rr_idx[0]
        rr_idx[0] += 1
        return {
            "val_ndcg10": ndcg_values[i],
            "val_p10": p10_values[i],
            "val_mrr": mrr_values[i],
            "evaluated_queries": len(val_queries),
        }

    def _fake_save(*args, **kwargs):
        # save_pretrained(path) — 记录相对输出目录的路径
        path = args[0] if args else kwargs.get("save_directory", "")
        save_calls.append(str(path))
        return None

    def _fake_optim(*args, **kwargs):
        m = SimpleNamespace(zero_grad=lambda: None, step=lambda: None, __class__=torch.optim.AdamW)
        return m

    def _fake_sched(*args, **kwargs):
        return SimpleNamespace(step=lambda: None)

    class _FakeOut:
        def __init__(self, batch_size=1):
            self.loss = torch.tensor(0.5, requires_grad=True)
            self.logits = torch.zeros(batch_size, 2)

    class FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def train(self): return self
        def eval(self): return self
        def to(self, d): return self
        def state_dict(self): return {}
        def load_state_dict(self, *a, **k): return None
        def save_pretrained(self, *a, **k):
            _fake_save(*a, **k)
        def zero_grad(self): return None
        def parameters(self): return [self.p]
        def __call__(self, **kwargs):
            bs = kwargs.get("input_ids", torch.zeros(1)).shape[0] if "input_ids" in kwargs else 1
            return _FakeOut(batch_size=bs)

    fake_model = FakeModule()
    fake_tok = _FakeTokenizer(with_token_type_ids=True)

    class _TokCls:
        @staticmethod
        def from_pretrained(*a, **k): return fake_tok
    class _ModelCls:
        @staticmethod
        def from_pretrained(*a, **k): return fake_model

    cfg = tbr.TrainConfig(
        model_name="any", train_epochs=epochs, batch_size=1, lr=1e-5,
        warmup_ratio=0.1, max_length=256, smoke_test=False,
        skip_test_eval=skip_test_eval,
    )

    with patch.object(tbr, "load_and_split", return_value=fake_split), \
         patch.object(tbr, "load_annotations", return_value=fake_anns), \
         patch.object(tbr, "build_pairs", side_effect=_fake_build_pairs), \
         patch.object(tbr, "evaluate", side_effect=_fake_evaluate), \
         patch.object(tbr, "rerank_val", side_effect=_fake_rerank_val), \
         patch.object(tbr, "AutoTokenizer", _TokCls), \
         patch.object(tbr, "AutoModelForSequenceClassification", _ModelCls), \
         patch.object(tbr.torch.optim, "AdamW", side_effect=_fake_optim), \
         patch.object(tbr, "get_linear_schedule_with_warmup", side_effect=_fake_sched):
        rc = tbr.main(cfg, out_dir)

    # 读取 metadata.json
    meta_path = out_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    # rerank_val 是否收到过 test_queries
    any_test_seen = any(
        set(arg["val_queries"]) & set(arg["test_queries_in_split"])
        for arg in rerank_call_args
    )

    return {
        "rc": rc,
        "save_calls": save_calls,
        "rerank_call_args": rerank_call_args,
        "rerank_used_test": any_test_seen,
        "metadata": metadata,
        "history": metadata.get("history", []),
        "best_epoch": metadata.get("best_epoch"),
        "best_val_ndcg10": metadata.get("best_val_ndcg10"),
        "selection_metric": metadata.get("selection_metric"),
        "checkpoint_selection_criterion": metadata.get("checkpoint_selection_criterion"),
        "train_queries": fake_split.train_queries,
        "val_queries": fake_split.val_queries,
        "test_queries": fake_split.test_queries,
    }


# --- Test 1: rerank_val 每 epoch 用 VAL 不用 TEST ---

def test_each_epoch_rerank_val_uses_val_queries_not_test():
    """rerank_val 每 epoch 被调用时传入的是 val_queries，永不混入 test_queries。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True,
        epochs=3,
        ndcg_values=[0.3, 0.5, 0.7],
        p10_values=[0.4, 0.5, 0.6],
        mrr_values=[0.35, 0.55, 0.65],
    )
    assert r["rc"] == 0
    assert len(r["rerank_call_args"]) == 3, f"每 epoch 应调 1 次 rerank_val，共 3 次，实际 {len(r['rerank_call_args'])}"
    for call in r["rerank_call_args"]:
        # 每次调用都只传 val_queries
        assert set(call["val_queries"]) == set(r["val_queries"]), f"rerank_val 应只收到 val_queries，实际: {call['val_queries']}"
        # 绝不含 test_queries
        assert set(call["val_queries"]) & set(r["test_queries"]) == set(), "rerank_val 不应收到任何 test_queries"
    assert r["rerank_used_test"] is False, "rerank_val 全程不得接触 TEST"


# --- Test 2: NDCG 高于前一 epoch → checkpoint 替换 ---

def test_higher_ndcg_replaces_checkpoint():
    """NDCG 持续升高：[0.2, 0.4, 0.6] → epoch 1/2/3 各触发一次 save_best，最终 best_epoch=3。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True,
        epochs=3,
        ndcg_values=[0.2, 0.4, 0.6],
        p10_values=[0.3, 0.5, 0.7],
        mrr_values=[0.25, 0.45, 0.65],
    )
    best_save_paths = [p for p in r["save_calls"] if "best" in p]
    assert len(best_save_paths) == 3, f"NDCG 持续升高应触发 3 次 best save，实际 {best_save_paths}"
    assert r["best_epoch"] == 3
    assert r["best_val_ndcg10"] == 0.6


# --- Test 3: NDCG 低于前一 epoch → 不替换 checkpoint ---

def test_lower_ndcg_does_not_replace_checkpoint():
    """NDCG 持续降低：[0.6, 0.4, 0.2] → 只有 epoch 1 触发 save，best_epoch 始终 =1。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True,
        epochs=3,
        ndcg_values=[0.6, 0.4, 0.2],
        p10_values=[0.7, 0.5, 0.3],
        mrr_values=[0.65, 0.45, 0.25],
    )
    best_save_paths = [p for p in r["save_calls"] if "best" in p]
    assert len(best_save_paths) == 1, f"NDCG 持续降低只应触发 epoch 1 的 best save，实际 {best_save_paths}"
    assert r["best_epoch"] == 1
    assert r["best_val_ndcg10"] == 0.6


# --- Test 4: NDCG 完全相同 → 保留更早 epoch ---

def test_equal_ndcg_keeps_earlier_epoch():
    """NDCG 有连续相等：[0.5, 0.5, 0.5] → strict '>' 语义下只 epoch 1 被选；
    变化场景 [0.5, 0.5, 0.6] → best_epoch=3；[0.5, 0.5, 0.5, 0.7] → best_epoch=4。"""
    # 子场景 A: 全部相等
    r_a = _run_main_with_selection_policy(
        skip_test_eval=True, epochs=3,
        ndcg_values=[0.5, 0.5, 0.5],
        p10_values=[0.4] * 3, mrr_values=[0.3] * 3,
    )
    best_save_a = [p for p in r_a["save_calls"] if "best" in p]
    assert len(best_save_a) == 1, f"全部 NDCG 相等时只应 epoch 1 save，实际 {best_save_a}"
    assert r_a["best_epoch"] == 1

    # 子场景 B: 先相等再高 → epoch 3
    r_b = _run_main_with_selection_policy(
        skip_test_eval=True, epochs=3,
        ndcg_values=[0.5, 0.5, 0.6],
        p10_values=[0.4] * 3, mrr_values=[0.3] * 3,
    )
    best_save_b = [p for p in r_b["save_calls"] if "best" in p]
    assert len(best_save_b) == 2, f"[0.5, 0.5, 0.6] 应触发 epoch1 + epoch3 两次 best save，实际 {best_save_b}"
    assert r_b["best_epoch"] == 3
    assert r_b["best_val_ndcg10"] == 0.6


# --- Test 5: val_acc 更高但 NDCG 更低 → 不替换 checkpoint ---

def test_higher_val_acc_but_lower_ndcg_does_not_replace_checkpoint():
    """accuracy 逐步升高但 NDCG 逐步降低 — checkpoint 选择完全不看 accuracy。
    仅 reranking metric 参与决策。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True,
        epochs=3,
        ndcg_values=[0.7, 0.5, 0.3],  # NDCG 递减
        p10_values=[0.8, 0.6, 0.4],
        mrr_values=[0.75, 0.55, 0.35],
        acc_values=[0.6, 0.8, 0.95],  # accuracy 递增（好起来了）
    )
    best_save_paths = [p for p in r["save_calls"] if "best" in p]
    assert len(best_save_paths) == 1, f"accuracy 变好但 NDCG 变差时只能 epoch 1 save best，实际 {best_save_paths}"
    assert r["best_epoch"] == 1
    assert r["best_val_ndcg10"] == 0.7


# --- Test 6: history / metadata 正确记录 best epoch 和 ranking metrics ---

def test_history_and_metadata_record_ranking_metrics():
    """跑完 3 epoch 递增 NDCG → history 每条都含 val_ndcg10 / val_p10 / val_mrr / evaluated_queries；
    metadata 有 selection_metric / checkpoint_selection_criterion / best_epoch / best_val_ndcg10 / best_val_p10 / best_val_mrr。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True,
        epochs=3,
        ndcg_values=[0.3, 0.5, 0.7],
        p10_values=[0.4, 0.5, 0.6],
        mrr_values=[0.35, 0.55, 0.65],
    )
    assert r["selection_metric"] == "val_ndcg@10"
    assert "current_ndcg10 > best_ndcg10" in r["checkpoint_selection_criterion"]
    assert r["best_epoch"] == 3
    assert r["best_val_ndcg10"] == 0.7
    # best_epoch 对应的 val_p10 / val_mrr（epoch 3）
    assert r["metadata"]["best_val_p10"] == 0.6
    assert r["metadata"]["best_val_mrr"] == 0.65
    # history 条目完整性
    hist = r["history"]
    assert len(hist) == 3
    for i, entry in enumerate(hist):
        assert "epoch" in entry
        assert "loss" in entry
        assert "val_acc" in entry
        assert "val_ndcg10" in entry
        assert "val_p10" in entry
        assert "val_mrr" in entry
        assert "evaluated_queries" in entry
        if i == 0:
            assert entry["val_ndcg10"] == 0.3
        elif i == 1:
            assert entry["val_ndcg10"] == 0.5
        else:
            assert entry["val_ndcg10"] == 0.7
    # metadata 顶层字段
    assert "split" in r["metadata"]
    assert r["metadata"]["test_pairs"] == "skipped"


# --- Test 7: --skip-test-eval 仍保证 TEST 完全不加载（回归） ---

def test_skip_test_eval_still_guarantees_test_not_loaded_after_policy_change():
    """新 checkpoint-selection 策略生效后，旧的 TEST isolation 仍完好：
    build_pairs 不被 test_queries 调用、rerank_val 不混入 test_queries、
    metadata.test_pairs = 'skipped'、rerank_val 调用次数 = epoch 数（且每次都只收 val_queries）。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True,
        epochs=3,
        ndcg_values=[0.3, 0.5, 0.7],
        p10_values=[0.4] * 3, mrr_values=[0.35] * 3,
    )
    assert r["rc"] == 0
    # metadata 顶层 test_pairs 必须是字符串 "skipped"
    assert r["metadata"]["test_pairs"] == "skipped"
    # rerank_val 全程零 TEST
    assert r["rerank_used_test"] is False
    # 每次 rerank_val 调用的 val_queries 都是 VAL split
    for call in r["rerank_call_args"]:
        assert set(call["val_queries"]) == set(r["val_queries"])
        assert set(call["val_queries"]) & set(r["test_queries"]) == set()
    # best_epoch 仍正常选出
    assert r["best_epoch"] == 3
    assert r["best_val_ndcg10"] == 0.7


# =====================================================================
# Boundary: checkpoint-baseline sentinel (epoch 1 无论 NDCG 是否为 0 都必须保存)
# =====================================================================


def test_epoch1_zerondcg_still_saves_checkpoint():
    """epoch 1 NDCG@10 = 0.0 — 必须保存为 baseline，best_epoch=1。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True,
        epochs=3,
        ndcg_values=[0.0, 0.2, 0.4],       # epoch 1 = 0.0
        p10_values=[0.0, 0.2, 0.4],
        mrr_values=[0.0, 0.2, 0.4],
    )
    best_save_paths = [p for p in r["save_calls"] if "best" in p]
    # 3 次保存：epoch1 (sentinel) + epoch2 (0.2 > 0.0) + epoch3 (0.4 > 0.2)
    assert len(best_save_paths) == 3, f"应触发 3 次 best save，实际 {best_save_paths}"
    assert r["best_epoch"] == 3
    assert r["best_val_ndcg10"] == 0.4
    # 关键：metadata 顶层 best_val_ndcg10 必须是 float，不得是 None
    assert isinstance(r["best_val_ndcg10"], float)


def test_all_zerondcg_keeps_epoch1_only():
    """三个 epoch NDCG 全为 0.0 → sentinel 只触发 epoch 1，后续 0.0 不覆盖 0.0（严格 > 不满足）。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True, epochs=3,
        ndcg_values=[0.0, 0.0, 0.0],
        p10_values=[0.0] * 3, mrr_values=[0.0] * 3,
    )
    best_save_paths = [p for p in r["save_calls"] if "best" in p]
    assert len(best_save_paths) == 1, f"全零 NDCG 时只能有 epoch 1 sentinel save，实际 {best_save_paths}"
    assert r["best_epoch"] == 1
    assert r["best_val_ndcg10"] == 0.0


def test_zero_then_positive_ndcg_replaces_properly():
    """NDCG 序列 [0.0, 0.0, 0.5] → epoch 1 sentinel 保存；epoch 2 不覆盖（0.0 不 > 0.0）；epoch 3 覆盖。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True, epochs=3,
        ndcg_values=[0.0, 0.0, 0.5],
        p10_values=[0.0, 0.0, 0.5], mrr_values=[0.0, 0.0, 0.5],
    )
    best_save_paths = [p for p in r["save_calls"] if "best" in p]
    assert len(best_save_paths) == 2, f"epoch1 sentinel + epoch3 win，应 2 次 save，实际 {best_save_paths}"
    assert r["best_epoch"] == 3
    assert r["best_val_ndcg10"] == 0.5


def test_metadata_best_metrics_match_saved_checkpoint_after_sentinel():
    """sentinel 触发 baseline 后，metadata.best_val_ndcg10 / best_val_p10 / best_val_mrr 必须与 best_epoch 对应的 history 条目一致。"""
    r = _run_main_with_selection_policy(
        skip_test_eval=True, epochs=3,
        ndcg_values=[0.0, 0.0, 0.5],
        p10_values=[0.1, 0.1, 0.55], mrr_values=[0.05, 0.05, 0.6],
    )
    assert r["best_epoch"] == 3
    assert r["best_val_ndcg10"] == 0.5
    assert r["metadata"]["best_val_p10"] == 0.55
    assert r["metadata"]["best_val_mrr"] == 0.6
    # checkpoint_selection_criterion 必须说明 sentinel 语义
    criterion = r["metadata"]["checkpoint_selection_criterion"]
    assert "best_ndcg10 is None" in criterion, f"criterion 未提及 sentinel: {criterion}"
    assert "baseline" in criterion.lower(), f"criterion 未提及 baseline: {criterion}"


# =====================================================================
# Logits / class-1 scoring contract
# rerank_val 必须用 softmax(logits)[:, 1] 排序 —— 锁定 num_labels=2、class-1=positive 的契约
# =====================================================================


def test_rerank_val_uses_class1_softmax_probability_for_scoring():
    """mock model 输出 [B, 2] logits，控制 class-0 / class-1 大小；断言 rerank_val 的 ranking 完全由 class-1 probability 决定。"""
    from scripts.train_bert_reranker import rerank_val

    # 两条 annotation-like 记录，同 query "q1"
    # repo_high: class-1 logit=5, class-0 logit=0 → softmax → prob1 ≈ 0.993
    # repo_low:  class-1 logit=0, class-0 logit=5 → softmax → prob1 ≈ 0.007
    ann_high = SimpleNamespace(
        query="q1", full_name="owner/high",
        description="high positive", language="Python",
        topics=["api"], judgment=2,
    )
    ann_low = SimpleNamespace(
        query="q1", full_name="owner/low",
        description="low negative", language=None,
        topics=[], judgment=0,
    )

    # fake tokenizer —— 输出固定 shape
    class _Tok:
        def __call__(self, queries, texts, **kw):
            B = len(queries) if isinstance(queries, list) else 1
            return {
                "input_ids": torch.zeros(B, 10, dtype=torch.int64),
                "attention_mask": torch.ones(B, 10, dtype=torch.int64),
            }

    # fake model —— logits 完全受控
    class _Out:
        def __init__(self, logits): self.logits = logits

    class _Model:
        def eval(self): return self
        def __call__(self, **enc):
            B = enc["input_ids"].shape[0]
            logits = torch.zeros(B, 2)
            # 第一个 pair → high repo → class-1 logit=5
            # 第二个 pair → low repo → class-1 logit=0
            logits[0, 1] = 5.0; logits[0, 0] = 0.0
            logits[1, 1] = 0.0; logits[1, 0] = 5.0
            return _Out(logits)

    # 调用 rerank_val —— 内部会 mock 掉 evaluator（否则需要完整 evaluator 环境）
    from unittest.mock import patch
    from mosquito.evaluation.annotation import ManualAnnotation as _MA

    with patch("scripts.train_bert_reranker.evaluate_annotations") as mock_eval:
        # mock evaluator 返回占位结果，我们只关心传入的 rankings dict
        class _Res:
            macro_ndcg_at_10 = 0.9
            macro_precision_at_10 = 1.0
            macro_mrr = 1.0
            evaluated_queries = 1
        mock_eval.return_value = _Res()

        # 调用 rerank_val（纯函数路径）
        result = rerank_val(
            model=_Model(), tokenizer=_Tok(), device="cpu",
            val_queries=["q1"],
            annotations=[ann_high, ann_low],
            max_length=256, batch_size=16,
        )

    # 验证 mock evaluator 收到的 ranking 里 owner/high 排第一
    call_args = mock_eval.call_args
    rankings_arg = call_args[0][0]  # 第一个 positional arg = rankings dict
    assert "q1" in rankings_arg
    ranking = rankings_arg["q1"]
    assert ranking[0] == "owner/high", f"class-1 probability 更高的 repo 必须排第一，实际 ranking={ranking}"
    assert ranking[1] == "owner/low"
    # 返回值应被 mock 占位结果填充
    assert result["val_ndcg10"] == 0.9
    assert result["val_p10"] == 1.0
    assert result["val_mrr"] == 1.0
    assert result["evaluated_queries"] == 1
