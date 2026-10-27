"""BERT Cross-Encoder 训练脚本。

用法：
    # 完整训练
    uv run --extra bert python scripts/train_bert_reranker.py

    # 小样本/smoke 测试（1 个 epoch，每个 split 只取 5 个 query）
    uv run --extra bert python scripts/train_bert_reranker.py --smoke-test

    # 指定输出目录和模型
    uv run --extra bert python scripts/train_bert_reranker.py --model bert-base-multilingual-cased --output data/bert/

切分冻结：12 train / 3 val / 3 test，按 query 分组，零泄漏。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from mosquito.evaluation.annotation import DEFAULT_MANUAL_ANNOTATIONS_PATH, load_annotations
from mosquito.evaluation.evaluator import evaluate_annotations
from mosquito.ranking.embedding.text import build_repo_text_from_fields

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 固定 seed — 训练可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 冻结的 query 切分（按正负样本分布分层后手工冻结）
# Test: 必须有丰富正样本的 query，用于最终评测
# Val: 用于调阈值/超参数
# Train: 训练
TEST_QUERIES = [
    "LLM agent framework with tool calling and memory",
    "open source tool to monitor changes on websites",
    "python web framework",
]
VAL_QUERIES = [
    "lightweight python API framework with async support",
    "fast fuzzy string matching library for python",
    "self hosted alternative to cloud AI chat applications",
]
TRAIN_QUERIES = [
    "actively maintained python web framework with good documentation",
    "beginner friendly machine learning library with clear examples",
    "lightweight local vector database for a personal AI project",
    "local RAG application framework for private documents",
    "minimal dependency markdown parser for python",
    "old but stable python library for parsing command line arguments",
    "production ready FastAPI authentication library",
    "python semantic search library that works well on CPU",
    "small maintained CLI library for building interactive terminal apps",
    "tool for evaluating RAG retrieval quality",
    "我想给个人 AI 助手加一个长期记忆系统，不想部署很重的服务",
    "适合本地运行的轻量级 Python 向量数据库",
]

# 二分类：judgment=2 → positive, judgment in (0,1) → negative
POSITIVE = 1
NEGATIVE = 0


@dataclass
class SplitMeta:
    train_queries: list[str]
    val_queries: list[str]
    test_queries: list[str]


@dataclass
class TrainConfig:
    model_name: str
    train_epochs: int
    batch_size: int
    lr: float
    warmup_ratio: float
    max_length: int
    smoke_test: bool
    skip_test_eval: bool = False


@dataclass
class Pair:
    query: str
    repo_text: str
    label: int  # 0 or 1


def load_and_split(path: str | Path = DEFAULT_MANUAL_ANNOTATIONS_PATH) -> SplitMeta:
    """验证切分完整性：三条清单必须覆盖全部 18 条标注 query。"""
    all_queries = set(TRAIN_QUERIES) | set(VAL_QUERIES) | set(TEST_QUERIES)
    recs = load_annotations(path)
    existing = {r.query for r in recs if r.judgment in (0, 1, 2)}
    missing = existing - all_queries
    extra = all_queries - existing
    if missing or extra:
        raise ValueError(f"切分不完整: missing={missing}, extra={extra}")
    if set(TRAIN_QUERIES) & set(VAL_QUERIES) or set(TRAIN_QUERIES) & set(TEST_QUERIES) or set(VAL_QUERIES) & set(TEST_QUERIES):
        raise ValueError("切分有泄漏：query 同时出现在多个 split")
    logger.info("切分: train=%d val=%d test=%d", len(TRAIN_QUERIES), len(VAL_QUERIES), len(TEST_QUERIES))
    return SplitMeta(TRAIN_QUERIES, VAL_QUERIES, TEST_QUERIES)


def build_pairs(queries: list[str], annotations: list) -> list[Pair]:
    """按 query 列表训练集构造 (query, repo_text, label) pairs。

    repo_text 调用单一 canonical builder build_repo_text_from_fields，
    与推理路径（BertRanker → build_repo_text）保持完全一致。
    """
    by_q = {}
    for r in annotations:
        if r.judgment not in (0, 1, 2):
            continue
        by_q.setdefault(r.query, []).append(r)

    pairs: list[Pair] = []
    for q in queries:
        for ann in by_q.get(q, []):
            label = POSITIVE if ann.judgment == 2 else NEGATIVE
            repo_text = build_repo_text_from_fields(
                full_name=ann.full_name,
                description=ann.description,
                language=ann.language,
                topics=ann.topics,
            )
            pairs.append(Pair(query=q, repo_text=repo_text, label=label))

    pos = sum(1 for p in pairs if p.label == POSITIVE)
    neg = sum(1 for p in pairs if p.label == NEGATIVE)
    logger.info("pairs: total=%d pos=%d neg=%d (%.0f%%)", len(pairs), pos, neg, 100 * pos / max(pos + neg, 1))
    return pairs


class CrossEncoderDataset(Dataset):
    def __init__(self, pairs: list[Pair], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        enc = self.tok(
            p.query,
            p.repo_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # 透传 tokenizer 原始输出的所有 key（含 token_type_ids）。
        # BatchEncoding 对不存在的 key（某些 tokenizer 不产出 token_type_ids）也安全。
        item: dict = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(p.label, dtype=torch.long)
        return item


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = batch.pop("label").to(device)
        model_kwargs = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**model_kwargs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        labels = batch.pop("label").to(device)
        model_kwargs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**model_kwargs)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def rerank_val(
    model, tokenizer, device,
    val_queries: list[str],
    annotations: list,
    max_length: int,
    batch_size: int = 16,
) -> dict[str, float]:
    """用当前 BERT model 对每个 VAL query 的已标注候选做 reranking evaluation。

    纯离线：候选池来自 VAL annotations（manual_annotations.jsonl 里 judgment ∈ {0,1,2} 的记录），
    用 canonical builder 构造 repo_text 与训练时完全一致；不调用 GitHub API、不访问 TEST。
    返回 {"val_ndcg10": float, "val_p10": float, "val_mrr": float, "evaluated_queries": int}。
    """
    from mosquito.evaluation.annotation import ManualAnnotation

    model.eval()

    # 按 query 分组已标注 VAL 记录
    by_q: dict[str, list[ManualAnnotation]] = {}
    for r in annotations:
        if r.query in set(val_queries) and r.judgment in (0, 1, 2):
            by_q.setdefault(r.query, []).append(r)

    rankings: dict[str, list[str]] = {}
    for query, recs in by_q.items():
        # 三元组：(full_name, query, repo_text) — full_name 用于 evaluator 匹配 labels
        triples = [
            (
                r.full_name,
                query,
                build_repo_text_from_fields(
                    full_name=r.full_name,
                    description=r.description,
                    language=r.language,
                    topics=r.topics,
                ),
            )
            for r in recs
        ]
        if not triples:
            continue

        # 批量打分：tokenizer 吃 (query, repo_text)，同 BertRanker 推理路径一致
        scores: list[float] = []
        for i in range(0, len(triples), batch_size):
            chunk = triples[i : i + batch_size]
            enc = tokenizer(
                [t[1] for t in chunk],   # query
                [t[2] for t in chunk],   # repo_text
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits  # [B, 2] for binary classifier
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())  # 正类概率作为 score

        ranked = sorted(zip(triples, scores), key=lambda x: x[1], reverse=True)
        rankings[query] = [t[0] for t, _ in ranked]  # full_name 列表 → evaluator 按 full_name 匹配 labels

    # 复用 evaluator.evaluate_annotations（NDCG: 0/1/2 graded；P@10/MRR: judgment >= 1）
    val_anns = [r for r in annotations if r.query in set(val_queries) and r.judgment in (0, 1, 2)]
    result = evaluate_annotations(rankings, val_anns, k=10, min_labeled=3)

    return {
        "val_ndcg10": result.macro_ndcg_at_10,
        "val_p10": result.macro_precision_at_10,
        "val_mrr": result.macro_mrr,
        "evaluated_queries": result.evaluated_queries,
    }


def main(cfg: TrainConfig, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 切分验证
    split = load_and_split()

    # 2. 加载标注 → 构造 pairs
    annotations = load_annotations()
    train_pairs = build_pairs(split.train_queries, annotations)
    val_pairs = build_pairs(split.val_queries, annotations)
    if cfg.skip_test_eval:
        test_pairs = []  # 完全不访问 TEST：不 build、不 ds/loader、不 evaluate
        logger.info("TEST: NOT LOADED (--skip-test-eval)")
    else:
        test_pairs = build_pairs(split.test_queries, annotations)

    if cfg.smoke_test:
        logger.info("SMOKE TEST: 每个 split 截断到 2 个 query")
        train_pairs = [p for p in train_pairs if p.query in set(split.train_queries[:2])]
        val_pairs = [p for p in val_pairs if p.query in set(split.val_queries[:2])]
        if not cfg.skip_test_eval:
            test_pairs = [p for p in test_pairs if p.query in set(split.test_queries[:2])]
        cfg.train_epochs = 1

    if not train_pairs:
        logger.error("训练 pairs 为空，退出")
        return 1

    # 3. 加载模型
    logger.info("加载模型 %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1}
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("device=%s, model params: %dM", device, sum(p.numel() for p in model.parameters()) // 1_000_000)

    # 4. DataLoader
    train_ds = CrossEncoderDataset(train_pairs, tokenizer, cfg.max_length)
    val_ds = CrossEncoderDataset(val_pairs, tokenizer, cfg.max_length) if val_pairs else None
    test_ds = CrossEncoderDataset(test_pairs, tokenizer, cfg.max_length) if (test_pairs and not cfg.skip_test_eval) else None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size) if test_ds else None

    # 5. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(train_loader) * cfg.train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

    # 6. 训练循环 — VAL reranking evaluation + checkpoint 基于 NDCG@10
    best_ndcg10: float | None = None  # None = 尚未有 checkpoint（epoch 1 无论 NDCG 多低都必须保存 baseline）
    best_epoch = 0
    history = []
    for epoch in range(cfg.train_epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_acc = evaluate(model, val_loader, device) if val_loader else 0.0

        # 每 epoch 都做 VAL reranking evaluation（纯离线、不访问 TEST）
        rr = rerank_val(
            model, tokenizer, device,
            split.val_queries, annotations,
            cfg.max_length, cfg.batch_size,
        )

        if cfg.skip_test_eval:
            test_acc = None
        else:
            test_acc = evaluate(model, test_loader, device) if test_loader else 0.0

        entry = {
            "epoch": epoch + 1,
            "loss": loss,
            "val_acc": val_acc,
            "val_ndcg10": rr["val_ndcg10"],
            "val_p10": rr["val_p10"],
            "val_mrr": rr["val_mrr"],
            "evaluated_queries": rr["evaluated_queries"],
        }
        if test_acc is not None:
            entry["test_acc"] = test_acc
        history.append(entry)

        # 日志
        if cfg.skip_test_eval:
            logger.info(
                "epoch %d/%d  loss=%.4f  val_acc=%.4f  val_ndcg@10=%.4f  val_p@10=%.4f  val_mrr=%.4f  TEST=skipped",
                epoch + 1, cfg.train_epochs, loss, val_acc,
                rr["val_ndcg10"], rr["val_p10"], rr["val_mrr"],
            )
        else:
            logger.info(
                "epoch %d/%d  loss=%.4f  val_acc=%.4f  val_ndcg@10=%.4f  val_p@10=%.4f  val_mrr=%.4f  test_acc=%.4f",
                epoch + 1, cfg.train_epochs, loss, val_acc,
                rr["val_ndcg10"], rr["val_p10"], rr["val_mrr"], test_acc,
            )

        # Checkpoint selection：
        #   - best_ndcg10 is None（尚未有 checkpoint）→ 必须保存 baseline（epoch 1 无论 NDCG 是否为 0 都必须入选）
        #   - current_ndcg10 > best_ndcg10 → 严格大于才覆盖（相等时保留更早 epoch）
        current_ndcg = rr["val_ndcg10"]
        if best_ndcg10 is None or current_ndcg > best_ndcg10:
            best_ndcg10 = current_ndcg
            best_epoch = epoch + 1
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
            if best_epoch == 1:
                logger.info("  → baseline saved: epoch 1  val_ndcg@10=%.4f (sentinel-triggered)", best_ndcg10)
            else:
                logger.info("  → new best: epoch %d  val_ndcg@10=%.4f", best_epoch, best_ndcg10)

    # 7. 保存 final + metadata
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    # 找 best_epoch 对应的 reranking metrics（history 里按 epoch 查）
    best_entry = next((h for h in history if h["epoch"] == best_epoch), None)
    meta = {
        "config": asdict(cfg),
        "split": asdict(split),
        "selection_metric": "val_ndcg@10",
        "checkpoint_selection_criterion": (
            "best_ndcg10 is None（首次 epoch-1 baseline 必须保存）"
            " or current_ndcg10 > best_ndcg10（仅严格大于时覆盖；相等时保留更早 epoch）"
        ),
        "best_epoch": best_epoch,
        "best_val_ndcg10": best_ndcg10,
        "best_val_p10": best_entry["val_p10"] if best_entry else None,
        "best_val_mrr": best_entry["val_mrr"] if best_entry else None,
        "history": history,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": "skipped" if cfg.skip_test_eval else len(test_pairs),
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("完成。best_epoch=%d  best_val_ndcg@10=%.4f，模型已保存到 %s", best_epoch, best_ndcg10, output_dir)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Cross-Encoder 训练")
    parser.add_argument("--model", default="bert-base-multilingual-cased", help="HuggingFace 模型名")
    parser.add_argument("--output", default="data/bert/", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256, help="cross-encoder joint encoding 上限（tokenize max_length）")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true", help="Formal Training 必须携带：完全不加载/构建/评估 TEST split")
    args = parser.parse_args()
    cfg = TrainConfig(
        model_name=args.model,
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=0.1,
        max_length=args.max_length,
        smoke_test=args.smoke_test,
        skip_test_eval=args.skip_test_eval,
    )
    raise SystemExit(main(cfg, Path(args.output)))