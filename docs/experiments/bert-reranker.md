# BERT Cross-Encoder Reranker — 冻结实验记录

本文档记录 **已完成且冻结** 的实验事实。任何数字均不可用于重新选模、调参或重训。

- 冻结日期：2026-10-27
- Checkpoint：`data/bert/best/`（epoch 3）
- 训练脚本：`scripts/train_bert_reranker.py`
- 评测脚本：`scripts/evaluate_bert_reranker.py`
- 原始元数据：`data/bert/metadata.json`

---

## 1. Formal Protocol（冻结，不可修改）

| 参数 | 值 |
|---|---|
| backbone | `bert-base-multilingual-cased` (mBERT) |
| epochs | 3（全部跑完，不使用 early stopping） |
| batch_size | 8 |
| max_length | 256 |
| lr | 2e-5 |
| seed | 42 |
| checkpoint selection | **VAL NDCG@10**（严格 `>`，tie 保留更早 epoch；epoch-1 sentinel baseline） |
| secondary VAL metrics | P@10、MRR（仅报告，不参与选择） |
| TEST | **训练期间完全未加载**（`--skip-test-eval`，5 道门禁） |

**Relevance 定义（不可修改）**

- NDCG@10：使用原始 graded judgment `0/1/2`
- P@10 / MRR：`judgment >= 1` 视为 relevant
- 二分类训练标签：`judgment == 2` → positive(1)，`judgment in {0,1}` → negative(0)

**Split（冻结）**

| Split | Queries | Pairs | Positive | Negative |
|---|---|---|---|---|
| TRAIN | 12 | 233 | 96 | 137 |
| VAL | 3 | 63 | 40 | 23 |
| TEST | 3 | 62 | 55 | 7 |

---

## 2. Formal Result（VAL，用于 checkpoint selection）

| 指标 | 值 |
|---|---|
| best_epoch | **3** |
| best VAL NDCG@10 | **0.8405** |
| VAL P@10 | 1.0 |
| VAL MRR | 1.0 |

每 epoch：

| Epoch | Train Loss | VAL Acc | VAL NDCG@10 | VAL P@10 | VAL MRR |
|---|---|---|---|---|---|
| 1 | 0.6990 | 0.6349 | 0.8243 | 1.0 | 1.0 |
| 2 | 0.6908 | 0.3810 | 0.8284 | 1.0 | 1.0 |
| 3 | 0.6801 | 0.3651 | 0.8405 | 1.0 | 1.0 |

> **解释谨慎**：VAL 只有 3 条 query（63 条 labeled records）。P@10 与 MRR 在如此小的集合上无区分度，不能作为模型排序能力的强证据；NDCG@10 是唯一参与选模的指标，且其绝对差（0.8243 → 0.8405）很小。

---

## 3. Frozen TEST Primary Result（Annotation Pool）

评测配置：frozen TEST split、annotation candidate pool、与训练 `rerank_val()` 完全相同的逻辑
（canonical repo text、`max_length=256`、`softmax(logits)[:, 1]`、同一 `evaluate_annotations`）。

| 指标 | 值 |
|---|---|
| evaluated TEST queries | 3 |
| labeled records | 62 |
| **TEST NDCG@10** | **0.9392** |
| TEST P@10 | 1.0 |
| TEST MRR | 1.0 |

**必须同时记录的限定条件**

- TEST relevant density 极高（55/62 = **88.7%** 为 `judgment >= 1`），候选集内几乎没有不相关项。
- 因此 **P@10 = 1.0 与 MRR = 1.0 是天花板效应，不表示模型达到完美排序**。
- NDCG@10 仍保留 graded 信息（0/1/2），是唯一有一定区分度的指标。
- 3 条 query 的样本量极小，置信区间很宽，不应据此做显著性声明。

---

## 4. Baseline Comparison（Exploratory，非严格 apples-to-apples）

**两类结果必须分开看：**

| 类别 | 状态 | 说明 |
|---|---|---|
| Primary frozen annotation-pool result | **主结果** | 见第 3 节 |
| Discovery-pipeline exploratory result | **探索性，不作为主结论** | 见下 |

**Discovery pipeline 对比的已知限制**

1. **rules** 是当前唯一可运行的**真实离线 baseline**。
2. **embedding / quality** 因 HuggingFace 权重（`BAAI/bge-small-zh-v1.5`）在本环境不可下载，退化为 **zero-vector fallback**，其指标**无数值意义，不可引用**。
3. **Candidate pool 不一致**：annotation pool 为 62 条已标注 repo；discovery pool 为检索返回的最多 50 条（且受 first-stage 截断影响）。
4. **BERT first-stage truncation**：`BertRanker` 先经 first-stage ranker 取 Top-N 再 rerank，zero-vector 下 first-stage 排序退化，导致 discovery-pool 的 BERT 结果同样受限。

**因此：**

> discovery-pipeline 上出现的 `0.9784` 等数字 **不得** 被表述为严格 baseline improvement，也不能作为论文/报告中的主要提升证据。它们仅用于说明 pipeline 可跑通及方向性观察。

---

## 5. Experiment Integrity

| 约束 | 状态 |
|---|---|
| TEST 仅在 protocol / model / checkpoint 冻结后执行 | ✅ |
| TEST 只运行一次 | ✅ |
| TEST 后无 retraining | ✅ |
| 无 hyperparameter tuning | ✅ |
| 无 checkpoint reselection（未比较 best/ 与 final/） | ✅ |
| 无 metric / relevance 定义修改 | ✅ |
| TEST 结果未用于任何模型决策 | ✅ |

---

## 6. 复现命令

```bash
# Formal Training（3 epochs，CPU，TEST 不加载）
uv run --extra bert python scripts/train_bert_reranker.py \
    --model data/models/bert-base-multilingual-cased \
    --output data/bert/ \
    --epochs 3 --batch-size 8 --max-length 256 --lr 2e-5 \
    --skip-test-eval
```

> TEST Final Evaluation 已在冻结后执行一次，产物见 `data/bert/test_eval_frozen.json`（本地，不纳入版本控制）。
