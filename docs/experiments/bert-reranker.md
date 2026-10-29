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

- **relevance 定义区分**（不可混淆）：
  - **二分类训练 target**：judgment==2 → positive(1)；judgment in {0,1} → negative(0)。这是 cross-encoder 用的 label scheme，不直接等价于 ranking evaluator。
  - **排序 evaluator relevance**：judgment >= 1 → relevant（P@10/MRR）；judgment 0/1/2 原始分级 → NDCG。当前 annotation 在 VAL/TEST 上 **没有任何 judgment=0 的记录**（VAL 63 条、TEST 62 条 judgment=0 都是 0）。
- **因此 VAL/TEST 的 relevant density（judgment >= 1）= 100%**，P@10 和 MRR **结构性饱和，对任何 ranker 都会输出 1.0**，完全无区分度。
- 之前文档中错误出现过 "TEST relevant density = 88.7%" — 这实际是 **high-relevance density（judgment==2 / 总量）**：VAL 40/63=63.5%，TEST 55/62=88.7%。NDCG@10 唯一能用到 graded 2/1 差异，是 VAL/TEST 上**唯一有区分力的主指标**。
- 3 条 query 的样本量极小，置信区间很宽，不应据此做显著性声明。

当前数据集本质上是 **graded reranking benchmark**（需要 judgment=0 才能成为完整 retrieval benchmark）。

---

## 3a. Apples-to-Apples Graded Reranking Benchmark（VAL，frozen candidate pool）

**目标**：在完全相同的 candidate pool 下、相同的 evaluator、相同的 relevance 定义下，对多个 ranker 做严格公平比较。所有 ranker 对同一 query 收到完全相同的 candidate IDs。

- Harness：`scripts/benchmark_rankers.py`
- Frozen candidate snapshot：`data/evaluation/repo_candidates.jsonl`（355 repos，gitignored）
- BERT 打分 API：`BertRanker.score_all()`（绕过 production first-stage + top_n 截断，全量 cross-encoder）
- Rules deterministic：`--evaluation-time 2026-10-27T00:00:00Z` 注入固定时间

**VAL 结果（2026-10-29 运行，3 queries，63 labeled records）**

| Ranker | NDCG@10 | P@10 | MRR |
|---|---|---|---|
| Rules | 0.8418 | 1.0 | 1.0 |
| BERT (score_all) | 0.8405 | 1.0 | 1.0 |

- Embedding / Quality：因 `BAAI/bge-small-zh-v1.5` 本地权重缺失，**未运行**（harness fail-fast，不 silent fallback）。

**明确结论**

- **当前结果不足以证明 BERT 优于 Rules**。差值仅 0.0013（BERT 略低），且 VAL 仅 3 条 query，统计上无区分力。
- 不描述为 BERT improvement；不据此重训、调参或改变冻结 BERT protocol。
- P@10 / MRR = 1.0 是结构性饱和（VAL judgment=0 为 0），无比较能力；NDCG@10 是唯一有分级信号的指标。
- BERT NDCG@10 = 0.8405 与 Formal Training 的 VAL 选模结果完全一致，验证了 `score_all()` 与 production scoring 等价。

**Snapshot Provenance / 可复现性限制**

- `data/evaluation/repo_candidates.jsonl` 由 `scripts/extract_repo_candidates.py` 从 `data/evaluation/discovery_cache.json` + annotations 提取，输出稳定排序（full_name 升序），SHA256 可比较。
- **Discovery cache 本身不是 git 内可复现 artifact**（含实时 GitHub 搜索结果，不在版本控制中）。同一冻结输入重新生成 snapshot 时内容排序稳定、SHA256 可比较，但上游 cache 一旦变化则 snapshot 内容不同。
- 论文 artifact release 时需另行发布 frozen candidate snapshot（含 manifest）以保证完全可复现。manifest sidecar 记录：snapshot SHA256、repo count、source annotations SHA256、source cache SHA256 + entries、extractor git commit、generation timestamp。

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
