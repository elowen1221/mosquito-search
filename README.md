# Mosquito

> Natural-language GitHub repository discovery and ranking.

Mosquito 把自然语言需求改写成更适合 GitHub Search 的查询，召回候选仓库，再用**多策略排序**（可解释规则、Embedding 语义、Quality 感知、BERT cross-encoder 精排）重新排序，并用人工标注做离线评测。

它想回答一个很实际的问题：

> 当用户只知道「我想找什么」，而不知道具体仓库名时，怎样找到既语义相关、又真正值得用的仓库？

## 当前架构

- **召回**：QueryPlanner → GitHub Search（约 50 个候选）→ 按 `full_name` 合并去重
- **排序**（同一候选池上可切换、可对比）：
  - `rules`：语言、更新时间、stars、forks、仓库规模的加权启发式
  - `embedding`：`bge-small-zh-v1.5`（fastembed / ONNX）语义相似度
  - `quality`：`0.6 × relevance + 0.4 × quality`
  - `bert`：二阶段精排 —— 先由 quality 取 Top-20 截断，再由 cross-encoder 逐条打分
- **服务**：FastAPI（`/discover`、`/health`）+ 移动优先 Web UI
- **评测**：20 个 query、359 条人工标注，NDCG@10 为主指标
- **规模**：206 个单元测试，覆盖排序 / API / 检索 / 缓存 / 评测 / 标注

## Pipeline

```text
自然语言 query
      ↓
QueryPlanner（1–4 个 GitHub 友好查询）
      ↓
GitHub Search API（最多约 50 个候选）
      ↓
合并 + 去重
      ↓
┌──────────┬────────────┬────────────┬──────────────────────┐
│ rules    │ embedding  │ quality    │ bert（二阶段精排）    │
│ 启发式   │ BGE 语义   │ 相关 + 质量 │ quality Top-20 → 精排 │
└──────────┴────────────┴────────────┴──────────────────────┘
      ↓
Top-K 返回
      ↓
人工相关性标注 + 离线评测（NDCG@10 / P@10 / MRR）
```

## 排序策略

### 1. Rule-based

可解释的仓库信号直接加权：语言匹配、更新时间、stars、forks、仓库规模。

### 2. Embedding

把 query 与仓库文本（`full_name` + description + language + topics）向量化，计算余弦相似度。

### 3. Quality-aware

```text
final_score = 0.6 × relevance + 0.4 × quality
```

用于缓解纯语义检索的常见问题：**「描述很像，但项目本身并不好用。」**

### 4. BERT cross-encoder（可选依赖，约 2GB）

`ranking=bert` 时先由 quality 截断到 Top-20，再由 cross-encoder 逐条打 relevance 分（`softmax(logits)[1]`）。
未安装 `--extra bert` 或未配置 `BERT_MODEL_PATH` 时，`ranking=bert` 返回 503，其他 ranking 不受影响。

```bash
uv run python scripts/train_bert_reranker.py                          # 训练
uv run python scripts/evaluate_bert_reranker.py --bert-model data/bert/best   # 评测对比
```

训练目标为二分类：judgment=2 → positive，judgment 0/1 → negative；排序评测则用原始 graded judgment（0/1/2）算 NDCG@10。细节见 [docs/experiments/bert-reranker.md](docs/experiments/bert-reranker.md)。

## 评测口径（重要）

项目里有**两套不同口径**的结果，不能混在一起看：

| 口径 | 说明 | 是否可比较 |
|---|---|---|
| 历史探索性评测 | 早期在 annotation pool 上的对比；其中 embedding / quality 因本地无 BGE 权重退化为 zero-vector，数值无意义，候选池口径也与 benchmark 不同 | ❌ 不可引用 |
| 当前 apples-to-apples benchmark | 固定 RepoCandidate snapshot，所有 ranker 收到**完全相同的候选 ID**，同一 evaluator | ✅ 目前只有 VAL |

当前 apples-to-apples 结果（VAL，3 个 query）：

| Ranker | NDCG@10 | P@10 | MRR |
|---|---:|---:|---:|
| rules | 0.8418 | 1.0 | 1.0 |
| bert | 0.8405 | 1.0 | 1.0 |

- 差距仅 0.0013，且 VAL 只有 3 个 query → **不足以证明 BERT 优于 rules**
- P@10 / MRR = 1.0 属结构性饱和：VAL 没有 judgment=0，所有候选都「相关」，这两个指标没有区分力
- embedding / quality 尚未进入该 benchmark（本地 BGE 权重未就位）
- 训练目标、数据切分、指标定义、benchmark 协议与 caveat 详见 [docs/experiments/bert-reranker.md](docs/experiments/bert-reranker.md)

## Query Planning 与可靠检索

在线接口与离线评测共用同一套 QueryPlanner 与候选聚合逻辑。

- 自然语言输入被确定性改写成 **1–4 个 GitHub 友好查询**，多查询结果按 `full_name` 合并去重
- 检索结果区分 `success` / `empty` / `rate_limited` / `network_error` / `api_error`
- 限流、5xx 与临时网络错误使用有界指数退避；失败不会被静默伪装成空结果
- 离线评测支持确定性本地缓存

## 技术栈

Python 3.12 · FastAPI · Pydantic · httpx · uv · fastembed / ONNX · PyTorch + Transformers（可选 BERT）· pytest

## Quick Start

```bash
uv sync

# 可选：含 BERT（约 2GB）
uv sync --extra bert

cp .env.example .env   # 按需填写 GITHUB_TOKEN / BERT_MODEL_PATH

uv run uvicorn mosquito.main:app --reload   # 打开 http://127.0.0.1:8000/
```

## API

### `POST /discover`

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `query` | string | 必填 | 自然语言搜索描述 |
| `limit` | int | 10 | 返回结果数量（1–50） |
| `ranking` | string | `rules` | `rules` / `embedding` / `quality` / `bert` |
| `compare` | bool | `false` | 同一候选池上返回四种排序结果；`bert` 未启用时 `comparison.bert` 为 `null` |

```json
{
  "query": "python web framework",
  "limit": 10,
  "ranking": "embedding",
  "compare": true
}
```

`GET /health` 返回最小 liveness 信息（版本、BERT 依赖是否可用、是否配置 token），不泄露路径或环境变量。

## 评测数据与标注

```bash
# 1. 构建候选池（多个 ranker 结果合并去重，标注时不暴露 ranker 分数）
uv run python scripts/build_judgment_pool.py

# 2. 交互式人工标注（逐条保存，可中断继续）
uv run python scripts/prepare_annotations.py
uv run python scripts/annotate.py
#    2=高度相关 1=部分相关 0=不相关 u=不确定 s=跳过 q=退出

# 3. 两套口径的评测
uv run python scripts/evaluate_annotations.py                    # 历史口径：P@10 / NDCG@10 / MRR
uv run python scripts/extract_repo_candidates.py \
    --output data/evaluation/repo_candidates.jsonl               # 冻结候选快照
uv run python scripts/benchmark_rankers.py --split VAL --rankers rules bert \
    --repos data/evaluation/repo_candidates.jsonl \
    --bert-checkpoint data/bert/best \
    --evaluation-time 2026-10-27T00:00:00Z \
    --output data/benchmark/val.json                             # apples-to-apples
```

## 项目结构

```text
src/mosquito/
├── main.py             FastAPI 入口（API + Web UI）
├── config.py
├── models/             请求/响应、RepoCandidate、排序结果模型
├── sources/            GitHub Search（重试与错误分类）
├── query_planning/     自然语言 → 1–4 个检索 query
├── ranking/
│   ├── protocol.py     Ranker 协议
│   ├── ranker.py       rules 组合排序
│   ├── rules.py
│   ├── embedding/      BGE 语义排序
│   ├── quality/        relevance + quality 加权
│   └── bert/           cross-encoder 二阶段精排
├── evaluation/         数据集、标注、指标、离线评测
├── services/           DiscoverService 编排
└── web/                移动优先 Web UI（纯 HTML/CSS/JS）

scripts/
├── build_judgment_pool.py      构建候选池
├── prepare_annotations.py
├── annotate.py                 交互式人工标注
├── evaluate_annotations.py     历史口径离线指标
├── evaluate_github.py          真实 GitHub 检索评测
├── train_bert_reranker.py      BERT cross-encoder 训练
├── evaluate_bert_reranker.py   BERT 对比评测
├── extract_repo_candidates.py  冻结 RepoCandidate snapshot
└── benchmark_rankers.py        固定候选池的 ranker 对比 benchmark

deploy/    systemd + nginx 部署配置
tests/     单元测试（不依赖网络与真实模型权重）
```

## 手机 Web UI

移动优先搜索页，零前端构建：

```bash
uv run uvicorn mosquito.main:app --host 0.0.0.0 --port 8000
# 手机浏览器访问 http://<your-server-ip>:8000/
```

包含自然语言搜索框、ranking 选择（quality / embedding / bert）、limit 与 compare 开关、3 个 demo preset query，以及展示 `full_name` / description / language / stars / score / reasons / GitHub 链接的 repo card。

## 生产部署

```bash
# 基础安装（约 200MB）/ 完整安装（含 BERT，约 2GB）
uv sync
uv sync --extra bert

cp .env.example .env
# 编辑 .env：GITHUB_TOKEN（可选）、BERT_MODEL_PATH（可选）

# 生产启动（必须绑定 127.0.0.1，由反向代理负责 HTTPS）
uv run uvicorn mosquito.main:app --host 127.0.0.1 --port 8000
```

### systemd（deploy/mosquito.service）

```bash
sudo cp deploy/mosquito.service /etc/systemd/system/
sudo useradd -r mosquito  # 首次部署
# 在仓库根目录执行：把代码同步到 /opt/mosquito
sudo mkdir -p /opt/mosquito && sudo rsync -a ./ /opt/mosquito/ && sudo chown -R mosquito:mosquito /opt/mosquito
sudo systemctl daemon-reload
sudo systemctl enable --now mosquito
sudo journalctl -u mosquito -f
```

### nginx 反向代理（deploy/nginx.conf）

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/mosquito
sudo ln -s /etc/nginx/sites-available/mosquito /etc/nginx/sites-enabled/
sudo certbot certonly --standalone -d your-domain.com
sudo nginx -t && sudo systemctl reload nginx
```

### 安全要求

- 服务绑定 `127.0.0.1`，由反向代理负责 HTTPS 和公网暴露
- GitHub Token、`BERT_MODEL_PATH` 等私密配置只能通过 `.env` / 环境变量注入
- `.env`、`data/bert/`、`*.bin` / `*.pt` / `*.safetensors` 已在 `.gitignore` 中排除
- `/health` 不泄露 token / 路径 / 环境变量

## Limitations

1. 标注集来自 Top-K pooled candidate set，不是对 GitHub 全量结果的穷举式标注
2. 负例稀少（20 个 query 中仅 7 条 judgment=0），P@10 与 MRR 很容易饱和
3. 排序效果受 GitHub Search 初始召回质量限制；没进候选池的仓库无法被后续排序「救回来」
4. BERT 与 benchmark 目前只在 VAL（3 个 query）上验证，样本量不足以支撑强结论

## Roadmap

- Profile-aware repository recommendation
- Personalized fit scoring by skill stack and learning goal
- Concise repository explanations for learners

## Tests

```bash
uv run pytest
```

排序与评测逻辑均为纯单元测试（假 embedder、stub BERT scorer，不依赖网络与真实模型权重）；GitHubSource 用 `httpx.MockTransport` 模拟；API 通过依赖覆盖注入假数据源。
