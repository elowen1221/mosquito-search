# Mosquito

> Natural-language GitHub repository discovery and ranking.

Mosquito 是一个面向 GitHub 仓库发现的轻量检索与排序项目：把自然语言需求改写成更适合 GitHub Search 的查询，获取候选仓库，再通过 **规则排序、Embedding 语义排序、质量感知排序** 三种策略进行重排，并用人工标注和离线指标评估结果。

这个项目想回答一个很实际的问题：

> 当用户只知道“我想找什么”，而不知道具体仓库名时，怎样找到既语义相关、又真正值得用的仓库？

## 当前状态

- 已跑通端到端发现链路
- 当前包含 20 个评测 query
- 共 359 条人工审阅记录：
  - 高度相关（2）：191
  - 部分相关（1）：160
  - 不相关（0）：7
  - 不确定：1
- 排序、API、检索、缓存、评测和标注流程已有 100+ 测试通过

### 离线评测结果

| Ranker | P@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| embedding | **0.994** | **0.899** | **1.000** |
| quality | 0.983 | 0.853 | 1.000 |
| rules | 0.978 | 0.824 | 1.000 |

当前结果里：

- **Embedding 排序整体语义匹配最好**
- **Quality-aware 排序在成熟框架类查询中有时更占优**
- **Rules 排序作为可解释 baseline 仍然有价值**

例如在错误分析中可以看到：

- 偏任务语义、描述性强的查询，embedding 更容易领先
- “成熟框架/稳定项目”类查询，quality-aware 往往更能利用 stars、更新时间、项目规模等质量信号
- 规则排序虽然整体指标最低，但解释最直接，适合作为 baseline

## Pipeline

```text
自然语言 query
      ↓
QueryPlanner
(1–4 个 GitHub 友好查询)
      ↓
GitHub Search API
(最多 50 个候选)
      ↓
合并 + 去重
      ↓
┌──────────────┬───────────────┬──────────────────┐
│ rules        │ embedding     │ quality          │
│ 启发式规则    │ BGE 向量语义   │ relevance+quality│
└──────────────┴───────────────┴──────────────────┘
      ↓
Top-K
      ↓
人工相关性标注 + 离线评测
(P@10 / NDCG@10 / MRR)
```

## 三种排序策略

### 1. Rule-based

使用可解释的仓库信号：

- 语言匹配
- 更新时间
- stars
- forks
- 仓库规模

### 2. Embedding

使用：

- `fastembed`（ONNX）
- `BAAI/bge-small-zh-v1.5`

将 query 与仓库文本向量化后计算语义相关性。

### 3. Quality-aware

把语义相关性和仓库质量结合：

```text
final_score = 0.6 × relevance + 0.4 × quality
```

它主要用于缓解纯语义检索的一个常见问题：**“描述很像，但项目本身并不好用。”**

## Query Planning 与可靠检索

在线接口和离线评测共用同一套 QueryPlanner 与候选聚合逻辑。

- 自然语言输入会被确定性地改写成 **1–4 个 GitHub 友好查询**
- 多个查询独立检索后按 `full_name` 合并去重
- 检索结果区分：
  - `success`
  - `empty`
  - `rate_limited`
  - `network_error`
  - `api_error`
- 限流、5xx 和临时网络错误使用有界指数退避
- 离线评测支持确定性本地缓存
- 检索失败不会被静默伪装为空结果

## 技术栈

- Python 3.12
- FastAPI
- Pydantic
- httpx
- uv
- fastembed / ONNX
- BAAI/bge-small-zh-v1.5
- pytest

## Quick Start

```bash
uv sync

# 可选：配置 GitHub Token
cp .env.example .env
# 编辑 .env，按需填写 GITHUB_TOKEN

uv run mosquito
# 或
uv run uvicorn mosquito.main:app --reload
```

## API

### `POST /discover`

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `query` | string | 必填 | 自然语言搜索描述 |
| `limit` | int | 10 | 返回结果数量（1–50） |
| `ranking` | string | `rules` | `rules` / `embedding` / `quality` |
| `compare` | bool | `false` | 是否同时返回三种排序结果 |

示例：

```json
{
  "query": "python web framework",
  "limit": 10,
  "ranking": "embedding",
  "compare": true
}
```

## 评测流程

### 1. 构建候选池

```bash
uv run python scripts/build_judgment_pool.py
```

三种 ranker 的候选结果会被合并去重。人工标注文件不暴露 ranker 分数，尽量减少直接排名信息对标注的影响。

### 2. 准备并进行人工标注

```bash
uv run python scripts/prepare_annotations.py
uv run python scripts/annotate.py
```

标注规则：

- `2`：高度相关
- `1`：部分相关
- `0`：不相关
- `u`：不确定
- `s`：跳过
- `q`：退出

标注会逐条保存，可中断后继续。

### 3. 离线评测

```bash
uv run python scripts/evaluate_annotations.py
```

当前输出包括：

- Precision@10
- NDCG@10
- MRR
- 已标注覆盖率
- relevant / highly relevant 数量

## 项目结构

```text
src/mosquito/
├── main.py
├── config.py
├── models/
├── sources/
├── query_planning/
├── ranking/
│   ├── protocol.py
│   ├── rules.py
│   ├── ranker.py
│   ├── embedding/
│   └── quality/
├── evaluation/
└── services/

scripts/
├── build_judgment_pool.py
├── prepare_annotations.py
├── annotate.py
├── evaluate_annotations.py
└── evaluate_github.py

tests/
└── ...
```

## Limitations

当前结果只能作为**阶段性离线评测**，不能直接理解成生产环境 benchmark。

1. 当前标注集来自 Top-K pooled candidate set，不是对 GitHub 全量结果的穷举式相关性标注
2. 当前标签明显偏正样本，只有 7 条显式负例，因此 P@10 和尤其 MRR 很容易饱和
3. 候选池以外的仓库没有被人工判断
4. 当前只有 20 个 query，仍需要更难的负例和更多查询类型
5. 最终 rerank 效果仍受 GitHub Search 初始候选生成质量限制；没有进入候选池的仓库无法被后续排序“救回来”

这些限制会作为后续迭代的一部分继续处理，而不是隐藏在结果里。

## Next Steps

- 扩展 query 集并加入 harder negatives
- 加强 candidate generation，并补 Recall@K / candidate-pool coverage 分析
- 探索 relevance 与 quality 的自适应权重
- 增加轻量 Demo UI 和更直观的排序解释

## Tests

```bash
uv run pytest
```

测试覆盖排序、API、QueryPlanner、GitHub 检索、缓存、评测指标、数据集与人工标注流程。
