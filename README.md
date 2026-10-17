# mosquito

GitHub 仓库发现 MVP：接收自然语言 query，通过 GitHub Search API 获取候选仓库，使用规则评分或 embedding 相似度排序后返回 Top 10。

在线接口与离线评测共用 QueryPlanner 和检索聚合逻辑：将自然语言规划为最多 4 个关键词查询，合并并去重候选仓库后排序。限流与临时故障有界重试；所有规划查询均失败时接口返回错误，不将失败伪装为空结果。在线接口目前不持久化检索缓存。

## 技术栈

- Python 3.12 + FastAPI + Pydantic + uv
- 规则排序：语言 / 更新时间 / stars / forks / 项目规模
- Embedding 排序：fastembed（ONNX）+ `BAAI/bge-small-zh-v1.5`（中英双语，首次使用自动下载）

## 快速开始

```bash
# 安装依赖
uv sync

# 可选：配置 GitHub Token（不配置则以匿名方式访问，有速率限制）
cp .env.example .env
# 编辑 .env，填入 GITHUB_TOKEN（可选）与 EMBEDDING_MODEL（可选）

# 启动服务
uv run mosquito
# 或
uv run uvicorn mosquito.main:app --reload
```

## 接口

### POST /discover

请求参数：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `query` | string | 必填 | 自然语言搜索描述 |
| `limit` | int | 10 | 返回候选数量上限（1-50） |
| `ranking` | string | `rules` | 排序方式：`rules` 规则评分 / `embedding` 向量相似度 / `quality` 质量感知 |
| `compare` | bool | `false` | 为 true 时同时返回所有排序结果（`comparison` 字段） |

请求示例：

```json
{
  "query": "python web framework",
  "limit": 10,
  "ranking": "embedding",
  "compare": true
}
```

响应示例：

```json
{
  "query": "python web framework",
  "total": 50,
  "results": [
    {
      "candidate": {
        "name": "fastapi",
        "full_name": "fastapi/fastapi",
        "url": "https://github.com/fastapi/fastapi",
        "description": "...",
        "language": "Python",
        "stars": 12345,
        "forks": 123,
        "open_issues": 12,
        "topics": ["python", "api"],
        "updated_at": "2026-08-01T10:00:00Z",
        "pushed_at": "2026-08-15T10:00:00Z",
        "size": 2048
      },
      "score": 0.83,
      "reasons": ["embedding 相似度: 0.8300"]
    }
  ],
  "comparison": {
    "rules": [ { "candidate": "...", "score": 70.0, "reasons": ["语言匹配: Python", "..."] } ],
    "embedding": [ { "candidate": "...", "score": 0.83, "reasons": ["embedding 相似度: 0.8300"] } ],
    "quality": [ { "candidate": "...", "score": 0.85, "relevance_score": 0.80, "quality_score": 0.94, "final_score": 0.85, "reasons": ["relevance: 0.7952", "quality: 0.9374 (...)"] } ]
  }
}
```

## 项目结构

```
src/mosquito/
├── main.py            # FastAPI 入口，POST /discover
├── config.py          # 配置：可选 GITHUB_TOKEN / EMBEDDING_MODEL
├── models/            # RepoCandidate / RankedRepo / QualityRankedRepo / 请求响应模型
├── sources/           # Source 抽象 + GitHubSource
├── ranking/
│   ├── protocol.py    # Ranker 协议（三种排序共用）
│   ├── rules.py       # 规则评分
│   ├── ranker.py      # 规则 Ranker
│   ├── embedding/     # EmbeddingRanker + 文本构造 + 编码器（懒加载/text hash 缓存）
│   └── quality/       # QualityScorer（质量特征归一化）+ QualityRanker（relevance+quality）
├── evaluation/        # 离线评测：metrics 纯函数 / dataset 加载 / evaluator
└── services/          # DiscoverService 编排
```

## 离线评测

对 rules / embedding / quality 三种排序在标注数据集上评测（P@10 / NDCG@10 / MRR）。

```bash
uv run python scripts/evaluate_github.py
# 可选参数：--dataset PATH（默认 data/evaluation/github_queries.jsonl）、--top-k N（默认 10）
```

- 标注数据：`data/evaluation/github_queries.jsonl`，每行 `{"query": ..., "judgments": {"owner/repo": 0|1|2}}`
- 评测器只接收已排序的 full_name 列表，不调用 GitHub API，与 ranking 实现解耦
- 指标为纯函数（无外部评测库），P@K 分母取 `min(k, len(ranked))`

### 构建标注候选池

从所有 ranker 的候选并集中生成待标注文件（人工标注对 ranker 身份盲测）：

```bash
uv run python scripts/build_judgment_pool.py
# 可选参数：--queries PATH（默认 data/evaluation/queries.jsonl）、--output PATH、--pool-depth N（默认 10）
```

- QueryPlanner 将每个 query 规划为 1-4 个 GitHub 搜索友好的关键词查询（确定性启发式，无 LLM）
- DiscoveryService 对每个规划查询执行带本地缓存与有界重试的 GitHub 发现：
  - 缓存键 = (source, query, limit)，成功与真空结果均缓存，命中后不再调用 GitHub
  - 限流 / 5xx / 网络错误按指数退避重试（默认 3 次），永久客户端错误不重试
  - 检索结果区分 success / empty / rate_limited / network_error / api_error，失败不会被当作零结果
- 对每个规划查询执行 GitHub 发现，按 full_name 并集去重，三种 ranker 复用同一候选集
- 取各自 Top-K 并集去重，按 full_name 中性排序
- 输出 `data/evaluation/pool_annotations.jsonl`：`query` + 仓库事实（description/language/topics）+ `judgment: null`，不含任何 ranker 来源信息
- 运行结束打印覆盖率报告（成功评估 / 有候选 / 真空 / 失败及原因 / 平均候选数 / 总唯一仓库）

### 人工相关性标注

标注方案：`2` 高度相关 / `1` 部分相关 / `0` 不相关 / `u` 不确定 / `s` 跳过 / `q` 退出。

```bash
# 1. 从候选池生成标注文件（确定性审阅顺序：top-rank > 分歧 > 多来源 > 查询覆盖 > 尾部）
uv run python scripts/prepare_annotations.py
# 可选参数：--pool PATH、--output PATH、--initial-target 150

# 2. 逐条标注（每条立即保存，自动恢复进度）
uv run python scripts/annotate.py
# 可选参数：--annotations PATH、--initial-target 150、--show-provenance

# 3. 基于已标注记录评测（仅用已标注记录，unsure/未标注不计入指标）
uv run python scripts/evaluate_annotations.py
# 可选参数：--annotations PATH、--top-k 10、--min-labeled 5
```

- 标注文件 `data/evaluation/manual_annotations.jsonl`：`query` + 仓库事实 + `retrieved_by`/`ranks` + `judgment` + 可选 `note`，不含任何 ranker 分数
- 同一仓库在不同 query 下独立标注，不跨 query 合并
- 第二审阅者：另存为 `manual_annotations_reviewer2.jsonl`（审阅者身份在文件层面分离）
- 评测报告每查询输出：已标注数 / 相关数（>=1）/ 高度相关数（==2）/ 覆盖率 / P@10 / NDCG@10 / MRR；宏平均仅统计标注充足的 query，并列出被排除的 query

## 测试

```bash
uv run pytest
```

- 三种排序均为纯单元测试（假 embedder，不依赖网络与模型）
- GitHubSource 用 `httpx.MockTransport` 模拟
- API 用依赖覆盖注入假数据源
- 评测指标 / 数据集加载 / 评测器均有独立单元测试
