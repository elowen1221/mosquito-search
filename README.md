# Mosquito

> A lightweight GitHub repository discovery and ranking system for natural-language queries.

Mosquito turns a natural-language query into GitHub-friendly search terms, retrieves repository candidates through the GitHub Search API, and reranks them with three strategies: **rule-based**, **embedding-based**, and **quality-aware** ranking.

The project focuses on one practical question:

> When users describe what they want in natural language, can we retrieve repositories that are not only semantically relevant, but also actually useful?

## Current Status

- End-to-end discovery pipeline is runnable.
- 20 evaluation queries are included.
- 359 candidate records were reviewed manually:
  - highly relevant (2): 191
  - partially relevant (1): 160
  - irrelevant (0): 7
  - unsure: 1
- 100+ tests pass across ranking, API, retrieval, cache, evaluation, and annotation workflow.

### Offline evaluation

| Ranker | P@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| embedding | **0.994** | **0.899** | **1.000** |
| quality | 0.983 | 0.853 | 1.000 |
| rules | 0.978 | 0.824 | 1.000 |

The current evaluation shows that **embedding ranking performs best overall on semantic matching**, while the **quality-aware ranker can outperform it on mature-framework queries** where repository quality signals matter more.

Examples observed during error analysis:

- semantic / task-specific queries: embedding tends to lead;
- mature framework queries: quality-aware ranking can lead;
- rule-based ranking remains useful as an interpretable baseline.

## Pipeline

```text
Natural-language query
        ↓
QueryPlanner
(1–4 GitHub-friendly queries)
        ↓
GitHub Search API
(up to 50 candidates)
        ↓
Merge + deduplicate
        ↓
┌──────────────┬───────────────┬──────────────────┐
│ rules        │ embedding     │ quality          │
│ heuristics   │ BGE vectors   │ relevance+quality│
└──────────────┴───────────────┴──────────────────┘
        ↓
Top-K results
        ↓
Manual relevance labels + offline evaluation
(P@10 / NDCG@10 / MRR)
```

## Ranking Strategies

### 1. Rule-based

Uses interpretable repository signals such as:

- language match
- update recency
- stars
- forks
- repository size

### 2. Embedding

Uses:

- `fastembed` (ONNX)
- `BAAI/bge-small-zh-v1.5`

Repository text is embedded and compared against the query for semantic relevance.

### 3. Quality-aware

Combines semantic relevance and repository quality:

```text
final_score = 0.6 × relevance + 0.4 × quality
```

This is intended to avoid a common failure mode of pure semantic search: retrieving something that sounds relevant but is too weak, stale, or low-quality to be useful.

## Query Planning & Reliable Retrieval

The online API and offline evaluation share the same QueryPlanner and discovery aggregation logic.

- Natural-language input is deterministically rewritten into **1–4 GitHub-friendly queries**.
- Planned queries are searched independently and merged by `full_name`.
- Discovery distinguishes `success`, `empty`, `rate_limited`, `network_error`, and `api_error`.
- Rate limits, 5xx responses, and transient network errors use bounded exponential backoff.
- Evaluation discovery supports deterministic local caching.
- A retrieval failure is never silently converted into an empty result.

## Tech Stack

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

# Optional: configure a GitHub token
cp .env.example .env
# Edit .env and set GITHUB_TOKEN if needed.

uv run mosquito
# or
uv run uvicorn mosquito.main:app --reload
```

## API

### `POST /discover`

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural-language search request |
| `limit` | int | 10 | Number of results to return (1–50) |
| `ranking` | string | `rules` | `rules` / `embedding` / `quality` |
| `compare` | bool | `false` | Return all ranking strategies for comparison |

Example:

```json
{
  "query": "python web framework",
  "limit": 10,
  "ranking": "embedding",
  "compare": true
}
```

## Evaluation Workflow

### Build a judgment pool

```bash
uv run python scripts/build_judgment_pool.py
```

Candidate repositories from all rankers are pooled and deduplicated. The annotation file does not expose ranker scores, reducing direct ranking bias during labeling.

### Prepare and annotate

```bash
uv run python scripts/prepare_annotations.py
uv run python scripts/annotate.py
```

Labels:

- `2`: highly relevant
- `1`: partially relevant
- `0`: irrelevant
- `u`: unsure
- `s`: skip
- `q`: quit

Annotations are saved incrementally and can be resumed.

### Evaluate

```bash
uv run python scripts/evaluate_annotations.py
```

The evaluator reports per-query and macro:

- Precision@10
- NDCG@10
- MRR
- judged coverage
- relevant / highly relevant counts

## Project Structure

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

The current numbers should **not** be interpreted as production-grade benchmark results.

1. The labeled set comes from a pooled Top-K candidate set rather than exhaustive relevance judgments over all GitHub results.
2. The current label distribution is highly positive (only 7 explicit negatives), which makes P@10 and especially MRR easy to saturate.
3. Repositories outside the pooled candidate set are not judged.
4. The evaluation currently covers 20 queries and should be expanded with harder negatives and more diverse query types.
5. Ranking quality can depend on GitHub Search candidate generation; reranking cannot recover repositories that never enter the candidate pool.

These limitations are part of the next iteration rather than hidden evaluation assumptions.

## Next Steps

- expand the query set and add harder negatives;
- improve candidate generation and recall-oriented evaluation;
- add Recall@K / candidate-pool coverage analysis;
- test adaptive weighting between semantic relevance and repository quality;
- add a lightweight demo UI and clearer result explanations.

## Tests

```bash
uv run pytest
```

The test suite covers ranking, API behavior, query planning, GitHub retrieval, caching, evaluation metrics, datasets, and annotation workflow.
