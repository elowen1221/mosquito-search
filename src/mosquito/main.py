"""mosquito FastAPI 入口：API + /health + 手机 Web UI。"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from mosquito.config import settings
from mosquito.models.discover import DiscoverRequest, DiscoverResponse
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.ranker import QualityRanker
from mosquito.ranking.ranker import Ranker
from mosquito.services.discover import DiscoverService
from mosquito.sources.github import GitHubSource

logger = logging.getLogger(__name__)

app = FastAPI(title="mosquito", version="0.2.0", description="GitHub 仓库发现 — rules / embedding / quality / bert")

# 尝试加载 BertRanker（lazy import），失败不影响现有 ranker
try:
    from mosquito.ranking.bert import ranker as _bert_mod

    _bert_available = _bert_mod._HF_AVAILABLE
    from mosquito.ranking.bert.ranker import BertRanker
except Exception:
    _bert_available = False
    BertRanker = None  # type: ignore[assignment]


def _make_bert_ranker() -> BertRanker | None:  # type: ignore[valid-type]
    if not _bert_available or BertRanker is None:
        return None
    model_path = settings.bert_model_path
    try:
        return BertRanker(model_name_or_path=model_path) if model_path else BertRanker()
    except Exception as exc:
        logger.warning("BertRanker 初始化失败，bert 模式不可用: %s", exc)
        return None


def get_discover_service() -> DiscoverService:
    return DiscoverService(
        source=GitHubSource(),
        rule_ranker=Ranker(),
        embedding_ranker=EmbeddingRanker(),
        quality_ranker=QualityRanker(),
        bert_ranker=_make_bert_ranker(),
    )


@app.post("/discover", response_model=DiscoverResponse)
async def discover(
    request: DiscoverRequest,
    service: DiscoverService = Depends(get_discover_service),
) -> DiscoverResponse:
    return await service.discover(request.query, request.limit, request.ranking, request.compare)


@app.get("/health")
async def health() -> dict:
    """最小 liveness 检查，不泄露 token / 路径 / 环境变量。

    字段语义：
    - bert_dependencies_available: torch + transformers 是否可以 import（不等同于模型真实可用）
    - bert_model_configured: 是否在配置中设置了 BERT_MODEL_PATH
    - github_authenticated: 是否配置了 GITHUB_TOKEN
    注意：health endpoint 不会主动加载任何大型模型权重。
    """
    return {
        "status": "ok",
        "version": app.version,
        "bert_dependencies_available": _bert_available,
        "bert_model_configured": bool(settings.bert_model_path),
        "github_authenticated": bool(settings.github_token),
    }


# --- 手机 Web UI ---

_WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (_WEB_DIR / "index.html").read_text(encoding="utf-8")