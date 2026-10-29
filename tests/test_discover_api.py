"""POST /discover 接口测试：通过依赖覆盖注入假数据源与假 embedder。"""

from fastapi.testclient import TestClient

from mosquito.main import app, get_discover_service
from mosquito.models.ranking import RankedRepo
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.ranker import QualityRanker
from mosquito.ranking.ranker import Ranker
from mosquito.services.discover import DiscoverService
from mosquito.sources.github import GitHubAuthError


class StubBertRanker:
    """假 BERT ranker（不加载 torch/transformers），分数固定 0.5 以区分来源。"""

    def rank(self, candidates: list, query: str) -> list[RankedRepo]:
        return [
            RankedRepo(candidate=c, score=0.5, reasons=["bert stub"])
            for c in sorted(candidates, key=lambda c: c.full_name, reverse=True)
        ]


class FakeSource:
    """不访问网络的假数据源。"""

    def __init__(self, candidates: list) -> None:
        self.candidates = candidates
        self.last_query: str | None = None
        self.queries: list[str] = []

    async def search(self, query: str, limit: int = 50) -> list:
        self.last_query = query
        self.queries.append(query)
        return self.candidates[:limit]


def build_client(candidates: list, embedder=None, bert_ranker=None) -> tuple[TestClient, FakeSource]:
    fake = FakeSource(candidates)
    service = DiscoverService(
        source=fake,
        rule_ranker=Ranker(),
        embedding_ranker=EmbeddingRanker(embedder=embedder),
        quality_ranker=QualityRanker(embedder=embedder),
        bert_ranker=bert_ranker,
    )

    def override() -> DiscoverService:
        return service

    app.dependency_overrides[get_discover_service] = override
    client = TestClient(app)
    return client, fake


def test_discover_returns_top_10(make_candidate):
    candidates = [make_candidate(f"repo-{i}", stars=1000 - i) for i in range(50)]
    client, fake = build_client(candidates)

    resp = client.post("/discover", json={"query": "python cli"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "python cli"
    assert body["total"] == 50
    assert len(body["results"]) == 10
    assert fake.queries[0] == "python cli"

    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    first = body["results"][0]
    assert "candidate" in first and "score" in first and "reasons" in first
    assert first["candidate"]["full_name"].startswith("owner/")


def test_discover_respects_custom_limit(make_candidate):
    candidates = [make_candidate(f"repo-{i}", stars=100) for i in range(20)]
    client, _ = build_client(candidates)

    resp = client.post("/discover", json={"query": "rust", "limit": 3})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 3


def test_discover_embedding_ranking(make_candidate, controlled_embedder):
    candidates = [make_candidate("a"), make_candidate("b")]
    client, _ = build_client(candidates, embedder=controlled_embedder)

    resp = client.post("/discover", json={"query": "python", "ranking": "embedding"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert all("embedding 相似度" in r["reasons"][0] for r in results)
    assert all(0.0 <= r["score"] <= 1.0 for r in results)


def test_discover_compare_returns_all(make_candidate, controlled_embedder):
    candidates = [make_candidate("a", stars=500), make_candidate("b", stars=10)]
    client, _ = build_client(candidates, embedder=controlled_embedder)

    resp = client.post("/discover", json={"query": "python", "compare": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["comparison"] is not None
    assert len(body["comparison"]["rules"]) == 2
    assert len(body["comparison"]["embedding"]) == 2
    assert len(body["comparison"]["quality"]) == 2
    # 默认 ranking=rules，results 与 comparison.rules 一致
    assert body["results"] == body["comparison"]["rules"]


def test_discover_compare_with_embedding_ranking(make_candidate, controlled_embedder):
    candidates = [make_candidate("a"), make_candidate("b")]
    client, _ = build_client(candidates, embedder=controlled_embedder)

    resp = client.post(
        "/discover", json={"query": "python", "ranking": "embedding", "compare": True}
    )
    body = resp.json()
    assert body["results"] == body["comparison"]["embedding"]


def test_discover_quality_ranking(make_candidate, controlled_embedder):
    candidates = [make_candidate("a", stars=500), make_candidate("b", stars=10)]
    client, _ = build_client(candidates, embedder=controlled_embedder)

    resp = client.post("/discover", json={"query": "python", "ranking": "quality"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    for r in results:
        assert "relevance_score" in r
        assert "quality_score" in r
        assert "final_score" in r
        assert r["score"] == r["final_score"]
        assert 0.0 <= r["relevance_score"] <= 1.0
        assert 0.0 <= r["quality_score"] <= 1.0
    finals = [r["final_score"] for r in results]
    assert finals == sorted(finals, reverse=True)


def test_discover_compare_with_quality_ranking(make_candidate, controlled_embedder):
    candidates = [make_candidate("a"), make_candidate("b")]
    client, _ = build_client(candidates, embedder=controlled_embedder)

    resp = client.post(
        "/discover", json={"query": "python", "ranking": "quality", "compare": True}
    )
    body = resp.json()
    assert body["results"] == body["comparison"]["quality"]


def test_discover_compare_includes_bert_when_enabled(make_candidate, controlled_embedder):
    candidates = [make_candidate("a", stars=500), make_candidate("b", stars=10)]
    client, _ = build_client(
        candidates, embedder=controlled_embedder, bert_ranker=StubBertRanker()
    )

    body = client.post(
        "/discover", json={"query": "python", "ranking": "bert", "compare": True}
    ).json()
    assert body["comparison"]["bert"] is not None
    assert len(body["comparison"]["bert"]) == 2
    assert all(r["reasons"] == ["bert stub"] for r in body["comparison"]["bert"])
    # ranking=bert 时 results 就是 BERT 排序
    assert body["results"] == body["comparison"]["bert"]


def test_discover_compare_bert_is_none_when_disabled(make_candidate, controlled_embedder):
    candidates = [make_candidate("a", stars=500), make_candidate("b", stars=10)]
    client, _ = build_client(candidates, embedder=controlled_embedder)

    resp = client.post("/discover", json={"query": "python", "compare": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["comparison"]["bert"] is None
    assert len(body["comparison"]["rules"]) == 2


def test_discover_rejects_empty_query(make_candidate):
    client, _ = build_client([])
    resp = client.post("/discover", json={"query": ""})
    assert resp.status_code == 422


def test_discover_rejects_invalid_limit(make_candidate):
    client, _ = build_client([])
    resp = client.post("/discover", json={"query": "python", "limit": 0})
    assert resp.status_code == 422


def test_discover_rejects_invalid_ranking(make_candidate):
    client, _ = build_client([])
    resp = client.post("/discover", json={"query": "python", "ranking": "llm"})
    assert resp.status_code == 422


def test_discover_plans_and_deduplicates_candidates(make_candidate):
    client, fake = build_client([make_candidate("a")])
    resp = client.post(
        "/discover", json={"query": "lightweight python API framework with async support"}
    )
    assert resp.status_code == 200
    assert len(fake.queries) > 1
    assert resp.json()["total"] == 1
    assert len(resp.json()["results"]) == 1


def test_discover_reports_source_failure():
    client, fake = build_client([])

    async def fail(query: str, limit: int = 50):
        raise GitHubAuthError("bad token")

    fake.search = fail
    resp = client.post("/discover", json={"query": "python"})
    assert resp.status_code == 502
    assert "bad token" in resp.json()["detail"]
