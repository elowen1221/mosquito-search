"""POST /recommend 接口测试：依赖覆盖注入假数据源（不访问网络）。"""

from fastapi.testclient import TestClient

from mosquito.main import app, get_discover_service, get_recommend_service
from mosquito.ranking.embedding.ranker import EmbeddingRanker
from mosquito.ranking.quality.ranker import QualityRanker
from mosquito.ranking.ranker import Ranker
from mosquito.recommendation.service import RecommendService
from mosquito.services.discover import DiscoverService

PROFILE = {
    "identity": "student",
    "learning_goal": "learn RAG",
    "tech_stack": ["Python"],
    "skill_level": "beginner",
    "time_budget": "few_weeks",
}


class FakeSource:
    """不访问网络的假数据源。"""

    def __init__(self, candidates: list) -> None:
        self.candidates = candidates
        self.queries: list[str] = []

    async def search(self, query: str, limit: int = 50) -> list:
        self.queries.append(query)
        return self.candidates[:limit]


def build_client(candidates: list) -> tuple[TestClient, FakeSource]:
    fake = FakeSource(candidates)
    service = RecommendService(source=fake, base_ranker=Ranker())
    discover_service = DiscoverService(
        source=fake,
        rule_ranker=Ranker(),
        embedding_ranker=EmbeddingRanker(),
        quality_ranker=QualityRanker(),
    )

    app.dependency_overrides[get_recommend_service] = lambda: service
    app.dependency_overrides[get_discover_service] = lambda: discover_service
    return TestClient(app), fake


def test_recommend_returns_fit_scores_and_reasons(make_candidate):
    candidates = [
        make_candidate("rag-tutorial", description="RAG demo with examples", topics=["rag"], size=800),
        make_candidate("rag-platform", language="Rust", description="Distributed platform", stars=500),
    ]
    client, fake = build_client(candidates)

    body = client.post(
        "/recommend",
        json={"query": "想找一个 RAG 项目练手", "limit": 2, "profile": PROFILE},
    ).json()

    assert fake.queries  # profile-aware 检索确实执行了
    assert body["profile"]["identity"] == "student"
    assert body["total"] == 2
    first = body["results"][0]
    assert first["repo"]["full_name"] == "owner/rag-tutorial"
    assert 0.0 <= first["fit_score"] <= 1.0
    assert first["difficulty"] == "easy"
    assert "learning" in first["recommended_for"]
    assert first["why_recommended"]
    assert set(first["fit_breakdown"]) == {
        "topic_fit",
        "stack_fit",
        "difficulty_fit",
        "learning_value",
    }

    scores = [r["final_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_profile_changes_recommendation_order(make_candidate):
    candidates = [
        make_candidate("rag-tutorial", description="RAG demo with examples", topics=["rag"], size=800),
        make_candidate("web-tutorial", description="Web backend demo", topics=["web"], size=800),
    ]
    client, _ = build_client(candidates)

    rag = client.post(
        "/recommend", json={"query": "找个项目练手", "profile": PROFILE}
    ).json()
    web = client.post(
        "/recommend",
        json={
            "query": "找个项目练手",
            "profile": {**PROFILE, "learning_goal": "learn web backend", "tech_stack": ["JavaScript"]},
        },
    ).json()

    assert rag["results"][0]["repo"]["full_name"] == "owner/rag-tutorial"
    assert web["results"][0]["repo"]["full_name"] == "owner/web-tutorial"


def test_discover_response_unaffected(make_candidate):
    candidates = [make_candidate(f"repo-{i}", stars=100 + i) for i in range(3)]
    client, _ = build_client(candidates)

    body = client.post("/discover", json={"query": "python cli", "limit": 3}).json()
    assert body["results"][0]["candidate"]["full_name"]
    assert "fit_score" not in body["results"][0]
    assert body["comparison"] is None
