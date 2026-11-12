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


# ---------- Web UI 静态页面 ----------


def test_web_ui_contains_profile_section():
    client = TestClient(app)
    html = client.get("/").text

    # Profile 输入区域
    for field_id in ("identity", "learning_goal", "tech_stack", "skill_level", "time_budget"):
        assert f'id="{field_id}"' in html
    # skill_level 三档可选
    for level in ("beginner", "intermediate", "advanced"):
        assert level in html
    # 页面主入口是 /recommend
    assert '"/recommend"' in html or "'/recommend'" in html


def test_web_ui_request_body_shape():
    """前端拼装的请求字段与后端模型一致。"""
    html = TestClient(app).get("/").text
    # 前端 fetch /recommend，body 含 query / profile 及各字段名
    for key in ("query", "profile", "identity", "learning_goal", "tech_stack", "skill_level", "time_budget"):
        assert key in html
    # 结果卡片展示所需字段
    for field in ("fit_score", "final_score", "difficulty", "recommended_for", "why_recommended"):
        assert field in html


def test_web_ui_persists_profile_and_recent_searches_locally():
    """Profile / 最近搜索只用浏览器 localStorage，不落服务端。"""
    html = TestClient(app).get("/").text
    assert "localStorage" in html
    assert "mosquito.profile" in html
    assert "mosquito.recent_searches" in html
    # 最近搜索上限 8 条
    assert "MAX_RECENT = 8" in html
    # Profile 默认折叠（摘要 + Edit profile）
    for marker in ("profileSummary", "Edit profile", "profileForm"):
        assert marker in html


def test_web_ui_uses_relative_api_paths_without_token_leak():
    """前端只走相对路径，可直接部署到任意域名；且不暴露 GitHub Token。"""
    html = TestClient(app).get("/").text
    assert "'/recommend'" in html
    assert "'/discover'" in html
    assert "127.0.0.1" not in html
    assert "localhost" not in html
    # 浏览器端不携带/不暴露任何 GitHub 凭证（GITHUB_TOKEN 仅作为服务端环境变量名提示，非凭证值）
    assert "Authorization" not in html
    assert "ghp_" not in html
