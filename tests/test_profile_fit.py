"""Profile 模型、profile-aware planning 与 fit scoring 单元测试。"""

import pytest
from pydantic import ValidationError

from mosquito.models.profile import Profile
from mosquito.query_planning.planner import QueryPlanner
from mosquito.recommendation.fit import assess


def make_profile(**overrides) -> Profile:
    base = {
        "identity": "student",
        "learning_goal": "learn RAG",
        "tech_stack": ["Python"],
        "skill_level": "beginner",
    }
    base.update(overrides)
    return Profile(**base)


# ---------- Profile 模型 ----------


def test_profile_defaults():
    profile = make_profile()
    assert profile.skill_level == "beginner"
    assert profile.time_budget is None


@pytest.mark.parametrize("skill_level", ["expert", "", "beginer"])
def test_profile_rejects_unknown_skill_level(skill_level):
    with pytest.raises(ValidationError):
        make_profile(skill_level=skill_level)


def test_profile_requires_identity_and_goal():
    with pytest.raises(ValidationError):
        Profile(identity="", learning_goal="learn RAG")
    with pytest.raises(ValidationError):
        Profile(identity="student", learning_goal="")


# ---------- Profile-aware Query Planning ----------


def test_planner_adds_profile_variants():
    planned = QueryPlanner().plan("想找一个 RAG 项目练手", make_profile())
    assert "rag python tutorial" in planned
    assert "rag python example" in planned
    assert "retrieval augmented generation python" in planned
    assert len(planned) <= 4


def test_planner_without_profile_unchanged():
    assert QueryPlanner().plan("想找一个 RAG 项目练手") == ["rag"]


# ---------- fit scoring ----------


def test_topic_and_stack_fit(make_candidate):
    profile = make_profile()
    rag_python = make_candidate(
        "rag-tutorial", description="Simple RAG demo", topics=["rag"], size=800
    )
    rust_platform = make_candidate(
        "big-platform", language="Rust", description="Distributed platform", size=80000
    )

    good = assess(rag_python, profile)
    bad = assess(rust_platform, profile)
    assert good.topic_fit == 1.0
    assert bad.topic_fit == 0.0
    assert good.stack_fit == 1.0
    assert bad.stack_fit == 0.3
    assert good.fit_score > bad.fit_score


def test_difficulty_tiers_and_skill_alignment(make_candidate):
    beginner = make_profile()
    advanced = make_profile(skill_level="advanced")
    small = make_candidate("tiny-demo", size=500)
    huge = make_candidate("large-service", size=50000)

    assert assess(small, beginner).difficulty == "easy"
    assert assess(huge, beginner).difficulty == "hard"
    assert assess(huge, beginner).difficulty_fit == 0.25
    assert assess(huge, advanced).difficulty_fit == 1.0


def test_learning_value_prefers_tutorial(make_candidate):
    profile = make_profile()
    tutorial = make_candidate("rag-examples", description="RAG examples and demos", size=800)
    plain = make_candidate("rag-core", description="Production RAG library", size=800)

    assert assess(tutorial, profile).learning_value == 1.0
    assert assess(plain, profile).learning_value == 0.35
    assert "learning" in assess(tutorial, profile).recommended_for
    assert "source_reading" in assess(plain, profile).recommended_for


def test_why_recommended_is_template_based(make_candidate):
    profile = make_profile()
    result = assess(make_candidate("rag-demo", description="RAG demo with examples", size=900), profile)

    assert any("Python" in reason for reason in result.why_recommended)
    assert any("RAG" in reason for reason in result.why_recommended)
    assert len(result.why_recommended) <= 4


def test_specific_rag_repo_beats_generic_tutorial_collection(make_candidate):
    """name 命中（专一仓库）应压过"多主题合集里的一个 rag topic"（泛 AI 教程合集不应 1.0）。"""
    profile = make_profile()
    specific = make_candidate(
        "all-in-rag", description="RAG 技术全栈指南", topics=["rag", "llm", "embedding"], size=2000
    )
    generic = make_candidate(
        "AI-Deep-Learning-Tutorials",
        description="Collection of AI courses: deep learning, ML, document extraction, RAG, AI agents",
        topics=[
            "ai-agents", "ai-governance", "artificial-intelligence", "autograding",
            "deep-learning", "document-ai", "eu-ai-act", "google-colab", "humanoid-robotics",
            "jupyter-notebook", "machine-learning", "mlops", "model-risk-management",
            "mujoco", "online-course", "predictive-maintenance", "python", "rag",
            "responsible-ai", "synapsa",
        ],
        size=2000,
    )

    assert assess(specific, profile).topic_fit == 1.0
    assert assess(generic, profile).topic_fit <= 0.6
    assert assess(specific, profile).fit_score > assess(generic, profile).fit_score


def test_description_only_full_concept_is_medium_match(make_candidate):
    """description 里出现完整概念（缩写展开）给中等证据，而不是满分。"""
    profile = make_profile()
    repo = make_candidate(
        "some-project", description="A project that uses retrieval augmented generation", size=2000
    )
    assert assess(repo, profile).topic_fit == 0.65


def test_different_profiles_shift_fit_score(make_candidate):
    repo = make_candidate("python-rag-demo", description="RAG demo", topics=["rag"], size=900)
    rag_student = make_profile()
    web_student = make_profile(learning_goal="learn web backend", tech_stack=["JavaScript"])

    assert assess(repo, rag_student).fit_score > assess(repo, web_student).fit_score
