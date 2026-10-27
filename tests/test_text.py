"""Repo 检索文本构造测试。"""

from mosquito.ranking.embedding.text import build_repo_text, build_repo_text_from_fields, normalize


def test_build_repo_text_full(make_candidate):
    repo = make_candidate("demo", description="A demo repo", language="Python", topics=["cli", "api"])
    text = build_repo_text(repo)
    assert "owner/demo" in text
    assert "a demo repo" in text
    assert "[语言: python]" in text
    assert "[主题: cli, api]" in text


def test_build_repo_text_skips_empty_fields(make_candidate):
    repo = make_candidate("demo", description=None, language=None, topics=[])
    assert build_repo_text(repo) == "owner/demo"


def test_build_repo_text_normalizes_whitespace(make_candidate):
    repo = make_candidate("demo", description="  A   demo   repo  ")
    text = build_repo_text(repo)
    assert "  " not in text
    assert text.startswith("owner/demo a demo repo")


def test_normalize():
    assert normalize("  Hello   World  ") == "hello world"
    assert normalize("") == ""


# --- canonical builder 额外测试（Task 2 regression） ---

def test_build_repo_text_from_fields_full():
    text = build_repo_text_from_fields(
        full_name="fastapi/fastapi",
        description="Modern, fast web framework for building APIs",
        language="Python",
        topics=["api", "web", "async"],
    )
    assert text == (
        "fastapi/fastapi modern, fast web framework for building "
        "apis [语言: python] [主题: api, web, async]"
    )


def test_build_repo_text_from_fields_skips_empty():
    text = build_repo_text_from_fields("owner/x")
    assert text == "owner/x"


def test_build_repo_text_from_fields_ignores_empty_topics():
    # 空字符串和 None topic 都要过滤
    text = build_repo_text_from_fields("owner/x", topics=["", None, "cli"])
    assert "[主题: cli]" in text
    assert "[主题: ,  , cli]" not in text


def test_build_repo_text_and_from_fields_agree(make_candidate):
    """RepoCandidate 路径与 kwargs 路径产出完全一致 → train/infer 永不漂移。"""
    repo = make_candidate("demo", description="Desc", language="Python", topics=["cli"])
    via_repo = build_repo_text(repo)
    via_kwargs = build_repo_text_from_fields(
        full_name=repo.full_name,
        description=repo.description,
        language=repo.language,
        topics=repo.topics,
    )
    assert via_repo == via_kwargs