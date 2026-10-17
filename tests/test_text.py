"""Repo 检索文本构造测试。"""

from mosquito.ranking.embedding.text import build_repo_text, normalize


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