"""text hash 缓存 key 测试：语义内容变化失效，非语义字段（stars）不影响缓存。"""


def test_text_change_invalidates_cache(counting_embedder, make_candidate):
    counting_embedder.embed_repos([make_candidate("a", description="first version")])
    assert counting_embedder.encode_calls == 1

    counting_embedder.embed_repos([make_candidate("a", description="second version")])
    assert counting_embedder.encode_calls == 2


def test_stars_change_keeps_cache(counting_embedder, make_candidate):
    counting_embedder.embed_repos([make_candidate("a", stars=100)])
    assert counting_embedder.encode_calls == 1

    counting_embedder.embed_repos([make_candidate("a", stars=99_999)])
    assert counting_embedder.encode_calls == 1


def test_lru_eviction(counting_embedder, make_candidate):
    counting_embedder._cache_size = 2
    counting_embedder.embed_repos([make_candidate("a"), make_candidate("b"), make_candidate("c")])
    assert len(counting_embedder._repo_cache) == 2

    # 被淘汰的 a 再次编码 → 重新 encode
    counting_embedder.embed_repos([make_candidate("a")])
    assert counting_embedder.encode_calls == 2