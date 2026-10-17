"""FastEmbedEmbedder 缓存逻辑测试：同一 repo / query 只编码一次。"""


def test_repo_cache_reuses_embeddings(counting_embedder, make_candidate):
    repos = [make_candidate("a"), make_candidate("b")]
    counting_embedder.embed_repos(repos)
    assert counting_embedder.encode_calls == 1

    # 相同 repo 再次编码：命中缓存
    counting_embedder.embed_repos(repos)
    assert counting_embedder.encode_calls == 1

    # 部分新增：只编码缺失的
    counting_embedder.embed_repos([make_candidate("a"), make_candidate("c")])
    assert counting_embedder.encode_calls == 2


def test_query_cache_reuses_embeddings(counting_embedder):
    counting_embedder.embed_query("python")
    counting_embedder.embed_query("python")
    assert counting_embedder.encode_calls == 1

    counting_embedder.embed_query("rust")
    assert counting_embedder.encode_calls == 2


def test_query_cache_normalizes_key(counting_embedder):
    counting_embedder.embed_query("  Python  ")
    counting_embedder.embed_query("python")
    assert counting_embedder.encode_calls == 1