"""RerankingContextCompressor with deterministic fake embeddings and a fake reranker."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gpt_researcher.context import rerank_compression as rc
from gpt_researcher.context.compression import ContextCompressor
from gpt_researcher.context.rerank_compression import (
    BatchedEmbeddings,
    RerankingContextCompressor,
)
from gpt_researcher.prompts import PromptFamily


class FakeEmbeddings(Embeddings):
    """Deterministic vectors: every text is close to the query, ordering by text length."""

    def __init__(self):
        self.calls = []

    def _vec(self, text):
        return [1.0, (len(text) % 97) / 1000.0]

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeReranker:
    def __init__(self):
        self.received = []

    async def arerank(self, query, docs, top_n):
        self.received.append((query, list(docs), top_n))
        reversed_docs = list(reversed(docs))[:top_n]
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": 1.0 - i / 100})
            for i, d in enumerate(reversed_docs)
        ]


def make_pages(n_pages=5, chars=20_000):
    pages = []
    for p in range(n_pages):
        words = " ".join(f"page{p}word{i}" for i in range(chars // 10))
        pages.append({"url": f"https://example.test/{p}", "title": f"Page {p}", "raw_content": words[:chars]})
    return pages


@pytest.fixture(autouse=True)
def _threshold(monkeypatch):
    monkeypatch.setenv("COMPRESSION_THRESHOLD", "8000")


async def test_splitter_and_filter_use_configured_params(monkeypatch):
    captured = {}

    class SpySplitter(RecursiveCharacterTextSplitter):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(rc, "RecursiveCharacterTextSplitter", SpySplitter)
    compressor = RerankingContextCompressor(
        documents=make_pages(),
        embeddings=FakeEmbeddings(),
        prompt_family=PromptFamily,
        chunk_size=2000,
        chunk_overlap=200,
        embedding_top_k=30,
    )
    retriever = compressor.get_contextual_retriever()
    assert captured == {"chunk_size": 2000, "chunk_overlap": 200}
    embeddings_filter = retriever.base_compressor.transformers[1]
    assert embeddings_filter.k == 30
    assert embeddings_filter.similarity_threshold == compressor.similarity_threshold


async def test_reranker_receives_top_k_and_output_has_rerank_top_k():
    reranker = FakeReranker()
    compressor = RerankingContextCompressor(
        documents=make_pages(),
        embeddings=FakeEmbeddings(),
        prompt_family=PromptFamily,
        embedding_top_k=30,
        reranker=reranker,
        rerank_top_k=8,
    )
    out = await compressor.async_get_context("query", max_results=8)
    assert len(reranker.received) == 1
    query, docs, top_n = reranker.received[0]
    assert query == "query"
    assert 8 < len(docs) <= 30
    assert top_n == 8
    assert out.count("Source: ") == 8
    assert out.count("Content: ") == 8


async def test_without_reranker_matches_plain_context_compressor():
    pages = make_pages()
    plain = ContextCompressor(documents=pages, embeddings=FakeEmbeddings(), prompt_family=PromptFamily)
    ours = RerankingContextCompressor(
        documents=pages,
        embeddings=FakeEmbeddings(),
        prompt_family=PromptFamily,
        chunk_size=1000,
        chunk_overlap=100,
        embedding_top_k=20,
        reranker=None,
    )
    expected = await plain.async_get_context("query", max_results=10)
    actual = await ours.async_get_context("query", max_results=10)
    assert actual == expected


async def test_fast_path_matches_parent():
    pages = [{"url": "https://a.test", "title": "A", "raw_content": "tiny"}]
    plain = ContextCompressor(documents=pages, embeddings=FakeEmbeddings(), prompt_family=PromptFamily)
    reranker = FakeReranker()
    ours = RerankingContextCompressor(
        documents=pages, embeddings=FakeEmbeddings(), prompt_family=PromptFamily, reranker=reranker
    )
    assert await ours.async_get_context("q", max_results=5) == await plain.async_get_context("q", max_results=5)
    assert reranker.received == []


async def test_batched_embeddings_splits_calls():
    inner = FakeEmbeddings()
    batched = BatchedEmbeddings(inner, batch_size=16)
    texts = [f"t{i}" for i in range(30)]
    vectors = batched.embed_documents(texts)
    assert len(vectors) == 30
    assert [len(c) for c in inner.calls] == [16, 14]
    assert vectors == [inner._vec(t) for t in texts]
    assert batched.embed_query("q") == [1.0, 0.0]


async def test_embedding_batch_size_wraps_embeddings():
    inner = FakeEmbeddings()
    compressor = RerankingContextCompressor(
        documents=make_pages(),
        embeddings=inner,
        prompt_family=PromptFamily,
        embedding_batch_size=16,
    )
    await compressor.async_get_context("query", max_results=10)
    assert inner.calls
    assert all(len(c) <= 16 for c in inner.calls)
