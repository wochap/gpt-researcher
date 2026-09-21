"""LlamaCppReranker against an httpx.MockTransport (offline)."""

import json

import httpx
from langchain_core.documents import Document

from gpt_researcher.context.reranker import (
    QWEN3_DOCUMENT_SUFFIX,
    QWEN3_QUERY_PREFIX,
    LlamaCppReranker,
)


def make_docs(n):
    return [Document(page_content=f"doc {i}", metadata={"source": f"s{i}"}) for i in range(n)]


class RerankServer:
    """Scores each document by the integer in its text, so ordering is predictable."""

    def __init__(self, status=200):
        self.requests = []
        self.status = status

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request, body))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        results = []
        for i, text in enumerate(body["documents"]):
            digits = "".join(ch for ch in text.split("doc ")[1] if ch.isdigit())
            results.append({"index": i, "relevance_score": int(digits) / 100.0})
        return httpx.Response(200, json={"id": "x", "model": body["model"], "results": results})


def make_reranker(server, **kwargs):
    defaults = dict(base_url="http://localhost:8001", model="Qwen/Qwen3-Reranker-4B", batch_size=8)
    defaults.update(kwargs)
    return LlamaCppReranker(transport=httpx.MockTransport(server.handler), **defaults)


async def test_payload_fields_and_qwen3_template():
    server = RerankServer()
    reranker = make_reranker(server, instruction="Find it", apply_qwen3_template=True)
    await reranker.arerank("what?", make_docs(2), top_n=2)
    request, body = server.requests[0]
    assert request.url.path == "/v1/rerank"
    assert request.method == "POST"
    assert body["model"] == "Qwen/Qwen3-Reranker-4B"
    assert body["top_n"] == 2
    assert body["query"].startswith(QWEN3_QUERY_PREFIX)
    assert "<Instruct>: Find it\n<Query>: what?\n" in body["query"]
    assert all(doc.endswith(QWEN3_DOCUMENT_SUFFIX) for doc in body["documents"])
    assert body["documents"][0].startswith("<Document>: doc 0")


async def test_template_off_by_default_sends_raw_text():
    server = RerankServer()
    reranker = make_reranker(server)
    assert reranker.apply_qwen3_template is False
    await reranker.arerank("what?", make_docs(2), top_n=2)
    _, body = server.requests[0]
    assert body["query"] == "what?"
    assert body["documents"] == ["doc 0", "doc 1"]


async def test_batches_merge_sort_and_truncate():
    server = RerankServer()
    reranker = make_reranker(server, batch_size=8)
    docs = make_docs(30)
    out = await reranker.arerank("q", docs, top_n=8)
    assert len(server.requests) == 4
    assert [len(body["documents"]) for _, body in server.requests] == [8, 8, 8, 6]
    assert len(out) == 8
    assert [d.page_content for d in out] == [f"doc {i}" for i in range(29, 21, -1)]
    scores = [d.metadata["rerank_score"] for d in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0].metadata["source"] == "s29"


async def test_http_error_falls_back_to_embedding_order():
    server = RerankServer(status=500)
    reranker = make_reranker(server)
    docs = make_docs(12)
    out = await reranker.arerank("q", docs, top_n=8)
    assert out == docs[:8]
    assert "rerank_score" not in out[0].metadata


async def test_endpoint_is_configurable():
    server = RerankServer()
    reranker = make_reranker(server, endpoint="rerank/")
    await reranker.arerank("q", make_docs(1), top_n=1)
    assert server.requests[0][0].url.path == "/rerank"


async def test_connect_error_falls_back_to_embedding_order():
    def boom(request):
        raise httpx.ConnectError("connection refused", request=request)

    reranker = LlamaCppReranker(
        base_url="http://localhost:8001", model="m", transport=httpx.MockTransport(boom)
    )
    docs = make_docs(5)
    assert await reranker.arerank("q", docs, top_n=3) == docs[:3]


async def test_malformed_response_falls_back_to_embedding_order():
    def handler(request):
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": "high"}]})

    reranker = LlamaCppReranker(
        base_url="http://localhost:8001", model="m", transport=httpx.MockTransport(handler)
    )
    docs = make_docs(3)
    out = await reranker.arerank("q", docs, top_n=2)
    assert out == docs[:2]
    assert "rerank_score" not in out[0].metadata

    reranker._transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"scores": []}))
    assert await reranker.arerank("q", docs, top_n=2) == docs[:2]


async def test_empty_docs():
    server = RerankServer()
    reranker = make_reranker(server)
    assert await reranker.arerank("q", [], top_n=8) == []
    assert server.requests == []
