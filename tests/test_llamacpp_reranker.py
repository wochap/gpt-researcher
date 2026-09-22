"""LlamaCppReranker against an httpx.MockTransport (offline)."""

import json

import httpx
import pytest
from langchain_core.documents import Document

from gpt_researcher.context import reranker as reranker_module
from gpt_researcher.context.reranker import (
    QWEN3_DOCUMENT_SUFFIX,
    QWEN3_QUERY_PREFIX,
    LlamaCppReranker,
)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record retry waits instead of sleeping, so retry tests run instantly."""
    waits = []

    async def fake_sleep(delay):
        waits.append(delay)

    monkeypatch.setattr(reranker_module, "_sleep", fake_sleep)
    return waits


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


def scripted_handler(steps):
    """Replay ``steps`` in order: a status code, a (status, headers) pair or an exception.

    Once the script runs out every request succeeds via RerankServer.
    """
    server = RerankServer()
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= len(steps):
            step = steps[len(calls) - 1]
            if isinstance(step, type) and issubclass(step, Exception):
                raise step("scripted failure", request=request)
            status, headers = step if isinstance(step, tuple) else (step, {})
            return httpx.Response(status, headers=headers, json={"error": {"message": f"status {status}"}})
        return server.handler(request)

    return handler, calls


async def test_retries_503_then_succeeds(sleeps):
    handler, calls = scripted_handler([503])
    reranker = LlamaCppReranker(base_url="http://localhost:8001", model="m", transport=httpx.MockTransport(handler))
    out = await reranker.arerank("q", make_docs(3), top_n=2)
    assert len(calls) == 2
    assert len(sleeps) == 1
    assert [d.page_content for d in out] == ["doc 2", "doc 1"]
    assert "rerank_score" in out[0].metadata


async def test_retry_after_header_is_honored(sleeps):
    handler, calls = scripted_handler([(503, {"Retry-After": "7"})])
    reranker = LlamaCppReranker(base_url="http://localhost:8001", model="m", transport=httpx.MockTransport(handler))
    await reranker.arerank("q", make_docs(2), top_n=2)
    assert len(calls) == 2
    assert sleeps == [7.0]


async def test_500_is_not_retried_and_body_is_logged(sleeps, caplog):
    def handler(request):
        return httpx.Response(
            500, json={"error": {"message": "input (739 tokens) is too large to process"}}
        )

    reranker = LlamaCppReranker(base_url="http://localhost:8001", model="m", transport=httpx.MockTransport(handler))
    docs = make_docs(4)
    with caplog.at_level("WARNING", logger="gpt_researcher.context.reranker"):
        out = await reranker.arerank("q", docs, top_n=2)
    assert out == docs[:2]
    assert sleeps == []
    assert "HTTP 500" in caplog.text
    assert "input (739 tokens) is too large to process" in caplog.text


async def test_retries_exhausted_falls_back(sleeps, caplog):
    handler, calls = scripted_handler([503] * 10)
    reranker = LlamaCppReranker(
        base_url="http://localhost:8001", model="m", max_retries=3, transport=httpx.MockTransport(handler)
    )
    docs = make_docs(5)
    with caplog.at_level("WARNING", logger="gpt_researcher.context.reranker"):
        out = await reranker.arerank("q", docs, top_n=3)
    assert out == docs[:3]
    assert len(calls) == 4
    assert len(sleeps) == 3
    # Exponential backoff with equal jitter: attempt n waits within [2**n / 2, 2**n].
    for n, wait in enumerate(sleeps):
        assert 2 ** n / 2 <= wait <= 2 ** n
    assert "HTTP 503" in caplog.text


async def test_retry_budget_stops_retries(sleeps):
    handler, calls = scripted_handler([(503, {"Retry-After": "120"})])
    reranker = LlamaCppReranker(
        base_url="http://localhost:8001", model="m", retry_max_wait=60, transport=httpx.MockTransport(handler)
    )
    docs = make_docs(2)
    assert await reranker.arerank("q", docs, top_n=2) == docs[:2]
    assert len(calls) == 1
    assert sleeps == []


async def test_connect_error_then_succeeds(sleeps):
    handler, calls = scripted_handler([httpx.ConnectError])
    reranker = LlamaCppReranker(base_url="http://localhost:8001", model="m", transport=httpx.MockTransport(handler))
    out = await reranker.arerank("q", make_docs(3), top_n=3)
    assert len(calls) == 2
    assert len(sleeps) == 1
    assert [d.page_content for d in out] == ["doc 2", "doc 1", "doc 0"]


async def test_transient_failure_in_later_batch_keeps_earlier_scores(sleeps):
    # Transient failure in the second batch only; the first batch's scores survive.
    server = RerankServer()
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 2:
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        return server.handler(request)

    reranker = LlamaCppReranker(
        base_url="http://localhost:8001", model="m", batch_size=2, transport=httpx.MockTransport(handler)
    )
    out = await reranker.arerank("q", make_docs(4), top_n=4)
    assert len(calls) == 3
    assert [d.page_content for d in out] == ["doc 3", "doc 2", "doc 1", "doc 0"]
