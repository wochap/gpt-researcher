"""Reranker client for the local GPU retrieval pipeline.

One provider is supported: ``llamacpp``, a llama.cpp ``llama-server`` started
with ``--rerank --pooling rank`` and exposing a Cohere/Jina-shaped rerank
endpoint (``POST /v1/rerank`` by default; the path is configurable because
llama.cpp builds have exposed ``/rerank``, ``/v1/rerank`` and
``/v1/reranking`` at different times).

Qwen3-Reranker GGUFs converted with llama.cpp's reranker path carry a
``tokenizer.chat_template.rerank`` entry, and llama-server applies it to every
query/document pair itself. Client-side templating is therefore off by
default; the switch is kept for GGUFs that lack that template.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Template pieces from the Qwen3-Reranker model card. The query carries the
# system prompt and instruction; each document carries the assistant suffix
# so the model scores the "yes" / "no" token that follows.
#
# Only used when ``apply_qwen3_template`` is on. llama-server already wraps
# each pair with the GGUF's ``tokenizer.chat_template.rerank`` (the same
# template, with the default instruction), so turning this on against such a
# GGUF nests the template twice. Verified empirically against llama.cpp b9190
# with giladgd/Qwen3-Reranker-4B-GGUF Q8_0: see the templating section of
# docs/docs/gpt-researcher/gptr/config.md for the observed scores.
QWEN3_QUERY_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN3_DOCUMENT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
DEFAULT_ENDPOINT = "/v1/rerank"


class LlamaCppReranker:
    """Rerank documents with a llama-server rerank endpoint in conservative batches."""

    def __init__(
        self,
        base_url: str,
        model: str,
        batch_size: int = 8,
        timeout: float = 30.0,
        instruction: str = DEFAULT_INSTRUCTION,
        apply_qwen3_template: bool = False,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.timeout = timeout
        self.instruction = instruction
        self.apply_qwen3_template = apply_qwen3_template
        self.api_key = api_key
        self.endpoint = "/" + (endpoint or DEFAULT_ENDPOINT).strip("/")
        self._transport = transport

    def format_query(self, query: str) -> str:
        if not self.apply_qwen3_template:
            return query
        return f"{QWEN3_QUERY_PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n"

    def format_document(self, text: str) -> str:
        if not self.apply_qwen3_template:
            return text
        return f"<Document>: {text}{QWEN3_DOCUMENT_SUFFIX}"

    def _client(self) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers,
            transport=self._transport,
        )

    async def _rerank_batch(
        self, client: httpx.AsyncClient, query: str, texts: List[str], offset: int
    ) -> List[Tuple[int, float]]:
        # llama-server validates ``query`` and ``documents``; ``top_n`` is
        # honored and ``model`` is ignored in single-model mode (it selects
        # the model in router mode), so both stay in the payload.
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        response = await client.post(self.endpoint, json=payload)
        response.raise_for_status()
        results = response.json()["results"]
        return [(offset + int(item["index"]), float(item["relevance_score"])) for item in results]

    async def arerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        """Return the ``top_n`` most relevant docs, each tagged with ``metadata["rerank_score"]``.

        On any HTTP or parsing failure the embedding order is kept and the
        first ``top_n`` documents are returned, so a reranker outage degrades
        to the pre-existing behavior instead of failing the research run.
        """
        if not docs:
            return []
        formatted_query = self.format_query(query)
        scores: List[Tuple[int, float]] = []
        try:
            async with self._client() as client:
                for offset in range(0, len(docs), self.batch_size):
                    batch = docs[offset:offset + self.batch_size]
                    texts = [self.format_document(d.page_content) for d in batch]
                    scores.extend(await self._rerank_batch(client, formatted_query, texts, offset))
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            logger.warning("Reranker request failed (%s); falling back to embedding order", exc)
            return docs[:top_n]

        # Scores are only used for relative ordering. Observed with llama.cpp
        # b9190 + Qwen3-Reranker-4B Q8_0 (cls.output has a no/yes pair): values
        # in (0, 1) that behave like a "yes" probability (0.998 exact match,
        # 1e-6 off-topic). Other builds or GGUFs may return a raw logit, so
        # nothing downstream may threshold ``rerank_score``.
        scores.sort(key=lambda item: item[1], reverse=True)
        reranked: List[Document] = []
        for index, score in scores[:top_n]:
            doc = docs[index]
            reranked.append(
                Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": score})
            )
        return reranked


def build_reranker(cfg: Any) -> Optional[LlamaCppReranker]:
    """Build the configured reranker, or ``None`` when reranking is disabled."""
    if not getattr(cfg, "reranker_enabled", False):
        return None
    provider = str(getattr(cfg, "reranker_provider", "llamacpp") or "llamacpp").lower()
    if provider != "llamacpp":
        logger.warning("Unknown RERANKER_PROVIDER %r; reranking disabled", provider)
        return None
    return LlamaCppReranker(
        base_url=getattr(cfg, "reranker_base_url", "http://localhost:8001"),
        model=getattr(cfg, "reranker_model", "Qwen/Qwen3-Reranker-4B"),
        batch_size=getattr(cfg, "reranker_batch_size", 8),
        timeout=getattr(cfg, "reranker_timeout", 30.0),
        instruction=getattr(cfg, "reranker_instruction", DEFAULT_INSTRUCTION),
        apply_qwen3_template=getattr(cfg, "reranker_apply_qwen3_template", False),
        api_key=os.environ.get("RERANKER_API_KEY"),
        endpoint=getattr(cfg, "reranker_endpoint", DEFAULT_ENDPOINT),
    )
