"""Embed -> rerank context compression for the local GPU pipeline.

This module extends :class:`~gpt_researcher.context.compression.ContextCompressor`
with configurable chunking, a cosine top-K stage and an optional reranker.
It is only used when ``RETRIEVAL_PIPELINE=local_gpu``; the default pipeline
is left untouched. The embedding model (Ollama) and the reranker
(llama-server) are assumed to stay resident on the GPU at the same time.

Pipeline per sub-query::

    scrape -> RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
    -> embeddings (batched) -> cosine top ``embedding_top_k``
    -> reranker -> top ``rerank_top_k`` -> LLM
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, List

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..memory.embeddings import OPENAI_EMBEDDING_MODEL
from ..prompts import PromptFamily
from ..utils.costs import estimate_embedding_cost
from .compression import ContextCompressor
from .reranker import LlamaCppReranker, build_reranker
from .retriever import SearchAPIRetriever


class BatchedEmbeddings(Embeddings):
    """Wrap an ``Embeddings`` so ``embed_documents`` sends at most ``batch_size`` texts per call.

    ``OllamaEmbeddings.embed_documents`` sends every chunk in one request.
    On a small GPU that spikes memory and can time out; batching keeps each
    request bounded. Queries pass through untouched.
    """

    def __init__(self, inner: Embeddings, batch_size: int):
        self.inner = inner
        self.batch_size = max(1, int(batch_size))

    def _batches(self, texts: List[str]) -> List[List[str]]:
        return [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for batch in self._batches(list(texts)):
            vectors.extend(self.inner.embed_documents(batch))
        return vectors

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for batch in self._batches(list(texts)):
            vectors.extend(await self.inner.aembed_documents(batch))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.inner.embed_query(text)

    async def aembed_query(self, text: str) -> List[float]:
        return await self.inner.aembed_query(text)


class RerankingContextCompressor(ContextCompressor):
    """``ContextCompressor`` with configurable chunking, top-K embedding filter and reranking."""

    def __init__(
        self,
        documents,
        embeddings,
        max_results: int = 5,
        similarity_threshold: float | None = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        embedding_top_k: int = 30,
        reranker: LlamaCppReranker | None = None,
        rerank_top_k: int = 8,
        embedding_batch_size: int = 0,
        **kwargs,
    ):
        super().__init__(
            documents,
            embeddings,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            prompt_family=prompt_family,
            **kwargs,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_top_k = embedding_top_k
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.embedding_batch_size = embedding_batch_size

    def _filter_embeddings(self) -> Embeddings:
        if self.embedding_batch_size and self.embedding_batch_size > 0:
            return BatchedEmbeddings(self.embeddings, self.embedding_batch_size)
        return self.embeddings

    def get_contextual_retriever(self) -> ContextualCompressionRetriever:
        """Build the retriever with the configured chunk size, overlap and top-K."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        relevance_filter = EmbeddingsFilter(
            embeddings=self._filter_embeddings(),
            k=self.embedding_top_k,
            similarity_threshold=self.similarity_threshold,
        )
        pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, relevance_filter])
        base_retriever = SearchAPIRetriever(pages=self.documents)
        return ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        """Embed, keep the cosine top-K, then rerank (when configured) and format."""
        # Fast path, identical to ContextCompressor.async_get_context: tiny
        # document sets are returned directly without embedding or reranking.
        total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
        chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

        if total_chars < chunk_threshold and len(self.documents) <= max_results:
            direct_docs = [
                Document(
                    page_content=doc.get('raw_content', '') or '',
                    metadata={
                        "title": doc.get("title", "") or "",
                        "source": doc.get("source") or doc.get("url") or "",
                    },
                )
                for doc in self.documents[:max_results]
            ]
            return self.prompt_family.pretty_print_docs(direct_docs, max_results)

        retriever = self.get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))

        relevant_docs = await asyncio.to_thread(retriever.invoke, query, **self.kwargs)

        if self.reranker is None:
            return self.prompt_family.pretty_print_docs(relevant_docs, max_results)

        reranked = await self.reranker.arerank(query, relevant_docs, self.rerank_top_k)
        return self.prompt_family.pretty_print_docs(reranked, self.rerank_top_k)


def retrieval_pipeline_enabled(cfg: Any) -> bool:
    """True when ``RETRIEVAL_PIPELINE=local_gpu`` selects the embed -> rerank path."""
    return str(getattr(cfg, "retrieval_pipeline", "default") or "default").lower() == "local_gpu"


def _build_pipeline(researcher: Any) -> dict:
    cfg = researcher.cfg
    return {
        "reranker": build_reranker(cfg),
        "chunk_size": int(getattr(cfg, "compression_chunk_size", 2000)),
        "chunk_overlap": int(getattr(cfg, "compression_chunk_overlap", 200)),
        "embedding_top_k": int(getattr(cfg, "embedding_top_k", 30)),
        "embedding_batch_size": int(getattr(cfg, "embedding_batch_size", 16)),
        "rerank_top_k": int(getattr(cfg, "reranker_top_k", 8)),
    }


async def get_reranked_context(researcher: Any, query: str, pages: list) -> str:
    """Entry point used by ``ContextManager`` when the local GPU pipeline is enabled.

    The reranker and compressor settings are built once per researcher and
    cached on it, so every sub-query shares the same HTTP settings.
    """
    pipeline = getattr(researcher, "_gptr_rerank_pipeline", None)
    if pipeline is None:
        pipeline = _build_pipeline(researcher)
        try:
            setattr(researcher, "_gptr_rerank_pipeline", pipeline)
        except AttributeError:
            pass

    cfg = researcher.cfg
    compressor = RerankingContextCompressor(
        documents=pages,
        embeddings=researcher.memory.get_embeddings(),
        similarity_threshold=getattr(cfg, "similarity_threshold", None),
        prompt_family=researcher.prompt_family,
        **pipeline,
        **getattr(researcher, "kwargs", {}),
    )
    max_results = pipeline["rerank_top_k"] if pipeline["reranker"] is not None else 10
    return await compressor.async_get_context(
        query=query, max_results=max_results, cost_callback=researcher.add_costs
    )
