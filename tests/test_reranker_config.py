"""Config defaults, env overrides and ContextManager routing for the local GPU pipeline."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from gpt_researcher.config.config import Config
from gpt_researcher.context.reranker import LlamaCppReranker, build_reranker
from gpt_researcher.context.rerank_compression import retrieval_pipeline_enabled
from gpt_researcher.skills import context_manager as cm_module
from gpt_researcher.skills.context_manager import ContextManager

ENV_KEYS = [
    "RETRIEVAL_PIPELINE", "COMPRESSION_CHUNK_SIZE", "COMPRESSION_CHUNK_OVERLAP", "EMBEDDING_TOP_K",
    "EMBEDDING_BATCH_SIZE", "RERANKER_ENABLED", "RERANKER_PROVIDER", "RERANKER_BASE_URL",
    "RERANKER_ENDPOINT", "RERANKER_MODEL", "RERANKER_TOP_K", "RERANKER_BATCH_SIZE",
    "RERANKER_TIMEOUT", "RERANKER_INSTRUCTION", "RERANKER_APPLY_QWEN3_TEMPLATE",
    "RERANKER_MAX_RETRIES", "RERANKER_RETRY_MAX_WAIT",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Some retriever tests replace gpt_researcher.retrievers.utils in
    # sys.modules with a stub and never restore it, which breaks
    # Config.parse_retrievers for every later test in the same session.
    # Restore the real module so this file passes regardless of order.
    stub = sys.modules.get("gpt_researcher.retrievers.utils")
    if stub is not None and not hasattr(stub, "get_all_retriever_names"):
        del sys.modules["gpt_researcher.retrievers.utils"]
        importlib.import_module("gpt_researcher.retrievers.utils")


def test_defaults():
    cfg = Config("default")
    assert cfg.retrieval_pipeline == "default"
    assert cfg.compression_chunk_size == 2000
    assert cfg.compression_chunk_overlap == 200
    assert cfg.embedding_top_k == 30
    assert cfg.embedding_batch_size == 16
    assert cfg.reranker_enabled is False
    assert cfg.reranker_provider == "llamacpp"
    assert cfg.reranker_base_url == "http://localhost:8001"
    assert cfg.reranker_endpoint == "/v1/rerank"
    assert cfg.reranker_model == "Qwen/Qwen3-Reranker-4B"
    assert cfg.reranker_top_k == 8
    assert cfg.reranker_batch_size == 8
    assert cfg.reranker_timeout == 30.0
    assert cfg.reranker_max_retries == 4
    assert cfg.reranker_retry_max_wait == 60.0
    assert cfg.reranker_apply_qwen3_template is False
    for removed in ("reranker_vllm_sleep_mode", "gpu_exclusive", "gpu_unload_timeout", "gpu_unload_strict"):
        assert not hasattr(cfg, removed)
    assert retrieval_pipeline_enabled(cfg) is False
    assert build_reranker(cfg) is None


def test_env_overrides_have_correct_types(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_PIPELINE", "local_gpu")
    monkeypatch.setenv("COMPRESSION_CHUNK_SIZE", "1500")
    monkeypatch.setenv("EMBEDDING_TOP_K", "40")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_BASE_URL", "http://localhost:9000/")
    monkeypatch.setenv("RERANKER_ENDPOINT", "/rerank")
    monkeypatch.setenv("RERANKER_MODEL", "local/Qwen3-Reranker-4B-INT4")
    monkeypatch.setenv("RERANKER_TOP_K", "5")
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "4")
    monkeypatch.setenv("RERANKER_TIMEOUT", "45.5")
    monkeypatch.setenv("RERANKER_MAX_RETRIES", "2")
    monkeypatch.setenv("RERANKER_RETRY_MAX_WAIT", "15.5")
    monkeypatch.setenv("RERANKER_APPLY_QWEN3_TEMPLATE", "true")
    cfg = Config("default")
    assert cfg.retrieval_pipeline == "local_gpu"
    assert cfg.compression_chunk_size == 1500 and isinstance(cfg.compression_chunk_size, int)
    assert cfg.embedding_top_k == 40
    assert cfg.embedding_batch_size == 0
    assert cfg.reranker_enabled is True
    assert cfg.reranker_top_k == 5
    assert cfg.reranker_batch_size == 4
    assert cfg.reranker_timeout == 45.5 and isinstance(cfg.reranker_timeout, float)
    assert cfg.reranker_max_retries == 2 and isinstance(cfg.reranker_max_retries, int)
    assert cfg.reranker_retry_max_wait == 15.5 and isinstance(cfg.reranker_retry_max_wait, float)
    assert cfg.reranker_apply_qwen3_template is True
    assert retrieval_pipeline_enabled(cfg) is True

    reranker = build_reranker(cfg)
    assert isinstance(reranker, LlamaCppReranker)
    assert reranker.base_url == "http://localhost:9000"
    assert reranker.endpoint == "/rerank"
    assert reranker.model == "local/Qwen3-Reranker-4B-INT4"
    assert reranker.batch_size == 4
    assert reranker.timeout == 45.5
    assert reranker.max_retries == 2
    assert reranker.retry_max_wait == 15.5
    assert reranker.apply_qwen3_template is True


@pytest.mark.parametrize("provider", ["cohere", "vllm"])
def test_build_reranker_unknown_provider_returns_none(monkeypatch, caplog, provider):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", provider)
    with caplog.at_level("WARNING", logger="gpt_researcher.context.reranker"):
        assert build_reranker(Config("default")) is None
    assert "Unknown RERANKER_PROVIDER" in caplog.text


def make_researcher(pipeline):
    return SimpleNamespace(
        cfg=SimpleNamespace(retrieval_pipeline=pipeline, similarity_threshold=0.35),
        verbose=False,
        websocket=None,
        memory=SimpleNamespace(get_embeddings=lambda: object()),
        prompt_family=object(),
        kwargs={},
        add_costs=lambda cost: None,
    )


async def test_context_manager_routes_to_reranked_context_only_when_enabled(monkeypatch):
    calls = []

    async def fake_get_reranked_context(researcher, query, pages):
        calls.append((query, pages))
        return "reranked"

    class FakeCompressor:
        def __init__(self, **kwargs):
            pass

        async def async_get_context(self, query, max_results, cost_callback):
            return "plain"

    monkeypatch.setattr(cm_module, "get_reranked_context", fake_get_reranked_context)
    monkeypatch.setattr(cm_module, "ContextCompressor", FakeCompressor)

    manager = ContextManager(make_researcher("local_gpu"))
    assert await manager.get_similar_content_by_query("q", [{"url": "u"}]) == "reranked"
    assert calls == [("q", [{"url": "u"}])]

    manager = ContextManager(make_researcher("default"))
    assert await manager.get_similar_content_by_query("q", []) == "plain"
    assert len(calls) == 1
