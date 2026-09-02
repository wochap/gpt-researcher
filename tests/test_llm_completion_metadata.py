from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpt_researcher.llm_provider.generic.base import GenericLLMProvider
from gpt_researcher.utils import llm as llm_utils
from gpt_researcher.utils.report_continuation import CompletionResult, PartialCompletionError


class FakeProvider:
    last_response_metadata = {"finish_reason": "max_tokens", "provider": "fake"}
    last_usage_metadata = None

    async def get_chat_response(self, messages, stream, websocket, **kwargs):
        return "response"


@pytest.mark.asyncio
async def test_chat_completion_metadata_is_opt_in(monkeypatch):
    monkeypatch.setattr(llm_utils, "get_llm", lambda *args, **kwargs: FakeProvider())
    arguments = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "llm_provider": "test-provider",
    }

    assert await llm_utils.create_chat_completion(**arguments) == "response"
    result = await llm_utils.create_chat_completion(**arguments, return_metadata=True)

    assert result == CompletionResult(
        "response", "length", {"finish_reason": "max_tokens", "provider": "fake"}
    )


class InterruptedStream:
    async def astream(self, messages, **kwargs):
        yield SimpleNamespace(
            content="received text", response_metadata={"finish_reason": "length"},
            usage_metadata=None,
        )
        raise ConnectionError("stream interrupted")


@pytest.mark.asyncio
async def test_provider_attaches_text_received_before_stream_failure():
    provider = GenericLLMProvider(InterruptedStream(), verbose=False)

    with pytest.raises(PartialCompletionError) as caught:
        await provider.stream_response([{"role": "user", "content": "hello"}])

    assert caught.value.partial_content == "received text"
    assert caught.value.response_metadata == {"finish_reason": "length"}
