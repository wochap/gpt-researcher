from __future__ import annotations

import pytest

from gpt_researcher.utils.report_continuation import (
    CompletionResult,
    IncompleteReportError,
    PartialCompletionError,
    REPORT_COMPLETION_MARKER,
    bounded_tail,
    complete_report_with_continuations,
    has_completion_marker,
    normalize_finish_reason,
    remove_completion_marker,
)


class Socket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("length", "length"),
        ("max-tokens", "length"),
        ("MAX_OUTPUT_TOKENS", "length"),
        ("stop", "stop"),
        ("end_turn", "stop"),
        (None, None),
        ("unknown", None),
        ("content_filter", "content_filter"),
    ],
)
def test_normalize_finish_reason(raw, expected):
    assert normalize_finish_reason(raw) == expected


def test_marker_and_tail_helpers():
    marked = f"report\n{REPORT_COMPLETION_MARKER}\n"
    assert has_completion_marker(marked)
    assert REPORT_COMPLETION_MARKER not in remove_completion_marker(marked)
    assert bounded_tail("abcdefgh", 4) == "efgh"
    with pytest.raises(ValueError):
        bounded_tail("text", 0)


@pytest.mark.asyncio
async def test_natural_completion_emits_clean_aggregate():
    socket = Socket()

    async def complete(messages):
        return CompletionResult(f"done{REPORT_COMPLETION_MARKER}", "end_turn")

    report = await complete_report_with_continuations(
        messages=[{"role": "user", "content": "write"}],
        complete=complete,
        websocket=socket,
    )

    assert report == "done"
    assert socket.messages == [
        {
            "type": "report_complete",
            "output": "done",
            "replace_chars": len(f"done{REPORT_COMPLETION_MARKER}"),
        }
    ]


@pytest.mark.asyncio
async def test_truncation_concatenates_once_and_bounds_each_generated_tail():
    requests = []
    results = iter(
        [
            CompletionResult("AAAAAA", "max_tokens"),
            CompletionResult("BBBBBB", "length"),
            CompletionResult(f"CCCC{REPORT_COMPLETION_MARKER}", "stop"),
        ]
    )

    async def complete(messages):
        requests.append(messages)
        return next(results)

    report = await complete_report_with_continuations(
        messages=[{"role": "user", "content": "ORIGINAL"}],
        complete=complete,
        max_continuations=2,
        tail_chars=4,
    )

    assert report == "AAAAAABBBBBBCCCC"
    assert requests[1][-2] == {"role": "assistant", "content": "AAAA"}
    assert requests[2][-2] == {"role": "assistant", "content": "BBBB"}
    assert all("ORIGINAL" in str(request) for request in requests)
    assert "AAAA" not in requests[2][-2]["content"]


@pytest.mark.asyncio
async def test_missing_metadata_uses_marker():
    async def complete(messages):
        return CompletionResult(f"complete{REPORT_COMPLETION_MARKER}")

    assert await complete_report_with_continuations(
        messages=[{"role": "user", "content": "write"}], complete=complete
    ) == "complete"


@pytest.mark.asyncio
async def test_exhaustion_preserves_partial_markdown():
    async def complete(messages):
        return CompletionResult("fragment", "length")

    with pytest.raises(IncompleteReportError) as caught:
        await complete_report_with_continuations(
            messages=[{"role": "user", "content": "write"}],
            complete=complete,
            max_continuations=1,
        )

    assert caught.value.partial_markdown == "fragmentfragment"
    assert caught.value.continuations == 1
    assert caught.value.finish_reason == "length"


@pytest.mark.asyncio
async def test_stop_without_marker_continues_until_budget_then_fails():
    calls = 0

    async def complete(messages):
        nonlocal calls
        calls += 1
        return CompletionResult("fragment", "stop")

    with pytest.raises(IncompleteReportError) as caught:
        await complete_report_with_continuations(
            messages=[{"role": "user", "content": "write"}],
            complete=complete,
            max_continuations=2,
        )

    assert calls == 3
    assert caught.value.partial_markdown == "fragmentfragmentfragment"
    assert caught.value.continuations == 2
    assert caught.value.finish_reason == "stop"


@pytest.mark.asyncio
async def test_streaming_interruption_preserves_all_received_fragments():
    calls = 0

    async def complete(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            return CompletionResult("first", "length")
        raise PartialCompletionError("connection lost", "partial", {"finish_reason": "length"})

    with pytest.raises(IncompleteReportError) as caught:
        await complete_report_with_continuations(
            messages=[{"role": "user", "content": "write"}],
            complete=complete,
        )

    assert caught.value.partial_markdown == "firstpartial"
    assert caught.value.continuations == 1
