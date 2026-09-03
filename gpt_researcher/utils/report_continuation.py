"""Reliable, bounded continuation for long report completions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence


REPORT_COMPLETION_MARKER = "<!-- GPT_RESEARCHER_REPORT_COMPLETE -->"

_TRUNCATED_REASONS = {
    "length",
    "max_tokens",
    "max_token",
    "max_output_tokens",
    "token_limit",
    "model_length",
}
_NATURAL_REASONS = {
    "stop",
    "end_turn",
    "end",
    "complete",
    "completed",
    "finished",
}


@dataclass(frozen=True)
class CompletionResult:
    """Text plus the provider metadata needed to judge completeness."""

    content: str
    finish_reason: str | None = None
    response_metadata: Mapping[str, Any] = field(default_factory=dict)


class PartialCompletionError(RuntimeError):
    """A streamed request failed after yielding some response text."""

    def __init__(
        self,
        message: str,
        partial_content: str,
        response_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_content = partial_content
        self.response_metadata = dict(response_metadata or {})


class IncompleteReportError(RuntimeError):
    """Report generation ended without proof that the report completed."""

    def __init__(
        self,
        partial_markdown: str,
        *,
        finish_reason: str | None,
        continuations: int,
        cause: BaseException | None = None,
    ) -> None:
        reason = finish_reason or "unavailable"
        message = (
            "Report generation is incomplete "
            f"(finish reason: {reason}; continuations attempted: {continuations})."
        )
        super().__init__(message)
        self.partial_markdown = remove_completion_marker(partial_markdown)
        self.finish_reason = finish_reason
        self.continuations = continuations
        self.cause = cause


def normalize_finish_reason(reason: Any) -> str | None:
    """Normalize common provider finish-reason spellings."""
    if reason is None:
        return None
    if hasattr(reason, "value"):
        reason = reason.value
    normalized = str(reason).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized or normalized in {"none", "null", "unknown"}:
        return None
    if normalized in _TRUNCATED_REASONS:
        return "length"
    if normalized in _NATURAL_REASONS:
        return "stop"
    return normalized


def extract_finish_reason(metadata: Mapping[str, Any] | None) -> str | None:
    """Extract a finish reason from common LangChain/provider metadata shapes."""
    if not metadata:
        return None
    for key in ("finish_reason", "stop_reason", "finishReason", "stopReason"):
        if metadata.get(key) is not None:
            return normalize_finish_reason(metadata[key])
    nested = metadata.get("generation_info")
    if isinstance(nested, Mapping):
        return extract_finish_reason(nested)
    return None


def remove_completion_marker(text: str) -> str:
    """Remove all private completion markers from user-visible Markdown."""
    return text.replace(REPORT_COMPLETION_MARKER, "")


def has_completion_marker(text: str) -> bool:
    return REPORT_COMPLETION_MARKER in text


def build_continuation_messages(
    original_messages: Sequence[Mapping[str, str]],
    generated_text: str,
) -> list[dict[str, str]]:
    """Keep the original request and the complete response generated so far."""
    messages = [dict(message) for message in original_messages]
    messages.extend(
        [
            {"role": "assistant", "content": generated_text},
            {
                "role": "user",
                "content": (
                    "Your previous response was cut off. Continue the report exactly "
                    "where it ended, without repeating any earlier text. Preserve the "
                    "report's existing structure and coherence. Only when the report "
                    "is fully complete, "
                    f"append {REPORT_COMPLETION_MARKER} on its own line."
                ),
            },
        ]
    )
    return messages


CompletionCallable = Callable[
    [list[dict[str, str]]], Awaitable[CompletionResult]
]


async def complete_report_with_continuations(
    *,
    messages: Sequence[Mapping[str, str]],
    complete: CompletionCallable,
    max_continuations: int = 2,
    websocket: Any | None = None,
) -> str:
    """Complete a report; only the explicit completion marker proves it finished.

    A ``stop`` finish reason without the marker is treated as truncation and
    retried within ``max_continuations``; if the marker never appears the
    accumulated text is surfaced via ``IncompleteReportError``.
    """
    if not 0 <= max_continuations <= 10:
        raise ValueError("max_continuations must be between 0 and 10")

    original_messages = [dict(message) for message in messages]
    original_messages.append(
        {
            "role": "user",
            "content": (
                "After the report is fully complete, append "
                f"{REPORT_COMPLETION_MARKER} on its own line."
            ),
        }
    )
    fragments: list[str] = []
    continuations = 0
    finish_reason: str | None = None

    while True:
        request_messages = (
            original_messages
            if not fragments
            else build_continuation_messages(original_messages, "".join(fragments))
        )
        try:
            result = await complete(request_messages)
        except PartialCompletionError as exc:
            partial = "".join(fragments) + exc.partial_content
            raise IncompleteReportError(
                partial,
                finish_reason=extract_finish_reason(exc.response_metadata),
                continuations=continuations,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise IncompleteReportError(
                "".join(fragments),
                finish_reason=finish_reason,
                continuations=continuations,
                cause=exc,
            ) from exc

        fragments.append(result.content)
        aggregate = "".join(fragments)
        finish_reason = normalize_finish_reason(result.finish_reason)

        # The marker is the only proof of completion. Providers such as
        # OmniRoute/Qwen have returned finish_reason "stop" for streams that
        # actually ended mid-output, so a "stop" without the marker must be
        # treated as truncation and continued within the bounded budget.
        if has_completion_marker(aggregate):
            cleaned = remove_completion_marker(aggregate)
            if websocket is not None:
                try:
                    await websocket.send_json(
                        {
                            "type": "report_complete",
                            "output": cleaned,
                            "replace_chars": len(aggregate),
                        }
                    )
                except Exception as exc:
                    raise IncompleteReportError(
                        aggregate,
                        finish_reason=finish_reason,
                        continuations=continuations,
                        cause=exc,
                    ) from exc
            return cleaned

        if continuations >= max_continuations:
            raise IncompleteReportError(
                aggregate,
                finish_reason=finish_reason,
                continuations=continuations,
            )

        continuations += 1
