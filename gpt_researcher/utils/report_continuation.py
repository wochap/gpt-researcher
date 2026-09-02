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


def bounded_tail(text: str, tail_chars: int) -> str:
    if tail_chars <= 0:
        raise ValueError("tail_chars must be positive")
    return text[-tail_chars:]


def build_continuation_messages(
    original_messages: Sequence[Mapping[str, str]],
    generated_text: str,
    tail_chars: int,
) -> list[dict[str, str]]:
    """Keep the original request and only the newest generated-text tail."""
    messages = [dict(message) for message in original_messages]
    messages.extend(
        [
            {"role": "assistant", "content": bounded_tail(generated_text, tail_chars)},
            {
                "role": "user",
                "content": (
                    "Continue the report exactly where the assistant text ends. "
                    "Do not repeat earlier text. When the report is fully complete, "
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
    tail_chars: int = 65_536,
    websocket: Any | None = None,
) -> str:
    """Complete a report, continuing only when completion is not established."""
    if not 0 <= max_continuations <= 10:
        raise ValueError("max_continuations must be between 0 and 10")
    if tail_chars <= 0:
        raise ValueError("tail_chars must be positive")

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
            else build_continuation_messages(
                original_messages, "".join(fragments), tail_chars
            )
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

        # Known provider metadata is authoritative. The marker is a fallback
        # only for providers that omit or expose an unfamiliar finish reason.
        complete_naturally = finish_reason == "stop"
        truncated = finish_reason == "length"
        metadata_unknown = finish_reason not in {"stop", "length"}
        if complete_naturally or (metadata_unknown and has_completion_marker(aggregate)):
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

        # Explicit truncation and missing/unknown metadata both need another
        # bounded attempt. Unknown metadata may complete via the marker.
        if truncated or metadata_unknown:
            continuations += 1
