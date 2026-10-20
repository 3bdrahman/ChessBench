"""Shared helpers for OpenAI-compatible chat-completion providers."""

import contextlib
import json
from typing import Any, Protocol, runtime_checkable

from chessbench.common.common_types import CompletionResult
from chessbench.common.exceptions import ProviderAPIError


class ChatMessage:
    """Minimal typed chat message from OpenAI-compatible API."""

    content: str | None
    tool_calls: list[Any] | None
    model_extra: dict[str, Any] | None
    reasoning: str | None
    reasoning_content: str | None


@runtime_checkable
class _HasChoices(Protocol):
    choices: list[Any]


@runtime_checkable
class _HasMessage(Protocol):
    message: ChatMessage


def extract_chat_message(response: object, provider: str, model: str) -> ChatMessage:
    """Return ``response.choices[0].message`` or raise a typed error.

    Providers return an empty ``choices`` list on some failure modes
    (content filters, upstream truncation); indexing blindly raises a bare
    IndexError that bypasses the typed exception hierarchy.
    """
    if not isinstance(response, _HasChoices):
        raise ProviderAPIError(
            provider=provider,
            status_code=500,
            detail="Chat completion response missing 'choices' attribute",
            raw_response={"provider": provider, "model": model},
        )
    choices = response.choices
    if not choices:
        raise ProviderAPIError(
            provider=provider,
            status_code=500,
            detail=(
                "Chat completion returned no choices "
                "(possible content filter or upstream truncation)"
            ),
            raw_response={"provider": provider, "model": model},
        )
    first_choice = choices[0]
    if not isinstance(first_choice, _HasMessage):
        raise ProviderAPIError(
            provider=provider,
            status_code=500,
            detail="Chat completion choice missing 'message' attribute",
            raw_response={"provider": provider, "model": model},
        )
    return first_choice.message


def usage_of(response: object) -> tuple[int | None, int | None]:
    """Extract ``(tokens_in, tokens_out)`` from a chat completion response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def message_text(message: object) -> str:
    """Extract text plus provider-specific reasoning content safely."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text_content = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        text_content = "".join(parts)
    else:
        text_content = ""

    model_extra = getattr(message, "model_extra", None)
    reasoning = (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or (
            model_extra.get("reasoning") or model_extra.get("reasoning_content")
            if isinstance(model_extra, dict)
            else None
        )
    )
    if isinstance(reasoning, str) and reasoning:
        return f"<reasoning>\n{reasoning}\n</reasoning>\n{text_content}"
    return text_content


def tool_calls_of(message: object) -> list[dict[str, Any]] | None:
    """Convert SDK tool calls into ChessBench's provider-neutral structure."""
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return None

    output: list[dict[str, Any]] = []
    for call in calls:
        function = getattr(call, "function", None)
        if function is None:
            continue
        arguments = getattr(function, "arguments", "")
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
        output.append(
            {
                "name": getattr(function, "name", ""),
                "arguments": arguments,
            }
        )
    return output or None


__all__ = [
    "CompletionResult",
    "extract_chat_message",
    "message_text",
    "tool_calls_of",
    "usage_of",
]
