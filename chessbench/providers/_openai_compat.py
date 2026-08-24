"""Shared helpers for OpenAI-compatible chat-completion providers."""

from typing import Any

from chessbench.common.common_types import CompletionResult
from chessbench.common.exceptions import ProviderAPIError


class ChatMessage:
    """Minimal typed chat message from OpenAI-compatible API."""
    content: str | None
    tool_calls: list[Any] | None
    model_extra: dict[str, Any] | None
    reasoning: str | None


def extract_chat_message(response: object, provider: str, model: str) -> ChatMessage:
    """Return ``response.choices[0].message`` or raise a typed error.

    Providers return an empty ``choices`` list on some failure modes
    (content filters, upstream truncation); indexing blindly raises a bare
    IndexError that bypasses the typed exception hierarchy.
    """
    choices = getattr(response, "choices", None)
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
    # The OpenAI SDK message object has the attributes we need
    return choices[0].message  # type: ignore[no-any-return]


def usage_of(response: object) -> tuple[int | None, int | None]:
    """Extract ``(tokens_in, tokens_out)`` from a chat completion response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


__all__ = ["CompletionResult", "extract_chat_message", "usage_of"]
