"""Shared helpers for OpenAI-compatible chat-completion providers."""

from typing import Any, Protocol, runtime_checkable

from chessbench.common.common_types import CompletionResult
from chessbench.common.exceptions import ProviderAPIError


class ChatMessage:
    """Minimal typed chat message from OpenAI-compatible API."""
    content: str | None
    tool_calls: list[Any] | None
    model_extra: dict[str, Any] | None
    reasoning: str | None


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


__all__ = ["CompletionResult", "extract_chat_message", "usage_of"]
