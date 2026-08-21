"""Common types shared between models and providers."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chess_fight import constants


@dataclass
class ModelInfo:
    """Information about a model from a provider."""
    id: str
    name: str
    provider: str
    context_window: int | None = None
    pricing_tier: str | None = None
    capabilities: list[str] = field(default_factory=list)


@dataclass
class ChatMessage:
    """Chat message for completion."""
    role: str
    content: str


@dataclass
class CompletionResult:
    """Result from a completion request."""
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    raw_response: dict[str, object] | None = None
    error: str | None = None
    error_type: str | None = None
    retry_count: int = 0
    validation_retries: int = 0
    tool_calls: list[dict[str, Any]] | None = None


# Capability flag added to models that can serve chess moves reliably.
CAP_CHESS = "chess"

# Tokens from the model id that indicate a non-chat / non-text model.
# These almost always fail at generating a legal UCI move.
_NON_CHAT_TOKENS = constants.NON_CHAT_TOKENS

# Tokens that indicate a chat model but one that doesn't follow instructions well
# for chess (small instruction-tuned or older completion models).
_WEAK_FOR_CHESS_TOKENS = constants.WEAK_FOR_CHESS_TOKENS

# Pattern for "free" pricing tier indicators (openrouter ":free" suffix, etc.).
_FREE_TIER_PATTERN = constants.FREE_TIER_PATTERN


# Default HTTP timeout for provider requests (seconds).
DEFAULT_HTTP_TIMEOUT: float = constants.DEFAULT_HTTP_TIMEOUT

# Default temperature for LLM completions (benchmark mode = 0.0 for reproducibility)
DEFAULT_TEMPERATURE: float = constants.DEFAULT_TEMPERATURE

# Default context window for models when not provided by the API.
# This is a sensible fallback; providers should override with API data where available.
DEFAULT_CONTEXT_WINDOW: int = constants.DEFAULT_CONTEXT_WINDOW


@dataclass
class ThinkingKeywordConfig:
    """Configuration for thinking analysis keywords.

    All keyword lists are case-insensitive and matched as substrings.
    """
    tactics: list[str] = field(default_factory=lambda: constants.THINKING_KEYWORDS["tactics"])
    strategy: list[str] = field(default_factory=lambda: constants.THINKING_KEYWORDS["strategy"])
    time_pressure: list[str] = field(default_factory=lambda: constants.THINKING_KEYWORDS["time_pressure"])
    material: list[str] = field(default_factory=lambda: constants.THINKING_KEYWORDS["material"])
    positional: list[str] = field(default_factory=lambda: constants.THINKING_KEYWORDS["positional"])
    king_safety: list[str] = field(default_factory=lambda: constants.THINKING_KEYWORDS["king_safety"])
    structured_indicators: list[str] = field(default_factory=lambda: constants.THINKING_KEYWORDS["structured_indicators"])


@dataclass
class MoveParseResult:
    """Result of parsing a move from LLM output."""
    uci: str | None = None
    san: str | None = None
    confidence: float = 0.0
    ambiguous: bool = False
    promotion_piece: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "uci": self.uci,
            "san": self.san,
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "promotion_piece": self.promotion_piece,
        }


def is_chess_capable(model: ModelInfo) -> bool:
    """Return True if a model is suitable for serving chess moves.

    A model is chess-capable if:
      * it advertises the ``chess`` capability, OR
      * its id/name doesn't contain any non-chat tokens (embedding, audio, etc.)
        and doesn't contain a known-weak-for-chess token.
    """
    if CAP_CHESS in model.capabilities:
        return True

    haystack = f"{model.id} {model.name}".lower()
    for token in _NON_CHAT_TOKENS:
        if token in haystack:
            return False
    for token in _WEAK_FOR_CHESS_TOKENS:
        if token in haystack:
            return False

    return model.context_window is None or model.context_window >= 256


def is_free_tier(model: ModelInfo) -> bool:
    """Return True if a model is marked as free-tier."""
    if model.provider in ("groq", "nim", "ollama", "stockfish"):
        return True
    if model.pricing_tier == "free":
        return True
    return bool(_FREE_TIER_PATTERN.search(f"{model.id} {model.name}"))


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    name: str
    requires_api_key: bool = True

    @abstractmethod
    async def list_models(self, api_key: str) -> list["ModelInfo"]:
        """List available models from the provider."""

    @abstractmethod
    async def complete(self, api_key: str, model: str, messages: list["ChatMessage"], **params: Any) -> "CompletionResult":
        """Complete a chat conversation."""

    @abstractmethod
    def validate_key(self, api_key: str) -> bool:
        """Validate an API key format (doesn't make network calls)."""
