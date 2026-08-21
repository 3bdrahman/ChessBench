"""Typed exception hierarchy for chess_fight.

All custom exceptions subclass :class:`ChessFightError` for single-point catching.
Retryable errors (rate limit, timeout, network) are distinguishable from
non-retryable ones (auth, quota, model not found) via :func:`is_retryable`.
"""

import logging

_log = logging.getLogger(__name__)


class ChessFightError(Exception):
    """Base exception for all chess_fight errors."""

    def log(self, logger: logging.Logger = _log) -> None:
        logger.error("%s: %s", type(self).__name__, self)


class ProviderError(ChessFightError):
    """Base class for all provider-related errors."""

    provider: str

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message)
        self.provider = provider


class NoProvidersConfiguredError(ProviderError):
    """Raised when no providers are registered at startup."""

    def __init__(self) -> None:
        super().__init__("No providers configured", "")


class ProviderUnavailableError(ProviderError):
    """Raised when a provider's service/binary is unreachable."""

    def __init__(self, provider: str, detail: str = "") -> None:
        msg = f"Provider '{provider}' unavailable" + (f": {detail}" if detail else "")
        super().__init__(msg, provider)


class AuthError(ProviderError):
    """Base class for authentication errors."""

    def __init__(self, provider: str, detail: str, http_status: int | None = None) -> None:
        super().__init__(f"Authentication failed for {provider}: {detail}", provider)
        self.detail = detail
        self.http_status = http_status


class InvalidApiKeyError(AuthError):
    """Raised when API key format is invalid or key is rejected."""

    def __init__(
        self,
        provider: str,
        got_prefix: str,
        expected_prefix: str,
        http_status: int = 401,
    ) -> None:
        detail = (
            f"Invalid API key format. Expected prefix '{expected_prefix}…', "
            f"got '{got_prefix}…'"
        )
        super().__init__(provider, detail, http_status)
        self.got_prefix = got_prefix
        self.expected_prefix = expected_prefix


class AuthenticationError(AuthError):
    """Raised when a valid-format key is rejected (expired, revoked, etc.)."""

    def __init__(self, provider: str, detail: str, http_status: int = 403) -> None:
        super().__init__(provider, detail, http_status)


class RateLimitError(ProviderError):
    """Raised when provider returns 429 (rate limited).

    If the response includes a ``Retry-After`` header, ``retry_after`` is set
    to that value in seconds. Otherwise it is ``None`` and callers should
    fall back to exponential backoff.
    """

    def __init__(
        self,
        provider: str,
        retry_after: float | None = None,
        http_status: int = 429,
        raw_response: dict[str, object] | None = None,
    ) -> None:
        msg = f"Rate limited by {provider}"
        if retry_after is not None:
            msg += f" (retry after {retry_after:.0f}s)"
        super().__init__(msg, provider)
        self.retry_after = retry_after
        self.http_status = http_status
        self.raw_response = raw_response


class NetworkError(ProviderError):
    """Base class for network-level errors (connection, DNS, TLS)."""

    def __init__(self, provider: str, host: str | None = None, detail: str = "") -> None:
        msg = f"Network error connecting to {provider}"
        if host:
            msg += f" at {host}"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, provider)
        self.host = host
        self.detail = detail


class TimeoutError(NetworkError):
    """Raised when a request exceeds its timeout.

    Distinct from ``asyncio.TimeoutError`` to avoid shadowing.
    """

    def __init__(
        self,
        provider: str,
        timeout_seconds: float,
        host: str | None = None,
    ) -> None:
        super().__init__(provider, host, f"timeout after {timeout_seconds}s")
        self.timeout_seconds = timeout_seconds


class ConnectionError(NetworkError):
    """Raised when TCP connection cannot be established."""

    def __init__(
        self,
        provider: str,
        host: str | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(provider, host, detail)


class ModelNotFoundError(ProviderError):
    """Raised when the requested model does not exist on the provider."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        available_models: list[str] | None = None,
    ) -> None:
        msg = f"Model '{model_id}' not found on {provider}"
        if available_models:
            msg += f". Available: {', '.join(available_models[:5])}"
            if len(available_models) > 5:
                msg += f" (+{len(available_models) - 5} more)"
        super().__init__(msg, provider)
        self.model_id = model_id
        self.available_models = available_models


class QuotaExceededError(ProviderError):
    """Raised when provider quota is exhausted (not just rate limit)."""

    def __init__(self, provider: str, detail: str = "") -> None:
        msg = f"Quota exceeded for {provider}"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, provider)
        self.detail = detail


class ProviderAPIError(ProviderError):
    """Catch-all for provider 4xx/5xx errors not mapped above."""

    def __init__(
        self,
        provider: str,
        status_code: int,
        detail: str,
        raw_response: dict[str, object] | None = None,
    ) -> None:
        super().__init__(f"{provider} API error {status_code}: {detail}", provider)
        self.status_code = status_code
        self.detail = detail
        self.raw_response = raw_response


class MoveValidationError(ChessFightError):
    """Raised when LLM output cannot be parsed as a legal UCI move."""

    def __init__(
        self,
        message: str,
        fen: str,
        legal_moves: list[str],
        raw_text: str,
    ) -> None:
        super().__init__(message)
        self.fen = fen
        self.legal_moves = legal_moves
        self.raw_text = raw_text


class MoveFormatError(MoveValidationError):
    """Raised when LLM output cannot be parsed due to bad format."""

class MoveExhaustedError(ChessFightError):
    """Raised when all retry attempts to get a valid move are exhausted.

    This replaces the old fallback behavior of playing the first legal move.
    """

    def __init__(
        self,
        message: str,
        fen: str,
        legal_moves: list[str],
        attempted_moves: list[str],
        raw_text: str,
    ) -> None:
        super().__init__(message)
        self.fen = fen
        self.legal_moves = legal_moves
        self.attempted_moves = attempted_moves
        self.raw_text = raw_text


class BenchmarkError(ChessFightError):
    """Base class for benchmark runner errors."""



class SetupError(BenchmarkError):
    """Raised during BenchmarkRunner initialization for config issues."""



class GameExecutionError(BenchmarkError):
    """Raised when a single game fails during benchmark execution.

    Per-game failures (timeout, evaluator crash, etc.) are recoverable: the
    benchmark runner logs the failure and continues with the remaining games.
    The legacy behavior of treating any ``GameExecutionError`` as fatal was
    overly aggressive and is corrected in the runner's classification.
    """

    def __init__(
        self,
        message: str,
        game_index: int,
        white: str,
        black: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.game_index = game_index
        self.white = white
        self.black = black
        self.cause = cause


class GameTimeoutError(GameExecutionError):
    """Raised when a single game exceeds its wall-clock budget.

    A per-game timeout is recoverable: the benchmark continues with the
    remaining games. Distinct from the HTTP-level :class:`TimeoutError` and
    from the old behavior where any ``GameExecutionError`` aborted the run.
    """

    def __init__(
        self,
        timeout_seconds: float,
        game_index: int,
        white: str,
        black: str,
    ) -> None:
        super().__init__(
            f"Game timed out after {timeout_seconds:g} seconds",
            game_index=game_index,
            white=white,
            black=black,
        )
        self.timeout_seconds = timeout_seconds


class LimiterExhaustedError(GameExecutionError):
    """Raised when the outbound rate limiter's queue-wait budget is exceeded.

    Indicates the limiter for a provider is permanently saturated: continuing
    to dispatch games to that provider would just queue them all behind the
    same stall, so the runner treats this as a fatal benchmark-wide signal.
    """

    def __init__(
        self,
        provider: str,
        game_index: int,
        white: str,
        black: str,
    ) -> None:
        super().__init__(
            f"Rate limit exceeded for {provider}, max queue time exceeded",
            game_index=game_index,
            white=white,
            black=black,
        )
        self.provider = provider


class FatalBenchmarkError(BenchmarkError):
    """Raised by the runner when it aborts the entire benchmark.

    Carries the unrecoverable cause (auth failure, rate limit, limiter
    saturation, etc.). Replaces the legacy pattern of overloading
    :class:`GameExecutionError` as the abort wrapper, so the UI can
    distinguish "the whole run was aborted" from "one game failed".
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


def is_retryable(exc: Exception) -> bool:
    """Return True if ``exc`` is a transient error worth retrying."""
    return (
        isinstance(exc, (RateLimitError, TimeoutError, ConnectionError, NetworkError))
        or (
            isinstance(exc, ProviderAPIError)
            and getattr(exc, "status_code", 0) >= 500
        )
    )


__all__ = [
    "AuthError",
    "AuthenticationError",
    "BenchmarkError",
    "ChessFightError",
    "ConnectionError",
    "FatalBenchmarkError",
    "GameExecutionError",
    "GameTimeoutError",
    "InvalidApiKeyError",
    "LimiterExhaustedError",
    "ModelNotFoundError",
    "MoveExhaustedError",
    "MoveFormatError",
    "MoveValidationError",
    "NetworkError",
    "NoProvidersConfiguredError",
    "ProviderAPIError",
    "ProviderError",
    "ProviderUnavailableError",
    "QuotaExceededError",
    "RateLimitError",
    "SetupError",
    "TimeoutError",
    "is_retryable",
]
