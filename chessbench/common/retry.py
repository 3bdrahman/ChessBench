"""Retry and backoff utilities for async operations.

Provides configurable retry policies with exponential backoff and jitter.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 2.0  # seconds
    max_delay: float = 30.0     # seconds
    jitter: bool = True         # add ±25% random jitter
    should_retry: Callable[[Exception], bool] | None = None

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        delay: float = min(self.initial_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            # ±25% jitter
            delay *= 1.0 + random.uniform(-0.25, 0.25)
        return delay


async def retry_async(
    policy: RetryPolicy,
    coro_factory: Callable[[], Awaitable[Any]],
) -> Any:
    """Call ``coro_factory`` in a loop with exponential backoff.

    Retries on exceptions where ``policy.should_retry(exc)`` returns True
    (or if ``should_retry`` is None, retries on all exceptions).
    Re-raises the last exception after ``max_attempts`` exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(policy.max_attempts):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            should_retry = (
                policy.should_retry(exc) if policy.should_retry else True
            )
            if not should_retry or attempt == policy.max_attempts - 1:
                raise

            delay = policy.delay_for_attempt(attempt)
            await asyncio.sleep(delay)

    # Should never reach here, but for type checker:
    assert last_exc is not None
    raise last_exc


# Preset policies for common scenarios
RETRY_TRANSIENT = RetryPolicy(
    max_attempts=3,
    initial_delay=2.0,
    max_delay=30.0,
    jitter=True,
    should_retry=lambda exc: exc.__class__.__name__ in (
        "RateLimitError",
        "TimeoutError",
        "ConnectionError",
        "NetworkError",
    ),
)

RETRY_RATE_LIMIT = RetryPolicy(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=60.0,
    jitter=True,
    should_retry=lambda exc: exc.__class__.__name__ == "RateLimitError",
)

RETRY_MOVE_PARSE = RetryPolicy(
    max_attempts=3,
    initial_delay=0.0,
    max_delay=0.0,
    jitter=False,
    should_retry=lambda exc: exc.__class__.__name__ == "MoveValidationError",
)


def is_retryable_error(exc: Exception) -> bool:
    """Return True if ``exc`` is a transient error worth retrying.

    This is a convenience that matches the behavior used in
    ``chessbench.common.exceptions.is_retryable``.
    """
    return exc.__class__.__name__ in (
        "RateLimitError",
        "TimeoutError",
        "ConnectionError",
        "NetworkError",
    )


__all__ = [
    "RETRY_MOVE_PARSE",
    "RETRY_RATE_LIMIT",
    "RETRY_TRANSIENT",
    "RetryPolicy",
    "is_retryable_error",
    "retry_async",
]
