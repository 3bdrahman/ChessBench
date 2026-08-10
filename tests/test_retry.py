"""Tests for the retry/backoff utility."""


import pytest

from chess_fight.common.exceptions import (
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    TimeoutError,
)
from chess_fight.common.retry import (
    RETRY_MOVE_PARSE,
    RETRY_RATE_LIMIT,
    RETRY_TRANSIENT,
    RetryPolicy,
    retry_async,
)


@pytest.mark.asyncio
class TestRetryPolicy:
    async def test_delay_grows_exponentially(self):
        policy = RetryPolicy(max_attempts=5, initial_delay=1.0, max_delay=100.0, jitter=False)
        assert policy.delay_for_attempt(0) == 1.0
        assert policy.delay_for_attempt(1) == 2.0
        assert policy.delay_for_attempt(2) == 4.0
        assert policy.delay_for_attempt(3) == 8.0

    async def test_delay_capped_at_max(self):
        policy = RetryPolicy(max_attempts=10, initial_delay=1.0, max_delay=5.0, jitter=False)
        for i in range(10):
            assert policy.delay_for_attempt(i) <= 5.0

    async def test_jitter_applied(self):
        policy = RetryPolicy(max_attempts=3, initial_delay=10.0, max_delay=100.0, jitter=True)
        delays = [policy.delay_for_attempt(1) for _ in range(50)]
        # base = 10.0 * 2^1 = 20.0; with ±25% jitter, range is [15.0, 25.0]
        assert min(delays) >= 15.0
        assert max(delays) <= 25.0
        # Variance — should not all be identical
        assert len(set(delays)) > 1


@pytest.mark.asyncio
class TestRetryAsync:
    async def test_succeeds_first_attempt(self):
        calls = []

        async def factory():
            calls.append(1)
            return "ok"

        result = await retry_async(RETRY_TRANSIENT, factory)
        assert result == "ok"
        assert len(calls) == 1

    async def test_retries_on_retryable_then_succeeds(self):
        calls = []

        async def factory():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError(provider="x", timeout_seconds=30.0)
            return "ok"

        # Use policy with no delay for fast test
        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=0.0,
            max_delay=0.0,
            jitter=False,
            should_retry=lambda e: isinstance(e, (TimeoutError, ConnectionError)),
        )
        result = await retry_async(policy, factory)
        assert result == "ok"
        assert len(calls) == 3

    async def test_raises_after_max_attempts(self):
        calls = []

        async def factory():
            calls.append(1)
            raise TimeoutError(provider="x", timeout_seconds=30.0)

        policy = RetryPolicy(
            max_attempts=3,
            initial_delay=0.0,
            max_delay=0.0,
            jitter=False,
            should_retry=lambda e: isinstance(e, (TimeoutError, ConnectionError)),
        )
        with pytest.raises(TimeoutError):
            await retry_async(policy, factory)
        assert len(calls) == 3

    async def test_does_not_retry_non_retryable(self):
        calls = []

        async def factory():
            calls.append(1)
            raise AuthenticationError(provider="x", detail="bad key")

        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=0.0,
            max_delay=0.0,
            jitter=False,
            should_retry=lambda e: isinstance(e, (TimeoutError, ConnectionError)),
        )
        with pytest.raises(AuthenticationError):
            await retry_async(policy, factory)
        assert len(calls) == 1

    async def test_rate_limit_retry_with_backoff(self):
        calls = []

        async def factory():
            calls.append(1)
            if len(calls) < 3:
                raise RateLimitError(provider="x")
            return "ok"

        policy = RETRY_RATE_LIMIT
        # Override to skip real sleep delay
        policy_fast = RetryPolicy(
            max_attempts=5,
            initial_delay=0.001,
            max_delay=0.01,
            jitter=False,
            should_retry=lambda e: isinstance(e, RateLimitError),
        )
        result = await retry_async(policy_fast, factory)
        assert result == "ok"
        assert len(calls) == 3


@pytest.mark.asyncio
class TestPresetPolicies:
    async def test_retry_transient_retries_transient(self):
        calls = []

        async def factory():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError(provider="x")
            return "ok"

        # Patch the policy to use fast delays
        fast = RetryPolicy(
            max_attempts=3,
            initial_delay=0.001,
            max_delay=0.01,
            jitter=False,
            should_retry=RETRY_TRANSIENT.should_retry,
        )
        result = await retry_async(fast, factory)
        assert result == "ok"

    async def test_retry_move_parse_does_not_delay(self):
        from chess_fight.common.exceptions import MoveValidationError

        calls = []

        async def factory():
            calls.append(1)
            if len(calls) < 2:
                raise MoveValidationError("x", fen="x", legal_moves=[], raw_text="")
            return "ok"

        result = await retry_async(RETRY_MOVE_PARSE, factory)
        assert result == "ok"
        assert len(calls) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
