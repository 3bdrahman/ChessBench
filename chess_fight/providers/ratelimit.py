"""Rate limiter for provider API calls."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    max_concurrent: int = 4
    max_queue_time_seconds: float = 300.0  # 5 minutes max wait


class TokenBucketRateLimiter:
    """Token bucket rate limiter with sliding window for requests and tokens."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_timestamps: deque[float] = deque()
        self.token_usage: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self.active_requests = 0
        self._lock = asyncio.Lock()

        # For Retry-After handling
        self._retry_after_until: float = 0

    async def acquire(self, tokens: int = 1) -> float | None:
        """
        Acquire permission to make a request.

        Returns:
            Time waited in seconds, or None if rate limited and max queue time exceeded.
        """
        wait_start = time.time()
        while True:
            if time.time() - wait_start > self.config.max_queue_time_seconds:
                return None
                
            async with self._lock:
                now = time.time()
                wait_time = 0.0

                # Check if we're in a Retry-After period
                if now < self._retry_after_until:
                    wait_time = self._retry_after_until - now
                
                # Check concurrent limit
                elif self.active_requests >= self.config.max_concurrent:
                    wait_time = 0.1
                
                else:
                    # Clean old timestamps (older than 1 minute)
                    cutoff = now - 60
                    while self.request_timestamps and self.request_timestamps[0] < cutoff:
                        self.request_timestamps.popleft()

                    # Check request rate limit
                    if len(self.request_timestamps) >= self.config.requests_per_minute:
                        wait_time = self.request_timestamps[0] + 60 - now
                    else:
                        # Clean old token usage
                        while self.token_usage and self.token_usage[0][0] < cutoff:
                            self.token_usage.popleft()

                        # Check token rate limit
                        tokens_used_last_minute = sum(t for _, t in self.token_usage)
                        if tokens_used_last_minute + tokens > self.config.tokens_per_minute:
                            wait_time = self._calculate_token_wait(tokens, cutoff, now)
                        else:
                            # All checks passed - record the request
                            self.request_timestamps.append(now)
                            self.token_usage.append((now, tokens))
                            self.active_requests += 1
                            return 0.0

            # If we reach here, we need to wait outside the lock
            if wait_time > 0:
                if time.time() + wait_time - wait_start > self.config.max_queue_time_seconds:
                    return None
                await asyncio.sleep(wait_time)

    def _calculate_token_wait(self, needed_tokens: int, cutoff: float, now: float) -> float:
        """Calculate how long to wait for enough tokens to be available."""
        # Simplified: wait until oldest tokens expire
        if self.token_usage:
            oldest_time = self.token_usage[0][0]
            return max(0, oldest_time + 60 - now)
        return 0

    def release(self) -> None:
        """Release a request slot."""
        self.active_requests = max(0, self.active_requests - 1)

    def set_retry_after(self, seconds: float) -> None:
        """Set a Retry-After delay from provider response."""
        self._retry_after_until = time.time() + seconds

    async def __aenter__(self) -> TokenBucketRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object | None) -> bool:
        self.release()
        return False


class ProviderRateLimiter:
    """Manages rate limiters for multiple providers."""

    def __init__(self, default_config: RateLimitConfig | None = None):
        self.default_config = default_config or RateLimitConfig()
        self.limiters: dict[str, TokenBucketRateLimiter] = {}
        self._configs: dict[str, RateLimitConfig] = {}

    def set_config(self, provider: str, config: RateLimitConfig) -> None:
        """Set custom config for a provider."""
        self._configs[provider] = config
        if provider in self.limiters:
            self.limiters[provider] = TokenBucketRateLimiter(config)

    def get_limiter(self, provider: str) -> TokenBucketRateLimiter:
        """Get or create limiter for a provider."""
        if provider not in self.limiters:
            config = self._configs.get(provider, self.default_config)
            self.limiters[provider] = TokenBucketRateLimiter(config)
        return self.limiters[provider]

    async def acquire(self, provider: str, tokens: int = 1) -> float | None:
        """Acquire permission for a provider."""
        limiter = self.get_limiter(provider)
        return await limiter.acquire(tokens)

    def release(self, provider: str) -> None:
        """Release a request slot for a provider."""
        if provider in self.limiters:
            self.limiters[provider].release()

    def handle_retry_after(self, provider: str, seconds: float) -> None:
        """Handle Retry-After header from provider."""
        limiter = self.get_limiter(provider)
        limiter.set_retry_after(seconds)
