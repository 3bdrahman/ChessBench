"""Tests for error handling in provider implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chess_fight.common.common_types import ChatMessage
from chess_fight.common.exceptions import (
    ConnectionError,
    InvalidApiKeyError,
    ModelNotFoundError,
    RateLimitError,
    TimeoutError,
)
from chess_fight.providers.anthropic import AnthropicProvider
from chess_fight.providers.ollama import OllamaProvider
from chess_fight.providers.openai import OpenAIProvider


class TestOpenAIErrorHandling:
    @pytest.fixture
    def provider(self):
        return OpenAIProvider()

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_typed_exception(self, provider):
        from openai import AuthenticationError as OpenAIAuthError

        err = OpenAIAuthError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )
        with patch("chess_fight.providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(side_effect=err)

            with pytest.raises(InvalidApiKeyError) as exc_info:
                await provider.complete(
                    "sk-bad-key",
                    "gpt-4o",
                    [ChatMessage(role="user", content="test")],
                )
        assert exc_info.value.provider == "openai"
        assert exc_info.value.http_status == 401

    @pytest.mark.asyncio
    async def test_rate_limit_raises_typed_exception(self, provider):
        from openai import RateLimitError as OpenAIRateLimitError

        response = MagicMock(status_code=429)
        response.headers = {"retry-after": "5"}
        err = OpenAIRateLimitError(
            message="Rate limit exceeded",
            response=response,
            body=None,
        )
        with patch("chess_fight.providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(side_effect=err)

            with pytest.raises(RateLimitError) as exc_info:
                await provider.complete(
                    "sk-test",
                    "gpt-4o",
                    [ChatMessage(role="user", content="test")],
                )
        assert exc_info.value.provider == "openai"
        assert exc_info.value.retry_after == 5.0

    @pytest.mark.asyncio
    async def test_timeout_raises_typed_exception(self, provider):
        from openai import APITimeoutError

        with patch("chess_fight.providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(side_effect=APITimeoutError(MagicMock()))

            with pytest.raises(TimeoutError) as exc_info:
                await provider.complete(
                    "sk-test",
                    "gpt-4o",
                    [ChatMessage(role="user", content="test")],
                )
        assert exc_info.value.provider == "openai"

    @pytest.mark.asyncio
    async def test_model_not_found_raises_typed_exception(self, provider):
        from openai import NotFoundError

        response = MagicMock(status_code=404)
        err = NotFoundError(
            message="Model not found",
            response=response,
            body=None,
        )
        with patch("chess_fight.providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(side_effect=err)

            with pytest.raises(ModelNotFoundError) as exc_info:
                await provider.complete(
                    "sk-test",
                    "gpt-99-fake",
                    [ChatMessage(role="user", content="test")],
                )
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model_id == "gpt-99-fake"


class TestAnthropicErrorHandling:
    @pytest.fixture
    def provider(self):
        return AnthropicProvider()

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_typed_exception(self, provider):
        from anthropic import AuthenticationError as AnthropicAuthError

        err = AnthropicAuthError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )
        with patch("chess_fight.providers.anthropic.AsyncAnthropic") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.messages.create = AsyncMock(side_effect=err)

            with pytest.raises(InvalidApiKeyError) as exc_info:
                await provider.complete(
                    "sk-ant-bad",
                    "claude-3-5-sonnet-20241022",
                    [ChatMessage(role="user", content="test")],
                )
        assert exc_info.value.provider == "anthropic"


class TestOllamaErrorHandling:
    @pytest.fixture
    def provider(self):
        return OllamaProvider()

    @pytest.mark.asyncio
    async def test_connection_error_raises_provider_unavailable(self, provider):
        import httpx

        with patch("chess_fight.providers.ollama.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

            from chess_fight.common.exceptions import ProviderUnavailableError
            with pytest.raises(ProviderUnavailableError) as exc_info:
                await provider.complete(
                    "",
                    "llama3.2",
                    [ChatMessage(role="user", content="test")],
                )
        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_timeout_raises_typed_exception(self, provider):
        import httpx

        with patch("chess_fight.providers.ollama.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

            with pytest.raises(TimeoutError) as exc_info:
                await provider.complete(
                    "",
                    "llama3.2",
                    [ChatMessage(role="user", content="test")],
                )
        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_404_raises_model_not_found(self, provider):
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 404
        http_error = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response,
        )
        with patch("chess_fight.providers.ollama.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=http_error)

            with pytest.raises(ModelNotFoundError):
                await provider.complete(
                    "",
                    "nonexistent-model",
                    [ChatMessage(role="user", content="test")],
                )


class TestProviderRetryBehavior:
    @pytest.fixture
    def provider(self):
        return OpenAIProvider()

    @pytest.mark.asyncio
    async def test_connection_error_raises_typed(self, provider):
        import httpx
        from openai import APIConnectionError

        with patch("chess_fight.providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=APIConnectionError(message="Connection failed", request=httpx.Request("GET", "https://api.openai.com"))
            )

            with pytest.raises(ConnectionError) as exc_info:
                await provider.complete(
                    "sk-test",
                    "gpt-4o",
                    [ChatMessage(role="user", content="test")],
                )
        assert exc_info.value.provider == "openai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
