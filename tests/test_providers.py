"""Tests for provider implementations with mocked HTTP calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chess_fight.common.common_types import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from chess_fight.providers.anthropic import AnthropicProvider
from chess_fight.providers.google import GoogleProvider
from chess_fight.providers.nim import NIMProvider
from chess_fight.providers.ollama import OllamaProvider
from chess_fight.providers.openai import OpenAIProvider
from chess_fight.providers.openrouter import OpenRouterProvider
from chess_fight.providers.registry import (
    PROVIDER_REGISTRY,
    get_provider,
    list_providers,
    register_provider,
)


@pytest.fixture(autouse=True)
def _clean_provider_registry():
    """Ensure provider registry is clean before and after each test."""
    original = dict(PROVIDER_REGISTRY)
    yield
    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(original)


class MockModelProvider(ModelProvider):
    """Mock provider for testing base functionality."""
    name = "mock_test"
    requires_api_key = False

    def __init__(self):
        self.list_models_called = False
        self.complete_called = False

    async def list_models(self, api_key: str) -> list:
        self.list_models_called = True
        return [
            ModelInfo(id="mock-model-1", name="Mock Model 1", provider="mock_test"),
            ModelInfo(id="mock-model-2", name="Mock Model 2", provider="mock_test"),
        ]

    async def complete(self, api_key: str, model: str, messages: list, **params) -> CompletionResult:
        self.complete_called = True
        return CompletionResult(
            text="e2e4",
            tokens_in=100,
            tokens_out=5,
            latency_ms=100
        )

    def validate_key(self, api_key: str) -> bool:
        return True


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_register_and_get_provider(self):
        """Test registering and retrieving a provider."""
        mock_cls = MockModelProvider
        mock_cls.name = "mock_test_register"
        register_provider(mock_cls)

        provider = get_provider("mock_test_register")
        assert provider is not None
        assert provider.name == "mock_test_register"

    def test_list_providers(self):
        providers = list_providers()
        expected = ["openai", "anthropic", "google", "nim", "openrouter", "ollama"]
        for p in expected:
            assert p in providers

    def test_get_unknown_provider_returns_none(self):
        assert get_provider("nonexistent_provider_xyz") is None


class TestOpenAIProvider:
    @pytest.fixture
    def provider(self):
        return OpenAIProvider()

    @pytest.mark.asyncio
    async def test_validate_key(self, provider):
        assert provider.validate_key("sk-test12345678901234567890") is True
        assert provider.validate_key("invalid") is False
        assert provider.validate_key("") is False

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        with patch("chess_fight.providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_models = AsyncMock()
            mock_models.data = [
                MagicMock(id="gpt-4o", object="model"),
                MagicMock(id="gpt-4o-mini", object="model"),
                MagicMock(id="text-embedding-ada-002", object="model"),
                MagicMock(id="whisper-1", object="model"),
            ]
            mock_client.models.list = AsyncMock(return_value=mock_models)

            models = await provider.list_models("sk-test12345678901234567890")

            assert len(models) == 2
            assert models[0].id == "gpt-4o"
            assert models[1].id == "gpt-4o-mini"
            assert all(m.provider == "openai" for m in models)

    @pytest.mark.asyncio
    async def test_complete(self, provider):
        with patch("chess_fight.providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="e2e4"))]
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=5)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await provider.complete(
                "sk-test12345678901234567890",
                "gpt-4o",
                [ChatMessage(role="user", content="test")],
                temperature=0.1,
                max_tokens=100
            )

            assert result.text == "e2e4"
            assert result.tokens_in == 100
            assert result.tokens_out == 5


class TestAnthropicProvider:
    @pytest.fixture
    def provider(self):
        return AnthropicProvider()

    @pytest.mark.asyncio
    async def test_validate_key(self, provider):
        assert provider.validate_key("sk-ant-test123456789012345678901234") is True
        assert provider.validate_key("invalid") is False
        assert provider.validate_key("") is False

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        with patch("chess_fight.providers.anthropic.AsyncAnthropic") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_models = AsyncMock()
            mock_models.data = [
                MagicMock(id="claude-3-5-sonnet-20241022", display_name="Claude 3.5 Sonnet"),
                MagicMock(id="claude-3-5-haiku-20241022", display_name="Claude 3.5 Haiku"),
                MagicMock(id="claude-3-opus-20240229", display_name="Claude 3 Opus"),
            ]
            mock_client.models.list = AsyncMock(return_value=mock_models)

            models = await provider.list_models("sk-ant-test123456789012345678901234")

            assert len(models) >= 3
            assert any(m.id == "claude-3-5-sonnet-20241022" for m in models)
            assert any(m.id == "claude-3-5-haiku-20241022" for m in models)
            assert all(m.provider == "anthropic" for m in models)
            assert all(m.context_window == 128000 for m in models)

    @pytest.mark.asyncio
    async def test_complete(self, provider):
        with patch("chess_fight.providers.anthropic.AsyncAnthropic") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="e2e4")]
            mock_response.usage = MagicMock(input_tokens=100, output_tokens=5)
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            result = await provider.complete(
                "sk-ant-test123456789012345678901234",
                "claude-3-5-sonnet-20241022",
                [ChatMessage(role="user", content="test")],
                temperature=0.1,
                max_tokens=100
            )

            assert result.text == "e2e4"
            assert result.tokens_in == 100
            assert result.tokens_out == 5


class TestGoogleProvider:
    @pytest.fixture
    def provider(self):
        return GoogleProvider()

    @pytest.mark.asyncio
    async def test_validate_key(self, provider):
        assert provider.validate_key("valid_key_12345678901234567890") is True
        assert provider.validate_key("") is False

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        with patch("chess_fight.providers.google.genai.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_model1 = MagicMock()
            mock_model1.name = "models/gemini-1.5-pro"
            mock_model1.supported_actions = ["generateContent"]

            mock_model2 = MagicMock()
            mock_model2.name = "models/gemini-1.5-flash"
            mock_model2.supported_actions = ["generateContent"]

            mock_model3 = MagicMock()
            mock_model3.name = "models/embedding-001"
            mock_model3.supported_actions = ["embedContent"]

            # The API returns an iterable of models directly (synchronous)
            mock_client.models.list.return_value = [mock_model1, mock_model2, mock_model3]

            models = await provider.list_models("valid_key_12345678901234567890")

            assert len(models) == 2
            assert models[0].id == "gemini-1.5-pro"
            assert models[1].id == "gemini-1.5-flash"
            assert all(m.provider == "google" for m in models)

    @pytest.mark.asyncio
    async def test_complete(self, provider):
        with patch("chess_fight.providers.google.genai.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "e2e4"
            mock_response.usage_metadata = MagicMock(
                prompt_token_count=100,
                candidates_token_count=5
            )
            mock_client.models.generate_content = AsyncMock(return_value=mock_response)

            result = await provider.complete(
                "valid_key_12345678901234567890",
                "gemini-1.5-pro",
                [ChatMessage(role="user", content="test")],
                temperature=0.1,
                max_tokens=100
            )

            assert result.text == "e2e4"
            assert result.tokens_in == 100
            assert result.tokens_out == 5


class TestNIMProvider:
    @pytest.fixture
    def provider(self):
        return NIMProvider()

    @pytest.mark.asyncio
    async def test_validate_key(self, provider):
        assert provider.validate_key("valid_key1") is True
        assert provider.validate_key("") is False

    @pytest.mark.asyncio
    async def test_complete(self, provider):
        with patch("chess_fight.providers.nim.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="e2e4"))]
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=5)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            with patch.dict("os.environ", {"NIM_BASE_URL": "https://test.nvidia.com/v1"}):
                result = await provider.complete(
                    "test_key",
                    "meta/llama-3.1-70b-instruct",
                    [ChatMessage(role="user", content="test")],
                    temperature=0.1,
                    max_tokens=100
                )

                assert result.text == "e2e4"
                assert result.tokens_in == 100
                assert result.tokens_out == 5


class TestOpenRouterProvider:
    @pytest.fixture
    def provider(self):
        return OpenRouterProvider()

    @pytest.mark.asyncio
    async def test_validate_key(self, provider):
        """Test API key validation."""
        assert provider.validate_key("sk-or-validkey1234567890") is True
        assert provider.validate_key("invalid") is False
        assert provider.validate_key("") is False

    @pytest.mark.asyncio
    async def test_complete(self, provider):
        with patch("chess_fight.providers.openrouter.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="e2e4"))]
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=5)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await provider.complete(
                "sk-or-validkey1234567890",
                "anthropic/claude-3.5-sonnet",
                [ChatMessage(role="user", content="test")],
                temperature=0.1,
                max_tokens=100
            )

            assert result.text == "e2e4"
            assert result.tokens_in == 100
            assert result.tokens_out == 5


class TestOllamaProvider:
    @pytest.fixture
    def provider(self):
        return OllamaProvider()

    @pytest.mark.asyncio
    async def test_validate_key(self, provider):
        assert provider.validate_key("") is True
        assert provider.validate_key("anything") is True

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        with patch("chess_fight.providers.ollama.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.2:latest"},
                    {"name": "qwen2.5:7b"},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)

            models = await provider.list_models("")

            assert len(models) == 2
            assert models[0].id == "llama3.2:latest"
            assert models[1].id == "qwen2.5:7b"
            assert all(m.provider == "ollama" for m in models)

    @pytest.mark.asyncio
    async def test_complete(self, provider):
        with patch("chess_fight.providers.ollama.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "e2e4", "prompt_eval_count": 100, "eval_count": 5}
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            result = await provider.complete(
                "",
                "llama3.2",
                [ChatMessage(role="user", content="test")],
                temperature=0.1,
                max_tokens=100
            )

            assert result.text == "e2e4"
            assert result.tokens_in == 100
            assert result.tokens_out == 5


class TestProviderInterface:
    def test_all_providers_implement_interface(self):
        for name in list_providers():
            provider = get_provider(name)
            assert provider is not None, f"Provider {name} not found"

            assert hasattr(provider, 'name')
            assert hasattr(provider, 'requires_api_key')
            assert hasattr(provider, 'list_models')
            assert hasattr(provider, 'complete')
            assert hasattr(provider, 'validate_key')

            import inspect
            assert inspect.iscoroutinefunction(provider.list_models)
            assert inspect.iscoroutinefunction(provider.complete)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
