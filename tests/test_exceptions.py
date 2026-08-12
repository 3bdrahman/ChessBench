"""Tests for the typed exception hierarchy."""

import pytest

from chess_fight.common.exceptions import (
    AuthenticationError,
    AuthError,
    BenchmarkError,
    ChessFightError,
    ConnectionError,
    GameExecutionError,
    InvalidApiKeyError,
    ModelNotFoundError,
    MoveValidationError,
    NetworkError,
    NoProvidersConfiguredError,
    ProviderAPIError,
    ProviderError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    SetupError,
    TimeoutError,
    is_retryable,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_chess_fight_error(self):
        for cls in (
            ProviderError,
            NoProvidersConfiguredError,
            ProviderUnavailableError,
            AuthError,
            InvalidApiKeyError,
            AuthenticationError,
            RateLimitError,
            NetworkError,
            TimeoutError,
            ConnectionError,
            ModelNotFoundError,
            QuotaExceededError,
            ProviderAPIError,
            MoveValidationError,
            BenchmarkError,
            SetupError,
            GameExecutionError,
        ):
            assert issubclass(cls, ChessFightError), f"{cls.__name__} must subclass ChessFightError"

    def test_provider_errors_inherit_from_provider_error(self):
        for cls in (
            NoProvidersConfiguredError,
            ProviderUnavailableError,
            AuthError,
            InvalidApiKeyError,
            AuthenticationError,
            RateLimitError,
            NetworkError,
            TimeoutError,
            ConnectionError,
            ModelNotFoundError,
            QuotaExceededError,
            ProviderAPIError,
        ):
            assert issubclass(cls, ProviderError), f"{cls.__name__} must subclass ProviderError"

    def test_auth_errors_inherit_from_auth_error(self):
        for cls in (InvalidApiKeyError, AuthenticationError):
            assert issubclass(cls, AuthError), f"{cls.__name__} must subclass AuthError"

    def test_network_errors_inherit_from_network_error(self):
        for cls in (TimeoutError, ConnectionError):
            assert issubclass(cls, NetworkError), f"{cls.__name__} must subclass NetworkError"

    def test_benchmark_errors_inherit_from_benchmark_error(self):
        for cls in (SetupError, GameExecutionError):
            assert issubclass(cls, BenchmarkError), f"{cls.__name__} must subclass BenchmarkError"


class TestInvalidApiKeyError:
    def test_fields_populated(self):
        exc = InvalidApiKeyError(
            provider="openai",
            got_prefix="sk-abcde…",
            expected_prefix="sk-",
        )
        assert exc.provider == "openai"
        assert exc.got_prefix == "sk-abcde…"
        assert exc.expected_prefix == "sk-"
        assert exc.http_status == 401
        assert "Invalid API key" in str(exc)
        assert "openai" in str(exc)


class TestRateLimitError:
    def test_with_retry_after(self):
        exc = RateLimitError(provider="openai", retry_after=12.5)
        assert exc.provider == "openai"
        assert exc.retry_after == 12.5
        assert exc.http_status == 429
        assert "12s" in str(exc)

    def test_without_retry_after(self):
        exc = RateLimitError(provider="openai")
        assert exc.retry_after is None


class TestTimeoutError:
    def test_fields(self):
        exc = TimeoutError(provider="anthropic", timeout_seconds=30.0)
        assert exc.provider == "anthropic"
        assert exc.timeout_seconds == 30.0
        assert exc.host is None
        assert "anthropic" in str(exc)
        assert "30.0" in str(exc)

    def test_with_host(self):
        exc = TimeoutError(provider="anthropic", timeout_seconds=30.0, host="api.anthropic.com")
        assert exc.host == "api.anthropic.com"


class TestConnectionError:
    def test_fields(self):
        exc = ConnectionError(provider="ollama", host="localhost:11434", detail="refused")
        assert exc.provider == "ollama"
        assert exc.host == "localhost:11434"
        assert exc.detail == "refused"


class TestModelNotFoundError:
    def test_with_available_models(self):
        models = ["gpt-4o", "gpt-4o-mini", "o1-preview"]
        exc = ModelNotFoundError(provider="openai", model_id="gpt-99", available_models=models)
        assert exc.provider == "openai"
        assert exc.model_id == "gpt-99"
        assert exc.available_models == models
        assert "gpt-4o" in str(exc)

    def test_without_available_models(self):
        exc = ModelNotFoundError(provider="openai", model_id="gpt-99")
        assert exc.available_models is None


class TestProviderAPIError:
    def test_fields(self):
        exc = ProviderAPIError(
            provider="openai",
            status_code=500,
            detail="Internal Server Error",
        )
        assert exc.provider == "openai"
        assert exc.status_code == 500
        assert exc.detail == "Internal Server Error"


class TestNoProvidersConfiguredError:
    def test_message(self):
        exc = NoProvidersConfiguredError()
        assert "No providers" in str(exc)


class TestProviderUnavailableError:
    def test_with_detail(self):
        exc = ProviderUnavailableError("ollama", "server not running")
        assert exc.provider == "ollama"
        assert "server not running" in str(exc)


class TestAuthenticationError:
    def test_fields(self):
        exc = AuthenticationError(provider="openai", detail="key revoked")
        assert exc.provider == "openai"
        assert exc.detail == "key revoked"
        assert exc.http_status == 403


class TestQuotaExceededError:
    def test_fields(self):
        exc = QuotaExceededError(provider="openai", detail="monthly cap")
        assert "Quota exceeded" in str(exc)
        assert "monthly cap" in str(exc)


class TestSetupError:
    def test_message(self):
        exc = SetupError("No players configured")
        assert "No players configured" in str(exc)


class TestGameExecutionError:
    def test_fields(self):
        cause = RateLimitError(provider="openai")
        exc = GameExecutionError(
            "Benchmark aborted",
            game_index=5,
            white="openai:gpt-4o",
            black="anthropic:claude-3-5-sonnet",
            cause=cause,
        )
        assert exc.game_index == 5
        assert exc.white == "openai:gpt-4o"
        assert exc.black == "anthropic:claude-3-5-sonnet"
        assert exc.cause is cause
        assert "Benchmark aborted" in str(exc)


class TestMoveValidationError:
    def test_fields(self):
        exc = MoveValidationError(
            "Invalid move",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            legal_moves=["e2e4", "d2d4"],
            raw_text="garbage",
        )
        assert exc.fen
        assert exc.legal_moves == ["e2e4", "d2d4"]
        assert exc.raw_text == "garbage"


class TestIsRetryable:
    def test_rate_limit_is_retryable(self):
        exc = RateLimitError(provider="openai")
        assert is_retryable(exc) is True

    def test_timeout_is_retryable(self):
        exc = TimeoutError(provider="openai", timeout_seconds=30.0)
        assert is_retryable(exc) is True

    def test_connection_error_is_retryable(self):
        exc = ConnectionError(provider="openai")
        assert is_retryable(exc) is True

    def test_network_error_base_is_retryable(self):
        exc = NetworkError(provider="openai")
        assert is_retryable(exc) is True

    def test_auth_error_is_not_retryable(self):
        exc = AuthenticationError(provider="openai", detail="bad")
        assert is_retryable(exc) is False

    def test_invalid_key_is_not_retryable(self):
        exc = InvalidApiKeyError(provider="openai", got_prefix="x", expected_prefix="sk-")
        assert is_retryable(exc) is False

    def test_model_not_found_is_not_retryable(self):
        exc = ModelNotFoundError(provider="openai", model_id="x")
        assert is_retryable(exc) is False

    def test_provider_api_error_is_not_retryable(self):
        exc = ProviderAPIError(provider="openai", status_code=400, detail="x")
        assert is_retryable(exc) is False

    def test_provider_api_error_500_is_retryable(self):
        exc = ProviderAPIError(provider="openai", status_code=500, detail="x")
        assert is_retryable(exc) is True


class TestChessFightErrorLog:
    def test_log_method_runs(self, caplog):
        exc = ChessFightError("test")
        with caplog.at_level("ERROR"):
            exc.log()
        assert "ChessFightError" in caplog.text
        assert "test" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
