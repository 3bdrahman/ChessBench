"""Live integration tests for real LLM API providers.

These tests execute real network calls against configured LLM provider endpoints
when an API key is present in the environment (e.g. via .env or OS environment).
If an API key is missing for a provider, the test for that provider is skipped.

Run explicitly with:
    pytest tests/test_live_providers.py
"""

import os
import chess
import pytest

from chessbench.common.common_types import ChatMessage
from chessbench.providers import list_providers
from chessbench.providers.chess_ai import ProviderChessAI


def _get_api_key_for_provider(provider_name: str) -> str | None:
    """Retrieve API key for provider from OS environment."""
    env_keys = [
        f"{provider_name.upper()}_API_KEY",
        f"{provider_name.lower()}_api_key",
    ]
    for k in env_keys:
        val = os.getenv(k)
        if val and val.strip():
            return val.strip()
    return None


class TestLiveProvidersIntegration:
    """Live API tests executed against real LLM provider endpoints when keys exist."""

    @pytest.mark.parametrize("provider_name", [
        "openai",
        "anthropic",
        "google",
        "openrouter",
        "groq",
        "nim",
        "together",
        "fireworks",
        "deepinfra",
    ])
    @pytest.mark.asyncio
    async def test_live_provider_completion(self, provider_name: str):
        api_key = _get_api_key_for_provider(provider_name)
        if not api_key:
            pytest.skip(f"No API key available for {provider_name} in environment")

        # Pick default model for provider
        default_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "google": "gemini-2.0-flash",
            "openrouter": "meta-llama/llama-3.3-70b-instruct",
            "groq": "llama-3.3-70b-versatile",
            "nim": "meta/llama-3.3-70b-instruct",
            "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "fireworks": "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "deepinfra": "meta-llama/Llama-3.3-70B-Instruct",
        }
        model_id = default_models.get(provider_name, "default")

        ai = ProviderChessAI(
            provider_name=provider_name,
            model_id=model_id,
            api_key=api_key,
            temperature=0.1,
            max_tokens=200,
        )

        board = chess.Board()
        res = await ai.get_move(board)

        assert res.raw_text is not None and len(res.raw_text) > 0
        assert res.tokens_in is not None and res.tokens_in > 0
        assert res.tokens_out is not None and res.tokens_out > 0
        assert res.latency_ms is not None and res.latency_ms > 0

        # Move should either be parsed or identified cleanly
        if res.uci is not None:
            assert res.move in board.legal_moves, (
                f"Live model {provider_name}:{model_id} returned illegal move {res.uci}"
            )
