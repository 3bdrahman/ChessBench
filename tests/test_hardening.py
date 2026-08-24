"""Regression tests for production-hardening fixes.

Each test locks a specific defect fixed during the hardening pass:
hangs, silent failures, injection, and dead registration paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import chess
import pytest

from chessbench.benchmark.export import _game_to_pgn_with_eval, _pgn_tag
from chessbench.benchmark.results_view import GameRecord
from chessbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chessbench.common.exceptions import ProviderAPIError
from chessbench.game.async_game import AsyncChessGame
from chessbench.move_parser import parse_move
from chessbench.providers.chess_ai import ProviderChessAI

# ---------------------------------------------------------------------------
# Move parser: castling and thinking-tag robustness
# ---------------------------------------------------------------------------

_CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


class TestCastlingParsing:
    def test_numeric_zero_castling_parses_as_kingside(self):
        board = chess.Board(_CASTLE_FEN)
        result = parse_move("I'll play 0-0 here", board)
        assert result.uci == "e1g1"

    def test_numeric_zero_queenside_castling(self):
        board = chess.Board(_CASTLE_FEN)
        result = parse_move("0-0-0", board)
        assert result.uci == "e1c1"

    def test_letter_o_castling_still_works(self):
        board = chess.Board(_CASTLE_FEN)
        result = parse_move("O-O", board)
        assert result.uci == "e1g1"

    def test_lowercase_letter_o_castling(self):
        board = chess.Board(_CASTLE_FEN)
        result = parse_move("o-o", board)
        assert result.uci == "e1g1"


class TestThinkingStripping:
    def test_unclosed_think_tag_to_end_of_text_is_stripped(self):
        text = "<think>I considered e4 then d5 then"
        board = chess.Board()
        result = parse_move(text, board)
        # The thinking runoff must not leak squares ("d5") as a move.
        assert result.uci is None or result.uci == ""

    def test_closed_think_tag_with_move_after(self):
        board = chess.Board()
        result = parse_move("<think>plan</think>e2e4", board)
        assert result.uci == "e2e4"


# ---------------------------------------------------------------------------
# Registry: direct provider lookup without warm-up
# ---------------------------------------------------------------------------

class TestRegistryAutoRegistration:
    def test_ensure_registers_all_builtin_providers(self):
        from chessbench.providers.registry import (
            PROVIDER_REGISTRY,
            ensure_providers_registered,
        )

        ensure_providers_registered()
        for name in ("openai", "anthropic", "google", "groq", "nim",
                     "together", "fireworks", "deepinfra", "ollama"):
            assert name in PROVIDER_REGISTRY, f"{name} missing after auto-registration"

    def test_provider_chess_ai_resolves_nim_directly(self):
        ai = ProviderChessAI(provider_name="nim", model_id="m", api_key="nvapi-x")
        assert ai.provider is not None


# ---------------------------------------------------------------------------
# OpenAI-compatible providers: empty choices must raise typed errors
# ---------------------------------------------------------------------------

class TestEmptyChoicesGuard:
    def test_empty_choices_raises_provider_api_error_not_index_error(self):
        from chessbench.providers._openai_compat import extract_chat_message

        response = MagicMock()
        response.choices = []
        with pytest.raises(ProviderAPIError):
            extract_chat_message(response, "openai", "gpt-4o")

    def test_normal_choices_returns_message(self):
        from chessbench.providers._openai_compat import extract_chat_message

        response = MagicMock()
        response.choices = [MagicMock(message="m")]
        assert extract_chat_message(response, "openai", "gpt-4o") == "m"

    def test_missing_usage_yields_none_tokens(self):
        from chessbench.providers._openai_compat import usage_of

        response = MagicMock(spec=[])  # no .usage attribute at all
        assert usage_of(response) == (None, None)


# ---------------------------------------------------------------------------
# Export: PGN header injection
# ---------------------------------------------------------------------------

def _make_game(**overrides) -> GameRecord:
    base = {
        "game_id": "g1",
        "white_player": 'white" player',
        "black_player": "black\\player",
        "white_provider": "openai",
        "black_provider": "anthropic",
        "opening_eco": "C60",
        "opening_name": 'Spanish" Variation',
        "opening_fen": chess.STARTING_FEN,
        "result": "1-0",
        "result_numeric": 1.0,
        "total_moves": 1,
        "game_duration_sec": 1.0,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "termination_reason": "checkmate",
        "moves": [],
    }
    base.update(overrides)
    return GameRecord(**base)


class TestPgnEscaping:
    def test_pgn_tag_escapes_quotes_and_backslashes(self):
        assert _pgn_tag('a"b\\c') == 'a\\"b\\\\c'

    def test_game_pgn_headers_are_escaped(self):
        lines = _game_to_pgn_with_eval(_make_game())
        white_header = next(ln for ln in lines if ln.startswith("[White "))
        # Inner double-quote must be escaped so the tag still terminates at
        # the real closing delimiter.
        assert white_header.endswith('[White "white\\" player"]')
        black_header = next(ln for ln in lines if ln.startswith("[Black "))
        assert black_header.startswith('[Black "black\\\\')


# ---------------------------------------------------------------------------
# Game loop: headless failure must terminate, never hang
# ---------------------------------------------------------------------------

class FailingAI:
    name = "failing:ai"

    async def get_move_with_result(self, fen: str):
        raise RuntimeError("provider down")

    def reset_game(self) -> None:
        pass


class LegalAI:
    name = "legal:ai"

    def __init__(self) -> None:
        self.calls = 0

    async def get_move_with_result(self, fen: str):
        self.calls += 1
        board = chess.Board(fen)
        move = next(iter(board.legal_moves))
        return move.uci(), None

    def reset_game(self) -> None:
        pass


class TestHeadlessFailureTermination:
    @pytest.mark.asyncio
    async def test_non_interactive_game_ends_on_move_failure(self):
        game = AsyncChessGame(FailingAI(), LegalAI(), interactive=False)
        seen: list = []

        async def cb(state):
            seen.append(state)

        stats = await game.play_game(cb, delay=0)
        assert stats.termination_reason == "error"
        assert stats.winner is None
        assert seen[-1].is_game_over

    @pytest.mark.asyncio
    async def test_illegal_placeholder_removed_in_headless_failure(self):
        from chessbench.common.exceptions import MoveExhaustedError

        class ExhaustedAI:
            name = "exhausted:ai"

            async def get_move_with_result(self, fen: str):
                raise MoveExhaustedError(
                    "no valid move", fen=fen, legal_moves=[], attempted_moves=["z9z9"], raw_text=""
                )

            def reset_game(self) -> None:
                pass

        game = AsyncChessGame(ExhaustedAI(), LegalAI(), interactive=False)

        async def cb(state):
            pass

        stats = await game.play_game(cb, delay=0)
        assert stats.termination_reason == "illegal_move"
        assert all(not m.is_illegal for m in game.moves)


# ---------------------------------------------------------------------------
# Runner: headless run completes even when a provider dies mid-game
# ---------------------------------------------------------------------------

class TestRunnerHeadlessResilience:
    @pytest.mark.asyncio
    async def test_run_benchmark_survives_dead_provider(self, tmp_path):
        from chessbench.common.common_types import (
            ChatMessage,
            CompletionResult,
            ModelInfo,
            ModelProvider,
        )
        from chessbench.providers.registry import register_provider

        @register_provider
        class DeadProvider(ModelProvider):
            name = "mockdead"
            requires_api_key = False

            def validate_key(self, api_key: str) -> bool:
                return True

            async def list_models(self, api_key: str) -> list[ModelInfo]:
                return [ModelInfo(id="a", name="a", provider="mockdead")]

            async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult:
                raise RuntimeError("simulated outage")

        config = BenchmarkConfig(
            players=["mockdead:a", "mockdead:b"],
            games_per_pairing=1,
            opening_book="startpos",
            max_parallel_games=1,
            api_keys={},
            output_dir=str(tmp_path),
        )
        runner = BenchmarkRunner(config)

        runner.players = {
            "mockdead:a": ProviderChessAI("mockdead", "a", ""),
            "mockdead:b": ProviderChessAI("mockdead", "b", ""),
        }

        await asyncio_wait_for(runner.run_benchmark())
        summary_path = runner.run_dir / "summary.json"
        assert summary_path.exists(), "summary must be written even when every game fails"


async def asyncio_wait_for(coro):
    import asyncio

    return await asyncio.wait_for(coro, timeout=30)


# ---------------------------------------------------------------------------
# Clock: the exact API the game loop uses
# ---------------------------------------------------------------------------

class TestGameClockApiContract:
    def test_remaining_seconds_exists_for_game_loop(self):
        from chessbench.game.clock import GameClock

        clock = GameClock.from_seconds(30, 0)
        assert callable(clock.remaining_seconds)
        assert clock.remaining_seconds(True) <= 30.0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([str(Path(__file__))]))
