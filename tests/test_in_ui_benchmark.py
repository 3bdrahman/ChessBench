"""Real integration tests for the in-process benchmark runner."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from chess_fight.benchmark.results_view import load_run
from chess_fight.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chess_fight.providers.chess_ai import ProviderChessAI
from chess_fight.providers.stockfish import StockfishProvider

_STUB_ENGINE = str((Path(__file__).parent / "fixtures" / "uci_stub_engine.py").resolve())
_FULL_CMD = f"{sys.executable} {_STUB_ENGINE}"


def _make_stub_provider() -> StockfishProvider:
    provider = StockfishProvider()
    provider.find_binary = lambda: _FULL_CMD  # type: ignore[method-assign]
    return provider


def _make_ai(spec: str) -> ProviderChessAI:
    provider_name, model_id = spec.split(":", 1)
    return ProviderChessAI(
        provider_name=provider_name,
        model_id=model_id,
        api_key="",
        temperature=0.0,
        max_tokens=100,
    )


@pytest.fixture
def stockfish_stub_provider():
    """Pre-configured StockfishProvider wired to the stub engine."""
    return _make_stub_provider()


@pytest.fixture
def patched_provider(stockfish_stub_provider):
    """Patch get_provider where ProviderChessAI uses it."""
    from chess_fight.providers import chess_ai as providers_chess_ai
    original = providers_chess_ai.get_provider
    def _patched(name: str):
        if name == "stockfish":
            return stockfish_stub_provider
        return original(name)
    with patch("chess_fight.providers.chess_ai.get_provider", side_effect=_patched):
        yield


class TestInProcessBenchmark:
    @pytest.mark.asyncio
    async def test_benchmark_emits_live_callbacks(self, tmp_path, patched_provider):
        """The UI callback is invoked for each move and the final state."""
        seen_states: list = []

        async def callback(state):
            seen_states.append(state)

        config = BenchmarkConfig(
            players=["stockfish:depth-4", "stockfish:depth-8"],
            games_per_pairing=1,
            max_parallel_games=1,
            opening_book="startpos",
            temperature=0.0,
            max_tokens=100,
            api_keys={"stockfish": ""},
            output_dir=str(tmp_path / "runs"),
        )

        runner = BenchmarkRunner(config)
        runner.players = {
            "stockfish:depth-4": _make_ai("stockfish:depth-4"),
            "stockfish:depth-8": _make_ai("stockfish:depth-8"),
        }

        await runner.run_benchmark_with_callback(callback)

        assert len(seen_states) >= 4
        final_states = [s for s in seen_states if s.is_game_over]
        assert len(final_states) == 2
        for fs in final_states:
            assert fs.winner is not None
            assert fs.stats.total_moves > 0

    @pytest.mark.asyncio
    async def test_benchmark_persists_jsonl_artifacts(self, tmp_path, patched_provider):
        """Real JSONL files are written and loadable via results_view."""
        async def noop_callback(state):
            pass

        config = BenchmarkConfig(
            players=["stockfish:depth-4", "stockfish:depth-8"],
            games_per_pairing=1,
            max_parallel_games=1,
            opening_book="startpos",
            temperature=0.0,
            max_tokens=100,
            api_keys={"stockfish": ""},
            output_dir=str(tmp_path / "runs"),
        )

        runner = BenchmarkRunner(config)
        runner.players = {
            "stockfish:depth-4": _make_ai("stockfish:depth-4"),
            "stockfish:depth-8": _make_ai("stockfish:depth-8"),
        }

        run_dir = await runner.run_benchmark_with_callback(noop_callback)

        assert (run_dir / "games.jsonl").is_file()
        assert (run_dir / "moves.jsonl").is_file()
        assert (run_dir / "summary.json").is_file()
        assert (run_dir / "games.pgn").is_file()

        run = load_run(run_dir)
        assert run is not None
        assert run.total_games == 2
        assert "stockfish:depth-4" in run.player_stats
        assert "stockfish:depth-8" in run.player_stats

    @pytest.mark.asyncio
    async def test_elo_leaderboard_computed(self, tmp_path, patched_provider):
        """Glicko-2 ratings are computed and attached to the run."""
        async def noop_callback(state):
            pass

        config = BenchmarkConfig(
            players=["stockfish:depth-4", "stockfish:depth-8"],
            games_per_pairing=2,
            max_parallel_games=1,
            opening_book="startpos",
            temperature=0.0,
            max_tokens=100,
            api_keys={"stockfish": ""},
            output_dir=str(tmp_path / "runs"),
        )

        runner = BenchmarkRunner(config)
        runner.players = {
            "stockfish:depth-4": _make_ai("stockfish:depth-4"),
            "stockfish:depth-8": _make_ai("stockfish:depth-8"),
        }

        await runner.run_benchmark_with_callback(noop_callback)

        run = load_run(runner.run_dir)
        assert run is not None

        # The internal ELO calculator should have ratings accessible via get_rating
        assert runner.elo.get_rating("stockfish:depth-4") is not None
        assert runner.elo.get_rating("stockfish:depth-8") is not None

        leaderboard = runner.elo.leaderboard()
        assert len(leaderboard) == 2
        for row in leaderboard:
            assert row["rating"] is not None
            assert row["deviation"] is not None
            assert row["ci_low"] < row["rating"] < row["ci_high"]

    @pytest.mark.asyncio
    async def test_callback_receives_completion_result(self, tmp_path, patched_provider):
        """The GameState passed to callback includes last_completion_result."""
        seen: list = []

        async def callback(state):
            seen.append(state)

        config = BenchmarkConfig(
            players=["stockfish:depth-4", "stockfish:depth-8"],
            games_per_pairing=1,
            max_parallel_games=1,
            opening_book="startpos",
            temperature=0.0,
            max_tokens=100,
            api_keys={"stockfish": ""},
            output_dir=str(tmp_path / "runs"),
        )

        runner = BenchmarkRunner(config)
        runner.players = {
            "stockfish:depth-4": _make_ai("stockfish:depth-4"),
            "stockfish:depth-8": _make_ai("stockfish:depth-8"),
        }

        await runner.run_benchmark_with_callback(callback)

        states_with_cr = [s for s in seen if s.last_completion_result is not None]
        assert len(states_with_cr) > 0
        # At least one state should have the full raw_response (proves the
        # callback plumbing works); fallback states with raw_response=None are
        # possible in edge cases but shouldn't dominate.
        states_with_raw = [
            s for s in states_with_cr if s.last_completion_result.raw_response is not None
        ]
        assert len(states_with_raw) > 0, "No state had raw_response populated"
        for s in states_with_raw:
            cr = s.last_completion_result
            assert cr.text is not None
            assert cr.latency_ms is not None
            assert cr.latency_ms >= 0
            assert cr.raw_response["provider"] == "stockfish"
