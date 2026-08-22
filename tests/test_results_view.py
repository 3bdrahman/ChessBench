"""Real tests for the benchmark results reader.

All assertions exercise real JSONL / summary.json files written to a real
temporary directory by the test itself — no `MagicMock` patching of
:mod:`chessbench.benchmark.results_view`. The fixtures are written in the
exact format the :class:`BenchmarkLogger` produces, so the reader is being
exercised the same way it will be in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chessbench.benchmark.logging import BenchmarkLogger
from chessbench.benchmark.results_view import (
    aggregate_leaderboard,
    list_run_dirs,
    list_runs,
    load_run,
)


def _simulate_run(run_dir: Path, *, white_wins: int, black_wins: int, draws: int) -> None:
    """Drive the real BenchmarkLogger end-to-end to produce a real run on disk."""
    logger = BenchmarkLogger(str(run_dir))
    logger.start_run({"games_per_pairing": white_wins + black_wins + draws, "test": True})

    games = [
        ("1-0", 1.0, white_wins),
        ("0-1", 0.0, black_wins),
        ("1/2-1/2", 0.5, draws),
    ]
    move_idx = 0

    for result, numeric, count in games:
        for _ in range(count):
            logger.start_game(
                white_player="openai:gpt-4o-mini",
                black_player="groq:llama-3.3-70b",
                white_provider="openai",
                black_provider="groq",
                opening_eco="C50",
                opening_name="Italian Game",
                opening_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            )
            import chess
            board = chess.Board()
            for ply in range(4):
                spec = "openai:gpt-4o-mini" if ply % 2 == 0 else "groq:llama-3.3-70b"
                move = list(board.legal_moves)[(move_idx + ply) % len(list(board.legal_moves))]
                uci = move.uci()
                san = board.san(move)
                logger.log_move(
                    move_number=ply + 1,
                    player=spec,
                    color="white" if ply % 2 == 0 else "black",
                    fen_before=board.fen(),
                    move_uci=uci,
                    move_san=san,
                    llm_latency_ms=150 + ply,
                    llm_tokens_in=100 * (ply + 1),
                    llm_tokens_out=8,
                    llm_raw_response=uci,
                    thinking_trace=None,
                    prompt_hash="hash" + str(ply),
                    validation_retries=0,
                )
                if ply == 0 and (result == "1-0"):
                    pass  # White captured nothing here deliberately
                board.push(move)
            logger.end_game(result, numeric, total_moves=4, game_duration_sec=1.5)
    logger.write_summary()


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """Build a fake ``runs/`` tree with three runs of varying content."""
    root = tmp_path / "runs"
    root.mkdir()
    _simulate_run(root / "20260807_120000", white_wins=2, black_wins=1, draws=0)
    _simulate_run(root / "20260807_130000", white_wins=0, black_wins=0, draws=3)
    # An empty run dir should be silently skipped — never fabricated.
    (root / "20260807_140000_empty").mkdir()
    return root


class TestListRunDirs:
    def test_lists_only_run_directories_newest_first(self, runs_root: Path):
        dirs = list_run_dirs(runs_root)
        assert [d.name for d in dirs] == [
            "20260807_140000_empty",
            "20260807_130000",
            "20260807_120000",
        ]

    def test_returns_empty_list_when_missing(self, tmp_path: Path):
        assert list_run_dirs(tmp_path / "nope") == []


class TestLoadRun:
    def test_loads_real_run_summary(self, runs_root: Path):
        run = load_run(runs_root / "20260807_120000")
        assert run is not None
        assert run.run_id == "20260807_120000"
        assert run.total_games == 3
        # Real per-player stats from real moves
        assert "openai:gpt-4o-mini" in run.player_stats
        assert "groq:llama-3.3-70b" in run.player_stats
        white_ps = run.player_stats["openai:gpt-4o-mini"]
        assert white_ps.wins == 2
        assert white_ps.losses == 1
        assert white_ps.games_played == 3

    def test_load_run_returns_none_for_empty(self, runs_root: Path):
        assert load_run(runs_root / "20260807_140000_empty") is None

    def test_load_run_returns_none_for_missing_dir(self, tmp_path: Path):
        assert load_run(tmp_path / "totally_missing") is None

    def test_pairings_aggregate_real_wins(self, runs_root: Path):
        run = load_run(runs_root / "20260807_120000")
        assert run is not None
        assert len(run.pairings) == 1
        p = run.pairings[0]
        assert p.white == "openai:gpt-4o-mini"
        assert p.black == "groq:llama-3.3-70b"
        assert p.games == 3
        assert p.white_wins == 2
        assert p.black_wins == 1
        assert p.draws == 0


class TestListRuns:
    def test_skips_empty_runs(self, runs_root: Path):
        runs = list_runs(runs_root)
        assert [r.run_id for r in runs] == ["20260807_130000", "20260807_120000"]

    def test_providers_seen_extracted_from_specs(self, runs_root: Path):
        runs = list_runs(runs_root)
        assert runs
        for run in runs:
            assert "openai" in run.providers_seen
            assert "groq" in run.providers_seen


class TestAggregateLeaderboard:
    def test_aggregates_across_runs(self, runs_root: Path):
        runs = list_runs(runs_root)
        rows = aggregate_leaderboard(runs)
        names = {r.player: r for r in rows}
        # Two runs total, 3 games each → 6 per player
        assert names["openai:gpt-4o-mini"].games == 6
        assert names["groq:llama-3.3-70b"].games == 6
        # Run 1: W=2, L=1, D=0. Run 2: W=0, L=0, D=3.
        assert names["openai:gpt-4o-mini"].wins == 2
        assert names["openai:gpt-4o-mini"].losses == 1
        assert names["openai:gpt-4o-mini"].draws == 3

    def test_score_pct_higher_for_stronger_side(self, runs_root: Path):
        runs = list_runs(runs_root)
        rows = {r.player: r for r in aggregate_leaderboard(runs)}
        white_score = rows["openai:gpt-4o-mini"].score_pct
        black_score = rows["groq:llama-3.3-70b"].score_pct
        # 7/12 vs 5/12 — strictly above
        assert white_score is not None and black_score is not None
        assert white_score > black_score


class TestRunSummarySerialization:
    def test_to_dict_round_trips_player_stats(self, runs_root: Path):
        run = load_run(runs_root / "20260807_120000")
        assert run is not None
        d = run.to_dict()
        assert "player_stats" in d
        assert "openai:gpt-4o-mini" in d["player_stats"]
        ps = d["player_stats"]["openai:gpt-4o-mini"]
        assert ps["wins"] == 2
        # Each player makes 2 moves per game × 3 games = 6 latency samples
        assert ps["latency_samples"] == 6


class TestRealRunsOnDiskReal:
    """If the project shipped real runs under ``runs/``, exercise them too."""
    def test_runs_directory_in_repo_loads_cleanly(self):
        repo_runs = Path("runs")
        if not repo_runs.is_dir():
            pytest.skip("no runs/ directory present")
        runs = list_runs(repo_runs)
        for run in runs:
            assert run.run_id
            assert run.total_games >= 0
            # No fabricated stats: every player name has a real provider prefix.
            for name in run.player_stats:
                assert ":" in name, f"player {name!r} has no provider prefix"
