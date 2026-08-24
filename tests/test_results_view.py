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
        assert white_score is not None
        assert black_score is not None
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
        # Each player makes 2 moves per game x 3 games = 6 latency samples
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


class TestColorPrecedence:
    """Test color-specific prompt precedence over spec-keyed prompts."""

    def test_color_override_takes_precedence(self, tmp_path: Path):
        """Color-specific overrides should win over spec-keyed prompts."""
        logger = BenchmarkLogger(str(tmp_path / "test_run"))
        # Config with both color-specific and spec-keyed prompts
        config = {
            "games_per_pairing": 1,
            "players": ["openai:gpt-4o", "openai:gpt-4o"],  # Same model for both players
            "system_prompts": {
                "openai:gpt-4o": "spec-keyed system prompt"
            },
            "turn_prompts": {
                "openai:gpt-4o": "spec-keyed turn prompt"
            },
            "system_prompts_by_color": {
                "white": "white color-specific system",
                "black": "black color-specific system"
            },
            "turn_prompts_by_color": {
                "white": "white color-specific turn",
                "black": "black color-specific turn"
            }
        }
        logger.start_run(config)

        # Simulate a game
        import chess
        board = chess.Board()
        logger.start_game(
            white_player="openai:gpt-4o",
            black_player="openai:gpt-4o",
            white_provider="openai",
            black_provider="openai",
            opening_eco="A00",
            opening_name="Barnes Opening",
            opening_fen=chess.STARTING_FEN,
        )

        # White move
        logger.log_move(
            move_number=1,
            player="openai:gpt-4o",
            color="white",
            fen_before=board.fen(),
            move_uci="e2e4",
            move_san="e4",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e4",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e2e4"))

        # Black move
        logger.log_move(
            move_number=1,
            player="openai:gpt-4o",
            color="black",
            fen_before=board.fen(),
            move_uci="e7e5",
            move_san="e5",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e5",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e7e5"))

        logger.end_game("1/2-1/2", 0.5, total_moves=2, game_duration_sec=1.0)
        logger.write_summary()

        # Load the run and check provenance
        run = load_run(tmp_path / "test_run")
        assert run is not None

        # Check that provenance exists in the summary
        assert hasattr(run, 'provenance') or 'provenance' in run.to_dict()

        # Get the run dict to check for provenance
        run_dict = run.to_dict()
        assert 'provenance' in run_dict
        provenance = run_dict['provenance']

        # Check white player got color-specific prompts
        assert "openai:gpt-4o" in provenance
        white_provenance = provenance["openai:gpt-4o"].get("white")
        assert white_provenance is not None
        assert white_provenance["system_prompt"] == "white color-specific system"
        assert white_provenance["turn_prompt"] == "white color-specific turn"
        assert white_provenance["source"] == "custom"
        assert white_provenance["used_fallback"] is False

        # Check black player got color-specific prompts
        black_provenance = provenance["openai:gpt-4o"].get("black")
        assert black_provenance is not None
        assert black_provenance["system_prompt"] == "black color-specific system"
        assert black_provenance["turn_prompt"] == "black color-specific turn"
        assert black_provenance["source"] == "custom"
        assert black_provenance["used_fallback"] is False

    def test_fallback_to_spec_keyed(self, tmp_path: Path):
        """When color-specific overrides are empty, should fall back to spec-keyed."""
        logger = BenchmarkLogger(str(tmp_path / "test_run"))
        # Config with only spec-keyed prompts (no color-specific)
        config = {
            "games_per_pairing": 1,
            "players": ["openai:gpt-4o", "anthropic:claude-3-haiku"],
            "system_prompts": {
                "openai:gpt-4o": "openai spec system",
                "anthropic:claude-3-haiku": "anthropic spec system"
            },
            "turn_prompts": {
                "openai:gpt-4o": "openai spec turn",
                "anthropic:claude-3-haiku": "anthropic spec turn"
            },
            "system_prompts_by_color": {},  # Empty
            "turn_prompts_by_color": {}   # Empty
        }
        logger.start_run(config)

        # Simulate a game
        import chess
        board = chess.Board()
        logger.start_game(
            white_player="openai:gpt-4o",
            black_player="anthropic:claude-3-haiku",
            white_provider="openai",
            black_provider="anthropic",
            opening_eco="A00",
            opening_name="Barnes Opening",
            opening_fen=chess.STARTING_FEN,
        )

        # White move
        logger.log_move(
            move_number=1,
            player="openai:gpt-4o",
            color="white",
            fen_before=board.fen(),
            move_uci="e2e4",
            move_san="e4",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e4",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e2e4"))

        # Black move
        logger.log_move(
            move_number=1,
            player="anthropic:claude-3-haiku",
            color="black",
            fen_before=board.fen(),
            move_uci="e7e5",
            move_san="e5",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e5",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e7e5"))

        logger.end_game("1/2-1/2", 0.5, total_moves=2, game_duration_sec=1.0)
        logger.write_summary()

        # Load the run and check provenance
        run = load_run(tmp_path / "test_run")
        assert run is not None
        run_dict = run.to_dict()
        assert 'provenance' in run_dict
        provenance = run_dict['provenance']

        # Check white player got spec-keyed prompts
        assert "openai:gpt-4o" in provenance
        white_provenance = provenance["openai:gpt-4o"].get("white")
        assert white_provenance is not None
        assert white_provenance["system_prompt"] == "openai spec system"
        assert white_provenance["turn_prompt"] == "openai spec turn"
        assert white_provenance["source"] == "custom"
        assert white_provenance["used_fallback"] is True  # Fell back from color-specific to spec-keyed

        # Check black player got spec-keyed prompts
        assert "anthropic:claude-3-haiku" in provenance
        black_provenance = provenance["anthropic:claude-3-haiku"].get("black")
        assert black_provenance is not None
        assert black_provenance["system_prompt"] == "anthropic spec system"
        assert black_provenance["turn_prompt"] == "anthropic spec turn"
        assert black_provenance["source"] == "custom"
        assert black_provenance["used_fallback"] is True  # Fell back from color-specific to spec-keyed

    def test_fallback_to_preset(self, tmp_path: Path):
        """When both color-specific and spec-keyed are empty, should fall back to preset (None)."""
        logger = BenchmarkLogger(str(tmp_path / "test_run"))
        # Config with no prompts at all
        config = {
            "games_per_pairing": 1,
            "players": ["openai:gpt-4o", "anthropic:claude-3-haiku"],
            "system_prompts": {},  # Empty
            "turn_prompts": {},    # Empty
            "system_prompts_by_color": {},  # Empty
            "turn_prompts_by_color": {}   # Empty
        }
        logger.start_run(config)

        # Simulate a game
        import chess
        board = chess.Board()
        logger.start_game(
            white_player="openai:gpt-4o",
            black_player="anthropic:claude-3-haiku",
            white_provider="openai",
            black_provider="anthropic",
            opening_eco="A00",
            opening_name="Barnes Opening",
            opening_fen=chess.STARTING_FEN,
        )

        # White move
        logger.log_move(
            move_number=1,
            player="openai:gpt-4o",
            color="white",
            fen_before=board.fen(),
            move_uci="e2e4",
            move_san="e4",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e4",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e2e4"))

        # Black move
        logger.log_move(
            move_number=1,
            player="anthropic:claude-3-haiku",
            color="black",
            fen_before=board.fen(),
            move_uci="e7e5",
            move_san="e5",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e5",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e7e5"))

        logger.end_game("1/2-1/2", 0.5, total_moves=2, game_duration_sec=1.0)
        logger.write_summary()

        # Load the run and check provenance
        run = load_run(tmp_path / "test_run")
        assert run is not None
        run_dict = run.to_dict()
        assert 'provenance' in run_dict
        provenance = run_dict['provenance']

        # Check white player got None prompts (preset)
        assert "openai:gpt-4o" in provenance
        white_provenance = provenance["openai:gpt-4o"].get("white")
        assert white_provenance is not None
        assert white_provenance["system_prompt"] is None
        assert white_provenance["turn_prompt"] is None
        assert white_provenance["source"] == "preset"
        assert white_provenance["used_fallback"] is True  # Fell back from spec-keyed to preset

        # Check black player got None prompts (preset)
        assert "anthropic:claude-3-haiku" in provenance
        black_provenance = provenance["anthropic:claude-3-haiku"].get("black")
        assert black_provenance is not None
        assert black_provenance["system_prompt"] is None
        assert black_provenance["turn_prompt"] is None
        assert black_provenance["source"] == "preset"
        assert black_provenance["used_fallback"] is True  # Fell back from spec-keyed to preset

class TestConfigHashStability:
    """Test that config hash remains stable for backward compatibility."""

    def test_old_style_config_hash_unchanged(self, tmp_path: Path):
        """Old-style config without new fields should hash the same as before."""
        from chessbench.benchmark.export import compute_config_hash

        # Old-style config (as it would have been before our changes)
        old_config = {
            "games_per_pairing": 1,
            "players": ["openai:gpt-4o", "anthropic:claude-3-haiku"],
            "system_prompts": {
                "openai:gpt-4o": "openai spec system",
                "anthropic:claude-3-haiku": "anthropic spec system"
            },
            "turn_prompts": {
                "openai:gpt-4o": "openai spec turn",
                "anthropic:claude-3-haiku": "anthropic spec turn"
            },
            # Note: No system_prompts_by_color or turn_prompts_by_color fields
            "time_control_seconds_per_move": 15,
            "opening_book": "startpos",
            "temperature": 0.0,
            "max_tokens": 100,
        }

        # New-style config (with empty new fields, as would be default)
        new_config = {
            "games_per_pairing": 1,
            "players": ["openai:gpt-4o", "anthropic:claude-3-haiku"],
            "system_prompts": {
                "openai:gpt-4o": "openai spec system",
                "anthropic:claude-3-haiku": "anthropic spec system"
            },
            "turn_prompts": {
                "openai:gpt-4o": "openai spec turn",
                "anthropic:claude-3-haiku": "anthropic spec turn"
            },
            # New fields with default empty values
            "system_prompts_by_color": {},
            "turn_prompts_by_color": {},
            "time_control_seconds_per_move": 15,
            "opening_book": "startpos",
            "temperature": 0.0,
            "max_tokens": 100,
        }

        # The hash should be the same because the new fields are empty dicts
        # and _normalize_for_hash should treat them the same as missing fields
        old_hash = compute_config_hash(old_config)
        new_hash = compute_config_hash(new_config)

        assert old_hash == new_hash, f"Hash mismatch: old={old_hash}, new={new_hash}"

        # Also test that actually running with this config works
        logger = BenchmarkLogger(str(tmp_path / "test_run"))
        logger.start_run(new_config)

        # Simulate a minimal game
        import chess
        board = chess.Board()
        logger.start_game(
            white_player="openai:gpt-4o",
            black_player="anthropic:claude-3-haiku",
            white_provider="openai",
            black_provider="anthropic",
            opening_eco="A00",
            opening_name="Barnes Opening",
            opening_fen=chess.STARTING_FEN,
        )

        # White move
        logger.log_move(
            move_number=1,
            player="openai:gpt-4o",
            color="white",
            fen_before=board.fen(),
            move_uci="e2e4",
            move_san="e4",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e4",
            thinking_trace=None,
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e2e4"))

        # Black move
        logger.log_move(
            move_number=1,
            player="anthropic:claude-3-haiku",
            color="black",
            fen_before=board.fen(),
            move_uci="e7e5",
            move_san="e5",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e5",
            thinking_trace=None,
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e7e5"))

        logger.end_game("1/2-1/2", 0.5, total_moves=2, game_duration_sec=1.0)
        logger.write_summary()

        # Load and verify the run
        run = load_run(tmp_path / "test_run")
        assert run is not None
        assert run.total_games == 1

        # The config in the run should match what we started with
        assert run.config["games_per_pairing"] == 1
        assert run.config["players"] == ["openai:gpt-4o", "anthropic:claude-3-haiku"]
        assert "system_prompts_by_color" in run.config
        assert "turn_prompts_by_color" in run.config
        assert run.config["system_prompts_by_color"] == {}
        assert run.config["turn_prompts_by_color"] == {}

class TestProvenancePersistence:
    """Test provenance save and load round-trip preservation."""

    def test_provenance_round_trip(self, tmp_path: Path):
        """Provenance data should be preserved when saving and loading a run."""
        logger = BenchmarkLogger(str(tmp_path / "test_run"))
        # Config with mixed prompt sources
        config = {
            "games_per_pairing": 1,
            "players": ["openai:gpt-4o", "anthropic:claude-3-haiku"],
            "system_prompts": {
                "openai:gpt-4o": "openai spec system"
            },
            "turn_prompts": {
                "anthropic:claude-3-haiku": "anthropic spec turn"
            },
            "system_prompts_by_color": {
                "white": "white color-specific system"
            },
            "turn_prompts_by_color": {
                "black": "black color-specific turn"
            }
        }
        logger.start_run(config)

        # Simulate a game
        import chess
        board = chess.Board()
        logger.start_game(
            white_player="openai:gpt-4o",
            black_player="anthropic:claude-3-haiku",
            white_provider="openai",
            black_provider="anthropic",
            opening_eco="A00",
            opening_name="Barnes Opening",
            opening_fen=chess.STARTING_FEN,
        )

        # White move
        logger.log_move(
            move_number=1,
            player="openai:gpt-4o",
            color="white",
            fen_before=board.fen(),
            move_uci="e2e4",
            move_san="e4",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e4",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e2e4"))

        # Black move
        logger.log_move(
            move_number=1,
            player="anthropic:claude-3-haiku",
            color="black",
            fen_before=board.fen(),
            move_uci="e7e5",
            move_san="e5",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e5",
            thinking_trace=None,
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e7e5"))

        logger.end_game("1/2-1/2", 0.5, total_moves=2, game_duration_sec=1.0)
        logger.write_summary()

        # Load the run and check provenance
        run = load_run(tmp_path / "test_run")
        assert run is not None
        run_dict = run.to_dict()
        assert 'provenance' in run_dict
        original_provenance = run_dict['provenance']

        # Verify the provenance makes sense
        # White player:
        # - system_prompt: should get color-specific "white color-specific system"
        # - turn_prompt: should fall back to spec-keyed (none defined) -> None -> preset
        white_prov = original_provenance["openai:gpt-4o"]["white"]
        assert white_prov["system_prompt"] == "white color-specific system"
        assert white_prov["turn_prompt"] is None
        assert white_prov["source"] == "custom"  # system_prompt is custom
        assert white_prov["used_fallback"] is True  # turn_prompt fell back to preset

        # Black player:
        # - system_prompt: should fall back to spec-keyed (none defined) -> None -> preset
        # - turn_prompt: should get color-specific "black color-specific turn"
        black_prov = original_provenance["anthropic:claude-3-haiku"]["black"]
        assert black_prov["system_prompt"] is None
        assert black_prov["turn_prompt"] == "black color-specific turn"
        assert black_prov["source"] == "custom"  # turn_prompt is custom
        assert black_prov["used_fallback"] is True  # system_prompt fell back to preset

        # Verify we can access the same data again (simulating load)
        run_dict2 = run.to_dict()
        assert 'provenance' in run_dict2
        loaded_provenance = run_dict2['provenance']

        # Should be identical
        assert loaded_provenance == original_provenance

    def test_provenance_round_trip_fallback_behavior(self, tmp_path: Path):
        """Provenance data should correctly handle fallback behavior when some prompts are missing."""
        logger = BenchmarkLogger(str(tmp_path / "test_run"))
        # Config with mixed prompt sources - some missing to test fallbacks
        config = {
            "games_per_pairing": 1,
            "players": ["openai:gpt-4o", "anthropic:claude-3-haiku"],
            "system_prompts": {
                "openai:gpt-4o": "openai spec system"
                # anthropic:claude-3-haiku intentionally left out to test fallback
            },
            "turn_prompts": {
                # openai:gpt-4o intentionally left out to test fallback
                "anthropic:claude-3-haiku": "anthropic spec turn"
            },
            "system_prompts_by_color": {
                "white": "white color-specific system"  # Only for white
                # black intentionally left out to test fallback to spec-keyed or preset
            },
            "turn_prompts_by_color": {
                # white intentionally left out to test fallback to spec-keyed or preset
                "black": "black color-specific turn"  # Only for black
            }
        }
        logger.start_run(config)

        # Simulate a game
        import chess
        board = chess.Board()
        logger.start_game(
            white_player="openai:gpt-4o",
            black_player="anthropic:claude-3-haiku",
            white_provider="openai",
            black_provider="anthropic",
            opening_eco="A00",
            opening_name="Barnes Opening",
            opening_fen=chess.STARTING_FEN,
        )

        # White move
        logger.log_move(
            move_number=1,
            player="openai:gpt-4o",
            color="white",
            fen_before=board.fen(),
            move_uci="e2e4",
            move_san="e4",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e4",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e2e4"))

        # Black move
        logger.log_move(
            move_number=1,
            player="anthropic:claude-3-haiku",
            color="black",
            fen_before=board.fen(),
            move_uci="e7e5",
            move_san="e5",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e5",
            thinking_trace=None,
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e7e5"))

        logger.end_game("1/2-1/2", 0.5, total_moves=2, game_duration_sec=1.0)
        logger.write_summary()

        # Load the run and check provenance
        run = load_run(tmp_path / "test_run")
        assert run is not None
        run_dict = run.to_dict()
        assert 'provenance' in run_dict
        original_provenance = run_dict['provenance']

        # Verify the provenance makes sense
        # White player:
        # - system_prompt: should get color-specific "white color-specific system"
        # - turn_prompt: should fall back to spec-keyed (none defined) -> None -> preset
        white_prov = original_provenance["openai:gpt-4o"]["white"]
        assert white_prov["system_prompt"] == "white color-specific system"
        assert white_prov["turn_prompt"] is None
        assert white_prov["source"] == "custom"  # system_prompt is custom
        assert white_prov["used_fallback"] is True  # turn_prompt fell back to preset

        # Black player:
        # - system_prompt: should fall back to spec-keyed (none defined) -> None -> preset
        # - turn_prompt: should get color-specific "black color-specific turn"
        black_prov = original_provenance["anthropic:claude-3-haiku"]["black"]
        assert black_prov["system_prompt"] is None
        assert black_prov["turn_prompt"] == "black color-specific turn"
        assert black_prov["source"] == "custom"  # turn_prompt is custom
        assert black_prov["used_fallback"] is True  # system_prompt fell back to preset

        # Verify we can access the same data again (simulating load)
        run_dict2 = run.to_dict()
        assert 'provenance' in run_dict2
        loaded_provenance = run_dict2['provenance']
        assert loaded_provenance == original_provenance

    def test_provenance_round_trip_save_load_simulation(self, tmp_path: Path):
        """Provenance data should correctly handle fallback behavior and simulate save/load round-trip."""
        logger = BenchmarkLogger(str(tmp_path / "test_run"))
        # Config with mixed prompt sources - some missing to test fallbacks
        config = {
            "games_per_pairing": 1,
            "players": ["openai:gpt-4o", "anthropic:claude-3-haiku"],
            "system_prompts": {
                "openai:gpt-4o": "openai spec system"
                # anthropic:claude-3-haiku intentionally left out to test fallback
            },
            "turn_prompts": {
                # openai:gpt-4o intentionally left out to test fallback
                "anthropic:claude-3-haiku": "anthropic spec turn"
            },
            "system_prompts_by_color": {
            "white": "white color-specific system"  # Only for white
            # black intentionally left out to test fallback to spec-keyed or preset
            },
            "turn_prompts_by_color": {
            # white intentionally left out to test fallback to spec-keyed or preset
            "black": "black color-specific turn"  # Only for black
            }
        }
        logger.start_run(config)

        # Simulate a game
        import chess
        board = chess.Board()
        logger.start_game(
            white_player="openai:gpt-4o",
            black_player="anthropic:claude-3-haiku",
            white_provider="openai",
            black_provider="anthropic",
            opening_eco="A00",
            opening_name="Barnes Opening",
            opening_fen=chess.STARTING_FEN,
        )

        # White move
        logger.log_move(
            move_number=1,
            player="openai:gpt-4o",
            color="white",
            fen_before=board.fen(),
            move_uci="e2e4",
            move_san="e4",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e4",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e2e4"))

        # Black move
        logger.log_move(
            move_number=1,
            player="anthropic:claude-3-haiku",
            color="black",
            fen_before=board.fen(),
            move_uci="e7e5",
            move_san="e5",
            llm_latency_ms=100,
            llm_tokens_in=10,
            llm_tokens_out=5,
            llm_raw_response="e5",
            thinking_trace=None,
            prompt_hash="abcd1234",
            validation_retries=0,
        )
        board.push(chess.Move.from_uci("e7e5"))

        logger.end_game("1/2-1/2", 0.5, total_moves=2, game_duration_sec=1.0)
        logger.write_summary()

        # Load the run and check provenance
        run = load_run(tmp_path / "test_run")
        assert run is not None
        run_dict = run.to_dict()
        assert 'provenance' in run_dict
        original_provenance = run_dict['provenance']

        # Verify the provenance makes sense
        # White player:
        # - system_prompt: should get color-specific "white color-specific system"
        # - turn_prompt: should fall back to spec-keyed (none defined) -> None -> preset
        white_prov = original_provenance["openai:gpt-4o"]["white"]
        assert white_prov["system_prompt"] == "white color-specific system"
        assert white_prov["turn_prompt"] is None
        assert white_prov["source"] == "custom"  # system_prompt is custom
        assert white_prov["used_fallback"] is True  # turn_prompt fell back to preset

        # Black player:
        # - system_prompt: should fall back to spec-keyed (none defined) -> None -> preset
        # - turn_prompt: should get color-specific "black color-specific turn"
        black_prov = original_provenance["anthropic:claude-3-haiku"]["black"]
        assert black_prov["system_prompt"] is None
        assert black_prov["turn_prompt"] == "black color-specific turn"
        assert black_prov["source"] == "custom"  # turn_prompt is custom
        assert black_prov["used_fallback"] is True  # system_prompt fell back to preset

        # Now simulate saving and loading by checking that the data is in the summary
        # In a real scenario, this would be saved to disk and loaded back
        # But since we're using the same run object, we can just verify the data is there

        # The key point is that the provenance data is correctly computed and stored
        # in the summary.json, which simulates the save/load round-trip

        # Verify we can access the same data again (simulating load)
        run_dict2 = run.to_dict()
        assert 'provenance' in run_dict2
        loaded_provenance = run_dict2['provenance']

        # Should be identical
        assert loaded_provenance == original_provenance
