"""Export formats for benchmark results (Parquet, CSV, PGN+eval)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import pandas as pd

from chessbench.benchmark.results_view import GameRecord, PairingResult, load_run
from chessbench.benchmark.results_view import PlayerStats as ViewPlayerStats


def _sanitize_for_csv(value: str | None) -> str | None:
    """Sanitize string to prevent CSV formula injection.

    Prefixes formula-starting characters with an apostrophe to prevent
    spreadsheet programs from interpreting them as formulas.
    """
    if value is None:
        return None
    s = str(value)
    if not s:
        return s
    # Characters that trigger formula interpretation in Excel/LibreOffice/Google Sheets
    if s[0] in ('=', '+', '-', '@', '\t', '\r', '\n'):
        return "'" + s
    return s


@dataclass
class ExportConfig:
    """Configuration for export."""
    include_pgn_eval: bool = True
    include_csv: bool = True
    include_parquet: bool = True
    compression: str = "snappy"  # for parquet


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute SHA256 hash of benchmark configuration."""
    # Normalize config for consistent hashing
    normalized = _normalize_for_hash(config)
    serialized = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _normalize_for_hash(obj: Any) -> Any:
    """Normalize object for consistent hashing."""
    if isinstance(obj, dict):
        # Skip empty dicts for the new prompt override fields to preserve hash compatibility
        items = []
        for k, v in obj.items():
            if k in ('system_prompts_by_color', 'turn_prompts_by_color') and isinstance(v, dict) and len(v) == 0:
                continue
            items.append((k, v))
        return {k: _normalize_for_hash(v) for k, v in sorted(items)}
    elif isinstance(obj, list):
        return [_normalize_for_hash(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def compute_git_hash() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=Path.cwd()
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def compute_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def compute_dependency_versions() -> dict[str, str]:
    """Get versions of key dependencies."""
    deps: dict[str, str] = {}
    for pkg in ['chess', 'pandas', 'numpy', 'pyarrow']:
        try:
            __import__(pkg)
            deps[pkg] = sys.modules[pkg].__version__
        except (ImportError, AttributeError):
            deps[pkg] = "unknown"
    return deps


def _compute_player_provenance(config: dict[str, Any], run_summary: Any) -> dict[str, Any]:
    """Compute per-player prompt provenance from config and run summary.

    Args:
        config: The benchmark configuration dict
        run_summary: The run summary containing game data

    Returns:
        Dict mapping player specs to their provenance info by color
    """
    provenance: dict[str, dict[str, dict[str, Any]]] = {}

    # Get the set of player specs from the config
    player_specs = set(config.get('players', []))

    # For each player, determine what colors they played as
    player_colors: dict[str, set[str]] = {spec: set() for spec in player_specs}
    for game in run_summary.games:
        white_spec = game.white_player
        black_spec = game.black_player
        if white_spec in player_colors:
            player_colors[white_spec].add('white')
        if black_spec in player_colors:
            player_colors[black_spec].add('black')

    # Compute provenance for each player
    for spec in player_specs:
        provenance[spec] = {}

        # Check what colors this player actually played as
        for color in player_colors[spec]:
            # Determine prompts with precedence: color-specific > spec-keyed > None
            system_prompt = None
            turn_prompt = None
            source_system = None
            source_turn = None

            # Try color-specific override first
            color_key = color.lower()
            if color_key in config.get('system_prompts_by_color', {}):
                system_prompt = config['system_prompts_by_color'][color_key]
                source_system = f"color-specific:{color_key}"
            if color_key in config.get('turn_prompts_by_color', {}):
                turn_prompt = config['turn_prompts_by_color'][color_key]
                source_turn = f"color-specific:{color_key}"

            # Fall back to spec-keyed prompts
            if system_prompt is None:
                system_prompt = config.get('system_prompts', {}).get(spec)
                if system_prompt is not None:
                    source_system = "spec-keyed"
            if turn_prompt is None:
                turn_prompt = config.get('turn_prompts', {}).get(spec)
                if turn_prompt is not None:
                    source_turn = "spec-keyed"

            # Determine if fallback occurred
            used_fallback = False
            # Check if we wanted color-specific but fell back to spec-keyed or None
            # Check if we wanted spec-keyed but fell back to None

            # For system prompt
            wanted_color_specific = color_key in config.get('system_prompts_by_color', {})
            wanted_spec_keyed = spec in config.get('system_prompts', {})
            got_color_specific = source_system and source_system.startswith("color-specific")
            got_spec_keyed = source_system == "spec-keyed"

            if (wanted_color_specific and not got_color_specific) or \
               (wanted_spec_keyed and not got_spec_keyed and system_prompt is None):
                used_fallback = True

            # For turn prompt
            wanted_color_specific_turn = color_key in config.get('turn_prompts_by_color', {})
            wanted_spec_keyed_turn = spec in config.get('turn_prompts', {})
            got_color_specific_turn = source_turn and source_turn.startswith("color-specific")
            got_spec_keyed_turn = source_turn == "spec-keyed"

            if (wanted_color_specific_turn and not got_color_specific_turn) or \
               (wanted_spec_keyed_turn and not got_spec_keyed_turn and turn_prompt is None):
                used_fallback = True

            # Compute strategy hash
            strategy_hash = ""
            if system_prompt is not None or turn_prompt is not None:
                # Use null byte as separator as specified in requirements
                system_part = system_prompt if system_prompt is not None else ""
                turn_part = turn_prompt if turn_prompt is not None else ""
                strategy_string = f"{system_part}\x00{turn_part}"
                strategy_hash = hashlib.sha256(strategy_string.encode()).hexdigest()[:16]

            # Determine source (simplified - we don't have access to preset versions)
            # For now, we'll say "custom" if we got a non-None prompt from config, else "preset"
            source = "custom"
            if system_prompt is None and turn_prompt is None:
                source = "preset"

            provenance[spec][color] = {
                "system_prompt": system_prompt,
                "turn_prompt": turn_prompt,
                "strategy_hash": strategy_hash,
                "source": source,
                "used_fallback": used_fallback,
            }

    return provenance


def create_reproducibility_metadata(config: dict[str, Any], run_summary: Any = None) -> dict[str, Any]:
    """Create comprehensive reproducibility metadata.

    Args:
        config: The benchmark configuration dict
        run_summary: Optional run summary to compute per-player provenance
    """
    metadata = {
        "config_hash": compute_config_hash(config),
        "config": config,
        "git_commit": compute_git_hash(),
        "python_version": compute_python_version(),
        "dependencies": compute_dependency_versions(),
        "timestamp_utc": datetime.now(UTC).isoformat() + 'Z',
    }

    # Add per-player provenance if run summary is provided
    if run_summary is not None:
        metadata["provenance"] = _compute_player_provenance(config, run_summary)

    return metadata


def export_parquet(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Export benchmark run to Parquet format."""
    run_dir = Path(run_dir)
    output_path = run_dir / "export.parquet" if output_path is None else Path(output_path)

    # Load run data
    run_summary = load_run(run_dir)
    if run_summary is None:
        raise ValueError(f"No valid run data found in {run_dir}")

    # Convert to DataFrames
    games_df = _games_to_dataframe(run_summary.games)
    moves_df = _moves_to_dataframe(run_summary.games)
    players_df = _players_to_dataframe(run_summary.player_stats)
    pairings_df = _pairings_to_dataframe(run_summary.pairings)

    # Write parquet files
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write multiple tables as separate parquet files
    base = output_path.with_suffix('')
    games_df.to_parquet(f"{base}_games.parquet", compression='snappy')
    moves_df.to_parquet(f"{base}_moves.parquet", compression='snappy')
    players_df.to_parquet(f"{base}_players.parquet", compression='snappy')
    pairings_df.to_parquet(f"{base}_pairings.parquet", compression='snappy')

    # Write metadata
    metadata = create_reproducibility_metadata(run_summary.config, run_summary)
    with open(f"{base}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_path


def export_csv(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Export benchmark run to CSV format."""
    run_dir = Path(run_dir)
    output_dir = run_dir / "export_csv" if output_dir is None else Path(output_dir)

    run_summary = load_run(run_dir)
    if run_summary is None:
        raise ValueError(f"No valid run data found in {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    games_df = _games_to_dataframe(run_summary.games)
    moves_df = _moves_to_dataframe(run_summary.games)
    players_df = _players_to_dataframe(run_summary.player_stats)
    pairings_df = _pairings_to_dataframe(run_summary.pairings)

    games_df.to_csv(output_dir / "games.csv", index=False)
    moves_df.to_csv(output_dir / "moves.csv", index=False)
    players_df.to_csv(output_dir / "players.csv", index=False)
    pairings_df.to_csv(output_dir / "pairings.csv", index=False)

    # Write metadata
    metadata = create_reproducibility_metadata(run_summary.config, run_summary)
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def export_pgn_with_eval(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Export games as PGN with Stockfish evaluation annotations."""
    run_dir = Path(run_dir)
    output_path = run_dir / "games_with_eval.pgn" if output_path is None else Path(output_path)

    run_summary = load_run(run_dir)
    if run_summary is None:
        raise ValueError(f"No valid run data found in {run_dir}")

    pgn_lines = []

    for game in run_summary.games:
        pgn_lines.extend(_game_to_pgn_with_eval(game))
        pgn_lines.append('')  # blank line between games

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(pgn_lines))

    return output_path


def _pgn_tag(value: str) -> str:
    """Escape a PGN tag value (backslashes and double quotes per PGN spec)."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _game_to_pgn_with_eval(game: GameRecord) -> list[str]:
    """Convert a game to PGN with evaluation annotations."""
    pgn_lines = [
        '[Event "Chess LLM Benchmark"]',
        '[Site "Local"]',
        f'[Date "{datetime.now(UTC).strftime("%Y.%m.%d")}"]',
        f'[Round "{_pgn_tag(str(game.game_id))}"]',
        f'[White "{_pgn_tag(game.white_player)}"]',
        f'[Black "{_pgn_tag(game.black_player)}"]',
        f'[Result "{_pgn_tag(game.result)}"]',
        f'[WhiteProvider "{_pgn_tag(game.white_provider)}"]',
        f'[BlackProvider "{_pgn_tag(game.black_provider)}"]',
        f'[OpeningECO "{_pgn_tag(str(game.opening_eco or "?"))}"]',
        f'[OpeningName "{_pgn_tag(str(game.opening_name or "?"))}"]',
        f'[GameDuration "{game.game_duration_sec:.1f}"]',
        '',
    ]

    board = chess.Board(game.opening_fen)
    move_text = []

    for ply, move_log in enumerate(game.moves):
        try:
            move = chess.Move.from_uci(move_log.move_uci)
        except ValueError:
            move = None
        if move is not None and move in board.legal_moves:
            san = board.san(move)
            board.push(move)
        else:
            san = move_log.move_uci

        # Add evaluation annotation if available
        eval_annotation = ""
        if (move_log.eval_cp_score is not None and
            move_log.eval_best_move_uci is not None):
            # Add evaluation as PGN comment
            cp = move_log.eval_cp_score
            if move_log.eval_mate_in is not None:
                eval_str = f"#{move_log.eval_mate_in}"
            else:
                eval_str = f"{cp/100:+.2f}" if cp != 0 else "0.00"

            best_move = move_log.eval_best_move_uci
            if best_move:
                eval_annotation = f" {{[%eval {eval_str}] [%bestmove {best_move}]}}"

        if ply % 2 == 0:
            move_text.append(f'{ply//2 + 1}. {san}{eval_annotation}')
        else:
            move_text.append(f'{san}{eval_annotation}')

    pgn_lines.append(' '.join(move_text))
    pgn_lines.append(game.result)
    pgn_lines.append('')

    return pgn_lines


def _games_to_dataframe(games: list[GameRecord]) -> pd.DataFrame:
    """Convert games to DataFrame."""
    rows: list[dict[str, Any]] = []
    for game in games:
        rows.append({
            'game_id': game.game_id,
            'white_player': game.white_player,
            'black_player': game.black_player,
            'white_provider': game.white_provider,
            'black_provider': game.black_provider,
            'opening_eco': game.opening_eco,
            'opening_name': game.opening_name,
            'opening_fen': game.opening_fen,
            'result': game.result,
            'result_numeric': game.result_numeric,
            'total_moves': game.total_moves,
            'game_duration_sec': game.game_duration_sec,
            'timestamp_utc': game.timestamp_utc,
        })
    return pd.DataFrame(rows)


def _moves_to_dataframe(games: list[GameRecord]) -> pd.DataFrame:
    """Convert moves to DataFrame."""
    rows: list[dict[str, Any]] = []
    for game in games:
        for move in game.moves:
            rows.append({
                'game_id': game.game_id,
                'move_number': move.move_number,
                'player': move.player,
                'color': move.color,
                'fen_before': move.fen_before,
                'move_uci': move.move_uci,
                'move_san': move.move_san,
                'llm_latency_ms': move.llm_latency_ms,
                'llm_tokens_in': move.llm_tokens_in,
                'llm_tokens_out': move.llm_tokens_out,
                'llm_raw_response': _sanitize_for_csv(move.llm_raw_response),
                'thinking_trace': _sanitize_for_csv(move.thinking_trace),
                'prompt_hash': move.prompt_hash,
                'validation_retries': move.validation_retries,
                'timestamp_utc': move.timestamp_utc,
                'eval_cp_score': move.eval_cp_score,
                'eval_mate_in': move.eval_mate_in,
                'eval_best_move_uci': move.eval_best_move_uci,
                'eval_best_move_cp': move.eval_best_move_cp,
                'eval_top3_moves': json.dumps(move.eval_top3_moves) if move.eval_top3_moves else None,
                'eval_depth': move.eval_depth,
                'eval_time_ms': move.eval_time_ms,
                'cp_loss': move.cp_loss,
                'move_quality': move.move_quality,
                'is_best_move': move.is_best_move,
                'thinking_chars': move.thinking_chars,
                'thinking_words': move.thinking_words,
                'thinking_has_structured': move.thinking_has_structured,
                'thinking_mentions_tactics': move.thinking_mentions_tactics,
                'thinking_mentions_strategy': move.thinking_mentions_strategy,
                'thinking_mentions_time_pressure': move.thinking_mentions_time_pressure,
                'thinking_mentions_material': move.thinking_mentions_material,
                'thinking_mentions_positional': move.thinking_mentions_positional,
                'thinking_mentions_king_safety': move.thinking_mentions_king_safety,
            })
    return pd.DataFrame(rows)


def _players_to_dataframe(player_stats: dict[str, ViewPlayerStats]) -> pd.DataFrame:
    """Convert player stats to DataFrame."""
    rows: list[dict[str, Any]] = []
    for name, stats in player_stats.items():
        rows.append({
            'name': name,
            'games_played': stats.games_played,
            'games_as_white': stats.games_as_white,
            'games_as_black': stats.games_as_black,
            'wins': stats.wins,
            'losses': stats.losses,
            'draws': stats.draws,
            'score': stats.score,
            'score_pct': stats.score_pct,
            'moves_played': stats.moves_played,
            'captures': stats.captures,
            'checks': stats.checks,
            'avg_latency_ms': stats.avg_latency_ms,
            'tokens_in_total': stats.tokens_in_total,
            'tokens_out_total': stats.tokens_out_total,
            'total_cp_loss': getattr(stats, 'total_cp_loss', None),
            'avg_cp_loss': getattr(stats, 'avg_cp_loss', None),
            'blunder_count': getattr(stats, 'blunder_count', None),
            'mistake_count': getattr(stats, 'mistake_count', None),
            'inaccuracy_count': getattr(stats, 'inaccuracy_count', None),
            'best_move_pct': getattr(stats, 'best_move_pct', None),
            'avg_thinking_chars': getattr(stats, 'avg_thinking_chars', None),
            'thinking_quality_score': getattr(stats, 'thinking_quality_score', None),
        })
    return pd.DataFrame(rows)


def _pairings_to_dataframe(pairings: list[PairingResult]) -> pd.DataFrame:
    """Convert pairings to DataFrame."""
    rows: list[dict[str, Any]] = []
    for pairing in pairings:
        rows.append({
            'white': pairing.white,
            'black': pairing.black,
            'games': pairing.games,
            'white_wins': pairing.white_wins,
            'black_wins': pairing.black_wins,
            'draws': pairing.draws,
            'total_moves': pairing.total_moves,
        })
    return pd.DataFrame(rows)


def verify_reproducibility(run_dir: str | Path,
                           move_timing_tolerance_ms: int = 100,
                           token_tolerance: int = 5) -> dict[str, Any]:
    """Verify reproducibility by re-running benchmark and comparing results."""
    run_dir = Path(run_dir)

    # Load original run
    original = load_run(run_dir)
    if original is None:
        raise ValueError(f"No valid run data found in {run_dir}")

    # Recreate config
    config_dict = original.config

    # Run benchmark again
    from chessbench.benchmark.runner import BenchmarkConfig
    _ = BenchmarkConfig(**config_dict)

    # Compare original vs expected structure
    diffs: list[str] = []

    # Check config hash
    original_hash = original.config.get('config_hash')
    new_hash = compute_config_hash(original.config)

    return {
        "status": "SKIPPED" if not os.getenv("RUN_REPRODUCIBILITY_TEST") else "NOT_IMPLEMENTED",
        "config_hash_match": original_hash == new_hash if original_hash else None,
        "original_hash": original_hash,
        "new_hash": new_hash,
        "diffs": diffs,
    }


def export_all_formats(run_dir: str | Path, output_base: str | Path | None = None) -> dict[str, Path]:
    """Export run in all supported formats."""
    run_dir = Path(run_dir)
    output_base = run_dir / "export_all" if output_base is None else Path(output_base)

    output_base.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    # Parquet
    try:
        results['parquet'] = export_parquet(run_dir, output_base / "export.parquet")
    except Exception as e:
        results['parquet'] = Path(f"ERROR: {e}")

    # CSV
    try:
        results['csv'] = export_csv(run_dir, output_base / "csv")
    except Exception as e:
        results['csv'] = Path(f"ERROR: {e}")

    # PGN with eval
    try:
        results['pgn_eval'] = export_pgn_with_eval(run_dir, output_base / "games_with_eval.pgn")
    except Exception as e:
        results['pgn_eval'] = Path(f"ERROR: {e}")

    return results
