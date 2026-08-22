"""Reproducibility verification for benchmark runs."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chessbench import constants
from chessbench.benchmark.export import (
    compute_config_hash,
    compute_dependency_versions,
    compute_git_hash,
)
from chessbench.benchmark.results_view import load_run
from chessbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner


@dataclass
class ReproductionReport:
    """Report from reproducibility verification."""
    status: str  # PASS, FAIL, SKIPPED, ERROR
    config_hash_match: bool | None = None
    original_hash: str | None = None
    new_hash: str | None = None
    diffs: list[str] = field(default_factory=list)
    error: str | None = None
    behavioral_checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "config_hash_match": self.config_hash_match,
            "original_hash": self.original_hash,
            "new_hash": self.new_hash,
            "diffs": self.diffs,
            "error": self.error,
            "behavioral_checks": self.behavioral_checks,
        }


async def _run_behavioral_check(
    original: Any,
    move_timing_tolerance_ms: int,
    token_tolerance: int,
    max_game_diffs: int,
) -> tuple[list[str], dict[str, Any]]:
    """Run a small subset of games to verify behavioral reproducibility.

    Returns:
        Tuple of (diffs, behavioral_checks)
    """
    diffs: list[str] = []
    behavioral_checks: dict[str, Any] = {
        "games_verified": 0,
        "moves_compared": 0,
        "move_matches": 0,
        "timing_diffs": [],
        "token_diffs": [],
    }

    try:
        # Recreate the config
        config = original.config

        # Run just 1 game per pairing (up to 3 pairings max)
        players = list(config.get('players', []))
        if not players:
            return diffs, behavioral_checks

        # Limit to 2 pairings, 1 game each
        pairings = []
        for i, white in enumerate(players[:2]):
            for j, black in enumerate(players[:2]):
                if i != j:
                    pairings.append((white, black))

        if not pairings:
            return diffs, behavioral_checks

        # Limit to 2 pairings
        pairings = pairings[:2]

        # Create config for verification run
        verify_config = BenchmarkConfig(
            time_control_seconds_per_move=config.get('time_control_seconds_per_move', constants.REPRODUCIBILITY_DEFAULTS["verification_time_control"]),
            opening_book="startpos",
            games_per_pairing=constants.REPRODUCIBILITY_DEFAULTS["verification_games_per_pairing"],
            colors=config.get('colors', 'alternating'),
            temperature=config.get('temperature', constants.DEFAULT_BENCHMARK_TEMPERATURE),
            max_tokens=config.get('max_tokens', constants.DEFAULT_MAX_TOKENS_BENCHMARK),
            seed=config.get('seed', constants.DEFAULT_SEED),
            max_parallel_games=1,
            move_timeout_seconds=constants.REPRODUCIBILITY_DEFAULTS["verification_move_timeout"],
            game_timeout_seconds=constants.REPRODUCIBILITY_DEFAULTS["verification_game_timeout"],
            players=players,
            output_dir="runs",
            api_keys=config.get('api_keys', {}),
        )

        # Initialize runner components
        runner = BenchmarkRunner(verify_config)

        # Verify a few games
        for white_spec, black_spec in pairings:
            if white_spec not in runner.players or black_spec not in runner.players:
                continue

            # Use startpos for verification
            opening = {
                'eco': 'START',
                'name': 'Starting Position',
                'moves': [],
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'ply': 0
            }

            # Run one game
            game_log = await runner.run_pairing(
                white_spec, black_spec, opening, 0
            )

            behavioral_checks["games_verified"] += 1

            # Compare with original run if available
            original_game = None
            for g in original.games:
                if (g.white_player == white_spec and g.black_player == black_spec) or \
                   (g.white_player == black_spec and g.black_player == white_spec):
                    original_game = g
                    break

            if original_game and original_game.moves:
                # Compare move sequences
                original_moves = [m.move_uci for m in original_game.moves]
                new_moves = [m.move_uci for m in game_log.moves]

                behavioral_checks["moves_compared"] += min(len(original_moves), len(new_moves))

                for i, (orig, new) in enumerate(zip(original_moves, new_moves, strict=True)):
                    if orig == new:
                        behavioral_checks["move_matches"] += 1
                    else:
                        diffs.append(f"Move mismatch at ply {i+1}: original={orig}, new={new}")

                # Compare timing if available
                if original_game.moves and game_log.moves:
                    for i, (orig, new) in enumerate(zip(original_game.moves, game_log.moves, strict=True)):
                        if hasattr(orig, 'llm_latency_ms') and hasattr(new, 'llm_latency_ms') and orig.llm_latency_ms is not None and new.llm_latency_ms is not None:
                            diff = abs(orig.llm_latency_ms - new.llm_latency_ms)
                            behavioral_checks["timing_diffs"].append(diff)
                            if diff > move_timing_tolerance_ms:
                                diffs.append(f"Timing diff at ply {i+1}: {diff}ms > {move_timing_tolerance_ms}ms tolerance")

                        # Compare token counts
                        if hasattr(orig, 'llm_tokens_in') and hasattr(new, 'llm_tokens_in') and orig.llm_tokens_in is not None and new.llm_tokens_in is not None:
                            diff = abs(orig.llm_tokens_in - new.llm_tokens_in)
                            behavioral_checks["token_diffs"].append(diff)
                            if diff > token_tolerance:
                                diffs.append(f"Token in diff at ply {i+1}: {diff} > {token_tolerance} tolerance")

                        if hasattr(orig, 'llm_tokens_out') and hasattr(new, 'llm_tokens_out') and orig.llm_tokens_out is not None and new.llm_tokens_out is not None:
                            diff = abs(orig.llm_tokens_out - new.llm_tokens_out)
                            behavioral_checks["token_diffs"].append(diff)
                            if diff > token_tolerance:
                                diffs.append(f"Token out diff at ply {i+1}: {diff} > {token_tolerance} tolerance")

        # Calculate summary stats
        if behavioral_checks["moves_compared"] > 0:
            behavioral_checks["move_match_rate"] = behavioral_checks["move_matches"] / behavioral_checks["moves_compared"]

    except Exception as e:
        diffs.append(f"Behavioral check failed: {e!s}")
        behavioral_checks["error"] = str(e)

    return diffs, behavioral_checks


async def verify_run_reproducibility(
    run_dir: str | Path,
    move_timing_tolerance_ms: int = constants.REPRODUCIBILITY_DEFAULTS["move_timing_tolerance_ms"],
    token_tolerance: int = constants.REPRODUCIBILITY_DEFAULTS["token_tolerance"],
    max_game_diffs: int = constants.REPRODUCIBILITY_DEFAULTS["max_game_diffs"],
    full_behavioral_check: bool = False,
) -> ReproductionReport:
    """
    Verify that a benchmark run can be reproduced.

    Args:
        run_dir: Directory containing the original run
        move_timing_tolerance_ms: Tolerance for move timing differences (ms)
        token_tolerance: Tolerance for token count differences
        max_game_diffs: Maximum number of game diffs to report

    Returns:
        ReproductionReport with verification results
    """
    run_dir = Path(run_dir)

    # Load original run
    original = load_run(run_dir)
    if original is None:
        return ReproductionReport(
            status="ERROR",
            error=f"No valid run data found in {run_dir}",
        )

    # Check config hash
    original_hash = original.config.get('config_hash')
    new_hash = compute_config_hash(original.config)

    diffs: list[str] = []

    if original_hash and original_hash != new_hash:
        diffs.append(f"Config hash mismatch: original={original_hash}, new={new_hash}")

    # Check git commit
    original_git = original.config.get('git_commit')
    current_git = compute_git_hash()
    if original_git and current_git and original_git != current_git:
        diffs.append(f"Git commit changed: original={original_git[:8]}, current={current_git[:8]}")

    # Check Python version
    original_py = original.config.get('python_version')
    current_py = f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
    if original_py and original_py != current_py:
        diffs.append(f"Python version changed: original={original_py}, current={current_py}")

    # Check dependency versions
    original_deps = original.config.get('dependencies', {})
    current_deps = compute_dependency_versions()
    for dep, orig_ver in original_deps.items():
        if dep in current_deps and orig_ver != current_deps[dep]:
            diffs.append(f"Dependency {dep} version changed: original={orig_ver}, current={current_deps[dep]}")

    # If only config hash check is needed (no API keys available)
    if not os.getenv("RUN_FULL_REPRODUCIBILITY") and not full_behavioral_check:
        return ReproductionReport(
            status="PASS" if not diffs else "FAIL",
            config_hash_match=(original_hash == new_hash) if original_hash else None,
            original_hash=original_hash,
            new_hash=new_hash,
            diffs=diffs,
        )

    # Full behavioral check - run a subset of games to verify behavioral reproducibility
    if full_behavioral_check or os.getenv("RUN_FULL_REPRODUCIBILITY"):
        behavioral_diffs, behavioral_checks = await _run_behavioral_check(
            original, move_timing_tolerance_ms, token_tolerance, max_game_diffs
        )
        diffs.extend(behavioral_diffs)

    status = "PASS" if not diffs else "FAIL"
    return ReproductionReport(
        status=status,
        config_hash_match=(original_hash == new_hash) if original_hash else None,
        original_hash=original_hash,
        new_hash=new_hash,
        diffs=diffs,
        behavioral_checks=behavioral_checks,
    )


def run_reproducibility_cli(run_dir: str | Path) -> int:
    """CLI entry point for reproducibility verification."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify benchmark reproducibility")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--move-tolerance", type=int, default=constants.REPRODUCIBILITY_DEFAULTS["move_timing_tolerance_ms"], help="Move timing tolerance (ms)")
    parser.add_argument("--token-tolerance", type=int, default=constants.REPRODUCIBILITY_DEFAULTS["token_tolerance"], help="Token count tolerance")
    parser.add_argument("--full-behavioral", action="store_true", help="Run full behavioral check (requires API keys)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    report = verify_run_reproducibility(
        args.run_dir,
        move_timing_tolerance_ms=args.move_tolerance,
        token_tolerance=args.token_tolerance,
        full_behavioral_check=args.full_behavioral,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Reproducibility Check: {report.status}")
        if report.config_hash_match is not None:
            print(f"  Config Hash Match: {'YES' if report.config_hash_match else 'NO'}")
            print(f"  Original Hash: {report.original_hash}")
            print(f"  New Hash:      {report.new_hash}")
        if report.diffs:
            print(f"  Differences ({len(report.diffs)}):")
            for diff in report.diffs:
                print(f"    - {diff}")
        if report.error:
            print(f"  Error: {report.error}")

    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_reproducibility_cli(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")))
