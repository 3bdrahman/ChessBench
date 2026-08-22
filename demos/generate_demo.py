"""Generate a showcase demo run when ``runs/`` has no real runs yet.

Produces a clearly-labeled demo benchmark directory using the *real*
``BenchmarkLogger`` + ``StockfishEvaluator`` so the JSONL schema is identical
to a real run. Episodes are genuine Stockfish-vs-Stockfish games (the real
chess engine on the host) starting from a small set of ECO openings so the
analytical dashboard has dimensional eval/quality data on day one.

The script is idempotent: it writes to ``runs/<DEMO_RUN_ID>/`` and only when
that directory is absent. Re-running it does not overwrite real runs.

Usage::

    python -m demos.generate_demo            # write the demo run if absent
    python -m demos.generate_demo --force    # overwrite the demo run

Run is tagged with ``is_demo: true`` in summary.json + every game's
config so the UI can mark it visually ("★ Demo run") and never confuse it
with a real benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess

from chessbench.benchmark.evaluator import StockfishEvaluator
from chessbench.benchmark.logging import BenchmarkLogger
from chessbench.benchmark.openings import OpeningBook

_log = logging.getLogger(__name__)

DEMO_RUN_ID = "demo_gpt4o_vs_sonnet"
DEMO_WHITE = "openai:gpt-4o"
DEMO_BLACK = "anthropic:claude-3-5-sonnet-20241022"
DEMO_WHITE_PROVIDER = "openai"
DEMO_BLACK_PROVIDER = "anthropic"

# Curated ECO openings that produce有趣, varied middlegames for a showcase.
DEMO_OPENINGS: list[tuple[str, str, str]] = [
    # (eco, name, FEN)
    ("C50", "Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("B20", "Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
    ("D02", "London System", "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"),
    ("A04", "Reti Opening", "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1"),
]

# Short, curated thinking-trace snippets used to populate the Thinking Trace
# analytics section. The blades of every trace are real LLM chess transcripts
# trimmed to ~200-400 chars. They ARE clearly tagged via is_demo=True at the
# run level — never serve a demo trace as a model output.
_THINKING_TRACES: list[str] = [
    "Looking at the position. My opponent just played ...e5, fighting for the center. "
    "Tactical idea: Nxe5 wins a pawn but Nxe5 Bb4+ forks my king and knight after the recapture — "
    "that's a zwischenzug I have to respect. So 1. Nxe5 is a known trap but wrong here. Strategic plan: "
    "I want to hold the center with d2-d4 eventually, develop the bishop to e2 or g2, then cxd5 in one go. "
    "Material is even. King safety is fine — I haven't castled yet but b1 is sheltered. "
    "I'll play Bc4, the natural Italian Game developing move, pinning the f-file conceptually and preparing O-O.",

    "My move. Opponent just played Nf6 attacking e4. Tactical scan: Nd5 forks the c7 pawn and the f6 knight — "
    "but Nxd5 exd5 reveals my bishop pair and I lose tempo. Better is Nc3, defending e4 and developing. "
    "Strategy: dark-square control. I want my knight on d5 long-term. Bishop is well placed on g7; "
    "if they castle short I have h6xBg7 ideas later. Time pressure is not a factor yet — move 7, plenty on clock. "
    "I'll castle kingside to secure my king before pushing d6-e5 for space on the queenside.",

    "Critical moment. Material down a pawn but I have compensation via bishop pair + active rook on the open c-file. "
    "Tactics first: Rxc3 Bxc3 Qxc3 wins back the pawn with check and threatens Qxc7. Let me verify — "
    "yes, the b3 rook is indirectly defended by my queen on a4. Calculation depth: 5 plies, the resulting position "
    "is +0.6 for me per my own analysis. Strategically I trade into an endgame where my bishop pair + passed d-pawn "
    "dominate. Opponent has no counterplay on the kingside. Time is tight (~45s on my clock) but the line is forced. "
    "Play Rxc3.",

    "Endgame. Opponent just played Kf6. Pure king-and-pawn. Tactical: my king is in the square of the a-pawn "
    "so opposition matters most. I need to seize the opposition with Kf3 — that forces their king back because "
    "whoever has to move loses. Strategic — the a-pawn is a long way from queening and my king is closer. "
    "Material equal but my king is more active, which in K+P endgames is decisive. "
    "No time pressure at all — this is move 38 and I have 2 minutes. Play Kf3."
]


def _is_demo_run(run_dir: Path) -> bool:
    """Return True if the given run directory is the seeded demo run."""
    summary = run_dir / "summary.json"
    if not summary.is_file():
        return False
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("is_demo"))


def demo_run_exists(runs_root: str | Path = "runs") -> bool:
    """Return True if the demo run directory already exists with real games."""
    path = Path(runs_root) / DEMO_RUN_ID
    if not path.is_dir():
        return False
    games = path / "games.jsonl"
    return games.is_file() and games.stat().st_size > 0


async def _play_stockfish_game(
    white_spec: str,
    black_spec: str,
    opening: tuple[str, str, str],
    target_plies: int,
    evaluator: StockfishEvaluator,
    logger: BenchmarkLogger,
    rng: random.Random,
) -> str:
    """Play a genuine Stockfish-vs-Stockfish game through the real engine.

    Uses ``chess.engine`` directly to drive moves — the same library the
    benchmark runner uses. Each move is logged through the real
    :meth:`BenchmarkLogger.log_move` so the JSONL schema is identical to a
    real run, including eval/quality fields.
    """
    import chess.engine

    eco, name, opening_fen = opening
    game_id = str(uuid.uuid4())[:8]
    logger.start_game(
        white_player=white_spec,
        black_player=black_spec,
        white_provider=white_spec.split(":")[0],
        black_provider=black_spec.split(":")[0],
        opening_eco=eco,
        opening_name=name,
        opening_fen=opening_fen,
    )
    # Override the logger's auto-generated game_id so our eval records line up.
    logger.current_game_id = game_id

    board = chess.Board(opening_fen)
    ply = 0

    engine = evaluator._engine  # noqa: SLF001 — internal access by design
    if engine is None:
        return game_id

    try:
        while not board.is_game_over(claim_draw=True) and ply < target_plies:
            color = "white" if board.turn == chess.WHITE else "black"
            spec = white_spec if board.turn == chess.WHITE else black_spec

            eval_result = await evaluator.evaluate(board)

            engine_limit = chess.engine.Limit(depth=rng.randint(8, 12))

            # Deliberately deviate from the best line on a fraction of moves
            # so demo runs show realistic move-quality variety (inaccuracies,
            # mistakes, blunders). Position evals stay real (Stockfish
            # evaluates before), and cp_loss records the deviation honestly.
            # Run config flags is_demo=True + synthetic_quality=true.
            candidate_moves: list[chess.Move] = []
            if eval_result and eval_result.top3_moves:
                for mv in eval_result.top3_moves:
                    try:
                        m = chess.Move.from_uci(mv["uci"])
                        if m in board.legal_moves:
                            candidate_moves.append(m)
                    except (ValueError, chess.IllegalMoveError):
                        continue
            all_legal = list(board.legal_moves)

            rng_roll = rng.random()
            if rng_roll < 0.75 or not candidate_moves:
                try:
                    result_play = await engine.play(board, engine_limit)
                except (chess.engine.EngineError, BrokenPipeError):
                    break
                move = result_play.move
            elif rng_roll < 0.90 and len(candidate_moves) >= 2:
                move = candidate_moves[1]
            elif rng_roll < 0.97:
                move = candidate_moves[2] if len(candidate_moves) >= 3 else candidate_moves[-1]
            else:
                off_lines = [m for m in all_legal if m not in candidate_moves]
                move = rng.choice(off_lines) if off_lines else candidate_moves[0]

            if move is None or move not in board.legal_moves:
                break

            fen_before = board.fen()
            san = board.san(move)

            thinking = _THINKING_TRACES[ply % len(_THINKING_TRACES)]

            top3 = eval_result.top3_moves if (eval_result and eval_result.top3_moves) else []
            chosen_cp_on_top3 = None
            for mv in top3:
                if mv.get("uci") == move.uci():
                    chosen_cp_on_top3 = mv.get("cp")
                    break

            board.push(move)
            if chosen_cp_on_top3 is not None:
                cp_after_played = chosen_cp_on_top3
                board.pop()
            else:
                after_eval = await evaluator.evaluate(board)
                board.pop()
                cp_after_played = after_eval.cp_score if after_eval else None

            best_cp_before = eval_result.best_move_cp if eval_result else None
            eval_cp_score_logged = (
                cp_after_played
                if cp_after_played is not None
                else (eval_result.cp_score if eval_result else None)
            )
            eval_best_move_cp_logged = (
                best_cp_before
                if best_cp_before is not None
                else (eval_result.best_move_cp if eval_result else None)
            )

            logger.log_move(
                move_number=ply + 1,
                player=spec,
                color=color,
                fen_before=fen_before,
                move_uci=move.uci(),
                move_san=san,
                llm_latency_ms=rng.randint(180, 2400),
                llm_tokens_in=rng.randint(280, 1200),
                llm_tokens_out=rng.randint(8, 280),
                llm_raw_response=f"<thinking>{thinking}</thinking>\n<uci>{move.uci()}</uci> {san}",
                thinking_trace=thinking,
                prompt_hash=str(uuid.uuid4())[:16],
                validation_retries=0,
                eval_cp_score=eval_cp_score_logged,
                eval_mate_in=eval_result.mate_in if eval_result else None,
                eval_best_move_uci=eval_result.best_move_uci if eval_result else None,
                eval_best_move_cp=eval_best_move_cp_logged,
                eval_top3_moves=eval_result.top3_moves if eval_result else None,
                eval_depth=eval_result.depth if eval_result else None,
                eval_time_ms=eval_result.time_ms if eval_result else None,
            )

            board.push(move)
            ply += 1
            await asyncio.sleep(0.001)

        # Determine a clean chess terminal.
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            result = "1/2-1/2"
            result_numeric = 0.5
            termination = "max_moves"
        elif outcome.winner is None:
            result = "1/2-1/2"
            result_numeric = 0.5
            termination = (
                outcome.termination.name.lower()
                if hasattr(outcome.termination, "name")
                else str(outcome.termination).lower()
            )
        elif outcome.winner == chess.WHITE:
            result = "1-0"
            result_numeric = 1.0
            termination = "checkmate" if outcome.termination == chess.Termination.CHECKMATE else outcome.termination.name.lower()
        else:
            result = "0-1"
            result_numeric = 0.0
            termination = "checkmate" if outcome.termination == chess.Termination.CHECKMATE else outcome.termination.name.lower()

        # Tiny game_duration estimate from the synthetic latencies
        # (the real runner measures wall-clock; here we approximate.)
        game_duration_sec = float(sum(rng.randint(2, 4) for _ in range(ply)))

        # If is_demo=True was injected in start_run config, preserve that flag.
        logger.end_game(
            result=result,
            result_numeric=result_numeric,
            total_moves=ply,
            game_duration_sec=game_duration_sec,
            termination_reason=termination,
        )
    finally:
        pass

    return game_id


async def generate_demo_run(
    runs_root: str | Path = "runs",
    *,
    games_per_opening: int = 1,
    target_plies: int = 80,
    force: bool = False,
) -> Path | None:
    """Write a single demo benchmark run to ``<runs_root>/<DEMO_RUN_ID>/``.

    Returns the run directory Path on success, or ``None`` if Stockfish is
    unavailable on this host or the demo run already exists and ``force`` is
    False.
    """
    if demo_run_exists(runs_root) and not force:
        _log.info("Demo run already exists at %s/%s — skipping", runs_root, DEMO_RUN_ID)
        return Path(runs_root) / DEMO_RUN_ID

    run_dir = Path(runs_root) / DEMO_RUN_ID
    if run_dir.exists() and force:
        import shutil
        shutil.rmtree(run_dir)

    logger = BenchmarkLogger(str(run_dir))
    # Tag the run as a demo so the UI can mark it visually and never confuse
    # it with a real benchmark. summary.json + per-game config carry this.
    config: dict[str, Any] = {
        "games_per_pairing": games_per_opening,
        "opening_book": "demo_seed",
        "colors": "alternating",
        "temperature": 0.0,
        "max_tokens": None,
        "reasoning_level": "mid",
        "seed": 42,
        "max_parallel_games": 1,
        "move_timeout_seconds": 120,
        "game_timeout_seconds": 7200,
        "players": [DEMO_WHITE, DEMO_BLACK],
        "output_dir": str(runs_root),
        "run_name": DEMO_RUN_ID,
        "is_demo": True,
        "synthetic_quality": True,
        "synthetic_thinking": True,
    }
    logger.start_run(config)

    rng = random.Random(42)

    async with StockfishEvaluator(depth=10, time_ms=60, multipv=3) as evaluator:
        if not evaluator._available:  # noqa: SLF001
            _log.warning("Stockfish not available — cannot generate demo run")
            return None

        for opening in DEMO_OPENINGS:
            for _ in range(games_per_opening):
                await _play_stockfish_game(
                    DEMO_WHITE, DEMO_BLACK, opening, target_plies, evaluator, logger, rng
                )

        await evaluator.stop()

    # Manual summary injection so the demo flag survives write_summary,
    # which rebuilds the summary dict from scratch.
    logger.write_summary()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["is_demo"] = True
    summary["run_id"] = DEMO_RUN_ID
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a demo benchmark run.")
    parser.add_argument("--runs-root", default="runs", help="runs/ directory")
    parser.add_argument("--games", type=int, default=1, help="games per opening")
    parser.add_argument("--plies", type=int, default=42, help="target plies per game")
    parser.add_argument("--force", action="store_true", help="overwrite existing demo run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = asyncio.run(generate_demo_run(
        args.runs_root,
        games_per_opening=args.games,
        target_plies=args.plies,
        force=args.force,
    ))
    if path is None:
        print(f"Demo run not generated (stockfish missing or already present at {args.runs_root}/{DEMO_RUN_ID}).")
    else:
        print(f"Demo run ready: {path}")


if __name__ == "__main__":
    main()
