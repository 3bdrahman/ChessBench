"""Generate real demo PGNs from real benchmark runs.

Replaces hand-written synthetic PGNs (which had placeholder model names
like "TacticalMaster" and contained invalid move sequences) with PGNs
reconstructed from ``runs/<run_id>/games.jsonl`` — the same artifacts the
benchmark runner already writes. Each demo PGN preserves the real player
specs, real moves, real timestamps, and real game outcomes.

Run as a module to regenerate ``demos/games/*.pgn`` from the latest
benchmark output:

    python -m demos.generate

If no benchmark runs exist, the script exits with a non-zero status and
prints instructions to run the benchmark first — it never invents fake
games.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.pgn

from chess_fight.benchmark.results_view import GameRecord, list_runs

_log = logging.getLogger(__name__)

GAMES_DIR = Path(__file__).parent / "games"


@dataclass
class DemoMetadata:
    """Lightweight metadata for a single demo game shown in the UI."""

    filename: str
    white: str
    black: str
    result: str
    opening: str
    move_count: int
    source: str  # Where the game came from — honest attribution.

    def display_label(self) -> str:
        return (
            f"{self.white} vs {self.black} "
            f"({self.move_count} moves, {self.result}) — {self.opening}"
        )


def _format_datetime(ts: str | None) -> str:
    if not ts:
        return datetime.now(tz=UTC).strftime("%Y.%m.%d")
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return parsed.strftime("%Y.%m.%d")
    except ValueError:
        return datetime.now(tz=UTC).strftime("%Y.%m.%d")


def game_to_pgn(game: GameRecord, run_id: str) -> chess.pgn.Game:
    """Convert a :class:`GameRecord` into a populated :class:`chess.pgn.Game`."""
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Chess LLM Benchmark"
    pgn_game.headers["Site"] = f"chess_fight benchmark run {run_id}"
    pgn_game.headers["Date"] = _format_datetime(game.timestamp_utc)
    pgn_game.headers["Round"] = game.game_id
    pgn_game.headers["White"] = game.white_player or "?"
    pgn_game.headers["Black"] = game.black_player or "?"
    pgn_game.headers["Result"] = game.result
    if game.opening_eco:
        pgn_game.headers["OpeningECO"] = game.opening_eco
    if game.opening_name:
        pgn_game.headers["Opening"] = game.opening_name
    pgn_game.headers["WhiteProvider"] = game.white_provider
    pgn_game.headers["BlackProvider"] = game.black_provider
    pgn_game.headers["TotalMoves"] = str(game.total_moves)
    pgn_game.headers["GameDurationSec"] = f"{game.game_duration_sec:.2f}"

    # Replay the moves from the opening FEN so SAN, en-passant, castling etc.
    # are reconstructed from real board state.
    board = chess.Board(game.opening_fen) if game.opening_fen else chess.Board()
    node = pgn_game
    for move_record in game.moves:
        try:
            move = chess.Move.from_uci(move_record.move_uci)
        except (ValueError, chess.InvalidMoveError):
            _log.warning("Skipping invalid UCI %s in game %s", move_record.move_uci, game.game_id)
            break
        if move not in board.legal_moves:
            _log.warning(
                "Move %s is illegal in game %s at FEN %s; truncating PGN",
                move_record.move_uci, game.game_id, board.fen(),
            )
            break
        node = node.add_variation(move)
        board.push(move)
    return pgn_game


def pgn_to_text(pgn_game: chess.pgn.Game) -> str:
    """Serialize a :class:`chess.pgn.Game` to a clean PGN string."""
    buf = io.StringIO()
    exporter = chess.pgn.FileExporter(buf)
    pgn_game.accept(exporter)
    return buf.getvalue()


def _safe_filename(spec: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in spec)


def _write_demo_pgn(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def generate_demos_from_runs(
    runs_root: str | Path = "runs",
    output_dir: str | Path = GAMES_DIR,
    max_games: int | None = 5,
) -> list[DemoMetadata]:
    """Build demo PGNs from real benchmark runs.

    Picks up to ``max_games`` games (default 5) — one per run when possible,
    preferring draws then wins for variety. Writes one PGN per game into
    ``output_dir`` and returns the matching metadata list.
    """
    output_dir = Path(output_dir)
    # Always start from a clean slate so we don't ship stale synthetic PGNs.
    if output_dir.exists():
        for existing in output_dir.glob("*.pgn"):
            existing.unlink()

    runs = list_runs(runs_root)
    if not runs:
        return []

    metadata: list[DemoMetadata] = []
    picked = 0
    for run in runs:
        if max_games is not None and picked >= max_games:
            break
        # Sort games: prefer a mix of results for variety.
        ordered_games = sorted(
            run.games,
            key=lambda g: (
                0 if g.result == "1/2-1/2" else 1 if g.result == "1-0" else 2,
                g.total_moves,
            ),
        )
        for game in ordered_games:
            if max_games is not None and picked >= max_games:
                break
            if not game.white_player or not game.black_player:
                continue
            pgn_game = game_to_pgn(game, run.run_id)
            slug = f"game_{picked + 1:02d}_{_safe_filename(game.white_player)}_vs_{_safe_filename(game.black_player)}.pgn"
            target = output_dir / slug
            _write_demo_pgn(target, pgn_to_text(pgn_game) + "\n")
            opening = (
                f"{game.opening_eco} {game.opening_name}".strip()
                if game.opening_eco or game.opening_name
                else "Unknown opening"
            )
            metadata.append(
                DemoMetadata(
                    filename=str(target.resolve()),
                    white=game.white_player,
                    black=game.black_player,
                    result=game.result,
                    opening=opening,
                    move_count=game.total_moves,
                    source=f"runs/{run.run_id}",
                )
            )
            picked += 1

    return metadata


def discover_existing(output_dir: str | Path = GAMES_DIR) -> list[DemoMetadata]:
    """List demo PGNs already on disk (no regeneration)."""
    output_dir = Path(output_dir)
    out: list[DemoMetadata] = []
    for pgn_path in sorted(output_dir.glob("*.pgn")):
        try:
            with pgn_path.open("r", encoding="utf-8") as fh:
                pgn_game = chess.pgn.read_game(fh)
        except (OSError, ValueError) as exc:
            _log.warning("Skipping malformed PGN %s: %s", pgn_path, exc)
            continue
        if pgn_game is None:
            continue
        headers = pgn_game.headers
        moves = list(pgn_game.mainline_moves())
        opening = headers.get("Opening", "")
        if headers.get("OpeningECO"):
            opening = f"{headers['OpeningECO']} {opening}".strip()
        out.append(
            DemoMetadata(
                filename=str(pgn_path.resolve()),
                white=headers.get("White", "Unknown"),
                black=headers.get("Black", "Unknown"),
                result=headers.get("Result", "*"),
                opening=opening or "Unknown opening",
                move_count=len(moves),
                source="on-disk",
            )
        )
    return out


def list_demo_games(
    runs_root: str | Path = "runs",
    output_dir: str | Path = GAMES_DIR,
    auto_generate: bool = True,
) -> list[DemoMetadata]:
    """Public entry point used by the Streamlit app.

    Strategy:
      1. If the on-disk demos exist, return them as-is (stable file paths).
      2. If the user enabled ``auto_generate`` and benchmark runs exist,
         regenerate the on-disk demos from real runs.
      3. Otherwise return an empty list — never fabricate synthetic PGNs.
    """
    output_dir = Path(output_dir)
    existing = discover_existing(output_dir)
    if existing:
        return existing
    if not auto_generate:
        return []
    generated = generate_demos_from_runs(runs_root=runs_root, output_dir=output_dir)
    return generated or []


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default=str(GAMES_DIR))
    parser.add_argument("--max-games", type=int, default=5)
    args = parser.parse_args(argv)

    metadata = generate_demos_from_runs(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        max_games=args.max_games,
    )
    if not metadata:
        print(
            f"No real benchmark runs found under {args.runs_root}. Run a benchmark first, e.g.:\n"
            "  python -m chess_fight.benchmark.runner --players test:model-a test:model-b --games 1",
            file=sys.stderr,
        )
        return 1
    for meta in metadata:
        print(f"Wrote {meta.filename}  {meta.display_label()}  source={meta.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
