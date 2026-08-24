"""Real benchmark-results reader.

Reads the JSONL / PGN / summary.json artifacts that
:mod:`chessbench.benchmark.runner` writes to ``runs/<run_id>/`` and turns
them into structured records the Streamlit UI can render — ELO leaderboard,
per-pairing win/loss, per-model token usage, per-move timing, etc.

Nothing in this module fabricates data. If a stat is missing from the
underlying run (e.g. a run never recorded tokens) it is reported as ``None``,
not estimated.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MoveRecord:
    """One move reconstructed from moves.jsonl."""

    game_id: str
    move_number: int
    player: str
    color: str
    fen_before: str
    move_uci: str
    move_san: str
    llm_latency_ms: int
    llm_tokens_in: int | None
    llm_tokens_out: int | None
    llm_raw_response: str
    thinking_trace: str | None
    prompt_hash: str
    validation_retries: int
    timestamp_utc: str
    eval_cp_score: int | None = None
    eval_mate_in: int | None = None
    eval_best_move_uci: str | None = None
    eval_best_move_cp: int | None = None
    eval_top3_moves: list[dict[str, Any]] | None = None
    eval_depth: int | None = None
    eval_time_ms: int | None = None
    cp_loss: float | None = None
    move_quality: str | None = None
    is_best_move: bool | None = None
    thinking_chars: int | None = None
    thinking_words: int | None = None
    thinking_has_structured: bool | None = None
    thinking_mentions_tactics: bool | None = None
    thinking_mentions_strategy: bool | None = None
    thinking_mentions_time_pressure: bool | None = None
    thinking_mentions_material: bool | None = None
    thinking_mentions_positional: bool | None = None
    thinking_mentions_king_safety: bool | None = None
    game_phase: str | None = None
    material_white: int | None = None
    material_black: int | None = None
    material_imbalance: int | None = None
    position_complexity: int | None = None
    illegal_attempts_count: int = 0
    attempted_illegal_moves: list[str] = field(default_factory=list)
    clock_remaining_sec: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoveRecord:
        return cls(
            game_id=str(data.get("game_id", "")),
            move_number=int(data.get("move_number", 0)),
            player=str(data.get("player", "")),
            color=str(data.get("color", "")),
            fen_before=str(data.get("fen_before", "")),
            move_uci=str(data.get("move_uci", "")),
            move_san=str(data.get("move_san", "")),
            llm_latency_ms=int(data.get("llm_latency_ms", 0) or 0),
            llm_tokens_in=data.get("llm_tokens_in"),
            llm_tokens_out=data.get("llm_tokens_out"),
            llm_raw_response=str(data.get("llm_raw_response", "")),
            thinking_trace=data.get("thinking_trace"),
            prompt_hash=str(data.get("prompt_hash", "")),
            validation_retries=int(data.get("validation_retries", 0) or 0),
            timestamp_utc=str(data.get("timestamp_utc", "")),
            eval_cp_score=data.get("eval_cp_score"),
            eval_mate_in=data.get("eval_mate_in"),
            eval_best_move_uci=data.get("eval_best_move_uci"),
            eval_best_move_cp=data.get("eval_best_move_cp"),
            eval_top3_moves=data.get("eval_top3_moves"),
            eval_depth=data.get("eval_depth"),
            eval_time_ms=data.get("eval_time_ms"),
            cp_loss=data.get("cp_loss"),
            move_quality=data.get("move_quality"),
            is_best_move=data.get("is_best_move"),
            thinking_chars=data.get("thinking_chars"),
            thinking_words=data.get("thinking_words"),
            thinking_has_structured=data.get("thinking_has_structured"),
            thinking_mentions_tactics=data.get("thinking_mentions_tactics"),
            thinking_mentions_strategy=data.get("thinking_mentions_strategy"),
            thinking_mentions_time_pressure=data.get("thinking_mentions_time_pressure"),
            thinking_mentions_material=data.get("thinking_mentions_material"),
            thinking_mentions_positional=data.get("thinking_mentions_positional"),
            thinking_mentions_king_safety=data.get("thinking_mentions_king_safety"),
            game_phase=data.get("game_phase"),
            material_white=data.get("material_white"),
            material_black=data.get("material_black"),
            material_imbalance=data.get("material_imbalance"),
            position_complexity=data.get("position_complexity"),
            illegal_attempts_count=int(data.get("illegal_attempts_count", 0) or 0),
            attempted_illegal_moves=data.get("attempted_illegal_moves", []) or [],
            clock_remaining_sec=data.get("clock_remaining_sec"),
        )


@dataclass
class GameRecord:
    """One game reconstructed from games.jsonl or moves.jsonl grouping."""

    game_id: str
    white_player: str
    black_player: str
    white_provider: str
    black_provider: str
    opening_eco: str | None
    opening_name: str | None
    opening_fen: str
    result: str  # "1-0" | "0-1" | "1/2-1/2"
    result_numeric: float  # 1.0, 0.5, 0.0
    total_moves: int
    game_duration_sec: float
    timestamp_utc: str
    moves: list[MoveRecord] = field(default_factory=list)
    termination_reason: str = "unknown"

    @property
    def winner_spec(self) -> str | None:
        if self.result_numeric == 1.0:
            return self.white_player
        if self.result_numeric == 0.0:
            return self.black_player
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["moves"] = [asdict(m) for m in self.moves]
        return d


@dataclass
class PlayerStats:
    """Aggregated stats for one player across a run."""

    name: str
    games_played: int = 0
    games_as_white: int = 0
    games_as_black: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    moves_played: int = 0
    captures: int = 0
    checks: int = 0
    total_latency_ms: int = 0
    latency_samples: int = 0
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    tokens_recorded_moves: int = 0

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def avg_latency_ms(self) -> float | None:
        if self.latency_samples == 0:
            return None
        return self.total_latency_ms / self.latency_samples

    @property
    def score_pct(self) -> float | None:
        if self.games_played == 0:
            return None
        return 100.0 * self.score / self.games_played


@dataclass
class PairingResult:
    """Per-pairing head-to-head record within a run."""

    white: str
    black: str
    games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_moves: int = 0


@dataclass
class RunSummary:
    """Aggregated summary of a single benchmark run directory."""

    run_id: str
    run_dir: Path
    timestamp_utc: str | None
    config: dict[str, Any]
    total_games: int
    total_moves: int
    total_duration_sec: float
    games: list[GameRecord]
    player_stats: dict[str, PlayerStats]
    pairings: list[PairingResult]
    providers_seen: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "timestamp_utc": self.timestamp_utc,
            "config": self.config,
            "total_games": self.total_games,
            "total_moves": self.total_moves,
            "total_duration_sec": self.total_duration_sec,
            "providers_seen": self.providers_seen,
            "player_stats": {k: asdict(v) for k, v in self.player_stats.items()},
            "pairings": [asdict(p) for p in self.pairings],
            "games": [g.to_dict() for g in self.games],
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Reading & parsing
# ---------------------------------------------------------------------------


def list_run_dirs(runs_root: str | Path = "runs") -> list[Path]:
    """List all run directories under ``runs_root`` (newest first).

    Skips files and the ``runs_root`` itself. Returns empty list if the
    directory does not exist (no fabricated runs).
    """
    root = Path(runs_root)
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return the parsed objects.

    Skips blank lines; logs and continues on malformed lines instead of
    silently dropping the whole file.
    """
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _log.warning("Skipping malformed line %d in %s: %s", line_no, path, exc)
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        _log.warning("Malformed summary.json in %s: %s", path, exc)
        return {}


def _build_games(
    games_jsonl: list[dict[str, Any]],
    moves_jsonl: list[dict[str, Any]],
) -> list[GameRecord]:
    """Reconstruct full GameRecord objects, preferring games.jsonl (richer)."""
    games: dict[str, GameRecord] = {}
    for raw in games_jsonl:
        moves_raw = raw.get("moves", []) or []
        record = GameRecord(
            game_id=str(raw.get("game_id", "")),
            white_player=str(raw.get("white_player", "")),
            black_player=str(raw.get("black_player", "")),
            white_provider=str(raw.get("white_provider", "")),
            black_provider=str(raw.get("black_provider", "")),
            opening_eco=raw.get("opening_eco"),
            opening_name=raw.get("opening_name"),
            opening_fen=str(raw.get("opening_fen", "")),
            result=str(raw.get("result", "*")),
            result_numeric=(
                float(raw["result_numeric"])
                if raw.get("result_numeric") is not None
                else 0.5
            ),
            total_moves=int(raw.get("total_moves", 0) or 0),
            game_duration_sec=float(raw.get("game_duration_sec", 0.0) or 0.0),
            timestamp_utc=str(raw.get("timestamp_utc", "")),
            termination_reason=str(raw.get("termination_reason", "unknown")),
            moves=[MoveRecord.from_dict(m) for m in moves_raw],
        )
        if record.game_id:
            games[record.game_id] = record

    # Backfill from moves.jsonl for any game missing from games.jsonl.
    by_game: dict[str, list[MoveRecord]] = defaultdict(list)
    for raw in moves_jsonl:
        by_game[str(raw.get("game_id", ""))].append(MoveRecord.from_dict(raw))
    for game_id, moves in by_game.items():
        if not game_id:
            continue
        if game_id in games:
            continue
        if not moves:
            continue
        first = moves[0]
        last = moves[-1]
        games[game_id] = GameRecord(
            game_id=game_id,
            white_player=(first.color == "white" and first.player) or "",
            black_player=(first.color == "black" and first.player) or "",
            white_provider=first.player.split(":", 1)[0] if ":" in first.player else "",
            black_provider="",
            opening_eco=None,
            opening_name=None,
            opening_fen=first.fen_before,
            result="*",
            result_numeric=0.5,
            total_moves=len(moves),
            game_duration_sec=0.0,
            timestamp_utc=last.timestamp_utc,
            termination_reason="unknown",
            moves=moves,
        )

    return list(games.values())


def _aggregate_player_stats(games: Iterable[GameRecord]) -> dict[str, PlayerStats]:
    stats: dict[str, PlayerStats] = {}
    for game in games:
        for spec, role in (
            (game.white_player, "white"),
            (game.black_player, "black"),
        ):
            if not spec:
                continue
            if spec not in stats:
                stats[spec] = PlayerStats(name=spec)
            s = stats[spec]
            s.games_played += 1
            if role == "white":
                s.games_as_white += 1
            else:
                s.games_as_black += 1
            if (game.result_numeric == 1.0 and role == "white") or (game.result_numeric == 0.0 and role == "black"):
                s.wins += 1
            elif game.result_numeric == 0.5:
                s.draws += 1
            else:
                s.losses += 1

    # Per-move stats come from the moves log
    for game in games:
        for move in game.moves:
            s_opt = stats.get(move.player)
            if s_opt is None:
                continue
            s2: PlayerStats = s_opt
            s2.moves_played += 1
            if move.llm_latency_ms > 0:
                s2.total_latency_ms += move.llm_latency_ms
                s2.latency_samples += 1
            if move.llm_tokens_in is not None:
                s2.tokens_in_total += int(move.llm_tokens_in)
                s2.tokens_recorded_moves += 1
            if move.llm_tokens_out is not None:
                s2.tokens_out_total += int(move.llm_tokens_out)
                if move.llm_tokens_in is None:
                    s2.tokens_recorded_moves += 1
            # Captures / checks require board context, recompute from FEN if present.
            if move.fen_before and move.move_uci:
                try:
                    import chess

                    board = chess.Board(move.fen_before)
                    chess_move = chess.Move.from_uci(move.move_uci)
                    if chess_move in board.legal_moves:
                        if board.is_capture(chess_move):
                            s2.captures += 1
                        if board.gives_check(chess_move):
                            s2.checks += 1
                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                    continue
    return stats


def _aggregate_pairings(games: Iterable[GameRecord]) -> list[PairingResult]:
    pairings: dict[tuple[str, str], PairingResult] = {}
    for game in games:
        if not game.white_player or not game.black_player:
            continue
        key = (game.white_player, game.black_player)
        p = pairings.setdefault(key, PairingResult(white=game.white_player, black=game.black_player))
        p.games += 1
        p.total_moves += game.total_moves
        if game.result_numeric == 1.0:
            p.white_wins += 1
        elif game.result_numeric == 0.0:
            p.black_wins += 1
        else:
            p.draws += 1
    return sorted(pairings.values(), key=lambda p: (p.white, p.black))


def _providers_seen(stats: dict[str, PlayerStats]) -> list[str]:
    providers: set[str] = set()
    for name in stats:
        if ":" in name:
            providers.add(name.split(":", 1)[0])
    return sorted(providers)


def load_run(run_dir: str | Path) -> RunSummary | None:
    """Load a single benchmark run directory.

    Returns ``None`` if the directory has no games.jsonl or moves.jsonl
    (so the caller can show "no real data yet" instead of fabricating).
    """
    path = Path(run_dir)
    if not path.is_dir():
        return None

    games_raw = _read_jsonl(path / "games.jsonl")
    moves_raw = _read_jsonl(path / "moves.jsonl")
    if not games_raw and not moves_raw:
        return None

    games = _build_games(games_raw, moves_raw)
    player_stats = _aggregate_player_stats(games)
    pairings = _aggregate_pairings(games)

    summary = _read_summary(path / "summary.json")
    total_games = int(summary.get("total_games", len(games)) or len(games))
    total_moves = int(summary.get("total_moves", sum(g.total_moves for g in games)) or 0)
    total_duration = float(summary.get("total_duration_sec", sum(g.game_duration_sec for g in games)) or 0.0)
    timestamp_utc = (
        summary.get("timestamp_utc")
        or (games[0].timestamp_utc if games else None)
        or _mtime_utc(path)
    )

    return RunSummary(
        run_id=path.name,
        run_dir=path.resolve(),
        timestamp_utc=timestamp_utc,
        config=summary.get("config", {}),
        total_games=total_games,
        total_moves=total_moves,
        total_duration_sec=total_duration,
        games=games,
        player_stats=player_stats,
        pairings=pairings,
        providers_seen=_providers_seen(player_stats),
        provenance=summary.get("provenance", {}),
    )


def _mtime_utc(path: Path) -> str:
    try:
        return datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z"
    except OSError:
        return ""


def list_runs(runs_root: str | Path = "runs") -> list[RunSummary]:
    """Load all runs under ``runs_root``, newest first.

    Skips directories with no real benchmark data (no games.jsonl or
    moves.jsonl). Does not fabricate summaries for empty runs.
    """
    out: list[RunSummary] = []
    for d in list_run_dirs(runs_root):
        try:
            summary = load_run(d)
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("Failed to load run %s: %s", d, exc)
            continue
        if summary is not None:
            out.append(summary)
    return out


# ---------------------------------------------------------------------------
# Aggregated leaderboard across all runs
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardRow:
    player: str
    games: int
    wins: int
    losses: int
    draws: int
    score_pct: float | None
    avg_latency_ms: float | None
    tokens_in: int
    tokens_out: int

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws


def aggregate_leaderboard(runs: list[RunSummary]) -> list[LeaderboardRow]:
    """Aggregate per-player totals across all runs."""
    acc: dict[str, LeaderboardRow] = {}
    for run in runs:
        for name, ps in run.player_stats.items():
            row = acc.setdefault(
                name,
                LeaderboardRow(
                    player=name,
                    games=0,
                    wins=0,
                    losses=0,
                    draws=0,
                    score_pct=None,
                    avg_latency_ms=None,
                    tokens_in=0,
                    tokens_out=0,
                ),
            )
            row.games += ps.games_played
            row.wins += ps.wins
            row.losses += ps.losses
            row.draws += ps.draws
            row.tokens_in += ps.tokens_in_total
            row.tokens_out += ps.tokens_out_total

    # Latency needs weighted averaging
    weighted: dict[str, tuple[float, int]] = dict.fromkeys(acc, (0.0, 0))
    for run in runs:
        for name, ps in run.player_stats.items():
            if ps.avg_latency_ms is None:
                continue
            cur_avg, cur_n = weighted[name]
            weighted[name] = (cur_avg + ps.avg_latency_ms * ps.latency_samples, cur_n + ps.latency_samples)
    for name, (total, n) in weighted.items():
        if n > 0:
            acc[name].avg_latency_ms = total / n

    for row in acc.values():
        if row.games > 0:
            row.score_pct = 100.0 * row.score / row.games

    return sorted(acc.values(), key=lambda r: r.score_pct or 0.0, reverse=True)


# ---------------------------------------------------------------------------
# Rich Analytics & Metric Computations
# ---------------------------------------------------------------------------


def compute_model_performance_matrix(run: RunSummary) -> list[dict[str, Any]]:
    """Compute comprehensive per-model metrics matrix across a benchmark run."""
    import math

    player_moves: dict[str, list[MoveRecord]] = defaultdict(list)
    for game in run.games:
        for move in game.moves:
            player_moves[move.player].append(move)

    matrix: list[dict[str, Any]] = []
    for player_name, ps in run.player_stats.items():
        moves = player_moves.get(player_name, [])
        losses = [m.cp_loss for m in moves if m.cp_loss is not None]
        blunders = sum(1 for m in moves if m.move_quality == "blunder")
        inaccuracies = sum(1 for m in moves if m.move_quality in ("inaccuracy", "mistake"))
        retries = sum(m.validation_retries + m.illegal_attempts_count for m in moves)

        acpl = float(sum(losses) / len(losses)) if losses else None
        accuracy_pct = float(sum(100.0 * math.exp(-0.005 * loss) for loss in losses) / len(losses)) if losses else None
        blunder_rate = float(100.0 * blunders / len(moves)) if moves else 0.0
        inaccuracy_rate = float(100.0 * inaccuracies / len(moves)) if moves else 0.0
        illegal_rate = float(100.0 * retries / len(moves)) if moves else 0.0

        matrix.append({
            "Player": player_name,
            "Games": ps.games_played,
            "Win Rate %": round(100.0 * ps.score / ps.games_played, 1) if ps.games_played > 0 else 0.0,
            "ACPL": round(acpl, 1) if acpl is not None else None,
            "Accuracy %": round(accuracy_pct, 1) if accuracy_pct is not None else None,
            "Blunder Rate %": round(blunder_rate, 1),
            "Inaccuracy Rate %": round(inaccuracy_rate, 1),
            "Illegal/Retry Rate %": round(illegal_rate, 1),
            "Avg Latency (s)": round(ps.avg_latency_ms / 1000.0, 2) if ps.avg_latency_ms else None,
            "Tokens Out Total": ps.tokens_out_total,
        })

    return sorted(matrix, key=lambda x: x["Accuracy %"] or 0.0, reverse=True)


def compute_phase_breakdown(run: RunSummary) -> dict[str, dict[str, float | None]]:
    """Aggregate Average Centipawn Loss (ACPL) and Blunder % segmented by board phase."""
    phase_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    phase_blunders: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for game in run.games:
        for move in game.moves:
            phase = move.game_phase or "middlegame"
            player = move.player
            phase_counts[player][phase] += 1

            if move.cp_loss is not None:
                phase_data[player][phase].append(move.cp_loss)
            if move.move_quality == "blunder":
                phase_blunders[player][phase] += 1

    result: dict[str, dict[str, float | None]] = {}
    for player, phases in phase_data.items():
        result[player] = {}
        for phase, losses in phases.items():
            result[player][f"{phase}_acpl"] = round(sum(losses) / len(losses), 1) if losses else None
            cnt = phase_counts[player][phase]
            blunders = phase_blunders[player][phase]
            result[player][f"{phase}_blunder_pct"] = round(100.0 * blunders / cnt, 1) if cnt > 0 else 0.0

    return result


def compute_thinking_quality_correlation(run: RunSummary) -> list[dict[str, Any]]:
    """Extract (thinking_words, cp_loss, player, move_san) data points for analysis."""
    points: list[dict[str, Any]] = []
    for game in run.games:
        for move in game.moves:
            if move.thinking_words is not None and move.cp_loss is not None:
                points.append({
                    "Player": move.player,
                    "Thinking Words": move.thinking_words,
                    "Centipawn Loss": move.cp_loss,
                    "Move": move.move_san,
                    "Quality": move.move_quality or "unknown",
                })
    return points


def format_provenance_for_ui(run: RunSummary) -> dict[str, Any]:
    """Format provenance data for UI display with custom-strategy indicators and short hashes.

    Returns:
        Dict mapping player specs to their formatted provenance info by color
    """
    if not hasattr(run, 'provenance') or not run.provenance:
        return {}

    formatted: dict[str, dict[str, dict[str, Any]]] = {}
    for player_spec, color_data in run.provenance.items():
        formatted[player_spec] = {}
        for color, provenance_info in color_data.items():
            # Extract strategy hash and create short version
            strategy_hash = provenance_info.get("strategy_hash", "")
            short_hash = strategy_hash[:8] if strategy_hash else ""

            # Determine if this is a custom strategy (not preset/followback)
            source = provenance_info.get("source", "preset")
            is_custom = source == "custom"

            # Determine if fallback was used
            used_fallback = provenance_info.get("used_fallback", False)

            formatted[player_spec][color] = {
                "strategy_hash": strategy_hash,
                "short_hash": short_hash,
                "is_custom": is_custom,
                "used_fallback": used_fallback,
                "source": source,
                "system_prompt": provenance_info.get("system_prompt"),
                "turn_prompt": provenance_info.get("turn_prompt"),
            }
    return formatted


def compute_retry_resilience(run: RunSummary) -> dict[str, Any]:
    """Compute retry recovery rates and error breakdown per model."""
    resilience: dict[str, dict[str, int]] = defaultdict(lambda: {"clean_first_try": 0, "recovered_retries": 0, "failed_retries": 0})
    for game in run.games:
        for move in game.moves:
            player = move.player
            retries = move.validation_retries + move.illegal_attempts_count
            if retries == 0:
                resilience[player]["clean_first_try"] += 1
            else:
                resilience[player]["recovered_retries"] += 1

    out: dict[str, Any] = {}
    for player, counts in resilience.items():
        total = counts["clean_first_try"] + counts["recovered_retries"] + counts["failed_retries"]
        clean_pct = round(100.0 * counts["clean_first_try"] / total, 1) if total > 0 else 100.0
        out[player] = {
            "First-Try Clean %": clean_pct,
            "Clean Moves": counts["clean_first_try"],
            "Recovered Moves": counts["recovered_retries"],
            "Total Moves": total,
        }
    return out
