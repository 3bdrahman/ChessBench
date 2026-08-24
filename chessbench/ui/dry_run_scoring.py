"""Pure logic for dry-run scoring: FEN parsing, centipawn loss calculation, aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chess

from chessbench.benchmark.evaluator import StockfishEvaluator
from chessbench.constants import MATE_THREAT_SCORE


@dataclass(frozen=True)
class DryRunCell:
    """Result for a single position-candidate pair."""

    fen: str
    move_uci: str
    cp_loss: int | None = None  # centipawn loss, lower is better
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None  # e.g., illegal move, format error, provider error


@dataclass(frozen=True)
class DryRunResult:
    """Aggregate result for a candidate across all positions."""

    label: str
    cells: list[DryRunCell] = field(default_factory=list)

    def avg_cp_loss(self) -> float | None:
        """Average centipawn loss over cells with valid cp_loss (excluding errors)."""
        losses = [
            c.cp_loss for c in self.cells if c.cp_loss is not None and c.error is None
        ]
        if not losses:
            return None
        return sum(losses) / len(losses)

    def accuracy(self, threshold: int = 50) -> float | None:
        """Percentage of moves with cp_loss <= threshold (and no error)."""
        valid = [c for c in self.cells if c.error is None]
        if not valid:
            return None
        within = [c for c in valid if c.cp_loss is not None and c.cp_loss <= threshold]
        return len(within) / len(valid) * 100.0

    def illegal_count(self) -> int:
        """Number of cells with error (illegal/format/provider errors)."""
        return sum(1 for c in self.cells if c.error is not None)

    def mean_latency_ms(self) -> float | None:
        """Average latency over cells with valid latency."""
        latencies = [
            c.latency_ms
            for c in self.cells
            if c.latency_ms is not None and c.error is None
        ]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    def total_tokens(self) -> int | None:
        """Sum of tokens_in + tokens_out over cells with valid tokens."""
        tokens = []
        for c in self.cells:
            if c.error is None:
                if c.tokens_in is not None:
                    tokens.append(c.tokens_in)
                if c.tokens_out is not None:
                    tokens.append(c.tokens_out)
        if not tokens:
            return None
        return sum(tokens)


def parse_fen_lines(text: str) -> tuple[list[chess.Board], list[tuple[int, str]]]:
    """
    Parse FEN strings, one per line.
    Returns (list of valid boards, list of (line_number, error_message)).
    Line numbers are 1-indexed.
    """
    boards: list[chess.Board] = []
    errors: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            board = chess.Board(line)
            if board.is_valid():
                boards.append(board)
            else:
                errors.append((i, f"Invalid FEN position: {line}"))
        except ValueError as e:
            errors.append((i, f"Invalid FEN syntax: {line} ({e})"))
    return boards, errors


def _cp_score_from_evaluator(
    board: chess.Board, evaluator: StockfishEvaluator
) -> int | None:
    """
    Get centipawn score from Stockfish evaluator, converting mate to large centipawn value.
    Returns None if evaluation fails.
    """
    # Note: StockfishEvaluator.evaluate is async, but we are in a synchronous context.
    # We'll need to run the async function in a sync way. However, the dry-run will be
    # called from a background thread in streamlit, so we can use asyncio.run.
    # But to keep this function pure and synchronous, we will assume the caller
    # will handle the async part and pass in a precomputed score? Actually, we
    # need to compute the score inside this module.
    #
    # Let's change approach: we will make this function async and have the caller
    # handle the async execution. However, to keep the pure logic synchronous,
    # we will instead create a helper that uses the evaluator synchronously by
    # running the async function in a new event loop. This is acceptable because
    # the dry-run will be run in a background thread anyway.
    #
    # Alternatively, we can expose an async function and let the caller deal with it.
    # For simplicity, we will make this function synchronous by using asyncio.run
    # inside, assuming we are in a thread with no running event loop.
    #
    # We'll do:
    #   try:
    #       loop = asyncio.get_event_loop()
    #   except RuntimeError:
    #       loop = None
    #   if loop and loop.is_running():
    #       # We are in an async context, we cannot run synchronously. We'll return None
    #       # and let the caller handle it? This is messy.
    #
    # Given the complexity, and since the dry-run will be run in a background thread
    # (like the benchmark), we can assume there is no running event loop and use
    # asyncio.run.
    #
    # We'll import asyncio inside the function to avoid top-level import if not needed.
    import asyncio

    async def _eval() -> int | None:
        async with evaluator:
            result = await evaluator.evaluate(board)
            if result is None:
                return None
            cp_score = result.cp_score
            mate_in = result.mate_in
            if mate_in is not None:
                # Convert mate to centipawn loss: mate in N is a huge advantage.
                # We'll use the sign of mate_in: positive means white mates, negative means black mates.
                # The score from white's perspective: if white is mating, score is large positive.
                # If black is mating, score is large negative.
                return MATE_THREAT_SCORE if mate_in > 0 else -MATE_THREAT_SCORE
            return cp_score

    try:
        return asyncio.run(_eval())
    except Exception:
        return None


def compute_centipawn_loss(
    fen_before: str,
    move_uci: str,
    evaluator: StockfishEvaluator,
) -> int | None:
    """
    Compute centipawn loss for a move given the starting FEN and the move UCI.
    Returns the centipawn loss (integer) or None if evaluation fails.
    """
    try:
        board_before = chess.Board(fen_before)
        if not board_before.is_valid():
            return None
        move = chess.Move.from_uci(move_uci)
        if move not in board_before.legal_moves:
            return None  # illegal move
        board_after = board_before.copy()
        board_after.push(move)
    except Exception:
        return None

    # Get evaluation before and after
    cp_before = _cp_score_from_evaluator(board_before, evaluator)
    cp_after = _cp_score_from_evaluator(board_after, evaluator)
    if cp_before is None or cp_after is None:
        return None

    # Player to move before the move
    player_is_white = board_before.turn == chess.WHITE
    # Score from perspective of the player who made the move
    score_before_pov = cp_before if player_is_white else -cp_before
    score_after_pov = cp_after if player_is_white else -cp_after
    loss = score_before_pov - score_after_pov
    return loss


def aggregate_results(cells: list[DryRunCell]) -> dict[str, Any]:
    """
    Aggregate a list of DryRunCell into summary metrics.
    Returns a dictionary with keys:
        avg_cp_loss, accuracy, illegal_count, mean_latency_ms, total_tokens
    """
    if not cells:
        return {
            "avg_cp_loss": None,
            "accuracy": None,
            "illegal_count": 0,
            "mean_latency_ms": None,
            "total_tokens": None,
        }

    valid_cells = [c for c in cells if c.error is None]
    illegal_count = len(cells) - len(valid_cells)

    # Average cp loss
    losses = [c.cp_loss for c in valid_cells if c.cp_loss is not None]
    avg_cp_loss = sum(losses) / len(losses) if losses else None

    # Accuracy: percentage of moves within 50 cp of best (i.e., cp_loss <= 50)
    accurate = [c for c in valid_cells if c.cp_loss is not None and c.cp_loss <= 50]
    accuracy = (len(accurate) / len(valid_cells) * 100.0) if valid_cells else None

    # Mean latency
    latencies = [c.latency_ms for c in valid_cells if c.latency_ms is not None]
    mean_latency_ms = sum(latencies) / len(latencies) if latencies else None

    # Total tokens (in + out)
    total_tokens_accum = 0
    has_token_data = False
    for c in valid_cells:
        if c.tokens_in is not None:
            total_tokens_accum += c.tokens_in
            has_token_data = True
        if c.tokens_out is not None:
            total_tokens_accum += c.tokens_out
            has_token_data = True
    total_tokens: int | None = total_tokens_accum if has_token_data else None

    return {
        "avg_cp_loss": avg_cp_loss,
        "accuracy": accuracy,
        "illegal_count": illegal_count,
        "mean_latency_ms": mean_latency_ms,
        "total_tokens": total_tokens,
    }


def format_cp_loss(cp_loss: int | None) -> str:
    """Format centipawn loss for display, e.g., '12' or '--'."""
    if cp_loss is None:
        return "--"
    return f"{cp_loss}"
