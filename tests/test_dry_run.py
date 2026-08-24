"""Unit tests for dry-run scoring logic."""
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import chess
import pytest

from chessbench.benchmark.evaluator import EvaluationResult, StockfishEvaluator
from chessbench.constants import MATE_THREAT_SCORE
from chessbench.ui.dry_run_scoring import (
    DryRunCell,
    DryRunResult,
    aggregate_results,
    compute_centipawn_loss,
    format_cp_loss,
    parse_fen_lines,
)


def test_parse_fen_lines_valid():
    text = "\n".join([
        chess.STARTING_FEN,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",
        "",  # empty line
        "invalid fen",
    ])
    boards, errors = parse_fen_lines(text)
    assert len(boards) == 2
    assert boards[0].fen() == chess.STARTING_FEN
    assert boards[1].fen() == "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
    assert len(errors) == 1
    assert errors[0][0] == 4  # line number
    assert "Invalid FEN syntax" in errors[0][1]


def test_parse_fen_lines_invalid_position():
    text = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4 5"  # extra field
    boards, errors = parse_fen_lines(text)
    assert len(boards) == 0
    assert len(errors) == 1
    assert "Invalid FEN syntax" in errors[0][1]


def test_format_cp_loss():
    assert format_cp_loss(12) == "12"
    assert format_cp_loss(-5) == "-5"
    assert format_cp_loss(None) == "--"
    assert format_cp_loss(0) == "0"


def test_aggregate_results_empty():
    agg = aggregate_results([])
    assert agg["avg_cp_loss"] is None
    assert agg["accuracy"] is None
    assert agg["illegal_count"] == 0
    assert agg["mean_latency_ms"] is None
    assert agg["total_tokens"] is None


def test_aggregate_results_with_data():
    cells = [
        DryRunCell(fen=chess.STARTING_FEN, move_uci="e2e4", cp_loss=10, latency_ms=100, tokens_in=5, tokens_out=3),
        DryRunCell(fen=chess.STARTING_FEN, move_uci="d2d4", cp_loss=20, latency_ms=200, tokens_in=4, tokens_out=2),
        DryRunCell(fen=chess.STARTING_FEN, move_uci="g1f3", cp_loss=None, latency_ms=50, tokens_in=3, tokens_out=1, error="illegal move"),
    ]
    agg = aggregate_results(cells)
    assert agg["avg_cp_loss"] == 15.0  # (10+20)/2
    assert agg["accuracy"] == 100.0  # both valid moves have cp_loss <= 50
    assert agg["illegal_count"] == 1
    assert agg["mean_latency_ms"] == 150.0  # (100+200)/2 = 150 (only valid cells)
    assert agg["total_tokens"] == 14  # (5+3)+(4+2) = 14


def test_dry_run_result_methods():
    cells = [
        DryRunCell(fen=chess.STARTING_FEN, move_uci="e2e4", cp_loss=10),
        DryRunCell(fen=chess.STARTING_FEN, move_uci="d2d4", cp_loss=20),
        DryRunCell(fen=chess.STARTING_FEN, move_uci="g1f3", cp_loss=30),
    ]
    result = DryRunResult(label="test", cells=cells)
    assert result.avg_cp_loss() == 20.0
    assert result.accuracy(threshold=25) == 66.66666666666666  # 2 out of 3 <=25
    assert result.illegal_count() == 0
    assert result.mean_latency_ms() is None  # no latency set
    assert result.total_tokens() is None  # no tokens set


@patch("chessbench.ui.dry_run_scoring.StockfishEvaluator")
def test_compute_centipawn_loss(mock_eval_class):
    # Create a mock evaluator instance
    mock_eval = Mock(spec=StockfishEvaluator)
    # Make the mock an async context manager
    mock_eval.__aenter__ = AsyncMock(return_value=mock_eval)
    mock_eval.__aexit__ = AsyncMock(return_value=False)
    # We'll define the evaluate method later per test case

    # Helper to create a mock evaluator with given return values for two positions
    def create_mock_evaluator(cp_before: int, cp_after: int):
        # Compute the board after e2e4 from startpos
        board_before = chess.Board(chess.STARTING_FEN)
        move = chess.Move.from_uci("e2e4")
        board_after = board_before.copy()
        board_after.push(move)
        async def mock_evaluate(board):
            if board.fen() == board_before.fen():
                return EvaluationResult(cp_score=cp_before, mate_in=None)
            elif board.fen() == board_after.fen():
                return EvaluationResult(cp_score=cp_after, mate_in=None)
            else:
                # For any other board, return 0 to avoid interfering with other tests
                return EvaluationResult(cp_score=0, mate_in=None)
        mock_eval.evaluate = mock_evaluate
        return mock_eval

    # Test a move that does not change the evaluation (should yield 0 loss)
    board_before = chess.Board(chess.STARTING_FEN)
    move = chess.Move.from_uci("e2e4")
    board_after = board_before.copy()
    board_after.push(move)
    with patch("chessbench.ui.dry_run_scoring.StockfishEvaluator", return_value=create_mock_evaluator(0, 0)):
        loss = compute_centipawn_loss(chess.STARTING_FEN, "e2e4", mock_eval)
        # Before: cp_score=0 -> score_before_pov = 0 (white to move)
        # After: cp_score=0 -> score_after_pov = 0 (white to move)
        # loss = 0 - 0 = 0
        assert loss == 0

    # Test with a known gain/loss: before 10, after 5 -> loss = 5
    with patch("chessbench.ui.dry_run_scoring.StockfishEvaluator", return_value=create_mock_evaluator(10, 5)):
        loss = compute_centipawn_loss(chess.STARTING_FEN, "e2e4", mock_eval)
        # Before: cp_score=10 -> score_before_pov = 10 (white)
        # After: cp_score=5 -> score_after_pov = 5 (white)
        # loss = 10 - 5 = 5 (positive loss means the move worsened the position by 5 cp)
        assert loss == 5

    # Test with mate
    board_before = chess.Board(chess.STARTING_FEN)
    move = chess.Move.from_uci("e2e4")
    board_after = board_before.copy()
    board_after.push(move)
    async def mock_evaluate_mate(board):
        fen = board.fen()
        if fen == board_before.fen():
            return EvaluationResult(cp_score=None, mate_in=1)  # white mates in 1
        elif fen == board_after.fen():
            return EvaluationResult(cp_score=None, mate_in=-1)  # black mates in 1
        else:
            return EvaluationResult(cp_score=0, mate_in=None)
    mock_eval.evaluate = mock_evaluate_mate
    mock_eval.__aenter__ = AsyncMock(return_value=mock_eval)
    mock_eval.__aexit__ = AsyncMock(return_value=False)

    with patch("chessbench.ui.dry_run_scoring.StockfishEvaluator", return_value=mock_eval):
        loss = compute_centipawn_loss(chess.STARTING_FEN, "e2e4", mock_eval)
        # Before: mate_in=1 -> cp_score = MATE_THREAT_SCORE
        # After: mate_in=-1 -> cp_score = -MATE_THREAT_SCORE
        # score_before_pov = MATE_THREAT_SCORE
        # score_after_pov = -MATE_THREAT_SCORE
        # loss = MATE_THREAT_SCORE - (-MATE_THREAT_SCORE) = 2 * MATE_THREAT_SCORE
        assert loss == 2 * MATE_THREAT_SCORE


if __name__ == "__main__":
    pytest.main([__file__])
