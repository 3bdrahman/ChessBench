"""Tests for prompt template rendering."""

import chess
import pytest

from chess_fight.models.chess_ai import ChessAI


class MockChessAI(ChessAI):
    """Mock ChessAI for testing prompt rendering without API keys."""

    def __init__(self):
        super().__init__()
        self.name = "MockAI"

    async def _get_move_from_model(self, fen: str) -> str:
        return "e2e4"


class TestPromptTemplate:
    """Tests for prompt template rendering."""

    def test_prompt_template_renders_without_keyerror(self):
        """Test that _create_prompt renders without KeyError for all placeholders."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        # Should not raise KeyError
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_all_required_sections(self):
        """Test that prompt contains all expected sections."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        required_sections = [
            "[GAME STATE]",
            "[INSTRUCTIONS]",
            "FEN:",
            "Turn:",
            "<think>",
            "<move>",
            "REASONING LEVEL:",
        ]

        for section in required_sections:
            assert section in prompt, f"Missing section: {section}"

    def test_prompt_contains_position_specific_info(self):
        """Test that prompt contains position-specific information."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        # Should contain color
        assert "White" in prompt or "Black" in prompt
        assert "FEN:" in prompt

    def test_prompt_for_different_colors(self):
        """Test prompt generation for both colors."""
        ai = MockChessAI()

        # White to move
        board = chess.Board()
        white_prompt = ai._create_prompt(board.fen())
        assert "White" in white_prompt

        # Black to move
        board.push(chess.Move.from_uci("e2e4"))
        black_prompt = ai._create_prompt(board.fen())
        assert "Black" in black_prompt

    def test_prompt_with_complex_position(self):
        """Test prompt generation with a complex mid-game position."""
        ai = MockChessAI()

        # Complex position with captures available
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        prompt = ai._create_prompt(board.fen())

        assert isinstance(prompt, str)
        assert len(prompt) > 300  # Should be substantial
        assert "FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" in prompt

    def test_prompt_no_placeholder_leakage(self):
        """Test that no template placeholders remain unsubstituted."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        # No unrendered placeholders should remain.
        # Check all variables that the v1_baseline template references.
        bad_patterns = [
            "{color}", "{fen}", "{ascii_board}",
            "{legal_moves_uci}", "{forcing_uci}",
            "{developing_uci}", "{positional_uci}",
            "{legal_moves_annotated}", "{last_move_san}",
            "{move_history_san}", "{white_pieces}", "{black_pieces}",
        ]

        for pattern in bad_patterns:
            assert pattern not in prompt, f"Unrendered placeholder: {pattern}"

    def test_prompt_only_computes_needed_variables(self):
        """Verify dead-weight evaluations are not called for v1_baseline."""
        from unittest.mock import patch

        ai = MockChessAI()
        board = chess.Board()

        # These evaluations are NOT referenced by v1_baseline and should
        # never be called during prompt construction.
        dead_methods = [
            "analyze_defense",
            "analyze_vulnerabilities",
            "analyze_captures",
            "analyze_king_safety",
            "analyze_undefended_pieces",
            "analyze_exposed_pieces",
            "get_material_count",
            "analyze_material_balance",
            "analyze_center_control",
            "analyze_development_status",
            "calculate_development_score",
        ]

        for method_name in dead_methods:
            with patch.object(ai.evaluator, method_name, wraps=getattr(ai.evaluator, method_name)) as mock_method:
                ai._create_prompt(board.fen())
                mock_method.assert_not_called(), f"{method_name} should not be called for v1_baseline"

    def test_create_messages_returns_system_and_user_roles(self):
        """Verify _create_messages returns system and user role ChatMessage list."""
        ai = MockChessAI()
        board = chess.Board()
        messages = ai._create_messages(board.fen())

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "professional chess engine" in messages[0].content
        assert "[GAME STATE]" in messages[1].content

    def test_rich_context_helpers(self):
        """Verify annotated legal moves, last move, move history, and piece locations."""
        ai = MockChessAI()
        board = chess.Board()

        annotated = ai._get_annotated_legal_moves(board)
        assert "e2e4 (e4)" in annotated or "g1f3 (Nf3)" in annotated

        last_move = ai._get_last_move_san(board)
        assert "None" in last_move

        board.push(chess.Move.from_uci("e2e4"))
        last_move = ai._get_last_move_san(board)
        assert "1. e4 (e2e4)" in last_move

        history = ai._get_move_history_san(board)
        assert "1. e4" in history

        w, b = ai._get_piece_locations_str(board)
        assert "King at e1" in w
        assert "King at e8" in b


class TestAnalyzePositionRepetition:
    """Regression tests for _analyze_position_repetition.

    Previously the function returned progress_score=1.0 unconditionally when
    history had fewer than 3 entries, masking early-game stagnation. The fix
    computes unique/total across the last 4 positions regardless of length.
    """

    def test_empty_history_returns_real_score(self):
        ai = MockChessAI()
        result = ai._analyze_position_repetition(chess.Board())
        assert result["progress_score"] == 1.0
        assert result["repetitions"] == 1
        assert result["is_stagnating"] is False

    def test_short_history_returns_real_score_not_always_one(self):
        ai = MockChessAI()
        # After one move, history has 1 entry; previously the function
        # short-circuited to progress_score=1.0 regardless of repetition.
        ai.move_history.append(chess.Board().fen().split(" ")[0])
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        result = ai._analyze_position_repetition(board)
        assert 0.0 < result["progress_score"] <= 1.0
        assert result["is_stagnating"] is False

    def test_repetition_triggers_stagnation_flag(self):
        ai = MockChessAI()
        # Simulate the board returning to the same FEN position.
        starting = chess.Board()
        ai.move_history.append(starting.fen().split(" ")[0])
        ai.move_history.append(starting.fen().split(" ")[0])
        ai.move_history.append(starting.fen().split(" ")[0])
        result = ai._analyze_position_repetition(starting)
        assert result["repetitions"] >= ai.stagnation_threshold
        assert result["is_stagnating"] is True
        assert result["progress_score"] < 1.0

    def test_all_unique_history_has_full_progress(self):
        ai = MockChessAI()
        board = chess.Board()
        ai.move_history.append(board.fen().split(" ")[0])
        board.push(chess.Move.from_uci("e2e4"))
        ai.move_history.append(board.fen().split(" ")[0])
        board.push(chess.Move.from_uci("e7e5"))
        ai.move_history.append(board.fen().split(" ")[0])
        board.push(chess.Move.from_uci("g1f3"))
        result = ai._analyze_position_repetition(board)
        assert result["progress_score"] == 1.0
        assert result["is_stagnating"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
