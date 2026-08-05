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
            "MOVE HISTORY ANALYSIS",
            "TACTICAL OPPORTUNITIES",
            "Material Status",
            "POSITION EVALUATION",
            "DEFENSE ANALYSIS",
            "VULNERABILITY ANALYSIS",
            "Legal moves by priority",
            "WINNING CAPTURES/CHECKS",
            "DEVELOPING MOVES",
            "POSITIONAL MOVES",
            "Decision Priority",
            "Best move given state of the game",
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

        # Should contain move categories (using actual template text)
        assert "WINNING CAPTURES" in prompt
        assert "DEVELOPING MOVES" in prompt
        assert "POSITIONAL MOVES" in prompt

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
        assert len(prompt) > 1000  # Should be substantial

        # Should mention captures
        assert "capture" in prompt.lower() or "CAPTURE" in prompt

    def test_prompt_no_placeholder_leakage(self):
        """Test that no template placeholders remain unsubstituted."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        # No unrendered placeholders should remain
        # Check for common placeholder patterns that shouldn't appear
        bad_patterns = [
            "{color}", "{position_repetitions}", "{stagnation_status}",
            "{position_progress}", "{material_tension}", "{position_dynamism}",
            "{development_score}", "{capture_analysis}", "{defense_analysis}",
            "{vulnerability_analysis}", "{material_count}", "{material_balance}",
            "{center_control}", "{development_status}", "{king_safety}",
            "{undefended_pieces}", "{exposed_pieces}", "{ascii_board}",
            "{forcing_moves}", "{developing_moves}", "{positional_moves}",
        ]

        for pattern in bad_patterns:
            assert pattern not in prompt, f"Unrendered placeholder: {pattern}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
