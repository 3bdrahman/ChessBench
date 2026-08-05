"""Tests for move parser."""

import chess
import pytest

from chess_fight.move_parser import extract_move, validate_move
from tests.fixtures.llm_outputs import LLM_OUTPUTS


class TestExtractMove:
    """Tests for extract_move function."""

    @pytest.mark.parametrize("fixture", LLM_OUTPUTS)
    def test_extract_move_from_fixtures(self, fixture):
        """Test extraction against curated LLM outputs."""
        result = extract_move(fixture["text"])
        assert result == fixture["expected"], f"Failed for {fixture['provider']}/{fixture['model']}: got {result}, expected {fixture['expected']}"

    def test_extract_move_with_legal_moves_filter(self):
        """Test extraction with legal moves validation."""
        board = chess.Board()
        legal_moves = list(board.legal_moves)

        # Valid move in position
        result = extract_move("I will play e2e4", legal_moves)
        assert result == "e2e4"

        # Invalid move (not legal in position)
        result = extract_move("I will play e2e5", legal_moves)
        assert result is None

    def test_extract_move_promotion(self):
        """Test extraction of promotion moves."""
        # Set up a promotion position
        board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
        legal_moves = list(board.legal_moves)

        result = extract_move("Promote with a7a8q", legal_moves)
        assert result == "a7a8q"

    def test_extract_move_strips_thinking(self):
        """Test that thinking blocks are stripped."""
        text = "<thinking>Long analysis here...</thinking>\ne2e4"
        result = extract_move(text)
        assert result == "e2e4"

    def test_extract_move_case_insensitive(self):
        """Test case insensitive extraction."""
        result = extract_move("E2E4")
        assert result == "e2e4"

    def test_extract_move_returns_none_for_no_move(self):
        """Test returns None when no valid move found."""
        result = extract_move("I don't know what to play")
        assert result is None

    def test_extract_move_first_valid(self):
        """Test returns first valid move when multiple present."""
        board = chess.Board()
        legal_moves = list(board.legal_moves)

        result = extract_move("I could play e2e4 or d2d4", legal_moves)
        assert result == "e2e4"


class TestValidateMove:
    """Tests for validate_move function."""

    def test_valid_move(self):
        """Test valid move passes."""
        board = chess.Board()
        result = validate_move("e2e4", board)
        assert result == "e2e4"

    def test_invalid_format(self):
        """Test invalid format raises."""
        board = chess.Board()
        with pytest.raises(ValueError, match="Invalid move format"):
            validate_move("e2", board)

    def test_illegal_move(self):
        """Test illegal move raises."""
        board = chess.Board()
        with pytest.raises(ValueError, match="Illegal move"):
            validate_move("e2e5", board)

    def test_strips_prefixes(self):
        """Test prefixes are stripped."""
        board = chess.Board()
        assert validate_move("move: e2e4", board) == "e2e4"
        assert validate_move("I choose e2e4", board) == "e2e4"
        assert validate_move("my move is e2e4", board) == "e2e4"
        assert validate_move("play e2e4", board) == "e2e4"
        assert validate_move("`e2e4`", board) == "e2e4"
        assert validate_move('"e2e4"', board) == "e2e4"

    def test_strips_trailing_artifacts(self):
        """Test trailing artifacts are stripped (fixes D2)."""
        board = chess.Board()
        # Trailing punctuation/quotes should be stripped
        assert validate_move("e2e4:", board) == "e2e4"
        assert validate_move("e2e4.", board) == "e2e4"
        assert validate_move('e2e4"', board) == "e2e4"
        assert validate_move("e2e4'", board) == "e2e4"
        assert validate_move("e2e4`", board) == "e2e4"
        assert validate_move("e2e4,", board) == "e2e4"
        assert validate_move("e2e4;", board) == "e2e4"

    def test_promotion_move(self):
        """Test promotion move validation."""
        board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
        result = validate_move("a7a8q", board)
        assert result == "a7a8q"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
