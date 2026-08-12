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

    def test_extract_move_freeform_thinking(self):
        """Test extraction when model uses untagged thinking process."""
        board = chess.Board("Q1bqkbnr/p2p1pp1/8/4p3/4P2p/2P2N2/P1P2PPP/RNBQKB1R b KQk - 0 8")
        text = (
            "Here's a thinking process:\n"
            "1. Analyze position:\n"
            "   - FEN: Q1bqkbnr/p2p1pp1/8/4p3/4P2p/2P2N2/P1P2PPP/RNBQKB1R b KQk - 0 8\n"
            "   - Turn: Black\n"
            "   - White pieces: Q1bqkbnr\n"
            "Candidate moves: d7d6, g8f6, f7f6\n"
            "I will play d7d6\n"
            "<move>d7d6</move>"
        )
        from chess_fight.move_parser import parse_move
        parsed = parse_move(text, board)
        assert parsed.uci == "d7d6"

    def test_parse_move_san_in_tag(self):
        """Test parsing SAN inside move tag."""
        board = chess.Board("Q1bqkbnr/p2p1pp1/8/4p3/4P2p/2P2N2/P1P2PPP/RNBQKB1R b KQk - 0 8")
        text = "Thinking about the game...\n<move>d6</move>"
        from chess_fight.move_parser import parse_move
        parsed = parse_move(text, board)
        assert parsed.uci == "d7d6"

    def test_parse_move_prefer_first_legal_uci(self):
        """First legal UCI mention wins over later alternatives mentioned."""
        board = chess.Board()
        from chess_fight.move_parser import parse_move
        result = parse_move("I think the best move would be e2e4, but I could also play d2d4", board)
        assert result.uci == "e2e4"

    def test_parse_move_rejects_alternative_clause_in_header(self):
        """The 'could also play X' clause must not steal the move-header slot."""
        board = chess.Board()
        from chess_fight.move_parser import parse_move
        text = "I'm confident in e2e4 but I could also play d2d4 as an alternative."
        result = parse_move(text, board)
        assert result.uci == "e2e4"

    def test_parse_move_keeps_inner_text_when_move_tag_unparseable(self):
        """A <move> tag whose contents cannot be parsed directly lets the
        fallback scanners run on the inner text so the model's stated move is
        still extracted."""
        board = chess.Board()
        board.push_uci("e2e4")  # Black to move
        from chess_fight.move_parser import parse_move
        text = "<thinking> analysis </thinking><move>final answer: e7e5</move>"
        result = parse_move(text, board)
        assert result.uci == "e7e5"

    def test_parse_move_castle_queenside_not_kingside(self):
        r"""`O-O-O` (queenside) and `O-O` (kingside) must produce the right
        castling UCI. The kingside regex `\bO[-\s]?O\b` matches inside `O-O-O`,
        so checking queenside first is required."""
        from chess_fight.move_parser import parse_move
        board = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')  # both castles available
        result = parse_move('O-O-O', board)
        assert result.uci == 'e1c1'
        result = parse_move('O-O', board)
        assert result.uci == 'e1g1'


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
