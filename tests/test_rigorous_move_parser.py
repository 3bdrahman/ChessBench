"""Rigorous property-based and fuzz testing for move parsing engine."""

import random
import string
import chess
import pytest

from chessbench.move_parser import extract_move, parse_move, validate_move


def _generate_random_board(depth: int = 15) -> chess.Board:
    """Generate a random legal chess board by making random legal moves."""
    board = chess.Board()
    for _ in range(random.randint(1, depth)):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
    return board


def _generate_noisy_llm_text(move_str: str, format_type: str = "tagged") -> str:
    """Wrap a move string in realistic, noisy LLM output noise."""
    garbage_before = "".join(random.choices(string.ascii_letters + " \n\t.,;:?!", k=random.randint(20, 100)))
    garbage_after = "".join(random.choices(string.ascii_letters + " \n\t.,;:?!", k=random.randint(20, 100)))
    
    if format_type == "tagged":
        return f"{garbage_before}\n<thinking>\nAnalyzing board...\n</thinking>\n<move>{move_str}</move>\n{garbage_after}"
    elif format_type == "san":
        return f"{garbage_before}\nI think the move to play is {move_str}. This controls the center.\n{garbage_after}"
    elif format_type == "markdown_code":
        return f"{garbage_before}\n```\n{move_str}\n```\n{garbage_after}"
    elif format_type == "bold":
        return f"{garbage_before}\nMy choice is **{move_str}**.\n{garbage_after}"
    else:
        return f"{garbage_before} {move_str} {garbage_after}"


class TestRigorousMoveParser:
    """Property-based tests ensuring move parser invariants hold under random noise."""

    def test_parser_never_raises_exception_on_arbitrary_gibberish(self):
        """Fuzz testing: parser must NEVER crash or raise unhandled exceptions on random text."""
        board = chess.Board()
        for seed in range(50):
            random.seed(seed)
            random_length = random.randint(0, 500)
            arbitrary_text = "".join(chr(random.randint(32, 126)) for _ in range(random_length))
            try:
                result = parse_move(arbitrary_text, board)
                assert result is not None
            except Exception as exc:
                pytest.fail(f"parse_move raised unexpected exception on fuzz input: {exc!r}")

    def test_parsed_move_is_always_100pct_legal(self):
        """Property: If parse_move returns a valid move, it MUST be legal in the position."""
        for seed in range(50):
            random.seed(seed)
            board = _generate_random_board(depth=15)
            if board.is_game_over() or not list(board.legal_moves):
                continue
            
            legal_moves = list(board.legal_moves)
            chosen_move = random.choice(legal_moves)
            
            # Format in random noise style
            format_style = random.choice(["tagged", "markdown_code", "bold"])
            move_repr = chosen_move.uci()
            noisy_text = _generate_noisy_llm_text(move_repr, format_type=format_style)
            
            result = parse_move(noisy_text, board)
            if result.uci is not None:
                parsed_move = chess.Move.from_uci(result.uci)
                assert parsed_move in board.legal_moves, (
                    f"Parsed move {result.uci} is illegal on board FEN: {board.fen()}"
                )

    def test_move_tags_always_take_precedence(self):
        """Property: Explicit <move>UCI</move> tags MUST override earlier distractor text."""
        board = chess.Board()
        
        distractor = "e2e4"
        actual = "d2d4"
        
        text = f"I am considering playing {distractor} because it controls e5. However, after deep analysis, my final decision is:\n<move>{actual}</move>"
        result = parse_move(text, board)
        
        assert result.uci == actual

    def test_extract_move_legal_filter_boundary(self):
        """Property: extract_move with legal_moves filter never returns illegal move."""
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        
        # 'e2e5' is an invalid/illegal move on initial board
        result = extract_move("I play e2e5", legal_moves)
        assert result is None
        
        # 'e2e4' is legal
        result = extract_move("I play e2e4", legal_moves)
        assert result == "e2e4"

    def test_promotion_parsing_all_pieces(self):
        """Property: Pawn promotion to Queen, Rook, Bishop, and Knight parses correctly."""
        promotion_fen = "8/P7/8/8/8/8/8/8 w - - 0 1"
        board = chess.Board(promotion_fen)
        
        for piece in ["q", "r", "b", "n"]:
            text = f"I will promote my pawn: <move>a7a8{piece}</move>"
            result = parse_move(text, board)
            assert result.uci == f"a7a8{piece}"

    def test_san_ambiguity_resolution(self):
        """Property: Disambiguated SAN moves (e.g. Rae1) map to exact UCI move."""
        # FEN with open 1st rank and 2 rooks on a1 and f1
        fen = "r2q1rk1/ppp2ppp/8/8/8/8/PPP2PPP/R4RK1 w - - 0 1"
        board = chess.Board(fen)
        result = parse_move("<move>Rae1</move>", board)
        assert result.uci == "a1e1"
