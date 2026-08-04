"""Chess AI base classes and types."""

from abc import ABC, abstractmethod
from typing import Optional, List
import chess


class ChessAI(ABC):
    """Abstract base class for chess AI implementations."""
    
    def __init__(self):
        self.move_history = []
        self.position_history = set()
        self.stagnation_threshold = 3
    
    @abstractmethod
    async def _get_move_from_model(self, fen: str) -> str:
        """Get raw move from the specific model implementation."""
        pass
    
    def _validate_move(self, move_str: str, board: chess.Board) -> str:
        """Validate and clean a move string."""
        move_str = move_str.strip().lower()
        
        # Remove common response artifacts from the start
        prefixes = ["move:", "i choose", "my move is", "play", "'", '"', "`"]
        for prefix in prefixes:
            if move_str.startswith(prefix):
                move_str = move_str[len(prefix):].strip()
        
        # Remove trailing artifacts
        suffixes = ["'", '"', "`", ".", ",", ":", ";"]
        for suffix in suffixes:
            if move_str.endswith(suffix):
                move_str = move_str[:-len(suffix)].strip()
        
        # Basic UCI format validation
        if not (4 <= len(move_str) <= 5):
            raise ValueError(f"Invalid move format: {move_str}")
        
        # Create chess.Move object
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            raise ValueError(f"Invalid UCI format: {move_str}")
        
        # Check if move is legal in current position
        if move not in board.legal_moves:
            legal_moves = [m.uci() for m in board.legal_moves]
            raise ValueError(f"Illegal move {move_str}. Legal moves are: {', '.join(legal_moves)}")
        
        return move_str
    
    async def get_move(self, fen: str) -> str:
        """Get move with position history tracking (async)."""
        board = chess.Board(fen)
        max_retries = 3
        errors = []
        
        for attempt in range(max_retries):
            try:
                move_str = await self._get_move_from_model(fen)
                validated_move = self._validate_move(move_str, board)
                
                # Track the position after making the move
                current_fen = board.fen().split(' ')[0]
                self.move_history.append(current_fen)
                
                return validated_move
            except ValueError as e:
                errors.append(f"Attempt {attempt + 1}: {str(e)}")
                continue
        
        # If we've exhausted retries, make a fallback move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            fallback_move = legal_moves[0].uci()
            current_fen = board.fen().split(' ')[0]
            self.move_history.append(current_fen)
            return fallback_move
        
        raise ValueError(f"Failed to get valid move after {max_retries} attempts. Errors: {'; '.join(errors)}")