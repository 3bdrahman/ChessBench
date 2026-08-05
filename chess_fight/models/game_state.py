"""Game state and move data structures."""

from dataclasses import dataclass


@dataclass
class GameMove:
    player: str
    move: str
    timestamp: float
    is_capture: bool
    is_check: bool


@dataclass
class GameStats:
    total_moves: int = 0
    capture_moves: int = 0
    check_moves: int = 0
    game_duration: float = 0
    winner: str | None = None
