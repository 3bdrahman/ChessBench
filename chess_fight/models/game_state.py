"""Game state and move data structures."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameMove:
    player: str
    move: str
    timestamp: float
    is_capture: bool
    is_check: bool
    reasoning: str | None = None


@dataclass
class GameStats:
    total_moves: int = 0
    capture_moves: int = 0
    check_moves: int = 0
    game_duration: float = 0
    winner: str | None = None
    termination_reason: str = "unknown"


@dataclass
class GameSummary:
    """Complete game summary for benchmark results."""
    game_id: str = ""
    white_player: str = ""
    black_player: str = ""
    white_provider: str = ""
    black_provider: str = ""
    opening_eco: str | None = None
    opening_name: str | None = None
    opening_fen: str = ""
    result: str = "*"
    result_numeric: float = 0.5
    total_moves: int = 0
    game_duration_sec: float = 0.0
    timestamp_utc: str = ""
    moves: list[Any] = field(default_factory=list)
    termination_reason: str = "unknown"


@dataclass
class PlayerStats:
    """Aggregated statistics for a single player."""
    name: str = ""
    games_played: int = 0
    games_as_white: int = 0
    games_as_black: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score: float = 0.0
    score_pct: float = 0.0
    moves_played: int = 0
    captures: int = 0
    checks: int = 0
    avg_latency_ms: float = 0.0
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    total_cp_loss: float | None = None
    avg_cp_loss: float | None = None
    blunder_count: int | None = None
    mistake_count: int | None = None
    inaccuracy_count: int | None = None
    best_move_pct: float | None = None
    avg_thinking_chars: float | None = None
    thinking_quality_score: float | None = None


@dataclass
class PairingStats:
    """Statistics for a pairing (two players)."""
    white: str = ""
    black: str = ""
    games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_moves: int = 0
