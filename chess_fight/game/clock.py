"""Game clock with Fischer increment support."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GameClock:
    """Game clock tracking time per player with optional Fischer increment.

    Time is tracked in milliseconds internally.
    """
    white_ms: int
    black_ms: int
    increment_ms: int = 0
    _start_time_ms: int = 0
    _current_player_white: bool = True
    _last_update_ms: int = 0

    def __post_init__(self) -> None:
        self._last_update_ms = 0
        self._current_player_white = True

    @classmethod
    def from_seconds(cls, seconds_per_move: int, increment_seconds: int = 0) -> GameClock:
        """Create clock from seconds per move and increment."""
        return cls(
            white_ms=seconds_per_move * 1000,
            black_ms=seconds_per_move * 1000,
            increment_ms=increment_seconds * 1000,
        )

    def start_turn(self, is_white: bool, current_time_ms: int = 0) -> None:
        """Mark the start of a player's turn."""
        self._current_player_white = is_white
        self._last_update_ms = current_time_ms

    def end_turn(self, is_white: bool, current_time_ms: int) -> None:
        """Mark the end of a player's turn and apply increment."""
        elapsed = current_time_ms - self._last_update_ms
        if is_white:
            self.white_ms = max(0, self.white_ms - elapsed)
        else:
            self.black_ms = max(0, self.black_ms - elapsed)

        # Apply increment after the move
        if self.increment_ms > 0:
            if is_white:
                self.white_ms += self.increment_ms
            else:
                self.black_ms += self.increment_ms

        self._last_update_ms = current_time_ms

    def remaining_ms(self, is_white: bool) -> int:
        """Get remaining time for a player in milliseconds."""
        if is_white:
            return max(0, self.white_ms)
        return max(0, self.black_ms)

    def remaining_seconds(self, is_white: bool) -> float:
        """Get remaining time for a player in seconds."""
        return self.remaining_ms(is_white) / 1000.0

    def is_time_up(self, is_white: bool) -> bool:
        """Check if a player's time has expired."""
        return self.remaining_ms(is_white) <= 0

    def get_state(self) -> dict[str, int]:
        """Get clock state for logging/UI."""
        return {
            "white_ms": self.white_ms,
            "black_ms": self.black_ms,
            "increment_ms": self.increment_ms,
        }

    def format_time(self, is_white: bool) -> str:
        """Format remaining time as MM:SS or HH:MM:SS."""
        ms = self.remaining_ms(is_white)
        total_seconds = ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
