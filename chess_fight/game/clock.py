"""Game clock with Fischer increment support using timedelta and monotonic timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import time
import chess.engine


class GameClock:
    """Game clock tracking time per player with Fischer increment support.

    Uses `datetime.timedelta` for duration calculations and `time.monotonic()`
    for precise, drift-free interval timing.
    """
    white_time: timedelta
    black_time: timedelta
    increment: timedelta
    _turn_start_monotonic: float | None

    def __init__(
        self,
        white_time: timedelta | float | int,
        black_time: timedelta | float | int,
        increment: timedelta | float | int = 0,
    ) -> None:
        """Initialize clock with timedelta objects or seconds (as int/float)."""
        self.white_time = white_time if isinstance(white_time, timedelta) else timedelta(seconds=float(white_time))
        self.black_time = black_time if isinstance(black_time, timedelta) else timedelta(seconds=float(black_time))
        self.increment = increment if isinstance(increment, timedelta) else timedelta(seconds=float(increment))
        self._turn_start_monotonic = None

    @classmethod
    def from_seconds(cls, seconds_per_move: float | int, increment_seconds: float | int = 0) -> GameClock:
        """Create clock from seconds per move and increment."""
        return cls(
            white_time=timedelta(seconds=float(seconds_per_move)),
            black_time=timedelta(seconds=float(seconds_per_move)),
            increment=timedelta(seconds=float(increment_seconds)),
        )

    @classmethod
    def from_timedelta(
        cls,
        white_time: timedelta,
        black_time: timedelta,
        increment: timedelta = timedelta(seconds=0),
    ) -> GameClock:
        """Create clock directly from timedelta objects."""
        return cls(white_time=white_time, black_time=black_time, increment=increment)

    @property
    def white_ms(self) -> int:
        return int(self.white_time.total_seconds() * 1000)

    @white_ms.setter
    def white_ms(self, val: int) -> None:
        self.white_time = timedelta(milliseconds=val)

    @property
    def black_ms(self) -> int:
        return int(self.black_time.total_seconds() * 1000)

    @black_ms.setter
    def black_ms(self, val: int) -> None:
        self.black_time = timedelta(milliseconds=val)

    @property
    def increment_ms(self) -> int:
        return int(self.increment.total_seconds() * 1000)

    @increment_ms.setter
    def increment_ms(self, val: int) -> None:
        self.increment = timedelta(milliseconds=val)

    def start_turn(self, is_white: bool, current_time_ms: int = 0) -> None:
        """Mark the start of a player's turn using monotonic timing."""
        self._turn_start_monotonic = time.monotonic()

    def end_turn(self, is_white: bool, current_time_ms: int = 0) -> None:
        """Mark the end of a player's turn and apply increment."""
        if self._turn_start_monotonic is not None:
            elapsed_seconds = time.monotonic() - self._turn_start_monotonic
            elapsed = timedelta(seconds=elapsed_seconds)
        else:
            elapsed = timedelta(seconds=0)

        if is_white:
            self.white_time = max(timedelta(seconds=0), self.white_time - elapsed + self.increment)
        else:
            self.black_time = max(timedelta(seconds=0), self.black_time - elapsed + self.increment)

        self._turn_start_monotonic = None

    def remaining_timedelta(self, is_white: bool) -> timedelta:
        """Get remaining time as a timedelta object."""
        rem = self.white_time if is_white else self.black_time
        return max(timedelta(seconds=0), rem)

    def remaining_ms(self, is_white: bool) -> int:
        """Get remaining time for a player in milliseconds."""
        return int(self.remaining_timedelta(is_white).total_seconds() * 1000)

    def remaining_seconds(self, is_white: bool) -> float:
        """Get remaining time for a player in seconds."""
        return self.remaining_timedelta(is_white).total_seconds()

    def is_time_up(self, is_white: bool) -> bool:
        """Check if a player's time has expired."""
        return self.remaining_timedelta(is_white) <= timedelta(seconds=0)

    def to_engine_limit(self) -> chess.engine.Limit:
        """Convert clock to python-chess engine Limit object."""
        return chess.engine.Limit(
            white_clock=self.white_time,
            black_clock=self.black_time,
            white_inc=self.increment,
            black_inc=self.increment,
        )

    def get_state(self) -> dict[str, int]:
        """Get clock state for logging/UI."""
        return {
            "white_ms": self.white_ms,
            "black_ms": self.black_ms,
            "increment_ms": self.increment_ms,
        }

    def format_time(self, is_white: bool) -> str:
        """Format remaining time as MM:SS or HH:MM:SS."""
        td = self.remaining_timedelta(is_white)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
