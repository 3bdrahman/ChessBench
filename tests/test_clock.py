"""Tests for GameClock time management."""

from datetime import timedelta
import time
import pytest
import chess.engine

from chess_fight.game.clock import GameClock


class TestGameClock:
    """Unit tests for GameClock."""

    def test_initialization_with_seconds(self):
        clock = GameClock.from_seconds(300, increment_seconds=5)
        assert clock.white_time == timedelta(seconds=300)
        assert clock.black_time == timedelta(seconds=300)
        assert clock.increment == timedelta(seconds=5)
        assert clock.white_ms == 300000
        assert clock.black_ms == 300000
        assert clock.increment_ms == 5000

    def test_initialization_with_timedelta(self):
        clock = GameClock.from_timedelta(
            white_time=timedelta(minutes=5),
            black_time=timedelta(minutes=5),
            increment=timedelta(seconds=3),
        )
        assert clock.remaining_seconds(True) == 300.0
        assert clock.remaining_seconds(False) == 300.0
        assert clock.increment_ms == 3000

    def test_turn_timing_with_increment(self):
        clock = GameClock.from_seconds(10, increment_seconds=2)
        clock.start_turn(is_white=True)
        time.sleep(0.01)
        clock.end_turn(is_white=True)

        # White time should be slightly less than 10s + 2s increment
        remaining = clock.remaining_seconds(is_white=True)
        assert 11.5 <= remaining <= 12.0

    def test_format_time(self):
        clock = GameClock.from_seconds(3661)  # 1 hour, 1 minute, 1 second
        assert clock.format_time(True) == "01:01:01"

        clock_short = GameClock.from_seconds(125)  # 2 minutes, 5 seconds
        assert clock_short.format_time(True) == "02:05"

    def test_is_time_up(self):
        clock = GameClock.from_seconds(0)
        assert clock.is_time_up(True) is True

        clock_positive = GameClock.from_seconds(10)
        assert clock_positive.is_time_up(True) is False

    def test_to_engine_limit(self):
        clock = GameClock.from_seconds(300, increment_seconds=5)
        limit = clock.to_engine_limit()
        assert isinstance(limit, chess.engine.Limit)
        assert limit.white_clock == timedelta(seconds=300)
        assert limit.black_clock == timedelta(seconds=300)
        assert limit.white_inc == timedelta(seconds=5)
        assert limit.black_inc == timedelta(seconds=5)
