"""Async game loop for non-blocking chess games."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import chess

from chess_fight.common.common_types import CompletionResult
from chess_fight.common.exceptions import MoveExhaustedError
from chess_fight.game.clock import GameClock
from chess_fight.models import ChessAI, GameMove, GameStats

_log = logging.getLogger(__name__)


@dataclass
class GameState:
    """Current game state for UI updates."""
    board: chess.Board
    moves: list[GameMove]
    stats: GameStats
    current_player: str
    is_game_over: bool
    winner: str | None = None
    game_duration: float = 0
    last_completion_result: CompletionResult | None = None
    fen_before: str | None = None
    clock_state: dict[str, Any] | None = None


class AsyncChessGame:
    """Async chess game that yields control to UI between moves."""

    def __init__(
        self,
        player1: ChessAI,
        player2: ChessAI,
        starting_fen: str | None = None,
        clock: GameClock | None = None
    ):
        self.board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self.player1 = player1
        self.player2 = player2
        self.moves: list[GameMove] = []
        self.stats = GameStats()
        self.start_time = time.time()
        self._cancelled = False
        self.clock = clock
        self._turn_start_time = 0.0

    def cancel(self) -> None:
        """Cancel the game."""
        self._cancelled = True

    async def play_game(
        self,
        ui_callback: Callable[[GameState], Awaitable[None]],
        delay: float = 0.1,
        move_timeout_seconds: float | None = None,
    ) -> GameStats:
        """Play a full game with async UI updates.

        Args:
            ui_callback: Callback for UI updates
            delay: Delay between moves for UI updates
            move_timeout_seconds: Optional timeout for each move in seconds
        """
        is_white = True

        # Start clock if provided
        if self.clock:
            self.clock.start_turn(True, 0)

        while not self.board.is_game_over() and not self._cancelled:
            current_player = self.player1 if len(self.moves) % 2 == 0 else self.player2
            is_white = len(self.moves) % 2 == 0
            fen_before = self.board.fen()

            # Start the player's turn on the clock
            state = GameState(
                board=self.board.copy(),
                moves=self.moves.copy(),
                stats=self.stats,
                current_player=current_player.name,
                is_game_over=False,
                fen_before=fen_before,
                clock_state=self.clock.get_state() if self.clock else None,
            )
            await ui_callback(state)

            if self.clock:
                now_ms = int(time.time() * 1000)
                self.clock.start_turn(is_white, now_ms)
                self._turn_start_time = now_ms

            # Get move from player with completion result
            try:
                move_str, completion_result = await current_player.get_move_with_result(fen_before)
            except MoveExhaustedError as exc:
                opponent = self.player2 if is_white else self.player1
                self.stats.winner = opponent.name
                self.stats.game_duration = time.time() - self.start_time
                final_state = GameState(
                    board=self.board.copy(),
                    moves=self.moves.copy(),
                    stats=self.stats,
                    current_player="",
                    is_game_over=True,
                    winner=f"{opponent.name} (Illegal Move Loss)",
                    game_duration=self.stats.game_duration,
                    clock_state=self.clock.get_state() if self.clock else None,
                )
                await ui_callback(final_state)
                return self.stats

            move = chess.Move.from_uci(move_str)

            # End the player's turn on the clock
            if self.clock:
                elapsed_ms = int(time.time() * 1000 - self._turn_start_time)
                self.clock.end_turn(is_white, int(time.time() * 1000))

                # We do not enforce time loss here to ensure the LLMs can play full games
                # even if they take a long time or hit rate limits.

            if move in self.board.legal_moves:
                game_move = GameMove(
                    player=current_player.name,
                    move=move_str,
                    timestamp=time.time(),
                    is_capture=self.board.is_capture(move),
                    is_check=self.board.gives_check(move),
                    reasoning=completion_result.text if completion_result else None
                )

                self.board.push(move)
                self.moves.append(game_move)
                self._update_stats(game_move)
            else:
                # This shouldn't happen due to validation in get_move
                raise ValueError(f"Illegal move {move_str}")

            # Update UI with completion result and clock state
            state = GameState(
                board=self.board.copy(),
                moves=self.moves.copy(),
                stats=self.stats,
                current_player=current_player.name,
                is_game_over=False,
                fen_before=fen_before,
                last_completion_result=completion_result,
                clock_state=self.clock.get_state() if self.clock else None,
            )
            await ui_callback(state)

            # Yield to UI
            await asyncio.sleep(delay)

        # Final state
        self.stats.game_duration = time.time() - self.start_time
        if self._cancelled and not self.board.is_game_over():
            self.stats.winner = "Cancelled"
        else:
            self.stats.winner = self._determine_winner()

        final_state = GameState(
            board=self.board.copy(),
            moves=self.moves.copy(),
            stats=self.stats,
            current_player="",
            is_game_over=True,
            winner=self.stats.winner,
            game_duration=self.stats.game_duration,
            clock_state=self.clock.get_state() if self.clock else None,
        )
        await ui_callback(final_state)

        return self.stats

    def _update_stats(self, move: GameMove) -> None:
        self.stats.total_moves += 1
        if move.is_capture:
            self.stats.capture_moves += 1
        if move.is_check:
            self.stats.check_moves += 1

    def _determine_winner(self) -> str:
        # claim_draw=True so the claimable draws (threefold repetition,
        # fifty-move rule) are recognized in addition to the automatic ones.
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return "Unknown"
        if outcome.winner is None:
            return "Draw"
        winner_name: str = self.player1.name if outcome.winner else self.player2.name
        return winner_name


async def run_game_async(
    white_ai: ChessAI,
    black_ai: ChessAI,
    ui_callback: Callable[[GameState], Awaitable[None]],
    delay: float = 0.1
) -> GameStats:
    """Convenience function to run a game."""
    game = AsyncChessGame(white_ai, black_ai)
    return await game.play_game(ui_callback, delay)
