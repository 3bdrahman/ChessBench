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
    # Pause/resume support
    is_paused: bool = False
    pause_reason: str | None = None
    pause_error: str | None = None
    paused_player: str | None = None
    paused_turn: int = 0


class AsyncChessGame:
    """Async chess game that yields control to UI between moves."""

    def __init__(
        self,
        player1: ChessAI,
        player2: ChessAI,
        starting_fen: str | None = None,
        clock: GameClock | None = None,
        max_moves: int = 512,
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
        self.max_moves = max_moves
        # Pause/resume support
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused
        self._paused = False
        self._pause_reason: str | None = None
        self._pause_error: str | None = None
        self._paused_player: str | None = None
        self._paused_turn: int = 0
        self._retry_current_turn = False

    def cancel(self) -> None:
        """Cancel the game."""
        self._cancelled = True
        self._pause_event.set()  # Unblock any waiting

    def pause(self, reason: str, error: str | None = None, player: str | None = None) -> None:
        """Pause the game with a reason."""
        self._paused = True
        self._pause_reason = reason
        self._pause_error = error
        self._paused_player = player
        self._paused_turn = len(self.moves)
        self._pause_event.clear()

    def resume(self, retry_current_turn: bool = True) -> None:
        """Resume the game from pause."""
        self._paused = False
        self._pause_reason = None
        self._pause_error = None
        self._paused_player = None
        self._paused_turn = 0
        self._retry_current_turn = retry_current_turn
        self._pause_event.set()

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def pause_reason(self) -> str | None:
        return self._pause_reason

    @property
    def pause_error(self) -> str | None:
        return self._pause_error

    @property
    def paused_player(self) -> str | None:
        return self._paused_player

    @property
    def paused_turn(self) -> int:
        return self._paused_turn

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

        while (not self.board.is_game_over(claim_draw=True) 
               and not self._cancelled 
               and len(self.moves) < self.max_moves):
            # Handle pause/resume - wait if paused
            if self._paused:
                # Send paused state to UI
                paused_state = GameState(
                    board=self.board.copy(),
                    moves=self.moves.copy(),
                    stats=self.stats,
                    current_player=self._paused_player or "",
                    is_game_over=False,
                    fen_before=self.board.fen(),
                    clock_state=self.clock.get_state() if self.clock else None,
                    is_paused=True,
                    pause_reason=self._pause_reason,
                    pause_error=self._pause_error,
                    paused_player=self._paused_player,
                    paused_turn=self._paused_turn,
                )
                await ui_callback(paused_state)
                
                # Wait for resume signal
                await self._pause_event.wait()
                
                # If retry is requested, we'll retry the same turn (don't advance move count)
                if not self._retry_current_turn:
                    # User chose not to retry - treat as cancellation
                    self._cancelled = True
                    break
                # If retry, continue to retry the same turn (loop continues without advancing)

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
                if move_timeout_seconds is not None:
                    move_str, completion_result = await asyncio.wait_for(
                        current_player.get_move_with_result(fen_before),
                        timeout=move_timeout_seconds
                    )
                else:
                    move_str, completion_result = await current_player.get_move_with_result(fen_before)
            except Exception as exc:
                _log.error("Player %s move execution failed on turn %d: %s", current_player.name, len(self.moves), exc)
                
                # Pause the game instead of terminating
                error_type = type(exc).__name__
                is_chess_loss = isinstance(exc, MoveExhaustedError)
                is_timeout = isinstance(exc, asyncio.TimeoutError)
                
                if is_chess_loss:
                    reason = "illegal_move"
                    error_msg = f"Illegal move: {exc}"
                elif is_timeout:
                    reason = "timeout"
                    error_msg = f"Move timeout after {move_timeout_seconds}s"
                else:
                    reason = "error"
                    error_msg = f"{error_type}: {exc}"
                
                self.pause(
                    reason=reason,
                    error=error_msg,
                    player=current_player.name
                )
                
                # Send paused state and wait for resume
                paused_state = GameState(
                    board=self.board.copy(),
                    moves=self.moves.copy(),
                    stats=self.stats,
                    current_player=current_player.name,
                    is_game_over=False,
                    fen_before=fen_before,
                    clock_state=self.clock.get_state() if self.clock else None,
                    is_paused=True,
                    pause_reason=reason,
                    pause_error=error_msg,
                    paused_player=current_player.name,
                    paused_turn=len(self.moves),
                )
                await ui_callback(paused_state)
                
                # Wait for resume
                await self._pause_event.wait()
                
                if not self._retry_current_turn:
                    # User chose not to retry - cancel game
                    self._cancelled = True
                    break
                # Retry the same turn - continue loop without advancing move count
                continue

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
        if self._cancelled and not self.board.is_game_over(claim_draw=True):
            self.stats.winner = "Cancelled"
            self.stats.termination_reason = "cancelled"
        elif len(self.moves) >= self.max_moves:
            self.stats.winner = "Draw (max moves reached)"
            self.stats.termination_reason = "max_moves"
        else:
            self.stats.winner = self._determine_winner()
            # Determine clean chess termination reason
            outcome = self.board.outcome(claim_draw=True)
            if outcome is not None:
                if outcome.termination == chess.Termination.CHECKMATE:
                    self.stats.termination_reason = "checkmate"
                elif outcome.termination == chess.Termination.STALEMATE:
                    self.stats.termination_reason = "stalemate"
                elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                    self.stats.termination_reason = "insufficient_material"
                elif outcome.termination == chess.Termination.FIFTY_MOVES:
                    self.stats.termination_reason = "fifty_moves"
                elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
                    self.stats.termination_reason = "threefold_repetition"
                elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                    self.stats.termination_reason = "seventyfive_moves"
                elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                    self.stats.termination_reason = "fivefold_repetition"
                elif outcome.termination == chess.Termination.VARIANT_WIN:
                    self.stats.termination_reason = "variant_win"
                elif outcome.termination == chess.Termination.VARIANT_LOSS:
                    self.stats.termination_reason = "variant_loss"
                elif outcome.termination == chess.Termination.VARIANT_DRAW:
                    self.stats.termination_reason = "variant_draw"
                else:
                    self.stats.termination_reason = "draw"
            else:
                self.stats.termination_reason = "unknown"

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
    delay: float = 0.1,
    max_moves: int = 512,
) -> GameStats:
    """Convenience function to run a game."""
    game = AsyncChessGame(white_ai, black_ai, max_moves=max_moves)
    return await game.play_game(ui_callback, delay)
