"""Async game loop for non-blocking chess games."""

import chess
import asyncio
import time
from typing import Optional, List, Callable, Awaitable
from dataclasses import dataclass
from models import GameMove, GameStats, ChessAI
from providers.base import CompletionResult


@dataclass
class GameState:
    """Current game state for UI updates."""
    board: chess.Board
    moves: List[GameMove]
    stats: GameStats
    current_player: str
    is_game_over: bool
    winner: Optional[str] = None
    game_duration: float = 0
    last_completion_result: Optional[CompletionResult] = None
    fen_before: Optional[str] = None


class AsyncChessGame:
    """Async chess game that yields control to UI between moves."""
    
    def __init__(self, player1: ChessAI, player2: ChessAI, starting_fen: Optional[str] = None):
        self.board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self.player1 = player1
        self.player2 = player2
        self.moves: List[GameMove] = []
        self.stats = GameStats()
        self.start_time = time.time()
        self._cancelled = False
    
    def cancel(self):
        """Cancel the game."""
        self._cancelled = True
    
    async def play_game(
        self, 
        ui_callback: Callable[[GameState], Awaitable[None]],
        delay: float = 0.1
    ) -> GameStats:
        """Play a full game with async UI updates."""
        
        while not self.board.is_game_over() and not self._cancelled:
            current_player = self.player1 if len(self.moves) % 2 == 0 else self.player2
            fen_before = self.board.fen()

            # Update UI with current state
            state = GameState(
                board=self.board.copy(),
                moves=self.moves.copy(),
                stats=self.stats,
                current_player=current_player.name,
                is_game_over=False,
                fen_before=fen_before,
            )
            await ui_callback(state)

            # Get move from player with completion result
            move_str, completion_result = await current_player.get_move_with_result(fen_before)
            move = chess.Move.from_uci(move_str)

            if move in self.board.legal_moves:
                game_move = GameMove(
                    player=current_player.name,
                    move=move_str,
                    timestamp=time.time(),
                    is_capture=self.board.is_capture(move),
                    is_check=self.board.gives_check(move)
                )

                self.board.push(move)
                self.moves.append(game_move)
                self._update_stats(game_move)
            else:
                # This shouldn't happen due to validation in get_move
                raise ValueError(f"Illegal move {move_str}")

            # Update UI with completion result
            state = GameState(
                board=self.board.copy(),
                moves=self.moves.copy(),
                stats=self.stats,
                current_player=current_player.name,
                is_game_over=False,
                fen_before=fen_before,
                last_completion_result=completion_result,
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
        return self.player1.name if outcome.winner else self.player2.name


async def run_game_async(
    white_ai: ChessAI,
    black_ai: ChessAI,
    ui_callback: Callable[[GameState], Awaitable[None]],
    delay: float = 0.1
) -> GameStats:
    """Convenience function to run a game."""
    game = AsyncChessGame(white_ai, black_ai)
    return await game.play_game(ui_callback, delay)