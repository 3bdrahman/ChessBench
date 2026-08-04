import chess
import time
import asyncio
from typing import Optional, List
from models import GameMove, GameStats, ChessAI
from providers.base import CompletionResult


class ChessGame:
    def __init__(self, player1: ChessAI, player2: ChessAI):
        self.board = chess.Board()
        self.player1 = player1
        self.player2 = player2
        self.moves: List[GameMove] = []
        self.stats = GameStats()
        self.start_time = time.time()
        self.current_completion_result: Optional[CompletionResult] = None

    def play_move(self) -> Optional[GameMove]:
        current_player = self.player1 if len(self.moves) % 2 == 0 else self.player2

        max_retries = 3
        for attempt in range(max_retries):
            try:
                move_str, completion_result = asyncio.run(
                    current_player.get_move_with_result(self.board.fen())
                )
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
                    self.current_completion_result = completion_result
                    return game_move

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Invalid move after {max_retries} attempts: {str(e)}")
                continue

        raise ValueError(f"Failed to get valid move after {max_retries} attempts")

    def _update_stats(self, move: GameMove) -> None:
        self.stats.total_moves += 1
        if move.is_capture:
            self.stats.capture_moves += 1
        if move.is_check:
            self.stats.check_moves += 1

    @property
    def is_game_over(self) -> bool:
        if self.board.is_game_over():
            self.stats.game_duration = time.time() - self.start_time
            self.stats.winner = self._determine_winner()
            return True
        return False

    def _determine_winner(self) -> str:
        # claim_draw=True so the claimable draws (threefold repetition,
        # fifty-move rule) are recognized in addition to the automatic ones.
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return "Unknown"
        if outcome.winner is None:
            return "Draw"
        return self.player1.name if outcome.winner else self.player2.name
