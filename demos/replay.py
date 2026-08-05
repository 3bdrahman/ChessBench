"""Replay engine for demo chess games."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import chess
import chess.pgn

from chess_fight.game.async_game import GameState
from chess_fight.models.game_state import GameMove, GameStats

_GAMES_DIR = Path(__file__).parent / "games"


def list_demo_games() -> list[dict]:
    """List all available demo games with metadata."""
    games = []
    for pgn_file in sorted(_GAMES_DIR.glob("*.pgn")):
        try:
            with open(pgn_file) as f:
                pgn_game = chess.pgn.read_game(f)
            if pgn_game is None:
                continue
            headers = pgn_game.headers
            move_count = len(list(pgn_game.mainline_moves()))
            games.append({
                "filename": str(pgn_file.resolve()),
                "white": headers.get("White", "Unknown"),
                "black": headers.get("Black", "Unknown"),
                "result": headers.get("Result", "*"),
                "opening": headers.get("Opening", ""),
                "move_count": move_count,
            })
        except Exception:
            continue
    return games


def _load_pgn_headers(pgn_file: str) -> tuple[list[chess.Move], str, str, str]:
    """Parse a PGN file and return (moves, result, white, black)."""
    with open(pgn_file) as f:
        pgn_game = chess.pgn.read_game(f)
    if pgn_game is None:
        raise ValueError(f"Could not parse PGN: {pgn_file}")
    moves = list(pgn_game.mainline_moves())
    result = pgn_game.headers.get("Result", "*")
    white = pgn_game.headers.get("White", "Unknown")
    black = pgn_game.headers.get("Black", "Unknown")
    return moves, result, white, black


class ReplayEngine:
    """Replays a recorded chess game through an async UI callback."""

    def __init__(self, pgn_file: str):
        self._moves, self._result, self._white, self._black = _load_pgn_headers(pgn_file)

    @property
    def move_count(self) -> int:
        return len(self._moves)

    @property
    def result(self) -> str:
        return self._result

    @property
    def white_player(self) -> str:
        return self._white

    @property
    def black_player(self) -> str:
        return self._black

    async def replay(
        self,
        ui_callback: Callable[[GameState], Awaitable],
        delay: float = 0.5,
    ) -> GameStats:
        """Replay the game, calling ui_callback for each move."""
        board = chess.Board()
        game_moves: list[GameMove] = []
        stats = GameStats()
        start = time.time()

        for i, move in enumerate(self._moves):
            player_name = self._white if i % 2 == 0 else self._black

            is_last = (i == len(self._moves) - 1)
            state = GameState(
                board=board.copy(),
                moves=game_moves.copy(),
                stats=stats,
                current_player=player_name,
                is_game_over=is_last,
                winner=self._get_game_winner() if is_last else None,
                game_duration=time.time() - start if is_last else 0,
                fen_before=board.fen(),
            )
            await ui_callback(state)

            game_move = GameMove(
                player=player_name,
                move=move.uci(),
                timestamp=time.time(),
                is_capture=board.is_capture(move),
                is_check=board.gives_check(move),
            )
            board.push(move)
            game_moves.append(game_move)
            stats.total_moves += 1
            if game_move.is_capture:
                stats.capture_moves += 1
            if game_move.is_check:
                stats.check_moves += 1

            await asyncio.sleep(delay)

        stats.game_duration = time.time() - start
        stats.winner = self._get_game_winner()

        state = GameState(
            board=board.copy(),
            moves=game_moves.copy(),
            stats=stats,
            current_player="",
            is_game_over=True,
            winner=stats.winner,
            game_duration=stats.game_duration,
            fen_before=board.fen(),
        )
        await ui_callback(state)

        return stats

    def _get_game_winner(self) -> str:
        if self._result == "1-0":
            return self._white
        if self._result == "0-1":
            return self._black
        return "Draw"


def load_demo_game(pgn_file: str) -> ReplayEngine:
    """Load a single demo game by PGN path."""
    return ReplayEngine(pgn_file)
