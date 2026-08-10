"""Models package.

Public API surface for chess models:

* :class:`ChessAI` — provider-agnostic abstract base class. Subclasses
  implement ``_get_move_from_model``; prompt construction, retry policy, and
  UCI validation all live in the base class.
* :class:`PositionEvaluator` — pure-Python chess position evaluation utilities
  used to build prompts and inspect board state.
* :class:`GameMove` / :class:`GameStats` — lightweight dataclasses describing
  the per-move and per-game history shown to the UI / logged by the benchmark
  runner.
"""

from .chess_ai import ChessAI
from .evaluation import PositionEvaluator
from .game_state import GameMove, GameStats

__all__ = [
    "ChessAI",
    "GameMove",
    "GameStats",
    "PositionEvaluator",
]
