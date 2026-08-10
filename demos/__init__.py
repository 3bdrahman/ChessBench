"""Demos package for chess replay functionality.

The demo games shown in the Streamlit UI are real, not synthetic. They
come from one of two real sources:

  1. PGNs reconstructed from the latest benchmark runs (``runs/<run_id>/``)
     via :func:`demos.generate.list_demo_games`.
  2. Hand-curated PGNs shipped in ``demos/games/`` (only populated by an
     explicit ``python -m demos.generate`` invocation).

The replay engine itself reads whatever PGNs are on disk and animates them
through the async game UI. Nothing in this module invents moves.
"""

from demos.generate import (
    DemoMetadata,
    generate_demos_from_runs,
    list_demo_games,
)
from demos.replay import ReplayEngine, load_demo_game

__all__ = [
    "DemoMetadata",
    "ReplayEngine",
    "generate_demos_from_runs",
    "list_demo_games",
    "load_demo_game",
]
