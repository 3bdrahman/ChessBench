"""ECO opening book for benchmark games."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess


def load_eco_openings() -> list[tuple[str, str, list[str]]]:
    """Load ECO openings from external JSON file."""
    data_path = Path(__file__).parent / "data" / "eco_openings.json"
    with open(data_path) as f:
        data = json.load(f)
    return [(item["eco"], item["name"], item["moves"]) for item in data]


# Load openings from external file
ECO_OPENINGS = load_eco_openings()


class OpeningBook:
    """Opening book for benchmark games."""

    def __init__(self) -> None:
        self.openings = ECO_OPENINGS
        self.opening_fens: list[dict[str, Any]] = []
        self._precompute_fens()

    def _precompute_fens(self) -> None:
        """Precompute FEN for each opening."""
        self.opening_fens = []
        self.invalid_openings = []
        for eco, name, moves in self.openings:
            board = chess.Board()
            valid = True
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    self.invalid_openings.append({
                        'eco': eco,
                        'name': name,
                        'reason': f'Illegal move {move_uci} in position {board.fen()}'
                    })
                    valid = False
                    break

            if valid:
                # Check if position has immediate mate
                if board.is_checkmate():
                    self.invalid_openings.append({
                        'eco': eco,
                        'name': name,
                        'reason': 'Immediate checkmate in opening position'
                    })
                    continue

                self.opening_fens.append({
                    'eco': eco,
                    'name': name,
                    'moves': moves,
                    'fen': board.fen(),
                    'ply': len(moves),
                    'category': eco[0],  # A, B, C, D, E
                })

    def get_random_opening(self) -> dict[str, Any]:
        """Get a random opening."""
        return random.choice(self.opening_fens)

    def get_opening_by_eco(self, eco: str) -> dict[str, Any] | None:
        """Get opening by ECO code."""
        for op in self.opening_fens:
            if op['eco'] == eco:
                return op
        return None

    def get_all_openings(self) -> list[dict[str, Any]]:
        """Get all openings."""
        return self.opening_fens.copy()

    def get_openings_by_category(self, category_prefix: str) -> list[dict[str, Any]]:
        """Get openings by ECO category (e.g., 'A0' for A00-A09)."""
        return [op for op in self.opening_fens if op['eco'].startswith(category_prefix)]

    def get_balanced_set(self, n: int) -> list[dict[str, Any]]:
        """Get a balanced set of n openings across categories (A, B, C, D, E)."""
        categories = defaultdict(list)
        for op in self.opening_fens:
            cat = op['category']  # A, B, C, D, E
            categories[cat].append(op)

        if not categories:
            return []

        # Distribute evenly across categories
        result = []
        per_cat = max(1, n // len(categories))
        for _cat, ops in categories.items():
            random.shuffle(ops)
            result.extend(ops[:per_cat])

        # Fill remaining
        if len(result) < n:
            remaining = [op for op in self.opening_fens if op not in result]
            random.shuffle(remaining)
            result.extend(remaining[:n - len(result)])

        random.shuffle(result)
        return result[:n]

    def get_openings_by_depth_range(self, min_ply: int, max_ply: int) -> list[dict[str, Any]]:
        """Get openings filtered by depth range (in plies)."""
        return [op for op in self.opening_fens if min_ply <= op['ply'] <= max_ply]

    def get_openings_by_main_category(self, category: str) -> list[dict[str, Any]]:
        """Get openings by ECO category (A, B, C, D, or E)."""
        return [op for op in self.opening_fens if op['category'] == category]

    def get_invalid_openings(self) -> list[dict[str, Any]]:
        """Get list of invalid openings with reasons."""
        return getattr(self, 'invalid_openings', [])
