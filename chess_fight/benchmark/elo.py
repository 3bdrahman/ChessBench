"""Bayesian ELO rating system for chess benchmark."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class GameResult:
    """Result of a single game."""
    white: str
    black: str
    result: float  # 1.0 = white win, 0.5 = draw, 0.0 = black win
    opening: str | None = None


@dataclass
class Rating:
    """Player rating with uncertainty."""
    rating: float
    deviation: float  # Standard deviation / uncertainty
    volatility: float = 0.06  # Glicko-2 volatility parameter
    mu: float = 0.0
    phi: float = 0.0
    sigma: float = 0.06

    def __post_init__(self):
        # Convert to Glicko-2 scale (1500 = 0, 173.7178 = 1 deviation)
        self.mu = (self.rating - 1500) / 173.7178
        self.phi = self.deviation / 173.7178
        self.sigma = self.volatility

    @property
    def display_rating(self) -> float:
        return float(1500 + 173.7178 * self.mu)

    @property
    def display_deviation(self) -> float:
        return float(173.7178 * self.phi)

    @property
    def confidence_interval_95(self) -> tuple:
        """95% confidence interval."""
        margin = 1.96 * self.display_deviation
        return (self.display_rating - margin, self.display_rating + margin)


class Glicko2:
    """Glicko-2 rating system implementation."""

    def __init__(self, tau: float = 0.5):
        self.tau = tau  # System constant for volatility change
        self.ratings: dict[str, Rating] = {}

    def add_player(self, name: str, rating: float = 1500, deviation: float = 350, volatility: float = 0.06):
        """Add a new player."""
        self.ratings[name] = Rating(rating, deviation, volatility)

    def get_rating(self, name: str) -> Rating:
        """Get player rating, creating default if not exists."""
        if name not in self.ratings:
            self.add_player(name)
        return self.ratings[name]

    def _g(self, phi: float) -> float:
        """g(phi) function."""
        return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)

    def _E(self, mu: float, mu_j: float, phi_j: float) -> float:
        """Expected score against opponent."""
        return 1 / (1 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def update_ratings(self, results: list[GameResult]):
        """Update ratings based on game results."""
        # Group results by player
        player_results = defaultdict(list)
        for r in results:
            player_results[r.white].append(('white', r))
            player_results[r.black].append(('black', r))

        # Update each player
        new_ratings = {}
        for player, games in player_results.items():
            rating = self.get_rating(player)

            # Calculate variance and delta
            v_inv: float = 0.0
            delta: float = 0.0

            for color, game in games:
                opp_name = game.black if color == 'white' else game.white
                opp_rating = self.get_rating(opp_name)

                g_phi = self._g(opp_rating.phi)
                e = self._E(rating.mu, opp_rating.mu, opp_rating.phi)

                v_inv += g_phi**2 * e * (1 - e)

                # Actual score
                score = game.result if color == 'white' else 1 - game.result

                delta += g_phi * (score - e)

            if v_inv == 0:
                new_ratings[player] = rating
                continue

            v = 1 / v_inv
            delta *= v

            # Update volatility (sigma) using Illinois algorithm
            new_sigma = self._update_volatility(rating, v, delta)

            # Update phi (deviation)
            new_phi = 1 / math.sqrt(1 / (rating.phi**2 + new_sigma**2) + 1/v)

            # Update mu (rating)
            new_mu = rating.mu + new_phi**2 * delta / v

            new_ratings[player] = Rating(
                rating=1500 + 173.7178 * new_mu,
                deviation=173.7178 * new_phi,
                volatility=new_sigma
            )
            new_ratings[player].mu = new_mu
            new_ratings[player].phi = new_phi
            new_ratings[player].sigma = new_sigma

        # Apply updates
        self.ratings.update(new_ratings)

    def _update_volatility(self, rating: Rating, v: float, delta: float) -> float:
        """Update volatility using Illinois algorithm."""
        a = math.log(rating.sigma**2)

        def f(x: float) -> float:
            ex = math.exp(x)
            return ex * (delta**2 - rating.phi**2 - v - ex) / (2 * (rating.phi**2 + v + ex)**2) - (x - a) / self.tau**2

        A: float = a
        B: float = 0.0

        if delta**2 > rating.phi**2 + v:
            B = math.log(delta**2 - rating.phi**2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        fA = f(A)
        fB = f(B)

        for _ in range(100):
            if abs(B - A) < 1e-6:
                break
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)

            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA /= 2
            B = C
            fB = fC

        return math.exp(A / 2)

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get sorted leaderboard."""
        board = []
        for name, rating in sorted(self.ratings.items(), key=lambda x: x[1].display_rating, reverse=True):
            ci = rating.confidence_interval_95
            board.append({
                'name': name,
                'rating': round(rating.display_rating, 1),
                'deviation': round(rating.display_deviation, 1),
                'ci_low': round(ci[0], 1),
                'ci_high': round(ci[1], 1),
                'volatility': round(rating.volatility, 4)
            })
        return board

    def export_json(self) -> str:
        """Export ratings to JSON."""
        data = {}
        for name, rating in self.ratings.items():
            data[name] = {
                'rating': rating.display_rating,
                'deviation': rating.display_deviation,
                'volatility': rating.volatility
            }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Glicko2':
        """Load ratings from JSON."""
        data = json.loads(json_str)
        glicko = cls()
        for name, vals in data.items():
            glicko.add_player(name, vals['rating'], vals['deviation'], vals['volatility'])
        return glicko


class BayesianElo:
    """Simplified Bayesian ELO (using Glicko-2 under the hood)."""

    def __init__(self):
        self.glicko = Glicko2()
        # cross_table() needs the raw game records; Glicko2 keeps only ratings.
        self.games: list[GameResult] = []

    def add_game(self, white: str, black: str, result: float, opening: str | None = None):
        """Add a game result."""
        for p in [white, black]:
            if p not in self.glicko.ratings:
                self.glicko.add_player(p)
        game = GameResult(white, black, result, opening)
        self.games.append(game)
        self.glicko.update_ratings([game])

    def get_rating(self, name: str) -> Rating:
        return self.glicko.get_rating(name)

    def leaderboard(self) -> list[dict[str, Any]]:
        return self.glicko.get_leaderboard()

    def cross_table(self, players: list[str]) -> list[list[float]]:
        """Head-to-head scoring matrix.

        Returns an n x n matrix where row i is `players[i]` as White and
        column j is `players[j]` as Black; cell value is White's mean score
        across all recorded games between that pairing (1.0 = White always
        won, 0.5 = all draws, 0.0 = White always lost). NaN marks unplayed
        pairings and the self-pairing diagonal, so downstream report code
        can distinguish "no games" from "draw rate".
        """
        n = len(players)
        index = {name: i for i, name in enumerate(players)}
        totals = [[0.0, 0] for _ in range(n * n)]
        for game in self.games:
            if game.white not in index or game.black not in index:
                continue
            i = index[game.white]
            j = index[game.black]
            totals[i * n + j][0] += game.result
            totals[i * n + j][1] += 1

        table: list[list[float]] = []
        for i in range(n):
            row: list[float] = []
            for j in range(n):
                if i == j:
                    row.append(float('nan'))
                    continue
                scored, count = totals[i * n + j]
                row.append(scored / count if count > 0 else float('nan'))
            table.append(row)
        return table


if __name__ == "__main__":
    # Quick test
    elo = BayesianElo()
    elo.add_game("GPT-4o", "Claude-3.5-Sonnet", 1.0)
    elo.add_game("Claude-3.5-Sonnet", "GPT-4o", 0.5)
    elo.add_game("GPT-4o", "Claude-3.5-Sonnet", 0.0)
    elo.add_game("GPT-4o", "Gemini-1.5-Pro", 1.0)

    for row in elo.leaderboard():
        print(f"{row['name']}: {row['rating']} ± {row['deviation']} (95% CI: {row['ci_low']}-{row['ci_high']})")
