"""Statistical analysis for benchmark results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PairingStats:
    """Statistical analysis for a single pairing."""
    games: int
    white_score: float
    black_score: float
    score_diff: float
    ci_95_low: float
    ci_95_high: float
    p_value: float
    white_player: str
    black_player: str


def binomial_test(wins: int, losses: int, draws: int, expected: float = 0.5) -> float:
    """
    Two-sided binomial test for deviation from expected score.

    Args:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        expected: Expected score under null hypothesis (default 0.5)

    Returns:
        p-value (two-sided)
    """
    n = wins + losses + draws
    if n == 0:
        return 1.0

    # Convert to effective wins (wins + 0.5 * draws)
    observed = wins + 0.5 * draws

    # Use normal approximation for binomial test
    # H0: p = expected, H1: p != expected
    p = expected
    q = 1 - p
    mean = n * p
    std = math.sqrt(n * p * q)

    if std == 0:
        return 1.0

    z = abs(observed - mean) / std
    # Two-sided p-value
    p_value = 2 * (1 - norm_cdf(z))
    return min(p_value, 1.0)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def binom_confidence_interval(wins: int, losses: int, draws: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval for binomial proportion (with draws as half-wins).

    Returns:
        (ci_low, ci_high) for the white player's score
    """
    n = wins + losses + draws
    if n == 0:
        return (0.0, 0.0)

    # Effective wins (wins + 0.5 * draws)
    k = wins + 0.5 * draws
    p_hat = k / n

    # Wilson score interval
    z = norm_ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    ci_low = centre - half_width
    ci_high = centre + half_width

    return (max(0.0, ci_low), min(1.0, ci_high))


def norm_ppf(p: float) -> float:
    """Percent point function (inverse CDF) for standard normal.

    Uses rational approximation from Abramowitz & Stegun 26.2.23.
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    # Coefficients for rational approximation (Abramowitz & Stegun 26.2.23)
    a = [0, -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    c = [0, -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [0, -7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]

    if p < 0.02425:
        # Rational approximation for lower region
        q = math.sqrt(-2 * math.log(p))
        return (((((a[6] * q + a[5]) * q + a[4]) * q + a[3]) * q + a[2]) * q + a[1]) * q + a[0]
    elif p > 0.97575:
        # Rational approximation for upper region
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((a[6] * q + a[5]) * q + a[4]) * q + a[3]) * q + a[2]) * q + a[1]) * q + a[0]
    else:
        # Rational approximation for central region
        q = p - 0.5
        r = q * q
        return (((((c[6] * r + c[5]) * r + c[4]) * r + c[3]) * r + c[2]) * r + c[1]) * q / \
               (((d[4] * r + d[3]) * r + d[2]) * r + d[1]) * r + 1


def compute_pairing_stats(white: str, black: str, white_wins: int, black_wins: int, draws: int) -> PairingStats:
    """Compute statistical analysis for a pairing."""
    games = white_wins + black_wins + draws

    if games == 0:
        return PairingStats(
            games=0, white_score=0, black_score=0, score_diff=0,
            ci_95_low=0, ci_95_high=0, p_value=1.0,
            white_player=white, black_player=black
        )

    white_score = (white_wins + 0.5 * draws) / games
    black_score = (black_wins + 0.5 * draws) / games
    score_diff = white_score - black_score

    ci_low, ci_high = binom_confidence_interval(white_wins, black_wins, draws)
    p_value = binomial_test(white_wins, black_wins, draws)

    return PairingStats(
        games=games,
        white_score=white_score,
        black_score=black_score,
        score_diff=score_diff,
        ci_95_low=ci_low,
        ci_95_high=ci_high,
        p_value=p_value,
        white_player=white,
        black_player=black
    )


def effective_sample_size(scores: list[float]) -> float:
    """
    Estimate effective sample size accounting for autocorrelation.

    Uses the formula: n_eff = n / (1 + 2 * sum_{k=1}^{n-1} rho_k)
    where rho_k is the autocorrelation at lag k.
    """
    n = len(scores)
    if n < 2:
        return float(n)

    # Compute autocorrelations up to lag 10 or n/2
    max_lag = min(10, n // 2)
    scores_arr = np.array(scores, dtype=float)
    mean_score = float(np.mean(scores_arr))
    var = float(np.var(scores_arr, ddof=1))

    if var == 0:
        return 1.0  # Constant values - no information beyond 1 sample

    autocorr_sum = 0.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        cov = float(np.mean((scores_arr[:-lag] - mean_score) * (scores_arr[lag:] - mean_score)))
        rho = cov / var
        autocorr_sum += rho

    denominator = 1 + 2 * autocorr_sum
    if denominator <= 0:
        return 1.0  # Negative or zero denominator means infinite/undefined ESS

    n_eff = n / denominator
    return max(1.0, n_eff)


def bootstrap_ci(ratings: list[float], n_bootstrap: int = 10000,
                 confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """
    Bootstrap confidence interval for rating statistics.

    Args:
        ratings: List of rating values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 default)
        seed: Random seed for reproducibility

    Returns:
        (ci_low, ci_high)
    """
    np.random.seed(seed)
    ratings_arr = np.array(ratings, dtype=float)
    n = len(ratings_arr)

    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(ratings_arr[0]), float(ratings_arr[0]))

    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(ratings_arr, size=n, replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    alpha = (1 - confidence) / 2
    ci_low = float(np.percentile(bootstrap_means, alpha * 100))
    ci_high = float(np.percentile(bootstrap_means, (1 - alpha) * 100))

    return (ci_low, ci_high)


def rating_convergence(ratings_history: dict[str, list[float]],
                       window: int = 10, threshold: float = 5.0) -> dict[str, bool]:
    """
    Check if ratings have converged.

    A rating is considered converged if the maximum change over the last
    `window` games is less than `threshold` Elo points.

    Args:
        ratings_history: Dict mapping player name to list of ratings over time
        window: Number of recent games to consider
        threshold: Maximum allowed rating change for convergence

    Returns:
        Dict mapping player name to convergence boolean
    """
    converged: dict[str, bool] = {}
    for player, history in ratings_history.items():
        if len(history) < window:
            converged[player] = False
        else:
            recent = history[-window:]
            max_change = max(recent) - min(recent)
            converged[player] = max_change < threshold
    return converged


def rating_stability_metric(ratings_history: list[float], window: int = 10) -> float:
    """
    Compute rating stability metric as the standard deviation of
    rating changes over the last `window` games.
    """
    if len(ratings_history) < 2:
        return 0.0

    ratings_arr = np.array(ratings_history, dtype=float)
    changes = np.diff(ratings_arr)
    if len(changes) < window:
        return float(np.std(changes, ddof=1))

    return float(np.std(changes[-window:], ddof=1))


def glicko2_bootstrap_ci(games: list[tuple[Any, ...]], players: list[str],
                          n_bootstrap: int = 1000, confidence: float = 0.95,
                          seed: int = 42) -> dict[str, tuple[float, float]]:
    """
    Bootstrap confidence intervals for Glicko-2 ratings.

    Args:
        games: List of (white, black, result, opening) tuples
        players: List of player names
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level
        seed: Random seed

    Returns:
        Dict mapping player name to (ci_low, ci_high)
    """
    from chessbench.benchmark.elo import BayesianElo

    np.random.seed(seed)
    n = len(games)
    if n == 0:
        return dict.fromkeys(players, (0.0, 0.0))

    bootstrap_ratings: dict[str, list[float]] = {p: [] for p in players}

    for _ in range(n_bootstrap):
        # Resample games with replacement
        indices = np.random.choice(n, size=n, replace=True)
        sample_games = [games[i] for i in indices]

        # Compute ratings for this bootstrap sample
        elo = BayesianElo()
        for white, black, result, opening in sample_games:
            elo.add_game(white, black, result, opening)
        elo.finalize_period()

        for p in players:
            rating = elo.get_rating(p)
            bootstrap_ratings[p].append(rating.display_rating)

    # Compute CIs
    alpha = (1 - confidence) / 2
    ci: dict[str, tuple[float, float]] = {}
    for p in players:
        ci[p] = (
            float(np.percentile(bootstrap_ratings[p], alpha * 100)),
            float(np.percentile(bootstrap_ratings[p], (1 - alpha) * 100))
        )
    return ci
