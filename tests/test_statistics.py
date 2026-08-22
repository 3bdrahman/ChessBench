"""Tests for statistical analysis module."""


import pytest

from chessbench.benchmark.statistics import (
    binom_confidence_interval,
    binomial_test,
    bootstrap_ci,
    compute_pairing_stats,
    effective_sample_size,
    rating_convergence,
    rating_stability_metric,
)


class TestBinomialTest:
    """Tests for binomial test."""

    def test_basic_binomial_test(self):
        """Test basic binomial test with known values."""
        # 5 wins, 0 losses, 0 draws - strong evidence against 0.5
        p = binomial_test(5, 0, 0)
        assert p < 0.05  # Should be significant

    def test_draw_binomial_test(self):
        """Test with draws."""
        # 2 wins, 2 losses, 2 draws = 3/6 = 0.5
        p = binomial_test(2, 2, 2)
        assert p > 0.5  # Not significant

    def test_no_games(self):
        """Test with no games."""
        p = binomial_test(0, 0, 0)
        assert p == 1.0

    def test_binomial_p_value_range(self):
        """Test that p-values are in [0, 1]."""
        for wins in [0, 1, 2, 5, 10]:
            for losses in [0, 1, 2, 5, 10]:
                for draws in [0, 1, 2]:
                    p = binomial_test(wins, losses, draws)
                    assert 0 <= p <= 1


class TestBinomConfidenceInterval:
    """Tests for binomial confidence interval."""

    def test_basic_ci(self):
        """Test basic Wilson confidence interval."""
        ci_low, ci_high = binom_confidence_interval(5, 0, 0)
        assert ci_low > 0.5  # Should be significantly above 0.5
        assert ci_high <= 1.0

    def test_draw_ci(self):
        """Test CI with draws."""
        ci_low, ci_high = binom_confidence_interval(2, 2, 2)
        # With 2W, 2L, 2D, score is 3/6 = 0.5
        assert ci_low <= 0.5 <= ci_high

    def test_zero_games_ci(self):
        """Test CI with zero games."""
        ci_low, ci_high = binom_confidence_interval(0, 0, 0)
        assert ci_low == 0.0
        assert ci_high == 0.0

    def test_ci_bounds(self):
        """Test that CI bounds are valid probabilities."""
        for wins in [0, 1, 3, 10]:
            for losses in [0, 1, 3]:
                for draws in [0, 1, 2]:
                    ci_low, ci_high = binom_confidence_interval(wins, losses, draws)
                    assert 0 <= ci_low <= 1
                    assert 0 <= ci_high <= 1
                    assert ci_low <= ci_high


class TestComputePairingStats:
    """Tests for compute_pairing_stats."""

    def test_basic_pairing_stats(self):
        """Test basic pairing statistics."""
        stats = compute_pairing_stats("A", "B", 3, 1, 2)
        assert stats.white_player == "A"
        assert stats.black_player == "B"
        assert stats.games == 6
        assert stats.white_score == 4/6  # (3 + 0.5*2)/6
        assert stats.black_score == 2/6

    def test_zero_games(self):
        """Test with zero games."""
        stats = compute_pairing_stats("A", "B", 0, 0, 0)
        assert stats.games == 0
        assert stats.white_score == 0
        assert stats.p_value == 1.0

    def test_all_draws(self):
        """Test with all draws."""
        stats = compute_pairing_stats("A", "B", 0, 0, 10)
        assert stats.white_score == 0.5
        assert stats.black_score == 0.5
        assert stats.score_diff == 0

    def test_significant_result(self):
        """Test significant pairing."""
        # A wins all 5 games
        stats = compute_pairing_stats("A", "B", 5, 0, 0)
        assert stats.p_value < 0.05
        assert stats.white_score == 1.0


class TestEffectiveSampleSize:
    """Tests for effective sample size."""

    def test_ess_independent(self):
        """Test ESS for alternating pattern (which has negative autocorrelation)."""
        # Alternating pattern has negative autocorrelation, which leads to invalid ESS
        # The function returns 1.0 for negative denominator (which is correct)
        scores = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        ess = effective_sample_size(scores)
        # Negative autocorrelation leads to denominator < 0, so ESS returns 1.0
        assert ess == 1.0

    def test_ess_constant(self):
        """Test ESS for constant series."""
        scores = [1.0] * 10
        ess = effective_sample_size(scores)
        assert ess == 1.0

    def test_ess_single(self):
        """Test ESS with single value."""
        ess = effective_sample_size([1.0])
        assert ess == 1.0

    def test_ess_empty(self):
        """Test ESS with empty list."""
        ess = effective_sample_size([])
        assert ess == 0.0


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap CI."""
        ratings = [1500, 1510, 1490, 1505, 1495]
        ci_low, ci_high = bootstrap_ci(ratings, n_bootstrap=1000)
        assert ci_low < ci_high
        assert ci_low > 1400
        assert ci_high < 1600

    def test_bootstrap_single(self):
        """Test bootstrap with single value."""
        ci_low, ci_high = bootstrap_ci([1500.0])
        assert ci_low == 1500.0
        assert ci_high == 1500.0

    def test_bootstrap_empty(self):
        """Test bootstrap with empty list."""
        ci_low, ci_high = bootstrap_ci([])
        assert ci_low == 0.0
        assert ci_high == 0.0


class TestRatingConvergence:
    """Tests for rating convergence."""

    def test_converged(self):
        """Test convergence detection."""
        history = {
            "A": [1500, 1502, 1501, 1503, 1502, 1501, 1502, 1503, 1502, 1501],
            "B": [1500, 1600, 1550, 1580, 1570, 1590, 1560, 1580, 1570, 1590],  # oscillating
        }
        converged = rating_convergence(history, window=5, threshold=5.0)
        assert converged["A"] is True
        assert converged["B"] is False

    def test_insufficient_history(self):
        """Test with insufficient history."""
        history = {"A": [1500, 1505]}
        converged = rating_convergence(history, window=10)
        assert converged["A"] is False


class TestRatingStabilityMetric:
    """Tests for rating stability metric."""

    def test_stable_ratings(self):
        """Test stability for stable ratings."""
        # Small fluctuations
        history = [1500, 1501, 1500, 1501, 1500, 1501]
        stability = rating_stability_metric(history, window=3)
        assert stability < 2.0  # Small fluctuations, should be small

    def test_unstable_ratings(self):
        """Test stability for unstable ratings."""
        history = [1500, 1600, 1400, 1600, 1400, 1600]
        stability = rating_stability_metric(history, window=3)
        assert stability > 50  # Large swings

    def test_empty_history(self):
        """Test with empty history."""
        stability = rating_stability_metric([])
        assert stability == 0.0

    def test_single_rating(self):
        """Test with single rating."""
        stability = rating_stability_metric([1500])
        assert stability == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
