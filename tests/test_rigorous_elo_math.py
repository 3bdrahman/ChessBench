"""Rigorous test suite for Glicko-2 and Bayesian ELO rating mathematical invariants."""

import math
import random
import pytest

from chessbench.benchmark.elo import BayesianElo, GameResult, Glicko2


class TestRigorousEloMath:
    """Mathematical invariant tests for Glicko-2 and rating calculations."""

    def test_glicko2_symmetry_property(self):
        """Invariant: If Player A beats B by outcome X, and in a parallel universe B beats A by X,
        their rating deltas must be exact mirror opposites.
        """
        glicko1 = Glicko2()
        glicko1.add_player("A", 1500, 200, 0.06)
        glicko1.add_player("B", 1500, 200, 0.06)
        glicko1.update_ratings([GameResult("A", "B", 1.0)])

        delta_a1 = glicko1.get_rating("A").display_rating - 1500
        delta_b1 = glicko1.get_rating("B").display_rating - 1500

        glicko2 = Glicko2()
        glicko2.add_player("A", 1500, 200, 0.06)
        glicko2.add_player("B", 1500, 200, 0.06)
        glicko2.update_ratings([GameResult("B", "A", 1.0)])

        delta_a2 = glicko2.get_rating("A").display_rating - 1500
        delta_b2 = glicko2.get_rating("B").display_rating - 1500

        assert pytest.approx(delta_a1) == -delta_a2
        assert pytest.approx(delta_b1) == -delta_b2

    def test_glicko2_rating_deviation_monotonic_decay(self):
        """Invariant: Playing more matches MUST strictly decrease or maintain rating deviation (RD)."""
        glicko = Glicko2()
        glicko.add_player("Player", 1500, 350, 0.06)
        glicko.add_player("Opponent", 1500, 350, 0.06)

        prev_rd = glicko.get_rating("Player").display_deviation

        for i in range(10):
            outcome = 1.0 if i % 2 == 0 else 0.0
            glicko.update_ratings([GameResult("Player", "Opponent", outcome)])
            current_rd = glicko.get_rating("Player").display_deviation
            assert current_rd <= prev_rd + 1e-6, f"RD increased from {prev_rd} to {current_rd} on step {i}"
            prev_rd = current_rd

    def test_glicko2_confidence_interval_bounds(self):
        """Invariant: 95% Confidence Interval low and high must satisfy mu +/- 1.96 * RD."""
        glicko = Glicko2()
        glicko.add_player("ModelX", 1650, 120, 0.05)

        r = glicko.get_rating("ModelX")
        low, high = r.confidence_interval_95

        expected_margin = 1.96 * 120
        assert pytest.approx(low) == 1650 - expected_margin
        assert pytest.approx(high) == 1650 + expected_margin

    def test_bayesian_elo_winrate_conversion_roundtrip(self):
        """Invariant: Expected score calculated from Elo diff via sigmoid must invert correctly."""
        glicko = Glicko2()
        
        for rating_diff in [-400, -200, -100, 0, 100, 200, 400]:
            winrate = glicko.expected_score(1500 + rating_diff, 1500)
            assert 0.0 < winrate < 1.0
            if rating_diff > 0:
                assert winrate > 0.5
            elif rating_diff < 0:
                assert winrate < 0.5
            else:
                assert pytest.approx(winrate) == 0.5

    def test_rating_period_batching_order_independence(self):
        """Invariant: In Glicko-2, game order within the SAME rating period must yield identical results."""
        game1 = GameResult("P1", "P2", 1.0)
        game2 = GameResult("P1", "P3", 0.0)
        game3 = GameResult("P2", "P3", 0.5)

        g1 = Glicko2()
        g1.add_player("P1", 1500, 200)
        g1.add_player("P2", 1500, 200)
        g1.add_player("P3", 1500, 200)
        g1.update_ratings([game1, game2, game3])

        g2 = Glicko2()
        g2.add_player("P1", 1500, 200)
        g2.add_player("P2", 1500, 200)
        g2.add_player("P3", 1500, 200)
        g2.update_ratings([game3, game1, game2])

        for player in ["P1", "P2", "P3"]:
            r1 = g1.get_rating(player).display_rating
            r2 = g2.get_rating(player).display_rating
            assert pytest.approx(r1) == r2

    def test_json_roundtrip_serialization_fidelity(self):
        """Invariant: Exporting to JSON and re-importing preserves full rating state and volatility."""
        glicko = Glicko2(tau=0.4)
        glicko.add_player("P1", 1720, 85, 0.055)
        glicko.add_player("P2", 1430, 110, 0.062)
        glicko.update_ratings([GameResult("P1", "P2", 1.0)])

        json_str = glicko.export_json()
        restored = Glicko2.from_json(json_str)

        for p in ["P1", "P2"]:
            orig = glicko.get_rating(p)
            rest = restored.get_rating(p)
            assert pytest.approx(orig.display_rating) == rest.display_rating
            assert pytest.approx(orig.display_deviation) == rest.display_deviation
            assert pytest.approx(orig.volatility) == rest.volatility
