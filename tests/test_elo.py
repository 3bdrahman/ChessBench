"""Tests for ELO/Glicko-2 rating system."""

import pytest

from benchmark.elo import BayesianElo, GameResult, Glicko2


class TestGlicko2:
    """Tests for Glicko-2 rating system."""

    def test_basic_rating_update(self):
        glicko = Glicko2()
        glicko.add_player("PlayerA", 1500, 350)
        glicko.add_player("PlayerB", 1500, 350)

        # PlayerA wins
        glicko.update_ratings([GameResult("PlayerA", "PlayerB", 1.0)])

        rating_a = glicko.get_rating("PlayerA")
        rating_b = glicko.get_rating("PlayerB")

        assert rating_a.display_rating > 1500
        assert rating_b.display_rating < 1500

    def test_draw_rating_update(self):
        glicko = Glicko2()
        glicko.add_player("PlayerA", 1500, 350)
        glicko.add_player("PlayerB", 1500, 350)

        # Draw
        glicko.update_ratings([GameResult("PlayerA", "PlayerB", 0.5)])

        rating_a = glicko.get_rating("PlayerA")
        rating_b = glicko.get_rating("PlayerB")

        assert abs(rating_a.display_rating - 1500) < 50
        assert abs(rating_b.display_rating - 1500) < 50

    def test_leaderboard_sorted(self):
        glicko = Glicko2()
        glicko.add_player("PlayerA", 1500, 350)
        glicko.add_player("PlayerB", 1600, 350)
        glicko.add_player("PlayerC", 1400, 350)

        board = glicko.get_leaderboard()
        assert board[0]['name'] == "PlayerB"
        assert board[1]['name'] == "PlayerA"
        assert board[2]['name'] == "PlayerC"

    def test_confidence_interval(self):
        glicko = Glicko2()
        glicko.add_player("PlayerA", 1500, 200)

        rating = glicko.get_rating("PlayerA")
        ci_low, ci_high = rating.confidence_interval_95

        assert ci_low < 1500 < ci_high
        assert ci_high - ci_low == 2 * 1.96 * rating.display_deviation

    def test_json_serialization(self):
        glicko = Glicko2()
        glicko.add_player("PlayerA", 1600, 200)
        glicko.add_player("PlayerB", 1400, 250)

        json_str = glicko.export_json()
        loaded = Glicko2.from_json(json_str)

        assert loaded.get_rating("PlayerA").display_rating == 1600
        assert loaded.get_rating("PlayerB").display_rating == 1400

    def test_many_games_rating_stability(self):
        """Test that ratings stabilize after many games."""
        glicko = Glicko2()
        glicko.add_player("Strong", 1800, 200)
        glicko.add_player("Weak", 1200, 200)

        # Strong player wins 20 games
        for _ in range(20):
            glicko.update_ratings([GameResult("Strong", "Weak", 1.0)])

        strong = glicko.get_rating("Strong")
        weak = glicko.get_rating("Weak")

        assert strong.display_rating > weak.display_rating
        assert strong.display_deviation < 200  # Should decrease
        assert weak.display_deviation < 200

    def test_volatility_convergence(self):
        """Test that volatility converges over time."""
        glicko = Glicko2(tau=0.5)
        glicko.add_player("Player", 1500, 350, 0.06)

        # Play many draws
        for _ in range(50):
            glicko.update_ratings([GameResult("Player", "Opponent", 0.5)])

        player = glicko.get_rating("Player")
        # Volatility should converge to a stable value
        assert 0.01 < player.volatility < 0.5


class TestBayesianElo:
    """Tests for BayesianElo wrapper."""

    def test_add_game(self):
        elo = BayesianElo()
        elo.add_game("White", "Black", 1.0)

        white_rating = elo.get_rating("White")
        black_rating = elo.get_rating("Black")

        assert white_rating.display_rating > black_rating.display_rating

    def test_leaderboard(self):
        elo = BayesianElo()
        elo.add_game("A", "B", 1.0)
        elo.add_game("B", "C", 1.0)

        board = elo.leaderboard()
        assert len(board) == 3
        assert board[0]['name'] == "A"

    def test_cross_table(self):
        """Head-to-head scores: rows=white, cols=black, cells=mean white score."""
        import math
        elo = BayesianElo()
        elo.add_game("A", "B", 1.0)
        elo.add_game("B", "A", 0.5)
        elo.add_game("A", "B", 0.0)

        table = elo.cross_table(["A", "B"])

        assert len(table) == 2
        assert all(len(row) == 2 for row in table)

        assert math.isnan(table[0][0])
        assert math.isnan(table[1][1])

        # A as white vs B averaged (1.0 + 0.0) / 2 = 0.5
        assert math.isclose(table[0][1], 0.5)
        assert math.isclose(table[1][0], 0.5)

    def test_cross_table_unplayed_pairings_are_nan(self):
        """Unplayed pairings are NaN, distinguishing them from draws."""
        elo = BayesianElo()
        elo.add_game("A", "B", 1.0)
        table = elo.cross_table(["A", "B", "C"])
        assert table[2][0] != table[2][0]
        assert table[0][2] != table[0][2]

    def test_cross_table_empty(self):
        """Empty game record yields an all-NaN matrix."""
        import math
        elo = BayesianElo()
        table = elo.cross_table(["A", "B"])
        assert math.isnan(table[0][1]) and math.isnan(table[1][0])

    def test_missing_player_handling(self):
        """Test that missing players are created with default rating."""
        elo = BayesianElo()
        # Don't pre-add players
        elo.add_game("NewPlayer1", "NewPlayer2", 1.0)

        rating1 = elo.get_rating("NewPlayer1")
        rating2 = elo.get_rating("NewPlayer2")

        assert rating1.display_rating != 1500  # Should have updated
        assert rating2.display_rating != 1500


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_game_result_creation(self):
        result = GameResult("White", "Black", 1.0, "A00")
        assert result.white == "White"
        assert result.black == "Black"
        assert result.result == 1.0
        assert result.opening == "A00"

    def test_game_result_defaults(self):
        result = GameResult("White", "Black", 0.5)
        assert result.opening is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
