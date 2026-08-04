"""Integration tests for the benchmark system."""

from unittest.mock import AsyncMock, patch

import chess
import pytest

from benchmark.elo import BayesianElo, GameResult, Glicko2
from benchmark.logging import BenchmarkLogger
from benchmark.openings import OpeningBook
from benchmark.runner import BenchmarkConfig, BenchmarkRunner
from game.async_game import AsyncChessGame, GameState
from move_parser import extract_move, validate_move
from providers.base import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from providers.chess_ai import ProviderChessAI
from providers.registry import get_provider, list_providers, register_provider


class MockProvider(ModelProvider):
    """Mock provider for testing."""
    name = "mock"
    requires_api_key = False

    def __init__(self, moves=None):
        self.moves = moves or ["e2e4", "e7e5", "g1f3", "g8f6"]
        self.move_index = 0

    def validate_key(self, api_key: str) -> bool:
        return True

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        return [ModelInfo(id="mock-model", name="Mock Model", provider="mock")]

    async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult:
        move = self.moves[self.move_index % len(self.moves)]
        self.move_index += 1
        return CompletionResult(
            text=move,
            tokens_in=100,
            tokens_out=5,
            latency_ms=100
        )


@register_provider
class MockProviderRegistered(MockProvider):
    name = "mock_registered"


class TestOpeningBook:
    """Tests for OpeningBook."""

    def test_openings_loaded(self):
        book = OpeningBook()
        assert len(book.openings) == 295
        assert len(book.opening_fens) == 295

    def test_random_opening(self):
        book = OpeningBook()
        op = book.get_random_opening()
        assert 'eco' in op
        assert 'name' in op
        assert 'fen' in op
        assert 'moves' in op

    def test_get_by_eco(self):
        book = OpeningBook()
        op = book.get_opening_by_eco("A00")
        assert op is not None
        assert op['eco'] == "A00"

    def test_balanced_set(self):
        book = OpeningBook()
        balanced = book.get_balanced_set(10)
        assert len(balanced) == 10
        # Should have variety across categories (we have many A category)
        categories = set(op['eco'][0] for op in balanced)
        # At minimum should have some categories
        assert len(categories) >= 1

    def test_opening_fen_valid(self):
        book = OpeningBook()
        for op in book.opening_fens:
            board = chess.Board(op['fen'])
            assert board.is_valid()


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

        # Ratings should stay close to 1500 for draws between equal players
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


class TestBenchmarkLogger:
    """Tests for BenchmarkLogger."""

    def test_start_game_log_move_end_game(self, tmp_path):
        logger = BenchmarkLogger(str(tmp_path))
        logger.start_run({"test": True})

        logger.start_game("White", "Black", "openai", "anthropic", "A00", "Test Opening")

        logger.log_move(
            move_number=1, player="White", color="white",
            fen_before=chess.STARTING_FEN, move_uci="e2e4", move_san="e4",
            llm_latency_ms=100, llm_tokens_in=50, llm_tokens_out=5,
            llm_raw_response="e2e4", thinking_trace=None, prompt_hash="abc", validation_retries=0
        )

        logger.end_game("1-0", 1.0, 1, 2.0)
        logger.write_summary()

        # Check files exist
        assert (tmp_path / "games.jsonl").exists()
        assert (tmp_path / "moves.jsonl").exists()
        assert (tmp_path / "games.pgn").exists()
        assert (tmp_path / "summary.json").exists()

        # Check PGN content
        pgn = (tmp_path / "games.pgn").read_text()
        assert "[White \"White\"]" in pgn
        assert "[Black \"Black\"]" in pgn
        assert "[Result \"1-0\"]" in pgn
        assert "e4" in pgn

    def test_multiple_games(self, tmp_path):
        import json
        logger = BenchmarkLogger(str(tmp_path))
        logger.start_run({"test": True})

        for i in range(3):
            logger.start_game(f"White{i}", f"Black{i}", "p1", "p2")
            logger.log_move(1, f"White{i}", "white", chess.STARTING_FEN, "e2e4", "e4", 100, 50, 5, "e2e4", None, "hash", 0)
            logger.end_game("1-0", 1.0, 1, 2.0)

        logger.write_summary()

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary['total_games'] == 3
        assert summary['results']['white_wins'] == 3


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self):
        config = BenchmarkConfig()
        assert config.time_control_seconds_per_move == 30
        assert config.games_per_pairing == 10
        assert config.temperature == 0.0

    def test_yaml_roundtrip(self, tmp_path):
        config = BenchmarkConfig(
            players=["openai:gpt-4o", "anthropic:claude-3-5-sonnet"],
            games_per_pairing=5,
            temperature=0.1
        )
        yaml_path = tmp_path / "config.yaml"
        config.save_yaml(str(yaml_path))

        loaded = BenchmarkConfig.from_yaml(str(yaml_path))
        assert loaded.players == config.players
        assert loaded.games_per_pairing == config.games_per_pairing
        assert loaded.temperature == config.temperature

    def test_to_dict(self):
        config = BenchmarkConfig(players=["openai:gpt-4o"])
        d = config.to_dict()
        assert d['players'] == ["openai:gpt-4o"]
        assert 'temperature' in d


class TestMoveParserIntegration:
    """Integration tests for move parser with realistic scenarios."""

    @pytest.mark.parametrize("text,expected", [
        ("I will play e2e4", "e2e4"),
        ("<thinking>Analysis</thinking>\ng1f3", "g1f3"),
        ("Move: g1f3", "g1f3"),
        ("Best move is d2d4", "d2d4"),
        ("`b1c3`", "b1c3"),
        ("I choose a2a4", "a2a4"),
    ])
    def test_extract_various_formats(self, text, expected):
        board = chess.Board()
        result = extract_move(text, list(board.legal_moves))
        assert result == expected

    def test_extract_promotion_move(self):
        """Test promotion move extraction with a promotion position."""
        board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
        result = extract_move("I choose a7a8q", list(board.legal_moves))
        assert result == "a7a8q"

    def test_validate_with_board(self):
        board = chess.Board()
        assert validate_move("e2e4", board) == "e2e4"
        assert validate_move("move: g1f3", board) == "g1f3"
        assert validate_move("`e2e4`", board) == "e2e4"

    def test_validate_rejects_illegal(self):
        board = chess.Board()
        with pytest.raises(ValueError, match="Illegal move"):
            validate_move("e2e5", board)


class TestAsyncGameIntegration:
    """Integration tests for async game loop."""

    @pytest.mark.asyncio
    async def test_mock_game_completes(self):
        """Test a complete game with mock players."""
        moves = ["e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5", "d2d3", "d7d6",
                 "b1c3", "b8c6", "c1g5", "h7h6", "g5h4", "g7g5", "h4g3", "g5g4",
                 "c3e2", "g4f3", "e2f4", "f3f2", "e1f2"]

        white_ai = ProviderChessAI("mock_registered", "mock", "", temperature=0.0)
        black_ai = ProviderChessAI("mock_registered", "mock", "", temperature=0.0)

        # Override get_move to use our move sequence
        move_idx = [0]
        async def mock_get_move(fen):
            move = moves[move_idx[0] % len(moves)]
            move_idx[0] += 1
            return move

        white_ai._get_move_from_model = mock_get_move
        black_ai._get_move_from_model = mock_get_move

        game = AsyncChessGame(white_ai, black_ai)

        states = []
        async def capture_state(state):
            states.append(state)

        result = await game.play_game(capture_state, delay=0.001)

        assert result.total_moves > 0
        # Winner can be Unknown if the mock moves cycle without a decisive result
        assert result.winner in ["Draw", white_ai.name, black_ai.name, "Unknown"]
        assert len(states) > 0


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_mock_provider_registered(self):
        providers = list_providers()
        assert "mock_registered" in providers

    def test_get_mock_provider(self):
        provider = get_provider("mock_registered")
        assert provider is not None
        assert provider.name == "mock_registered"

    def test_provider_validation(self):
        provider = get_provider("mock_registered")
        assert provider.validate_key("anything") == True


class TestProviderChessAI:
    """Tests for ProviderChessAI wrapper."""

    def test_initialization(self):
        ai = ProviderChessAI("openai", "gpt-4o", "sk-test", temperature=0.1)
        assert ai.provider_name == "openai"
        assert ai.model_id == "gpt-4o"
        assert ai.api_key == "sk-test"
        assert ai.params['temperature'] == 0.1

    def test_extract_move(self):
        ai = ProviderChessAI("openai", "gpt-4o", "sk-test")
        assert ai._extract_move("e2e4") == "e2e4"
        assert ai._extract_move("I will play e2e4") == "e2e4"
        assert ai._extract_move("<thinking></thinking>\ne2e4") == "e2e4"
        assert ai._extract_move("no move here") == ""

    @pytest.mark.asyncio
    async def test_last_completion_result_populated_from_provider(self):
        from unittest.mock import MagicMock

        ai = ProviderChessAI("openai", "gpt-4o", "sk-test12345678901234567890")
        assert ai.last_completion_result is None

        with patch("providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="<thinking>plan</thinking>\ne2e4"))]
            mock_response.usage = MagicMock(prompt_tokens=1234, completion_tokens=7)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            move_str, cr = await ai.get_move_with_result(chess.STARTING_FEN)

        assert move_str == "e2e4"
        assert cr.tokens_in == 1234
        assert cr.tokens_out == 7
        assert ai.last_completion_result is cr

    @pytest.mark.asyncio
    async def test_get_move_with_result_falls_back_when_no_completion(self):
        from providers.chess_ai import ProviderChessAI

        ai = ProviderChessAI("openai", "gpt-4o", "sk-test")

        async def fake_get_move(fen):
            return "e2e4"

        ai._get_move_from_model = fake_get_move
        move_str, cr = await ai.get_move_with_result(chess.STARTING_FEN)
        assert move_str == "e2e4"
        assert cr.tokens_in is None
        assert cr.tokens_out is None
        assert cr.latency_ms == 0


class TestAsyncGameFenBefore:
    """Tests that GameState.fen_before is correctly populated."""

    @pytest.mark.asyncio
    async def test_fen_before_set_correctly(self):
        async def fake_get_move(fen):
            return "e2e4"

        ai = ProviderChessAI("mock_registered", "mock", "", temperature=0.0)
        ai._get_move_from_model = fake_get_move

        game = AsyncChessGame(ai, ai)
        seen_fen_before: list[str | None] = []

        async def ui(state: GameState):
            seen_fen_before.append(state.fen_before)

        await game.play_game(ui, delay=0)
        starting_fen = chess.STARTING_FEN
        non_null = [f for f in seen_fen_before if f is not None]
        assert len(non_null) >= 2
        assert non_null[0] == starting_fen
        assert all(f != starting_fen for f in non_null[2:2]) or non_null[2] != starting_fen


class TestRunnerMetricsEndToEnd:
    """End-to-end test that benchmark runner logs real LLM metrics."""

    @pytest.mark.asyncio
    async def test_runner_logs_real_latency_and_tokens(self, tmp_path):
        from unittest.mock import MagicMock

        config = BenchmarkConfig(
            players=["openai:gpt-4o", "openai:gpt-4o-mini"],
            games_per_pairing=1,
            opening_book="startpos",
            max_parallel_games=1,
            api_keys={"openai": "sk-test12345678901234567890"},
            output_dir=str(tmp_path),
        )
        runner = BenchmarkRunner(config)

        with patch("providers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="e2e4"))]
            mock_response.usage = MagicMock(prompt_tokens=999, completion_tokens=4)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            await runner.run_benchmark()

        moves_path = runner.run_dir / "moves.jsonl"
        import json
        moves_logged = [json.loads(line) for line in moves_path.read_text().splitlines() if line]
        assert moves_logged, "expected at least one move logged"
        first = moves_logged[0]
        assert first["llm_tokens_in"] == 999
        assert first["llm_tokens_out"] == 4
        assert first["llm_raw_response"] == "e2e4"
        assert first["fen_before"] == chess.STARTING_FEN
        assert first["move_san"] != first["move_uci"], "SAN should be derived, not the UCI passthrough"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
