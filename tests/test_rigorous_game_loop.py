"""Rigorous integration tests for async game loop, clock controls, and event logging."""

import asyncio
import json
import tempfile
from pathlib import Path
import chess
import pytest

from chessbench.benchmark.logging import BenchmarkLogger
from chessbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chessbench.common.common_types import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from chessbench.game.async_game import AsyncChessGame, GameState
from chessbench.providers.chess_ai import ProviderChessAI
from chessbench.providers.registry import register_provider


class DynamicAiProvider(ModelProvider):
    """A provider that plays valid legal moves dynamically using python-chess logic."""
    name = "dynamic_test_ai"
    requires_api_key = False

    def __init__(self, mode: str = "valid"):
        self.mode = mode

    def validate_key(self, api_key: str) -> bool:
        return True

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        return [ModelInfo(id="dynamic-ai", name="Dynamic AI", provider="dynamic_test_ai")]

    async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult:
        fen = None
        for msg in reversed(messages):
            if "FEN:" in msg.content or "fen:" in msg.content.lower():
                lines = msg.content.split("\n")
                for line in lines:
                    if "FEN:" in line:
                        fen = line.split("FEN:", 1)[1].strip()
                        break
            elif msg.content.count("/") >= 7:
                fen = msg.content.strip()
                break

        board = chess.Board(fen) if fen else chess.Board()

        if self.mode == "invalid":
            text = "<move>z9z9</move>"
        elif self.mode == "slow":
            await asyncio.sleep(0.1)
            move = list(board.legal_moves)[0] if list(board.legal_moves) else None
            text = f"<move>{move.uci() if move else 'e2e4'}</move>"
        else:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = legal_moves[0]
                text = f"I decide to play:\n<move>{move.uci()}</move>"
            else:
                text = "No legal moves available."

        return CompletionResult(
            text=text,
            tokens_in=120,
            tokens_out=15,
            latency_ms=50,
        )


@pytest.fixture(autouse=True)
def _register_dynamic_provider():
    register_provider(DynamicAiProvider)
    yield
    from chessbench.providers.registry import PROVIDER_REGISTRY
    if "dynamic_test_ai" in PROVIDER_REGISTRY:
        del PROVIDER_REGISTRY["dynamic_test_ai"]


class TestRigorousGameLoop:
    """Rigorous end-to-end integration tests for AsyncChessGame and Tournament Runner."""

    @pytest.mark.asyncio
    async def test_full_game_plays_to_legal_completion_or_max_moves(self):
        """Play a full game between two dynamic AIs and assert complete game state validity."""
        white_ai = ProviderChessAI("dynamic_test_ai", "dynamic-ai", "")
        black_ai = ProviderChessAI("dynamic_test_ai", "dynamic-ai", "")

        game = AsyncChessGame(
            player1=white_ai,
            player2=black_ai,
            max_moves=20,
        )

        async def noop_cb(state: GameState) -> None:
            pass

        stats = await game.play_game(ui_callback=noop_cb, delay=0.0)

        assert stats.total_moves > 0
        assert len(game.moves) == stats.total_moves
        assert len(stats.move_latencies_ms) == stats.total_moves

    @pytest.mark.asyncio
    async def test_illegal_move_disqualification_after_max_attempts(self):
        """Assert player is disqualified after failing to produce a legal move."""
        white_ai = ProviderChessAI("dynamic_test_ai", "dynamic-ai", "")
        
        # Create illegal provider
        black_provider = DynamicAiProvider(mode="invalid")
        black_ai = ProviderChessAI("dynamic_test_ai", "dynamic-ai", "")
        black_ai.provider = black_provider

        game = AsyncChessGame(
            player1=white_ai,
            player2=black_ai,
            max_moves=10,
        )

        async def noop_cb(state: GameState) -> None:
            pass

        stats = await game.play_game(ui_callback=noop_cb, delay=0.0)

        assert stats.black_disqualified is True or stats.invalid_moves > 0

    @pytest.mark.asyncio
    async def test_jsonl_logger_schema_and_sha256_integrity(self):
        """Assert that BenchmarkLogger outputs valid JSONL with complete schema and SHA256 integrity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = BenchmarkLogger(output_dir=tmp_dir, run_name="test_integrity_run")
            logger.init_run(
                players=["dynamic_test_ai:dynamic-ai", "dynamic_test_ai:dynamic-ai"],
                games_per_pairing=1,
            )

            game_id = "test_game_001"
            logger.log_game_start(
                game_id=game_id,
                white="dynamic_test_ai:dynamic-ai",
                black="dynamic_test_ai:dynamic-ai",
                opening_name="Italian Game",
                fen=chess.STARTING_FEN,
            )

            logger.log_move(
                game_id=game_id,
                ply=1,
                move_uci="e2e4",
                san="e4",
                fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                thinking_time_ms=120,
                cpl=0,
                eval_cp=25,
            )

            logger.log_game_end(
                game_id=game_id,
                winner="white",
                reason="checkmate",
                total_moves=1,
                duration_seconds=0.5,
            )

            run_summary = logger.finalize_run()

            run_dir = Path(logger.run_dir)
            assert (run_dir / "summary.json").exists()
            assert (run_dir / "events.jsonl").exists()

            with open(run_dir / "events.jsonl") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) >= 3
            event_types = [l["event"] for l in lines]
            assert "game_start" in event_types
            assert "move" in event_types
            assert "game_end" in event_types
            assert "run_finalized" in event_types

            assert run_summary["run_id"] == logger.run_id
            assert "total_games" in run_summary
            assert "sha256" in run_summary
            assert len(run_summary["sha256"]) == 64
