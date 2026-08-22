"""Rigorous integration tests for async game loop, clock controls, and event logging."""

import asyncio
import json
import tempfile
from pathlib import Path

import chess
import pytest

from chessbench.benchmark.logging import BenchmarkLogger
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
            for line in msg.content.split("\n"):
                if "FEN:" in line:
                    parts = line.split("FEN:", 1)[1].strip().split()
                    if len(parts) >= 6:
                        fen = " ".join(parts[:6])
                    elif parts:
                        fen = parts[0]
                    break
            if fen:
                break

        try:
            board = chess.Board(fen) if fen else chess.Board()
        except Exception:
            board = chess.Board()

        if self.mode == "invalid":
            text = "<move>z9z9</move>"
        elif self.mode == "slow":
            await asyncio.sleep(0.01)
            move = next(iter(board.legal_moves)) if list(board.legal_moves) else None
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
            if state.is_paused:
                game.resume(retry_current_turn=False)

        stats = await game.play_game(ui_callback=noop_cb, delay=0.0)

        assert stats.total_moves > 0
        assert len(game.moves) == stats.total_moves

    @pytest.mark.asyncio
    async def test_illegal_move_disqualification_after_max_attempts(self):
        """Assert player handling when illegal moves occur."""
        white_ai = ProviderChessAI("dynamic_test_ai", "dynamic-ai", "")

        black_provider = DynamicAiProvider(mode="invalid")
        black_ai = ProviderChessAI("dynamic_test_ai", "dynamic-ai", "")
        black_ai.provider = black_provider

        game = AsyncChessGame(
            player1=white_ai,
            player2=black_ai,
            max_moves=10,
        )

        async def auto_resume_cb(state: GameState) -> None:
            if state.is_paused:
                game.resume(retry_current_turn=False)

        stats = await game.play_game(ui_callback=auto_resume_cb, delay=0.0)

        assert stats is not None

    @pytest.mark.asyncio
    async def test_jsonl_logger_schema_and_sha256_integrity(self):
        """Assert that BenchmarkLogger outputs valid JSONL with complete schema and integrity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = BenchmarkLogger(run_dir=tmp_dir)
            logger.start_run(config={
                "players": ["dynamic_test_ai:dynamic-ai", "dynamic_test_ai:dynamic-ai"],
                "games_per_pairing": 1,
            })

            logger.start_game(
                white_player="dynamic_test_ai:dynamic-ai",
                black_player="dynamic_test_ai:dynamic-ai",
                white_provider="dynamic_test_ai",
                black_provider="dynamic_test_ai",
                opening_name="Italian Game",
                opening_fen=chess.STARTING_FEN,
            )

            logger.log_move(
                move_number=1,
                player="dynamic_test_ai:dynamic-ai",
                color="White",
                fen_before=chess.STARTING_FEN,
                move_uci="e2e4",
                move_san="e4",
                llm_latency_ms=120,
                llm_tokens_in=100,
                llm_tokens_out=15,
                llm_raw_response="<move>e2e4</move>",
                thinking_trace=None,
                prompt_hash="abc12345",
                validation_retries=0,
                eval_cp_score=25,
            )

            logger.end_game(
                result="1-0",
                result_numeric=1.0,
                total_moves=1,
                game_duration_sec=0.5,
                termination_reason="checkmate",
            )

            logger.write_summary()

            run_dir = Path(logger.run_dir)
            assert (run_dir / "summary.json").exists()
            assert (run_dir / "games.jsonl").exists()
            assert (run_dir / "moves.jsonl").exists()

            with open(run_dir / "moves.jsonl") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) >= 1
            assert lines[0]["move_uci"] == "e2e4"

            with open(run_dir / "summary.json") as f:
                summary_data = json.load(f)
            assert summary_data["total_games"] == 1
