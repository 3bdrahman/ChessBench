"""Adversarial evaluation mode (LLM vs Stockfish at calibrated depths)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import chess

from chessbench.benchmark.evaluator import StockfishEvaluator
from chessbench.game.async_game import AsyncChessGame
from chessbench.game.clock import GameClock
from chessbench.providers import get_provider
from chessbench.providers.chess_ai import ProviderChessAI


@dataclass
class AdversarialConfig:
    """Configuration for adversarial evaluation."""
    stockfish_depths: list[int] = field(default_factory=lambda: [8, 12, 16, 20])
    games_per_depth: int = 4
    colors: str = "alternating"  # "alternating", "white", "black"
    time_control_seconds: int = 30
    max_parallel_games: int = 2

    def __post_init__(self) -> None:
        pass


@dataclass
class DepthResult:
    """Result for a single depth."""
    depth: int
    games: int
    llm_score: float  # LLM score vs Stockfish
    llm_cp_loss: float
    draw_rate: float
    win_rate: float
    loss_rate: float


@dataclass
class AdversarialReport:
    """Complete adversarial evaluation report."""
    model_name: str
    model_provider: str
    depth_results: list[DepthResult]
    equivalent_depth: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "depth_results": [
                {
                    "depth": dr.depth,
                    "games": dr.games,
                    "llm_score": dr.llm_score,
                    "llm_cp_loss": dr.llm_cp_loss,
                    "draw_rate": dr.draw_rate,
                    "win_rate": dr.win_rate,
                    "loss_rate": dr.loss_rate,
                }
                for dr in self.depth_results
            ],
            "equivalent_depth": self.equivalent_depth,
        }


class AdversarialEvaluator:
    """Evaluates LLMs against Stockfish at calibrated depths."""

    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.evaluator = StockfishEvaluator()

    async def evaluate_model(
        self,
        model_spec: str,  # e.g., "openai:gpt-4o"
        games_per_depth: int | None = None,
    ) -> AdversarialReport:
        """Run adversarial evaluation for a single model."""
        provider_name, model_id = model_spec.split(':', 1)

        # Create provider AI
        provider = get_provider(provider_name)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Get API key from environment or config
        api_key = os.getenv(f"{provider_name.upper()}_API_KEY", "")

        llm_ai = ProviderChessAI(
            provider_name=provider_name,
            model_id=model_id,
            api_key=api_key,
            temperature=0.0,
            max_tokens=100
        )

        games_per_depth = games_per_depth or self.config.games_per_depth
        depths = self.config.stockfish_depths

        all_depth_results = []

        for depth in depths:
            print(f"  Testing {model_spec} vs Stockfish depth {depth}...")

            # Create Stockfish AI at this depth
            stockfish_ai = self._create_stockfish_ai(depth)

            # Run games
            results = await self._run_depth_match(
                llm_ai, stockfish_ai, depth, games_per_depth
            )

            all_depth_results.append(results)

        # Calculate equivalent depth
        equivalent_depth = self._calculate_equivalent_depth(all_depth_results)

        return AdversarialReport(
            model_name=model_spec,
            model_provider=provider_name,
            depth_results=all_depth_results,
            equivalent_depth=equivalent_depth,
        )

    def _create_stockfish_ai(self, depth: int) -> ProviderChessAI:
        """Create a Stockfish AI at specified depth."""
        provider = get_provider("stockfish")
        if not provider:
            raise ValueError("Stockfish provider not available")

        return ProviderChessAI(
            provider_name="stockfish",
            model_id=f"depth-{depth}",
            api_key="",
            temperature=0.0,
            max_tokens=100
        )

    async def _run_depth_match(
        self,
        llm_ai: ProviderChessAI,
        stockfish_ai: ProviderChessAI,
        depth: int,
        games: int,
    ) -> DepthResult:
        """Run games between LLM and Stockfish at a specific depth."""
        # We'll use a simplified version of the runner logic
        wins = 0
        losses = 0
        draws = 0
        total_games = 0

        for game_idx in range(games):
            # Determine colors
            if self.config.colors == "alternating":
                llm_is_white = (game_idx % 2 == 0)
            elif self.config.colors == "white":
                llm_is_white = True
            else:
                llm_is_white = False

            white_ai = llm_ai if llm_is_white else stockfish_ai
            black_ai = stockfish_ai if llm_is_white else llm_ai

            # Create game
            board = chess.Board()
            clock = GameClock.from_seconds(self.config.time_control_seconds)
            game = AsyncChessGame(white_ai, black_ai, clock=clock)
            game.board = board.copy()

            # Play game with evaluation
            await self.evaluator.start()

            # Simple game loop with evaluation
            cp_losses = []
            total_games = 0
            while not game.board.is_game_over():
                current_player = white_ai if len(game.moves) % 2 == 0 else black_ai
                fen_before = game.board.fen()

                # Get move with evaluation
                move_str, _completion_result = await current_player.get_move_with_result(fen_before)
                move = chess.Move.from_uci(move_str)

                # Evaluate position before move
                eval_result = await self.evaluator.evaluate(game.board)

                if move in game.board.legal_moves:
                    game.board.push(move)

# Track cp loss for LLM moves
            if ((len(game.moves) % 2 == 0 and llm_is_white) or (len(game.moves) % 2 == 1 and not llm_is_white)) and eval_result and eval_result.cp_score is not None and eval_result.best_move_cp is not None:
                cp_loss = eval_result.best_move_cp - eval_result.cp_score
                cp_losses.append(cp_loss)

            # Determine result
            outcome = game.board.outcome(claim_draw=False)
            if outcome:
                if outcome.winner is None:
                    draws += 1
                elif (outcome.winner and llm_is_white) or (not outcome.winner and not llm_is_white):
                    wins += 1
                else:
                    losses += 1

            total_games += 1
            await self.evaluator.stop()

        games_played = wins + losses + draws
        llm_score = (wins + 0.5 * draws) / games_played if games_played > 0 else 0
        avg_cp_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0

        return DepthResult(
            depth=depth,
            games=games_played,
            llm_score=llm_score,
            llm_cp_loss=avg_cp_loss,
            draw_rate=draws / games_played if games_played > 0 else 0,
            win_rate=wins / games_played if games_played > 0 else 0,
            loss_rate=losses / games_played if games_played > 0 else 0,
        )

    def _calculate_equivalent_depth(self, depth_results: list[DepthResult]) -> float | None:
        """Calculate equivalent Stockfish depth via interpolation."""
        if len(depth_results) < 2:
            return None

        # Sort by depth
        depth_results.sort(key=lambda x: x.depth)

        # Find where score crosses 0.5 (equal to Stockfish)
        for i in range(len(depth_results) - 1):
            d1 = depth_results[i]
            d2 = depth_results[i + 1]

            if d1.llm_score <= 0.5 <= d2.llm_score or d2.llm_score <= 0.5 <= d1.llm_score:
                # Linear interpolation
                t = (0.5 - d1.llm_score) / (d2.llm_score - d1.llm_score) if d2.llm_score != d1.llm_score else 0.5
                return d1.depth + t * (d2.depth - d1.depth)

        # If never crosses 0.5, extrapolate
        # (This is a rough estimate)
        return None


