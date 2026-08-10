"""Base ChessAI class with prompt construction and async move negotiation.

Concrete providers live in :mod:`chess_fight.providers` — this module is the
provider-agnostic base class. Subclasses only need to implement
``_get_move_from_model``; the prompt format, retry policy, and validation are
all handled here.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import chess

from chess_fight.common.common_types import CompletionResult
from chess_fight.common.exceptions import (
    MoveExhaustedError,
    MoveValidationError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from chess_fight.models.evaluation import PositionEvaluator
from chess_fight.prompts import prompt_registry

_log = logging.getLogger(__name__)


class ChessAI(ABC):
    def __init__(self, name: str | None = None, prompt_version: str = "v1_baseline"):
        self.name = name or self.__class__.__name__
        self.move_history: list[str] = []
        self.position_history: set[str] = set()
        self.stagnation_threshold = 3

        self.last_completion_result: CompletionResult | None = None
        self.prompt_version = prompt_version

        # Initialize position evaluator
        self.evaluator = PositionEvaluator()

        # Load prompt template from registry
        self.prompt_template = prompt_registry.get(prompt_version)
        if self.prompt_template is None:
            raise ValueError(f"Unknown prompt version: {prompt_version}. Available: {prompt_registry.list_versions()}")

    def _get_piece_locations(self, board: chess.Board) -> tuple[list[str], list[str]]:
        return self.evaluator.get_piece_locations(board)

    def _get_material_count(self, board: chess.Board) -> str:
        eval_result = self.evaluator.get_material_count(board)
        return str(eval_result)

    def _analyze_material_tension(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_material_tension(board)
        return str(eval_result)

    def _annotate_moves(self, board: chess.Board) -> str:
        return self.evaluator.annotate_moves(board)

    def _analyze_position_repetition(self, board: chess.Board) -> dict[str, Any]:
        current_fen = board.fen().split(' ')[0]

        recent_history = [*self.move_history[-7:], current_fen]
        repetitions = sum(1 for pos in recent_history if pos == current_fen)

        is_stagnating = repetitions >= self.stagnation_threshold

        recent_positions = [*self.move_history[-3:], current_fen]
        unique_positions = len(set(recent_positions))
        progress_score = unique_positions / len(recent_positions)

        return {
            "repetitions": repetitions,
            "is_stagnating": is_stagnating,
            "progress_score": progress_score
        }

    def _analyze_position_progress(self, board: chess.Board, move: chess.Move) -> float:
        return self.evaluator.analyze_position_progress(board, move)

    def _analyze_position_dynamism(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_position_dynamism(board)
        return str(eval_result)

    def _get_castling_rights(self, board: chess.Board) -> str:
        return self.evaluator.get_castling_rights(board)

    def _analyze_capture_value(self, board: chess.Board, move: chess.Move) -> int:
        return self.evaluator.analyze_capture_value(board, move)

    def _calculate_development_score(self, board: chess.Board) -> str:
        eval_result = self.evaluator.calculate_development_score(board)
        return str(eval_result)

    def _analyze_captures(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_captures(board)
        return str(eval_result)

    def _analyze_threats(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_threats(board)
        return str(eval_result)

    def _evaluate_capture(self, board: chess.Board, move: chess.Move) -> float:
        return self.evaluator.evaluate_capture(board, move)

    def _categorize_moves(self, board: chess.Board) -> dict[str, str]:
        moves_dict = self.evaluator.categorize_moves(board)
        return {
            'forcing_moves': str(moves_dict['forcing_moves']),
            'developing_moves': str(moves_dict['developing_moves']),
            'positional_moves': str(moves_dict['positional_moves']),
        }

    def _analyze_defense(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_defense(board)
        return str(eval_result)

    def _analyze_vulnerabilities(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_vulnerabilities(board)
        return str(eval_result)

    def _analyze_king_safety(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_king_safety(board)
        return str(eval_result)

    def _is_pinned(self, board: chess.Board, square: int) -> bool:
        return self.evaluator.is_pinned(board, square)

    def _analyze_pawn_structure(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_pawn_structure(board)
        return str(eval_result)

    def _analyze_undefended_pieces(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_undefended_pieces(board)
        return str(eval_result)

    def _analyze_exposed_pieces(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_exposed_pieces(board)
        return str(eval_result)

    def _analyze_material_balance(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_material_balance(board)
        return str(eval_result)

    def _analyze_center_control(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_center_control(board)
        return str(eval_result)

    def _analyze_development_status(self, board: chess.Board) -> str:
        eval_result = self.evaluator.analyze_development_status(board)
        return str(eval_result)

    def _create_prompt(self, fen: str) -> str:
        board = chess.Board(fen)
        moves = self._categorize_moves(board)
        position_analysis = self._analyze_position_repetition(board)

        context = {
            "color": "White" if board.turn == chess.WHITE else "Black",
            "position_repetitions": position_analysis["repetitions"],
            "stagnation_status": "STAGNATING - Force dynamic play!" if position_analysis["is_stagnating"] else "Normal",
            "position_progress": f"{position_analysis['progress_score']:.2f}",
            "material_tension": str(self.evaluator.analyze_material_tension(board)),
            "position_dynamism": str(self.evaluator.analyze_position_dynamism(board)),
            "development_score": str(self.evaluator.calculate_development_score(board)),
            "defense_analysis": str(self.evaluator.analyze_defense(board)),
            "vulnerability_analysis": str(self.evaluator.analyze_vulnerabilities(board)),
            "capture_analysis": str(self.evaluator.analyze_captures(board)),
            "king_safety": str(self.evaluator.analyze_king_safety(board)),
            "undefended_pieces": str(self.evaluator.analyze_undefended_pieces(board)),
            "exposed_pieces": str(self.evaluator.analyze_exposed_pieces(board)),
            "ascii_board": str(board),
            "material_count": str(self.evaluator.get_material_count(board)),
            "material_balance": str(self.evaluator.analyze_material_balance(board)),
            "center_control": str(self.evaluator.analyze_center_control(board)),
            "development_status": str(self.evaluator.analyze_development_status(board)),
            "forcing_moves": moves['forcing_moves'],
            "developing_moves": moves['developing_moves'],
            "positional_moves": moves['positional_moves'],
        }

        assert self.prompt_template is not None, "Prompt template should be initialized"
        return self.prompt_template.render(context)

    def _validate_move(self, move_str: str, board: chess.Board) -> str:
        from chess_fight.move_parser import validate_move

        result = validate_move(move_str, board)
        if result is None:
            raise MoveValidationError(
                f"Invalid move: {move_str}",
                fen=board.fen(),
                legal_moves=[m.uci() for m in board.legal_moves],
                raw_text=move_str,
            )
        return result

    def _is_valid_square(self, square: str) -> bool:
        if len(square) != 2:
            return False
        file, rank = square[0], square[1]
        return (
            file in 'abcdefgh' and
            rank in '12345678'
        )

    async def get_move(self, fen: str) -> str:
        board = chess.Board(fen)
        max_network_retries = 10
        max_validation_retries = 3
        network_attempts = 0
        validation_attempts = 0
        errors: list[str] = []
        attempted_moves: list[str] = []

        while True:
            if network_attempts >= max_network_retries or validation_attempts >= max_validation_retries:
                break
                
            try:
                move_str = await self._get_move_from_model(fen, validation_attempts)
                attempted_moves.append(move_str)
                validated_move = self._validate_move(move_str, board)

                current_fen = board.fen().split(' ')[0]
                self.move_history.append(current_fen)

                return validated_move
            except (RateLimitError, TimeoutError) as exc:
                network_attempts += 1
                wait = getattr(exc, "retry_after", None) or (2.0 ** network_attempts)
                wait = min(wait, 60.0)
                _log.info(
                    "get_move retry network_attempt=%d/%d fen=%s error=%s wait=%.1fs",
                    network_attempts, max_network_retries, fen, type(exc).__name__, wait
                )
                await asyncio.sleep(wait)
                errors.append(f"Network Attempt {network_attempts}: {type(exc).__name__}: {exc}")
            except MoveValidationError as exc:
                validation_attempts += 1
                errors.append(f"Validation Attempt {validation_attempts}: MoveValidationError: {exc}")
            except ProviderError as exc:
                _log.error("get_move non-retryable error fen=%s error=%s", fen, exc)
                raise
            except ValueError as exc:
                validation_attempts += 1
                errors.append(f"Validation Attempt {validation_attempts}: {exc}")

        legal_moves = list(board.legal_moves)
        legal_moves_uci = [m.uci() for m in legal_moves]

        raise MoveExhaustedError(
            f"Failed to get valid move after {validation_attempts} validation / {network_attempts} network attempts. Errors: {'; '.join(errors)}",
            fen=fen,
            legal_moves=legal_moves_uci,
            attempted_moves=attempted_moves,
            raw_text=errors[-1] if errors else "",
        )

    async def get_move_with_result(self, fen: str) -> tuple[str, "CompletionResult"]:
        board = chess.Board(fen)
        max_network_retries = 10
        max_validation_retries = 3
        network_attempts = 0
        validation_attempts = 0
        errors: list[str] = []
        attempted_moves: list[str] = []

        while True:
            if network_attempts >= max_network_retries or validation_attempts >= max_validation_retries:
                break
                
            try:
                move_str = await self._get_move_from_model(fen, validation_attempts)
                attempted_moves.append(move_str)
                validated_move = self._validate_move(move_str, board)

                current_fen = board.fen().split(' ')[0]
                self.move_history.append(current_fen)

                return validated_move, self.last_completion_result or CompletionResult(
                    text=move_str,
                    tokens_in=None,
                    tokens_out=None,
                    latency_ms=0,
                    raw_response=None,
                )
            except (RateLimitError, TimeoutError) as exc:
                network_attempts += 1
                wait = getattr(exc, "retry_after", None) or (2.0 ** network_attempts)
                wait = min(wait, 60.0)
                _log.info(
                    "get_move_with_result retry network_attempt=%d/%d fen=%s error=%s wait=%.1fs",
                    network_attempts, max_network_retries, fen, type(exc).__name__, wait
                )
                await asyncio.sleep(wait)
                errors.append(f"Network Attempt {network_attempts}: {type(exc).__name__}: {exc}")
            except MoveValidationError as exc:
                validation_attempts += 1
                errors.append(f"Validation Attempt {validation_attempts}: MoveValidationError: {exc}")
            except ProviderError as exc:
                _log.error("get_move_with_result non-retryable error fen=%s error=%s", fen, exc)
                raise
            except ValueError as exc:
                validation_attempts += 1
                errors.append(f"Validation Attempt {validation_attempts}: {exc}")

        legal_moves = list(board.legal_moves)
        legal_moves_uci = [m.uci() for m in legal_moves]

        raise MoveExhaustedError(
            f"Failed to get valid move after {validation_attempts} validation / {network_attempts} network attempts. Errors: {'; '.join(errors)}",
            fen=fen,
            legal_moves=legal_moves_uci,
            attempted_moves=attempted_moves,
            raw_text=errors[-1] if errors else "",
        )

    @abstractmethod
    async def _get_move_from_model(self, fen: str, validation_attempt: int = 0) -> str:
        """Return the model's UCI move suggestion for the given FEN position.

        Implementations should populate ``self.last_completion_result`` with
        the raw provider response so downstream consumers (logging, UI)
        can read tokens, latency, and the model's free-form reasoning.
        """

