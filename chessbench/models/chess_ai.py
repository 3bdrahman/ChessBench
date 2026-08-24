"""Base ChessAI class with prompt construction and async move negotiation.

Concrete providers live in :mod:`chessbench.providers` — this module is the
provider-agnostic base class. Subclasses only need to implement
``_get_move_from_model``; the prompt format, retry policy, and validation are
all handled here.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import chess

from chessbench.common.common_types import CompletionResult
from chessbench.common.exceptions import (
    MoveExhaustedError,
    MoveFormatError,
    MoveValidationError,
    ProviderError,
    is_retryable,
)
from chessbench.models.evaluation import PositionEvaluator
from chessbench.prompts import PromptTemplate, prompt_registry

if TYPE_CHECKING:
    from chessbench.common.common_types import ChatMessage

_log = logging.getLogger(__name__)


def get_reasoning_directive(level: str) -> str:
    """Return reasoning directive string for given reasoning level.

    Args:
        level: Reasoning level ("low", "mid", "high").

    Returns:
        The reasoning directive string for the given level.
        For unknown/invalid level, falls back to "high".
    """
    reasoning_directives = {
        "low": (
            "\n\n[REASONING LEVEL: LOW]\n"
            "Be extremely fast and concise. Keep reasoning under 30 words, or output the move directly in <move>uci_move</move> tags."
        ),
        "mid": (
            "\n\n[REASONING LEVEL: MID]\n"
            "Provide concise strategic and tactical reasoning (under 150 words) in <think> tags before your chosen move in <move>uci_move</move> tags."
        ),
        "high": (
            "\n\n[REASONING LEVEL: HIGH]\n"
            "Perform deep step-by-step tactical calculation, candidate move evaluation, and king safety analysis in <think> tags before your move in <move>uci_move</move> tags."
        ),
    }
    return reasoning_directives.get(level, reasoning_directives["high"])


class ChessAI(ABC):
    def __init__(
        self,
        name: str | None = None,
        prompt_version: str = "v1_baseline",
        reasoning_level: str = "high",
        system_prompt: str | None = None,
        turn_prompt: str | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.move_history: list[str] = []
        self.position_history: set[str] = set()
        self.stagnation_threshold = 3

        self.last_completion_result: CompletionResult | None = None
        self.prompt_version = prompt_version
        self.prompt_template: PromptTemplate

        if reasoning_level not in ("low", "mid", "high"):
            reasoning_level = "high"
        self.reasoning_level = reasoning_level

        # Initialize position evaluator
        self.evaluator = PositionEvaluator()

        if system_prompt is not None and turn_prompt is not None:
            from chessbench.prompts import create_safe_prompt_template
            template, validation = create_safe_prompt_template(system_prompt, turn_prompt)
            self.prompt_template = template
            self.used_fallback_prompt = validation.used_fallback
            self.fallback_reason = validation.fallback_reason
        else:
            self.used_fallback_prompt = False
            self.fallback_reason = None
            pt = prompt_registry.get(prompt_version)
            if pt is None:
                raise ValueError(f"Unknown prompt version: {prompt_version}. Available: {prompt_registry.list_versions()}")
            self.prompt_template = pt

    def reset_game(self) -> None:
        """Reset per-game history (move_history, position_history, last_completion_result)."""
        self.move_history.clear()
        self.position_history.clear()
        self.last_completion_result = None

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
            'forcing_moves': "\n".join(moves_dict['forcing_moves'].pv) if moves_dict['forcing_moves'].pv else "None",
            'developing_moves': "\n".join(moves_dict['developing_moves'].pv) if moves_dict['developing_moves'].pv else "None",
            'positional_moves': "\n".join(moves_dict['positional_moves'].pv) if moves_dict['positional_moves'].pv else "None",
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

    def _get_annotated_legal_moves(self, board: chess.Board) -> str:
        """Format legal moves pairing UCI with SAN, e.g., 'c8b7 (Bxb7), f6e5 (fxe5)'."""
        moves = [f"{m.uci()} ({board.san(m)})" for m in board.legal_moves]
        return ", ".join(moves)

    def _get_last_move_san(self, board: chess.Board) -> str:
        """Format opponent's previous move in SAN and UCI, e.g. '2... Nc6 (b8c6)'."""
        if not board.move_stack:
            return "None (First move of the game)"
        last_move = board.peek()
        temp = board.copy()
        temp.pop()
        move_num = (len(board.move_stack) + 1) // 2
        prefix = f"{move_num}." if temp.turn == chess.WHITE else f"{move_num}..."
        return f"{prefix} {temp.san(last_move)} ({last_move.uci()})"

    def _get_move_history_san(self, board: chess.Board, max_moves: int = 10) -> str:
        """Reconstruct SAN game history, e.g. '1. e4 e5 2. Nf3 Nc6'."""
        if not board.move_stack:
            return "None (Starting position)"
        try:
            root = board.root()
            temp = root.copy()
            san_moves: list[str] = []
            for i, move in enumerate(board.move_stack):
                san = temp.san(move)
                temp.push(move)
                if i % 2 == 0:
                    san_moves.append(f"{(i//2)+1}. {san}")
                else:
                    san_moves[-1] += f" {san}"

            if max_moves and len(san_moves) > max_moves:
                return f"... {' '.join(san_moves[-max_moves:])}"
            return " ".join(san_moves)
        except Exception:
            return "Not available"

    def _get_piece_locations_str(self, board: chess.Board) -> tuple[str, str]:
        """Format piece locations for White and Black."""
        w, b = self.evaluator.get_piece_locations(board)
        return ", ".join(w), ", ".join(b)

    def _get_prompt_context(self, board: chess.Board) -> dict[str, Any]:
        """Compute the demand-driven prompt context dictionary for the given board position."""
        assert self.prompt_template is not None, "Prompt template should be initialized"
        from chessbench.prompts.sample_context import build_sample_context

        return build_sample_context(
            board,
            move_history=self.move_history,
            reasoning_level=self.reasoning_level,
            stagnation_threshold=self.stagnation_threshold,
            prompt_template=self.prompt_template,
            evaluator=self.evaluator,
        )

    def _get_reasoning_directive(self) -> str:
        """Return reasoning directive string for current reasoning level."""
        return get_reasoning_directive(self.reasoning_level)

    def _create_prompt(self, fen: str) -> str:
        assert self.prompt_template is not None, "Prompt template should be initialized"
        board = chess.Board(fen)
        context = self._get_prompt_context(board)
        base_prompt = self.prompt_template.render(context)
        return base_prompt + self._get_reasoning_directive()

    def _create_messages(self, fen: str) -> list["ChatMessage"]:
        """Create structured ChatMessage list (system and user role messages)."""
        assert self.prompt_template is not None, "Prompt template should be initialized"
        board = chess.Board(fen)
        context = self._get_prompt_context(board)

        try:
            messages = self.prompt_template.render_messages(
                context,
                include_output_contract=True,
                system_suffix=self._get_reasoning_directive()
            )
        except Exception as exc:
            _log.warning("Prompt rendering failed: %s. Using default fallback prompt.", exc)
            from chessbench.prompts import create_safe_prompt_template
            fallback_template, _ = create_safe_prompt_template(None, None)
            self.prompt_template = fallback_template
            self.used_fallback_prompt = True
            self.fallback_reason = f"Runtime rendering error: {exc}"
            messages = self.prompt_template.render_messages(
                context,
                include_output_contract=True,
                system_suffix=self._get_reasoning_directive()
            )

        return messages

    def _validate_move(self, move_str: str, board: chess.Board) -> str:
        from chessbench.move_parser import parse_move

        result = parse_move(move_str, board)

        if result is None or result.uci is None:
            # Check if it outputted an illegal move
            raw_result = parse_move(move_str, None)
            if raw_result and raw_result.uci:
                raise MoveValidationError(
                    f"You attempted an ILLEGAL move: {raw_result.uci}. This move is not valid in the current position.",
                    fen=board.fen(),
                    legal_moves=[m.uci() for m in board.legal_moves],
                    raw_text=move_str,
                )
            else:
                raise MoveFormatError(
                    f"Could not extract legal move from response: {move_str[:100]}...",
                    fen=board.fen(),
                    legal_moves=[m.uci() for m in board.legal_moves],
                    raw_text=move_str,
                )
        return result.uci

    def _is_valid_square(self, square: str) -> bool:
        if len(square) != 2:
            return False
        file, rank = square[0], square[1]
        return (
            file in 'abcdefgh' and
            rank in '12345678'
        )

    async def _invoke_get_move_from_model(self, fen: str, validation_attempt: int, network_attempts: int) -> str:
        # Subclass implementations may have different signatures; try progressively fewer args
        try:
            return await self._get_move_from_model(fen, validation_attempt, network_attempts)
        except TypeError:
            try:
                return await self._get_move_from_model(fen, validation_attempt)
            except TypeError:
                return await self._get_move_from_model(fen)

    async def get_move(self, fen: str) -> str:
        board = chess.Board(fen)
        max_network_retries = 3
        max_validation_retries = 3
        network_attempts = 0
        validation_attempts = 0
        errors: list[str] = []
        attempted_moves: list[str] = []

        while True:
            if network_attempts >= max_network_retries or validation_attempts >= max_validation_retries:
                break

            try:
                move_str = await self._invoke_get_move_from_model(fen, validation_attempts, network_attempts)
                attempted_moves.append(move_str)
                validated_move = self._validate_move(move_str, board)

                current_fen = board.fen().split(' ')[0]
                self.move_history.append(current_fen)

                return validated_move
            except ProviderError as exc:
                if is_retryable(exc):
                    network_attempts += 1
                    wait = getattr(exc, "retry_after", None) or (2.0 ** min(network_attempts, 3))
                    wait = min(wait, 15.0)
                    _log.info(
                        "get_move retry network_attempt=%d/%d fen=%s error=%s wait=%.1fs",
                        network_attempts, max_network_retries, fen, type(exc).__name__, wait
                    )
                    await asyncio.sleep(wait)
                    errors.append(f"Network Attempt {network_attempts}: {type(exc).__name__}: {exc}")
                else:
                    _log.error("get_move non-retryable error fen=%s error=%s", fen, exc)
                    raise exc
            except MoveValidationError as exc:
                validation_attempts += 1
                errors.append(f"Validation Attempt {validation_attempts}: MoveValidationError: {exc}")
            except Exception as exc:
                if is_retryable(exc):
                    network_attempts += 1
                    wait = 2.0 ** min(network_attempts, 3)
                    wait = min(wait, 15.0)
                    _log.warning("get_move unexpected retryable error: %s", exc)
                    await asyncio.sleep(wait)
                    errors.append(f"Unexpected Error Attempt {network_attempts}: {exc}")
                else:
                    _log.error("get_move non-retryable exception fen=%s error=%s", fen, exc)
                    raise exc

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
        max_network_retries = 3
        max_validation_retries = 3
        network_attempts = 0
        validation_attempts = 0
        errors: list[str] = []
        attempted_moves: list[str] = []

        while True:
            if network_attempts >= max_network_retries or validation_attempts >= max_validation_retries:
                break

            try:
                # We pass network_attempts so the provider can correctly populate the UI metric
                move_str = await self._invoke_get_move_from_model(fen, validation_attempts, network_attempts)
                attempted_moves.append(move_str)
                validated_move = self._validate_move(move_str, board)

                current_fen = board.fen().split(' ')[0]
                self.move_history.append(current_fen)

                cr = self.last_completion_result or CompletionResult(
                    text=move_str,
                    tokens_in=None,
                    tokens_out=None,
                    latency_ms=0,
                    raw_response=None,
                )
                cr.validation_retries = validation_attempts
                return validated_move, cr
            except ProviderError as exc:
                if is_retryable(exc):
                    network_attempts += 1
                    wait = getattr(exc, "retry_after", None) or (2.0 ** min(network_attempts, 3))
                    wait = min(wait, 15.0)
                    _log.info(
                        "get_move_with_result retry network_attempt=%d/%d fen=%s error=%s wait=%.1fs",
                        network_attempts, max_network_retries, fen, type(exc).__name__, wait
                    )
                    await asyncio.sleep(wait)
                    errors.append(f"Network Attempt {network_attempts}: {type(exc).__name__}: {exc}")
                else:
                    _log.error("get_move_with_result non-retryable error fen=%s error=%s", fen, exc)
                    raise exc
            except MoveValidationError as exc:
                validation_attempts += 1
                errors.append(f"Validation Attempt {validation_attempts}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                if is_retryable(exc):
                    network_attempts += 1
                    wait = 2.0 ** min(network_attempts, 3)
                    wait = min(wait, 15.0)
                    _log.warning("get_move_with_result unexpected retryable error: %s", exc)
                    await asyncio.sleep(wait)
                    errors.append(f"Unexpected Error Attempt {network_attempts}: {exc}")
                else:
                    _log.error("get_move_with_result non-retryable exception fen=%s error=%s", fen, exc)
                    raise exc

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
    async def _get_move_from_model(self, fen: str, validation_attempt: int = 0, network_attempts: int = 0) -> str:
        """Return the model's UCI move suggestion for the given FEN position.

        Implementations should populate ``self.last_completion_result`` with
        the raw provider response so downstream consumers (logging, UI)
        can read tokens, latency, and the model's free-form reasoning.
        """

