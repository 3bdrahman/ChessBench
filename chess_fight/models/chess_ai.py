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
    is_retryable,
)
from chess_fight.models.evaluation import PositionEvaluator
from chess_fight.prompts import prompt_registry

_log = logging.getLogger(__name__)


class ChessAI(ABC):
    def __init__(
        self,
        name: str | None = None,
        prompt_version: str = "v1_baseline",
        reasoning_level: str = "mid",
    ):
        self.name = name or self.__class__.__name__
        self.move_history: list[str] = []
        self.position_history: set[str] = set()
        self.stagnation_threshold = 3

        self.last_completion_result: CompletionResult | None = None
        self.prompt_version = prompt_version
        
        if reasoning_level not in ("low", "mid", "high"):
            reasoning_level = "mid"
        self.reasoning_level = reasoning_level

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

        # Determine which variables the active template actually references
        # so we only compute what's needed — no dead-weight evaluation.
        needed = self.prompt_template.referenced_variables()

        # --- Always-cheap variables (trivial to compute) ---
        context: dict[str, Any] = {
            "fen": board.fen(),
            "color": "White" if board.turn == chess.WHITE else "Black",
            "reasoning_level": self.reasoning_level,
        }

        # --- Rich context helpers ---
        if "legal_moves_annotated" in needed:
            context["legal_moves_annotated"] = self._get_annotated_legal_moves(board)

        if "last_move_san" in needed:
            context["last_move_san"] = self._get_last_move_san(board)

        if "move_history_san" in needed:
            context["move_history_san"] = self._get_move_history_san(board)

        if needed & {"white_pieces", "black_pieces"}:
            w_str, b_str = self._get_piece_locations_str(board)
            context["white_pieces"] = w_str
            context["black_pieces"] = b_str

        # --- Move categorization (needed by all current templates) ---
        # Shared dependency: categorize_moves is needed by the UCI-list
        # variables AND the PositionEval object variables.
        needs_moves = needed & {
            "forcing_moves", "developing_moves", "positional_moves",
            "legal_moves_uci", "forcing_uci", "developing_uci", "positional_uci",
        }
        moves = None
        if needs_moves:
            moves = self.evaluator.categorize_moves(board)

            if needed & {"forcing_moves", "developing_moves", "positional_moves"}:
                context["forcing_moves"] = moves["forcing_moves"]
                context["developing_moves"] = moves["developing_moves"]
                context["positional_moves"] = moves["positional_moves"]

            if needed & {"legal_moves_uci", "forcing_uci", "developing_uci", "positional_uci"}:
                def extract_uci_moves(pos_eval: "PositionEval") -> list[str]:
                    uci_moves = []
                    for desc in pos_eval.pv or []:
                        if "[" in desc and "]" in desc:
                            uci = desc[desc.rfind("[") + 1 : desc.rfind("]")]
                            if len(uci) in (4, 5):
                                uci_moves.append(uci)
                    return uci_moves

                forcing_uci = extract_uci_moves(moves["forcing_moves"])
                developing_uci = extract_uci_moves(moves["developing_moves"])
                positional_uci = extract_uci_moves(moves["positional_moves"])

                context["legal_moves_uci"] = " ".join(forcing_uci + developing_uci + positional_uci)
                context["forcing_uci"] = " ".join(forcing_uci)
                context["developing_uci"] = " ".join(developing_uci)
                context["positional_uci"] = " ".join(positional_uci)

        # --- Board representation ---
        if "ascii_board" in needed:
            context["ascii_board"] = str(board)

        # --- Repetition / stagnation analysis ---
        if needed & {"position_repetitions", "stagnation_status", "position_progress"}:
            position_analysis = self._analyze_position_repetition(board)
            context["position_repetitions"] = position_analysis["repetitions"]
            context["stagnation_status"] = (
                "STAGNATING - Force dynamic play!" if position_analysis["is_stagnating"] else "Normal"
            )
            context["position_progress"] = f"{position_analysis['progress_score']:.2f}"

        # --- Evaluations: only computed when the template references them ---
        _eval_map: dict[str, Any] = {
            "material_tension": lambda: str(self.evaluator.analyze_material_tension(board)),
            "position_dynamism": lambda: str(self.evaluator.analyze_position_dynamism(board)),
            "development_score": lambda: str(self.evaluator.calculate_development_score(board)),
            "defense_analysis": lambda: str(self.evaluator.analyze_defense(board)),
            "vulnerability_analysis": lambda: str(self.evaluator.analyze_vulnerabilities(board)),
            "capture_analysis": lambda: str(self.evaluator.analyze_captures(board)),
            "king_safety": lambda: str(self.evaluator.analyze_king_safety(board)),
            "undefended_pieces": lambda: str(self.evaluator.analyze_undefended_pieces(board)),
            "exposed_pieces": lambda: str(self.evaluator.analyze_exposed_pieces(board)),
            "material_count": lambda: str(self.evaluator.get_material_count(board)),
            "material_balance": lambda: str(self.evaluator.analyze_material_balance(board)),
            "center_control": lambda: str(self.evaluator.analyze_center_control(board)),
            "development_status": lambda: str(self.evaluator.analyze_development_status(board)),
        }
        for var_name in needed & _eval_map.keys():
            context[var_name] = _eval_map[var_name]()

        return context

    def _get_reasoning_directive(self) -> str:
        """Return reasoning directive string for current reasoning level."""
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
        return reasoning_directives.get(self.reasoning_level, reasoning_directives["mid"])

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
        messages = self.prompt_template.render_messages(context)

        reasoning_directive = self._get_reasoning_directive()
        if messages and messages[0].role == "system":
            messages[0].content += reasoning_directive
        elif messages:
            messages[0].content += reasoning_directive

        return messages

    def _validate_move(self, move_str: str, board: chess.Board) -> str:
        from chess_fight.move_parser import parse_move

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
                raise MoveValidationError(
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

                return validated_move, self.last_completion_result or CompletionResult(
                    text=move_str,
                    tokens_in=None,
                    tokens_out=None,
                    latency_ms=0,
                    raw_response=None,
                )
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
                errors.append(f"Validation Attempt {validation_attempts}: MoveValidationError: {exc}")
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
    async def _get_move_from_model(self, fen: str, validation_attempt: int = 0) -> str:
        """Return the model's UCI move suggestion for the given FEN position.

        Implementations should populate ``self.last_completion_result`` with
        the raw provider response so downstream consumers (logging, UI)
        can read tokens, latency, and the model's free-form reasoning.
        """

