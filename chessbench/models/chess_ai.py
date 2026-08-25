"""Base ChessAI class with simple prompt construction and async move negotiation.

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
from chessbench.prompts import render_messages

if TYPE_CHECKING:
    from chessbench.common.common_types import ChatMessage

_log = logging.getLogger(__name__)


class ChessAI(ABC):
    def __init__(
        self,
        name: str | None = None,
        reasoning_level: str = "high",
        system_prompt: str | None = None,
        turn_prompt: str | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.move_history: list[str] = []
        self.position_history: set[str] = set()

        self.last_completion_result: CompletionResult | None = None

        if reasoning_level not in ("low", "mid", "high"):
            reasoning_level = "high"
        self.reasoning_level = reasoning_level

        # Custom prompts override the default (for A/B testing if needed)
        self.custom_system_prompt = system_prompt
        self.custom_turn_prompt = turn_prompt

    def reset_game(self) -> None:
        """Reset per-game history (move_history, position_history, last_completion_result)."""
        self.move_history.clear()
        self.position_history.clear()
        self.last_completion_result = None

    def _create_messages(self, fen: str) -> list["ChatMessage"]:
        """Create structured ChatMessage list (system and user role messages)."""
        color = "White" if chess.Board(fen).turn == chess.WHITE else "Black"

        # Use custom prompts if provided, otherwise use defaults
        if self.custom_system_prompt and self.custom_turn_prompt:
            board = chess.Board(fen)
            legal_uci = " ".join(m.uci() for m in board.legal_moves)
            ctx = {
                "color": color,
                "fen": fen,
                "ascii_board": str(board),
                "legal_moves_uci": legal_uci,
            }
            system_content = self.custom_system_prompt.format(**ctx)
            user_content = self.custom_turn_prompt.format(**ctx)

            reasoning_directives = {
                "low": "Be concise. Reasoning under 30 words.",
                "mid": "Concise strategic & tactical reasoning (under 150 words).",
                "high": "Deep step-by-step tactical calculation and candidate move evaluation.",
            }
            reasoning = reasoning_directives.get(self.reasoning_level, reasoning_directives["high"])
            system_content += f"\n\n[REASONING LEVEL: {self.reasoning_level.upper()}]\n{reasoning}"

            from chessbench.common.common_types import ChatMessage
            return [
                ChatMessage(role="system", content=system_content),
                ChatMessage(role="user", content=user_content),
            ]

        # Use default prompts
        return render_messages(fen, color, self.reasoning_level)

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