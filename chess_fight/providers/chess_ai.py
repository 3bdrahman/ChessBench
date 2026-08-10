"""Provider-agnostic ChessAI wrapper.

Delegates chat completion to a registered :class:`ModelProvider` and parses
the free-form text response into a UCI move via :mod:`chess_fight.move_parser`.
"""

from typing import Any

import chess

from chess_fight.common.common_types import ChatMessage, CompletionResult
from chess_fight.common.exceptions import (
    MoveValidationError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from chess_fight.models.chess_ai import ChessAI
from chess_fight.models.thinking import extract_and_analyze_thinking
from chess_fight.move_parser import extract_move
from chess_fight.providers.registry import get_provider


class ProviderChessAI(ChessAI):
    """ChessAI implementation using the provider abstraction layer."""

    def __init__(self, provider_name: str, model_id: str, api_key: str, **params: Any) -> None:
        super().__init__()
        self.provider_name = provider_name
        self.model_id = model_id
        self.api_key = api_key
        self.params = params  # temperature, max_tokens, etc.

        provider = get_provider(provider_name)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")
        self.provider = provider

        self.name = f"{provider_name}:{model_id}"

    async def _get_move_from_model(self, fen: str, validation_attempt: int = 0) -> str:
        prompt = self._create_prompt(fen)
        if validation_attempt > 0:
            prompt += (
                f"\n\n[SYSTEM WARNING]: Your previous attempt failed because you either reasoned "
                f"for too long without outputting a move, or your move was invalid. "
                f"You MUST output a legal UCI move enclosed in <move></move> tags immediately."
            )

        # Pass FEN explicitly so providers like Stockfish can use it directly
        # instead of trying to parse it from the full prompt.
        params = dict(self.params)
        params["fen"] = fen

        retry_count = 0
        while True:
            try:
                result = await self.provider.complete(
                    self.api_key,
                    self.model_id,
                    [ChatMessage(role="user", content=prompt)],
                    **params
                )
                break
            except (RateLimitError, TimeoutError) as exc:
                retry_count += 1
                # Populate last_completion_result with error info before re-raising
                self.last_completion_result = CompletionResult(
                    text=str(exc),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    raw_response=getattr(exc, "raw_response", None),
                    latency_ms=getattr(exc, "latency_ms", 0),
                    retry_count=retry_count,
                )
                raise
            except ProviderError as exc:
                self.last_completion_result = CompletionResult(
                    text=str(exc),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    raw_response=getattr(exc, "raw_response", None),
                    latency_ms=getattr(exc, "latency_ms", 0),
                    retry_count=retry_count,
                )
                raise

        # Extract and analyze thinking from the result
        extract_and_analyze_thinking(result.text)

        self.last_completion_result = result
        if self.last_completion_result:
            self.last_completion_result.retry_count = retry_count
        board = chess.Board(fen)
        move = extract_move(result.text, list(board.legal_moves))
        if not move:
            raise MoveValidationError(
                f"Could not extract legal move from response: {result.text[:100]}",
                fen=fen,
                legal_moves=[m.uci() for m in board.legal_moves],
                raw_text=result.text,
            )
        return move

    def _extract_move(self, text: str) -> str:
        """Extract a UCI move from raw LLM text.

        Thin wrapper over :func:`chess_fight.move_parser.extract_move` kept
        for backward compatibility — returns ``""`` when no legal-looking
        move is found, matching the historical contract.
        """
        return extract_move(text) or ""

