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


from chess_fight.constants import REASONING_MAX_TOKENS


class ProviderChessAI(ChessAI):
    """ChessAI implementation using the provider abstraction layer."""

    def __init__(
        self,
        provider_name: str,
        model_id: str,
        api_key: str,
        reasoning_level: str = "mid",
        **params: Any,
    ) -> None:
        if "reasoning_level" in params:
            reasoning_level = params.pop("reasoning_level")
        super().__init__(reasoning_level=reasoning_level)
        self.provider_name = provider_name
        self.model_id = model_id
        self.api_key = api_key
        self.params = params  # temperature, max_tokens, etc.

        provider = get_provider(provider_name)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")
        self.provider = provider

        self.name = f"{provider_name}:{model_id}"

    async def _get_move_from_model(self, fen: str, validation_attempt: int = 0, network_attempts: int = 0) -> str:
        board = chess.Board(fen)
        legal_moves_uci = " ".join(m.uci() for m in board.legal_moves)

        if validation_attempt == 0:
            messages = self._create_messages(fen)
        elif validation_attempt == 1:
            # First retry: append a stern warning to the full prompt and enforce Structured Output via tools
            prompt = self._create_prompt(fen)
            prompt += (
                f"\n\n[SYSTEM WARNING]: Your previous attempt FAILED. You either "
                f"reasoned for too long without outputting a move, or your output "
                f"format was wrong. You MUST use the provided tool to output your move.\n"
                f"Legal moves: {legal_moves_uci}"
            )
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            # Second+ retry: explicit instruction demanding tool usage without truncating thinking.
            color = "White" if board.turn == chess.WHITE else "Black"
            prompt = (
                f"You are playing chess as {color}.\n"
                f"FEN: {fen}\n"
                f"Legal moves (UCI): {legal_moves_uci}\n\n"
                f"CRITICAL INSTRUCTION: You MUST use the make_chess_move tool to submit your move."
            )
            messages = [ChatMessage(role="user", content=prompt)]

        # Pass FEN explicitly so providers like Stockfish can use it directly
        # instead of trying to parse it from the full prompt.
        params = dict(self.params)
        params["fen"] = fen
        params["reasoning_level"] = self.reasoning_level

        # Inject Tool constraints on Retry
        if validation_attempt >= 1:
            params["tools"] = [{
                "type": "function",
                "function": {
                    "name": "make_chess_move",
                    "description": "Submit your chosen chess move.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "Brief strategic plan or explanation"
                            },
                            "uci_move": {
                                "type": "string",
                                "enum": [m.uci() for m in board.legal_moves],
                                "description": "The exact UCI format of your chosen move"
                            }
                        },
                        "required": ["reasoning", "uci_move"]
                    }
                }
            }]
            params["tool_choice"] = {"type": "function", "function": {"name": "make_chess_move"}}

        # Determine max_tokens: increase on retries so reasoning models
        # (like Nemotron or Qwen-Thinking) never get cut off mid-thought.
        base_max_tokens = REASONING_MAX_TOKENS.get(self.reasoning_level, 1024)
        if validation_attempt >= 1:
            # Give ample headroom on retries so thinking preambles don't cause truncation
            retry_max_tokens = max(1536, int(base_max_tokens * 1.5))
        else:
            retry_max_tokens = base_max_tokens
        if params.get("max_tokens") is None:
            params["max_tokens"] = retry_max_tokens

        while True:
            try:
                result = await self.provider.complete(
                    self.api_key,
                    self.model_id,
                    messages,
                    **params
                )
                break
            except (RateLimitError, TimeoutError) as exc:
                # Populate last_completion_result with error info before re-raising
                self.last_completion_result = CompletionResult(
                    text=str(exc),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    raw_response=getattr(exc, "raw_response", None),
                    latency_ms=getattr(exc, "latency_ms", 0),
                    retry_count=network_attempts + 1,
                )
                raise
            except ProviderError as exc:
                self.last_completion_result = CompletionResult(
                    text=str(exc),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    raw_response=getattr(exc, "raw_response", None),
                    latency_ms=getattr(exc, "latency_ms", 0),
                    retry_count=network_attempts + 1,
                )
                raise

        # Extract and analyze thinking from the result
        extract_and_analyze_thinking(result.text)

        self.last_completion_result = result
        if self.last_completion_result:
            self.last_completion_result.retry_count = network_attempts
        board = chess.Board(fen)
        
        # Check for tool call extracted move
        if result.tool_calls:
            for tool_call in result.tool_calls:
                args = tool_call.get("arguments", {})
                if isinstance(args, dict) and "uci_move" in args:
                    uci_move = args["uci_move"]
                    if uci_move in [m.uci() for m in board.legal_moves]:
                        return uci_move

        from chess_fight.move_parser import parse_move
        parsed = parse_move(result.text, board)
        
        if not parsed.uci:
            raw_result = parse_move(result.text, None)
            if raw_result and raw_result.uci:
                raise MoveValidationError(
                    f"You attempted an ILLEGAL move: {raw_result.uci}. This move is not valid in the current position.",
                    fen=fen,
                    legal_moves=[m.uci() for m in board.legal_moves],
                    raw_text=result.text,
                )
            else:
                raise MoveValidationError(
                    f"Could not extract legal move from response: {result.text[:200]}...",
                    fen=fen,
                    legal_moves=[m.uci() for m in board.legal_moves],
                    raw_text=result.text,
                )
        return parsed.uci

    def _extract_move(self, text: str) -> str:
        """Extract a UCI move from raw LLM text.

        Thin wrapper over :func:`chess_fight.move_parser.extract_move` kept
        for backward compatibility — returns ``""`` when no legal-looking
        move is found, matching the historical contract.
        """
        return extract_move(text) or ""

