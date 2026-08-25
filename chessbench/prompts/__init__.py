"""Single solid prompt for chess benchmarking — no prompt engineering complexity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from chessbench.common.common_types import ChatMessage


# ─── The One Prompt That Works ───
SYSTEM_PROMPT = (
    "You are a strong chess engine. Play the best move for {color}. "
    "Output your reasoning in  tags, then your move in <move> tags as UCI (e.g., e2e4)."
)

TURN_PROMPT = """Position:
{ascii_board}

FEN: {fen}
Your color: {color}

Legal moves (UCI): {legal_moves_uci}

Select the best move for {color}."""

# Backward compatibility aliases for UI components
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
DEFAULT_TURN_PROMPT = TURN_PROMPT


def build_prompt_context(board: chess.Board, color: str) -> dict[str, str]:
    """Build minimal context for the prompt — only what's actually needed."""
    legal_uci = " ".join(m.uci() for m in board.legal_moves)
    return {
        "color": color,
        "fen": board.fen(),
        "ascii_board": str(board),
        "legal_moves_uci": legal_uci,
    }


def render_messages(fen: str, color: str, reasoning_level: str = "high") -> list[ChatMessage]:
    """Render the complete chat messages for a given position."""
    board = chess.Board(fen)
    ctx = build_prompt_context(board, color)

    reasoning_directives = {
        "low": "Be concise. Reasoning under 30 words.",
        "mid": "Concise strategic & tactical reasoning (under 150 words).",
        "high": "Deep step-by-step tactical calculation and candidate move evaluation.",
    }
    reasoning = reasoning_directives.get(reasoning_level, reasoning_directives["high"])

    system_content = SYSTEM_PROMPT.format(**ctx) + f"\n\n[REASONING LEVEL: {reasoning_level.upper()}]\n{reasoning}"
    user_content = TURN_PROMPT.format(**ctx)

    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ]


# ─── Minimal validation for UI backward compatibility ───
@dataclass
class PromptValidationResult:
    """Minimal validation result for UI compatibility."""
    is_valid: bool
    errors: list[str]
    used_fallback: bool
    warnings: list[str] = None
    missing_variables: list[str] = None
    unrecognized_variables: list[str] = None
    suggestions: list[str] = None
    estimated_tokens: int = 0
    rendered_tokens_estimate: int = 0
    fallback_reason: str | None = None
    sanitized_system_prompt: str = ""
    sanitized_turn_prompt: str = ""

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.missing_variables is None:
            self.missing_variables = []
        if self.unrecognized_variables is None:
            self.unrecognized_variables = []
        if self.suggestions is None:
            self.suggestions = []


def validate_prompt_text(
    system_prompt: str | None,
    turn_prompt: str | None,
) -> PromptValidationResult:
    """Minimal prompt validation — checks basic sanity only."""
    errors: list[str] = []

    sys_text = (system_prompt or "").strip()
    turn_text = (turn_prompt or "").strip()

    if not sys_text:
        errors.append("System prompt cannot be empty.")

    if not turn_text:
        errors.append("Turn prompt cannot be empty.")

    # Check for required placeholders
    required_vars = {"color", "fen", "ascii_board", "legal_moves_uci"}
    all_text = sys_text + turn_text
    import re
    found_vars = set(re.findall(r"\{(\w+)\}", all_text))
    missing = required_vars - found_vars
    if missing:
        errors.append(f"Missing required placeholders: {', '.join(sorted(missing))}")

    is_valid = len(errors) == 0

    return PromptValidationResult(
        is_valid=is_valid,
        errors=errors,
        used_fallback=not is_valid,
    )