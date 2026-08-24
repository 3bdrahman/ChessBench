"""
Pure logic for the Prompt Workbench.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

# We'll avoid importing streamlit here to keep this module pure and testable without Streamlit.
# We'll import chess only in the UI functions that need it.


class ValidatablePrompt(Protocol):
    """Protocol for prompt validation results with required attributes."""
    is_valid: bool
    errors: list[str]
    used_fallback: bool


@dataclass
class ValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    errors: list[str]
    used_fallback: bool


def compute_budget_state(rendered: int, window: int | None) -> Literal["ok", "warning", "error", "standard"]:
    """
    Compare rendered token estimate vs model context window.
    Returns:
        "ok": >50% headroom (rendered < 50% of window)
        "warning": <20% headroom (rendered > 80% of window)
        "error": over-budget (rendered > window)
        "standard": window is None or falsy (use standard)
    """
    if not window:
        return "standard"
    if rendered > window:
        return "error"
    # Calculate headroom percentage: (window - rendered) / window * 100
    headroom_pct = (window - rendered) / window * 100
    if headroom_pct > 50:
        return "ok"
    if headroom_pct < 20:
        return "warning"
    return "ok"  # fallback, though the above should cover all cases


def can_launch_match(
    v1: ValidatablePrompt,
    v2: ValidatablePrompt,
) -> tuple[bool, str | None]:
    """
    Determine if we can launch a match based on two validation results.
    Returns (can_launch, error_message). If can_launch is True, error_message is None.
    """
    errors = []
    if not v1.is_valid:
        errors.append("P1 Prompt Invalid:\n" + "\n".join(f"- {e}" for e in v1.errors))
    if not v2.is_valid:
        errors.append("P2 Prompt Invalid:\n" + "\n".join(f"- {e}" for e in v2.errors))
    if v1.used_fallback:
        errors.append("P1 would use fallback prompt (validation failed silently).")
    if v2.used_fallback:
        errors.append("P2 would use fallback prompt (validation failed silently).")

    if errors:
        return False, "\n\n".join(errors)
    return True, None


def is_ab_eligible(
    model_1_config: dict,
    model_2_config: dict,
    sys_1: str,
    turn_1: str,
    sys_2: str,
    turn_2: str,
) -> bool:
    """
    Determine if the match is eligible for A/B mode.
    A/B mode is allowed only when:
        model_1 == model_2 (same provider and model_id)
        AND (sys_1, turn_1) != (sys_2, turn_2)
    """
    same_model = (
        model_1_config.get("provider") == model_2_config.get("provider")
        and model_1_config.get("model_id") == model_2_config.get("model_id")
    )
    if not same_model:
        return False
    same_strategy = (sys_1 == sys_2) and (turn_1 == turn_2)
    return not same_strategy
