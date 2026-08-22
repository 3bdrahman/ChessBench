"""Structured prompt system with validation, prompt injection defense, and fallbacks."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger(__name__)

# Default Fallback Prompts (guaranteed to be safe, valid, and contains all required variables)
DEFAULT_SYSTEM_PROMPT = "You are a professional chess engine playing as {color}."

DEFAULT_TURN_PROMPT = """Position:
{ascii_board}

Board FEN: {fen}

Legal Moves:
Forcing: {forcing_moves}
Developing: {developing_moves}
Positional: {positional_moves}

Select the best move for {color}.
You MUST format your response EXACTLY like this:
<think>
(Your reasoning here)
</think>
<move>
(Your chosen move in purely lower-case UCI notation, e.g. e2e4)
</move>"""

# Required variable placeholder groups
REQUIRED_BOARD_VARS = {"fen", "ascii_board", "board", "color"}
REQUIRED_MOVE_VARS = {
    "legal_moves",
    "legal_moves_uci",
    "legal_moves_annotated",
    "forcing_moves",
    "developing_moves",
    "positional_moves",
    "forcing_uci",
    "developing_uci",
    "positional_uci",
}

# Prompt injection threat patterns
PROMPT_INJECTION_PATTERNS = [
    (r"ignore\s+(?:all\s+)?previous\s+instructions", "Instruction override attempt detected"),
    (r"disregard\s+(?:all\s+)?prior\s+instructions", "Instruction disregard attempt detected"),
    (r"you\s+are\s+no\s+longer\s+playing\s+chess", "Role hijacking attempt detected"),
    (r"override\s+system\s+prompt", "System prompt override attempt detected"),
    (r"system\s*:\s*you\s+are", "Role injection attempt detected"),
    (r"<\/move>\s*<move>", "Tag breakout injection attempt detected"),
]


@dataclass
class PromptValidationResult:
    """Result of validating custom user prompts."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: str | None = None
    sanitized_system_prompt: str = ""
    sanitized_turn_prompt: str = ""


@dataclass
class PromptSection:
    """A section of a prompt template with priority for truncation."""
    name: str
    content_template: str
    required: bool = True
    priority: int = 0  # Lower = more important, dropped last during truncation
    is_system: bool = False

    def render(self, context: dict[str, Any]) -> str:
        """Render the section with the given context."""
        try:
            return self.content_template.format(**context)
        except KeyError as err:
            return f"[MISSING CONTEXT: {err}]"
        except Exception as err:
            return f"[FORMAT ERROR: {err}]"


@dataclass
class PromptTemplate:
    """A structured prompt template with versioned sections."""
    sections: list[PromptSection]
    version: str
    model_hints: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None
    used_fallback: bool = False
    fallback_reason: str | None = None

    def referenced_variables(self) -> set[str]:
        """Return the set of variable names referenced by all sections."""
        variables: set[str] = set()
        for section in self.sections:
            variables.update(re.findall(r"\{(\w+)\}", section.content_template))
        return variables

    def render(self, context: dict[str, Any], truncate: bool = True) -> str:
        """Render the full prompt with optional truncation."""
        rendered_parts = []
        total_estimated_tokens = 0

        sorted_sections = sorted(self.sections, key=lambda s: s.priority)

        for section in sorted_sections:
            rendered = section.render(context)
            estimated_tokens = len(rendered) // 4

            if truncate and self.max_tokens and total_estimated_tokens + estimated_tokens > self.max_tokens:
                if section.required:
                    remaining = self.max_tokens - total_estimated_tokens
                    if remaining > 50:
                        rendered = rendered[:remaining * 4] + "... [TRUNCATED]"
                        rendered_parts.append(rendered)
            else:
                rendered_parts.append(rendered)
                total_estimated_tokens += estimated_tokens

        return "\n\n".join(rendered_parts)

    def render_messages(self, context: dict[str, Any], truncate: bool = True) -> list[Any]:
        """Render prompt split into system and user ChatMessage objects."""
        from chessbench.common.common_types import ChatMessage

        system_parts = []
        user_parts = []

        sorted_sections = sorted(self.sections, key=lambda s: s.priority)
        for section in sorted_sections:
            rendered = section.render(context)
            if section.is_system:
                system_parts.append(rendered)
            else:
                user_parts.append(rendered)

        messages = []
        if system_parts:
            messages.append(ChatMessage(role="system", content="\n\n".join(system_parts)))
        if user_parts:
            messages.append(ChatMessage(role="user", content="\n\n".join(user_parts)))
        return messages

    def hash(self) -> str:
        """Generate a hash of this template for logging/versioning."""
        content = f"{self.version}|" + "|".join(
            f"{s.name}:{s.content_template}:{s.required}:{s.priority}:{s.is_system}"
            for s in self.sections
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def sanitize_prompt_text(text: str) -> tuple[str, list[str]]:
    """Sanitize prompt text against known prompt injection patterns.
    
    Returns (sanitized_text, list_of_warnings).
    """
    if not text:
        return "", []

    warnings: list[str] = []
    sanitized = text

    for pattern, desc in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            warnings.append(f"Prompt injection warning: {desc}")
            # Neutralize the suspicious injection phrase by wrapping in quotes
            sanitized = re.sub(
                pattern,
                lambda m: f'"[SANITIZED: {m.group(0)}]"',
                sanitized,
                flags=re.IGNORECASE,
            )

    return sanitized, warnings


def validate_prompt_text(
    system_prompt: str | None,
    turn_prompt: str | None,
) -> PromptValidationResult:
    """Validate system and turn prompts against safety, placeholders, and syntax rules."""
    errors: list[str] = []
    warnings: list[str] = []

    sys_text = (system_prompt or "").strip()
    turn_text = (turn_prompt or "").strip()

    if not sys_text:
        errors.append("System prompt cannot be empty.")

    if not turn_text:
        errors.append("Turn prompt cannot be empty.")

    # Sanitize inputs
    sanitized_sys, sys_warns = sanitize_prompt_text(sys_text)
    sanitized_turn, turn_warns = sanitize_prompt_text(turn_text)
    warnings.extend(sys_warns)
    warnings.extend(turn_warns)

    # Check variable placeholders
    sys_vars = set(re.findall(r"\{(\w+)\}", sys_text))
    turn_vars = set(re.findall(r"\{(\w+)\}", turn_text))
    all_vars = sys_vars | turn_vars

    # Must contain at least one board position variable
    if not (all_vars & REQUIRED_BOARD_VARS):
        errors.append(
            f"Prompt must include at least one board position placeholder: {', '.join('{' + v + '}' for v in sorted(REQUIRED_BOARD_VARS))}"
        )

    # Must contain at least one legal moves variable
    if not (all_vars & REQUIRED_MOVE_VARS):
        errors.append(
            f"Prompt must include at least one legal moves placeholder: {', '.join('{' + v + '}' for v in sorted(REQUIRED_MOVE_VARS))}"
        )

    # Check for formatting syntax errors (unmatched single braces or invalid format keys)
    dummy_context = {
        "color": "White",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "ascii_board": "r n b q k b n r\np p p p p p p p\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\nP P P P P P P P\nR N B Q K B N R",
        "board": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "legal_moves": "e2e4, d2d4",
        "legal_moves_uci": "e2e4 d2d4",
        "legal_moves_annotated": "1. e2e4",
        "forcing_moves": "e2e4",
        "developing_moves": "g1f3",
        "positional_moves": "d2d4",
        "forcing_uci": "e2e4",
        "developing_uci": "g1f3",
        "positional_uci": "d2d4",
        "reasoning_level": "mid",
        "last_move_san": "None",
        "move_history_san": "",
        "white_pieces": "",
        "black_pieces": "",
        "stagnation_status": "Normal",
        "position_progress": "0.0",
        "material_tension": "None",
        "position_dynamism": "Low",
    }

    try:
        sanitized_sys.format(**dummy_context)
    except (KeyError, ValueError, IndexError) as exc:
        errors.append(f"System prompt template format error: {exc}")

    try:
        sanitized_turn.format(**dummy_context)
    except (KeyError, ValueError, IndexError) as exc:
        errors.append(f"Turn prompt template format error: {exc}")

    # Check for required output format instructions in turn prompt
    if "<move>" not in sanitized_turn.lower():
        warnings.append("Turn prompt lacks <move> tag specification. Auto-appending move output instructions.")
        sanitized_turn += "\n\nFormat your final move inside <move>uci_move</move> tags."

    is_valid = len(errors) == 0

    if not is_valid:
        fallback_reason = "; ".join(errors)
        _log.warning("Prompt validation failed: %s. Using default fallback prompt.", fallback_reason)
        return PromptValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            used_fallback=True,
            fallback_reason=fallback_reason,
            sanitized_system_prompt=DEFAULT_SYSTEM_PROMPT,
            sanitized_turn_prompt=DEFAULT_TURN_PROMPT,
        )

    return PromptValidationResult(
        is_valid=True,
        errors=[],
        warnings=warnings,
        used_fallback=False,
        fallback_reason=None,
        sanitized_system_prompt=sanitized_sys,
        sanitized_turn_prompt=sanitized_turn,
    )


def create_safe_prompt_template(
    system_prompt: str | None = None,
    turn_prompt: str | None = None,
) -> tuple[PromptTemplate, PromptValidationResult]:
    """Validate prompt inputs and return a safe PromptTemplate alongside its validation result."""
    validation = validate_prompt_text(system_prompt, turn_prompt)

    sections = [
        PromptSection(
            name="system",
            content_template=validation.sanitized_system_prompt,
            is_system=True,
            priority=0,
        ),
        PromptSection(
            name="turn",
            content_template=validation.sanitized_turn_prompt,
            is_system=False,
            priority=1,
        ),
    ]

    template = PromptTemplate(
        sections=sections,
        version="custom_safe" if validation.is_valid else "fallback",
        used_fallback=validation.used_fallback,
        fallback_reason=validation.fallback_reason,
    )

    return template, validation


class PromptRegistry:
    """Registry of named prompt templates for A/B testing."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _load_template_from_yaml(self, path: Path) -> PromptTemplate | None:
        """Load a prompt template from a YAML file."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            sections = []
            for s in data.get("sections", []):
                sections.append(PromptSection(
                    name=s["name"],
                    content_template=s["content_template"],
                    required=s.get("required", True),
                    priority=s.get("priority", 0),
                    is_system=s.get("is_system", False),
                ))

            return PromptTemplate(
                sections=sections,
                version=data.get("version", path.stem),
                model_hints=data.get("model_hints", {}),
                max_tokens=data.get("max_tokens"),
            )
        except Exception:
            return None

    def _register_defaults(self) -> None:
        """Register the built-in prompt versions from YAML files."""
        templates_dir = Path(__file__).parent / "templates"
        if templates_dir.exists():
            for yaml_file in templates_dir.glob("*.yaml"):
                template = self._load_template_from_yaml(yaml_file)
                if template:
                    self.register(template.version, template)

        if not self._templates:
            self._register_hardcoded_defaults()

    def _register_hardcoded_defaults(self) -> None:
        """Register hardcoded defaults as fallback."""
        default_template, _ = create_safe_prompt_template(DEFAULT_SYSTEM_PROMPT, DEFAULT_TURN_PROMPT)
        default_template.version = "v1_baseline"
        self.register("v1_baseline", default_template)

    def register(self, name: str, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self._templates[name] = template

    def get(self, name: str) -> PromptTemplate | None:
        """Get a prompt template by name."""
        return self._templates.get(name)

    def list_versions(self) -> list[str]:
        """List all registered versions."""
        return list(self._templates.keys())


# Global registry instance
prompt_registry = PromptRegistry()
