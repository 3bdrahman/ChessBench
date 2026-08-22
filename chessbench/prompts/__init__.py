"""Structured 10x Prompt Engineering Engine with validation, injection defense, variable analytics, presets, and background output contract enforcement."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger(__name__)

# System Output Contract appended in the background by the engine
SYSTEM_OUTPUT_CONTRACT = """

[SYSTEM OUTPUT FORMAT CONTRACT]
You MUST format your output response strictly as follows:
<think>
(Include your tactical calculations and strategic reasoning here)
</think>
<move>
(Your chosen move in purely lower-case UCI notation, e.g. e2e4)
</move>

Failure to follow this exact format will result in move disqualification. The move inside <move> MUST be one of the legal moves provided."""

# Default Fallback Prompts (guaranteed safe, valid, and containing all required variables)
DEFAULT_SYSTEM_PROMPT = "You are a professional chess engine playing as {color}."

DEFAULT_TURN_PROMPT = """Position:
{ascii_board}

Board FEN: {fen}

Legal Moves:
Forcing: {forcing_moves}
Developing: {developing_moves}
Positional: {positional_moves}

Select the best move for {color}."""

# Categorized Variable Registry
KNOWN_VARIABLES: dict[str, dict[str, str]] = {
    "color": {"category": "Mandatory Player Side", "description": "Side to move ('White' or 'Black')"},
    "fen": {"category": "Mandatory Board State", "description": "Standard Forsyth-Edwards Notation string"},
    "ascii_board": {"category": "Mandatory Board State", "description": "8x8 ASCII grid of the current board"},
    "board": {"category": "Mandatory Board State", "description": "Alias for FEN board representation"},
    "forcing_moves": {"category": "Mandatory Legal Moves", "description": "Checks and captures categorized"},
    "developing_moves": {"category": "Mandatory Legal Moves", "description": "Piece development moves"},
    "positional_moves": {"category": "Mandatory Legal Moves", "description": "Pawn structure & positional moves"},
    "legal_moves_uci": {"category": "Mandatory Legal Moves", "description": "Space-separated raw UCI legal moves"},
    "legal_moves_annotated": {"category": "Mandatory Legal Moves", "description": "Legal moves with SAN annotations"},
    "legal_moves": {"category": "Mandatory Legal Moves", "description": "Comma-separated legal move list"},
    "forcing_uci": {"category": "Mandatory Legal Moves", "description": "UCI forcing moves string"},
    "developing_uci": {"category": "Mandatory Legal Moves", "description": "UCI developing moves string"},
    "positional_uci": {"category": "Mandatory Legal Moves", "description": "UCI positional moves string"},
    "last_move_san": {"category": "Rich Context", "description": "Opponent's last move in SAN notation"},
    "move_history_san": {"category": "Rich Context", "description": "Full game move history in SAN"},
    "white_pieces": {"category": "Rich Context", "description": "Square list of White pieces"},
    "black_pieces": {"category": "Rich Context", "description": "Square list of Black pieces"},
    "stagnation_status": {"category": "Rich Context", "description": "Repetition & draw stagnation flag"},
    "position_progress": {"category": "Rich Context", "description": "Progress score towards game resolution"},
    "material_tension": {"category": "Rich Context", "description": "Material tension evaluation"},
    "position_dynamism": {"category": "Rich Context", "description": "Positional dynamism rating"},
}

MANDATORY_COLOR_VARS = {"color"}
MANDATORY_BOARD_VARS = {"fen", "ascii_board", "board"}
MANDATORY_MOVE_VARS = {
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
    (r"allow\s+illegal\s+move", "Rule degradation attack detected"),
]


@dataclass
class PromptValidationResult:
    """Detailed result of prompt AST & safety validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_variables: list[str] = field(default_factory=list)
    unrecognized_variables: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
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
        """Render prompt split into system and user ChatMessage objects with background structured I/O contract."""
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
        
        user_content = "\n\n".join(user_parts) if user_parts else ""
        # Background structured I/O contract enforcement
        if SYSTEM_OUTPUT_CONTRACT.strip() not in user_content:
            user_content += SYSTEM_OUTPUT_CONTRACT

        if user_content:
            messages.append(ChatMessage(role="user", content=user_content))
        return messages

    def hash(self) -> str:
        """Generate a hash of this template for logging/versioning."""
        content = f"{self.version}|" + "|".join(
            f"{s.name}:{s.content_template}:{s.required}:{s.priority}:{s.is_system}"
            for s in self.sections
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def sanitize_prompt_text(text: str) -> tuple[str, list[str]]:
    """Sanitize prompt text against known prompt injection patterns."""
    if not text:
        return "", []

    warnings: list[str] = []
    sanitized = text

    for pattern, desc in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            warnings.append(f"Prompt injection threat: {desc}")
            sanitized = re.sub(
                pattern,
                lambda m: f'"[SANITIZED_INJECTION: {m.group(0)}]"',
                sanitized,
                flags=re.IGNORECASE,
            )

    return sanitized, warnings


def _find_closest_variable(typo: str) -> str | None:
    """Find closest known variable name for typo suggestions."""
    typo_lower = typo.lower()
    for known in KNOWN_VARIABLES:
        if known.startswith(typo_lower[:3]) or typo_lower in known:
            return known
    return None


def validate_prompt_text(
    system_prompt: str | None,
    turn_prompt: str | None,
) -> PromptValidationResult:
    """Strictly validate system and turn prompts against safety, mandatory variables, typos, and syntax rules."""
    errors: list[str] = []
    warnings: list[str] = []
    missing_vars: list[str] = []
    unrecognized_vars: list[str] = []
    suggestions: list[str] = []

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

    # Parse variable placeholders
    sys_vars = set(re.findall(r"\{(\w+)\}", sys_text))
    turn_vars = set(re.findall(r"\{(\w+)\}", turn_text))
    all_vars = sys_vars | turn_vars

    # Check for unrecognized variables / typos
    for v in sorted(all_vars):
        if v not in KNOWN_VARIABLES:
            unrecognized_vars.append(f"{{{v}}}")
            closest = _find_closest_variable(v)
            if closest:
                suggestions.append(f"Unrecognized placeholder `{{{v}}}` — did you mean `{{{closest}}}`?")
            else:
                suggestions.append(f"Unrecognized placeholder `{{{v}}}`")

    # Mandatory Category 1: Player Color ({color})
    if not (all_vars & MANDATORY_COLOR_VARS):
        missing_vars.append("{color}")
        errors.append("Missing mandatory player color placeholder: `{color}`")

    # Mandatory Category 2: Board Position ({fen} or {ascii_board} or {board})
    if not (all_vars & MANDATORY_BOARD_VARS):
        missing_vars.append("{fen} or {ascii_board}")
        errors.append("Missing mandatory board position placeholder: `{fen}` or `{ascii_board}`")

    # Mandatory Category 3: Legal Moves ({forcing_moves}, {legal_moves_uci}, etc.)
    if not (all_vars & MANDATORY_MOVE_VARS):
        missing_vars.append("{forcing_moves} or {legal_moves_uci}")
        errors.append("Missing mandatory legal moves placeholder: `{forcing_moves}` or `{legal_moves_uci}`")

    # Estimate token count (chars / 4)
    estimated_tokens = (len(sanitized_sys) + len(sanitized_turn)) // 4

    # Syntax test against dummy context
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

    # Add dummy entries for unrecognized variables to check python formatting syntax
    for unk in unrecognized_vars:
        key = unk.strip("{}")
        dummy_context[key] = f"[{key}_value]"

    try:
        sanitized_sys.format(**dummy_context)
    except (KeyError, ValueError, IndexError) as exc:
        errors.append(f"System prompt syntax format error: {exc}")

    try:
        sanitized_turn.format(**dummy_context)
    except (KeyError, ValueError, IndexError) as exc:
        errors.append(f"Turn prompt syntax format error: {exc}")

    is_valid = len(errors) == 0

    if not is_valid:
        fallback_reason = "; ".join(errors)
        _log.warning("Prompt validation failed: %s. Using default fallback prompt.", fallback_reason)
        return PromptValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            missing_variables=missing_vars,
            unrecognized_variables=unrecognized_vars,
            suggestions=suggestions,
            estimated_tokens=estimated_tokens,
            used_fallback=True,
            fallback_reason=fallback_reason,
            sanitized_system_prompt=DEFAULT_SYSTEM_PROMPT,
            sanitized_turn_prompt=DEFAULT_TURN_PROMPT,
        )

    return PromptValidationResult(
        is_valid=True,
        errors=[],
        warnings=warnings,
        missing_variables=[],
        unrecognized_variables=unrecognized_vars,
        suggestions=suggestions,
        estimated_tokens=estimated_tokens,
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
    """Registry of named prompt templates for A/B testing and presets."""

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
        """Register built-in prompt versions from YAML files."""
        templates_dir = Path(__file__).parent / "templates"
        if templates_dir.exists():
            for yaml_file in sorted(templates_dir.glob("*.yaml")):
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
        """List all registered version names."""
        return list(self._templates.keys())

    def get_preset_prompts(self, version_name: str) -> tuple[str, str]:
        """Convert a registered preset template into a (system_prompt, turn_prompt) tuple."""
        tmpl = self.get(version_name)
        if not tmpl:
            return DEFAULT_SYSTEM_PROMPT, DEFAULT_TURN_PROMPT

        sys_parts = [s.content_template for s in tmpl.sections if s.is_system]
        turn_parts = [s.content_template for s in tmpl.sections if not s.is_system]

        sys_str = "\n\n".join(sys_parts) if sys_parts else DEFAULT_SYSTEM_PROMPT
        turn_str = "\n\n".join(turn_parts) if turn_parts else DEFAULT_TURN_PROMPT

        return sys_str, turn_str


# Global registry instance
prompt_registry = PromptRegistry()
