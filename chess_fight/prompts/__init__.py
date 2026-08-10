"""Structured prompt system with versioning and A/B testing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptSection:
    """A section of a prompt template with priority for truncation."""
    name: str
    content_template: str
    required: bool = True
    priority: int = 0  # Lower = more important, dropped last during truncation

    def render(self, context: dict[str, Any]) -> str:
        """Render the section with the given context."""
        try:
            return self.content_template.format(**context)
        except KeyError:
            return f"[MISSING CONTEXT: {self.name}]"


@dataclass
class PromptTemplate:
    """A structured prompt template with versioned sections."""
    sections: list[PromptSection]
    version: str
    model_hints: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None

    def render(self, context: dict[str, Any], truncate: bool = True) -> str:
        """Render the full prompt with optional truncation."""
        rendered_parts = []
        total_estimated_tokens = 0

        # Sort sections by priority (lower first)
        sorted_sections = sorted(self.sections, key=lambda s: s.priority)

        for section in sorted_sections:
            rendered = section.render(context)
            estimated_tokens = len(rendered) // 4  # Rough estimate: 4 chars per token

            if truncate and self.max_tokens and total_estimated_tokens + estimated_tokens > self.max_tokens:
                if section.required:
                    # Required section - truncate it instead of dropping
                    remaining = self.max_tokens - total_estimated_tokens
                    if remaining > 50:  # Only include if we have meaningful space
                        rendered = rendered[:remaining * 4] + "... [TRUNCATED]"
                        rendered_parts.append(rendered)
                # Skip optional sections when over budget
            else:
                rendered_parts.append(rendered)
                total_estimated_tokens += estimated_tokens

        return "\n\n".join(rendered_parts)

    def hash(self) -> str:
        """Generate a hash of this template for logging/versioning."""
        content = f"{self.version}|" + "|".join(
            f"{s.name}:{s.content_template}:{s.required}:{s.priority}"
            for s in self.sections
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


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

        # Fallback: if no YAML templates found, use hardcoded defaults
        if not self._templates:
            self._register_hardcoded_defaults()

    def _register_hardcoded_defaults(self) -> None:
        """Register hardcoded defaults as fallback."""
        # v1_baseline - Original prompt structure
        self.register("v1_baseline", PromptTemplate(
            sections=DEFAULT_SECTIONS,
            version="v1_baseline",
            model_hints={"description": "Original baseline prompt"},
            max_tokens=2000,
        ))

        # v2_tactical_focus - Emphasize tactical patterns
        tactical_sections = [
            PromptSection(
                name="role",
                content_template="You are playing chess as {color}. Focus on tactical precision.",
                required=True,
                priority=0,
            ),
            PromptSection(
                name="tactical_priority",
                content_template="""CRITICAL TACTICAL PRIORITIES:
1. CHECKMATE threats (immediate or forced)
2. WINNING MATERIAL sequences
3. FORCING moves (checks, captures, threats)
4. DEFEND against opponent's tactics""",
                required=True,
                priority=1,
            ),
            PromptSection(
                name="position_analysis",
                content_template="""Current position:
Stagnation: {stagnation_status} | Progress: {position_progress}
Tension: {material_tension} | Dynamism: {position_dynamism}""",
                required=True,
                priority=2,
            ),
        ] + [s for s in DEFAULT_SECTIONS if s.name in ("legal_moves", "instructions")]

        self.register("v2_tactical_focus", PromptTemplate(
            sections=tactical_sections,
            version="v2_tactical_focus",
            model_hints={"description": "Tactical emphasis, shorter context"},
            max_tokens=1500,
        ))

        # v3_minimal - Minimal prompt for fast inference
        minimal_sections = [
            PromptSection(
                name="role",
                content_template="You are {color}. Respond with UCI move only.",
                required=True,
                priority=0,
            ),
            PromptSection(
                name="board",
                content_template="Position: {ascii_board}",
                required=True,
                priority=1,
            ),
            PromptSection(
                name="legal_moves",
                content_template="Legal: {forcing_moves}\n{developing_moves}\n{positional_moves}",
                required=True,
                priority=2,
            ),
            PromptSection(
                name="instructions",
                content_template="""Format:
<think>reasoning</think>
<move>uci_move</move>""",
                required=True,
                priority=0,
            ),
        ]

        self.register("v3_minimal", PromptTemplate(
            sections=minimal_sections,
            version="v3_minimal",
            model_hints={"description": "Minimal prompt for fast models"},
            max_tokens=800,
        ))

    def register(self, name: str, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self._templates[name] = template

    def get(self, name: str) -> PromptTemplate | None:
        """Get a prompt template by name."""
        return self._templates.get(name)

    def list_versions(self) -> list[str]:
        """List all registered versions."""
        return list(self._templates.keys())


# Built-in prompt sections (kept as fallback if YAML files are missing)
DEFAULT_SECTIONS = [
    PromptSection(
        name="role",
        content_template="You are a professional chess engine playing as {color}.",
        required=True,
        priority=0,
    ),
    PromptSection(
        name="board_state",
        content_template="""[GAME STATE]
FEN: {fen}
Turn: {color}""",
        required=True,
        priority=1,
    ),
    PromptSection(
        name="instructions",
        content_template="""[INSTRUCTIONS]
Analyze the position and select the best move. 

You must format your response EXACTLY like this:
<think>
(Your reasoning here)
</think>
<move>
(Your chosen move in purely lower-case UCI notation, e.g. e2e4)
</move>

Failure to follow this exact format will result in disqualification.""",
        required=True,
        priority=0,
    ),
]


# Global registry instance
prompt_registry = PromptRegistry()
