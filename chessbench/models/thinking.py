"""Thinking trace analysis for LLM reasoning quality."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from chessbench.common.common_types import ThinkingKeywordConfig


@dataclass
class ThinkingTrace:
    """Analyzed thinking trace from LLM response."""
    text: str
    char_count: int
    word_count: int
    has_structured_reasoning: bool
    mentions_tactics: bool
    mentions_strategy: bool
    mentions_time_pressure: bool
    mentions_material: bool
    mentions_positional: bool
    mentions_king_safety: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "has_structured_reasoning": self.has_structured_reasoning,
            "mentions_tactics": self.mentions_tactics,
            "mentions_strategy": self.mentions_strategy,
            "mentions_time_pressure": self.mentions_time_pressure,
            "mentions_material": self.mentions_material,
            "mentions_positional": self.mentions_positional,
            "mentions_king_safety": self.mentions_king_safety,
        }


# Default keyword configuration
DEFAULT_THINKING_CONFIG = ThinkingKeywordConfig()


def extract_thinking(text: str) -> str:
    """Extract thinking content from LLM response.

    Looks for <think>...</think> or <thinking>...</thinking> tags or similar markers.
    If no tags are found, strips the <move> block and returns the remaining text.
    """
    # Try <think> or <thinking>
    match = re.search(r'<(?:think|thinking)>(.*?)</(?:think|thinking)>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try ...
    match = re.search(r'事情(.*?)回答', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try [THINKING]...[/THINKING]
    match = re.search(r'\[THINKING\](.*?)\[/THINKING\]', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # If no tags found, fallback: remove <move> block and return whatever is left
    no_move = re.sub(r'<move>.*?</move>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return no_move.strip()


def analyze_thinking(text: str, config: ThinkingKeywordConfig | None = None) -> ThinkingTrace:
    """Analyze thinking text for reasoning quality features.

    Args:
        text: The thinking text to analyze
        config: Optional custom keyword configuration. Uses DEFAULT_THINKING_CONFIG if not provided.
    """
    if config is None:
        config = DEFAULT_THINKING_CONFIG

    if not text:
        return ThinkingTrace(
            text="",
            char_count=0,
            word_count=0,
            has_structured_reasoning=False,
            mentions_tactics=False,
            mentions_strategy=False,
            mentions_time_pressure=False,
            mentions_material=False,
            mentions_positional=False,
            mentions_king_safety=False,
        )

    text_lower = text.lower()
    char_count = len(text)
    word_count = len(text.split())

    # Check for structured reasoning (numbered steps, logical connectors)
    has_structured = any(indicator in text_lower for indicator in config.structured_indicators)

    # Check for tactical mentions
    mentions_tactics = any(kw in text_lower for kw in config.tactics)

    # Check for strategic mentions
    mentions_strategy = any(kw in text_lower for kw in config.strategy)

    # Check for time pressure mentions
    mentions_time = any(kw in text_lower for kw in config.time_pressure)

    # Check for material mentions
    mentions_material = any(kw in text_lower for kw in config.material)

    # Check for positional mentions
    mentions_positional = any(kw in text_lower for kw in config.positional)

    # Check for king safety mentions
    mentions_king = any(kw in text_lower for kw in config.king_safety)

    return ThinkingTrace(
        text=text.strip(),
        char_count=char_count,
        word_count=word_count,
        has_structured_reasoning=has_structured,
        mentions_tactics=mentions_tactics,
        mentions_strategy=mentions_strategy,
        mentions_time_pressure=mentions_time,
        mentions_material=mentions_material,
        mentions_positional=mentions_positional,
        mentions_king_safety=mentions_king,
    )


def extract_and_analyze_thinking(text: str, config: ThinkingKeywordConfig | None = None) -> ThinkingTrace:
    """Extract thinking from response and analyze it."""
    thinking_text = extract_thinking(text)
    return analyze_thinking(thinking_text, config)


