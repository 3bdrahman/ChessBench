"""
Strategy persistence for the Prompt Workbench.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml


def _get_strategy_dir() -> Path:
    return Path.home() / ".chessbench" / "strategies"


def _ensure_strategy_dir() -> Path:
    strategy_dir = _get_strategy_dir()
    strategy_dir.mkdir(parents=True, exist_ok=True)
    return strategy_dir


def _strategy_path(name: str) -> Path:
    # Sanitize filename: keep only alphanumeric, underscore, hyphen, and dot.
    # Replace any other character with underscore.
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    # Prevent path traversal by ensuring the base name is safe and not empty.
    if not safe_name or safe_name.startswith('.') or safe_name == '..':
        raise ValueError(f"Invalid strategy name: {name}")
    path = _ensure_strategy_dir() / f"{safe_name}.yaml"
    return path


@dataclass(frozen=True)
class StrategyMeta:
    """Metadata for a saved strategy."""
    name: str
    description: str | None = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now().isoformat())


def save_strategy(name: str, system_prompt: str, turn_prompt: str, description: str | None = None) -> Path:
    """Save a named strategy to disk. Returns the path to the saved file."""
    meta = StrategyMeta(name=name, description=description)
    data = {
        "meta": asdict(meta),
        "system_prompt": system_prompt,
        "turn_prompt": turn_prompt,
    }
    path = _strategy_path(name)
    with path.open('w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)
    return path


def list_strategies() -> list[StrategyMeta]:
    """List all saved strategies, sorted by creation time (newest first)."""
    _ensure_strategy_dir()
    strategies: list[StrategyMeta] = []
    for file in _get_strategy_dir().glob("*.yaml"):
        try:
            with file.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            meta = data.get("meta", {})
            strategies.append(
                StrategyMeta(
                    name=meta.get("name", ""),
                    description=meta.get("description"),
                    created_at=meta.get("created_at", ""),
                )
            )
        except Exception:
            # Skip corrupt files
            continue
    # Sort by created_at descending
    strategies.sort(key=lambda x: x.created_at, reverse=True)
    return strategies


def load_strategy(name: str) -> tuple[str, str]:
    """Load a named strategy. Returns (system_prompt, turn_prompt)."""
    path = _strategy_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Strategy not found: {name}")
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data["system_prompt"], data["turn_prompt"]


def delete_strategy(name: str) -> None:
    """Delete a named strategy."""
    path = _strategy_path(name)
    if path.exists():
        path.unlink()


def export_yaml(name: str) -> str:
    """Export a named strategy as YAML string."""
    sys_prompt, turn_prompt = load_strategy(name)
    # We also want to include the meta? The spec says round-trip, so we should include meta.
    # But the import_yaml function expects to be able to import from the exported string.
    # Let's include the meta as well.
    meta = next((m for m in list_strategies() if m.name == name), None)
    data = {
        "meta": asdict(meta) if meta else {},
        "system_prompt": sys_prompt,
        "turn_prompt": turn_prompt,
    }
    return yaml.dump(data, default_flow_style=False)


def import_yaml(yaml_text: str, name: str | None = None) -> None:
    """Import a strategy from YAML text. If name is provided, use it; otherwise, use the name from the YAML."""
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise ValueError("Invalid YAML: expected a dictionary")
    system_prompt = data.get("system_prompt")
    turn_prompt = data.get("turn_prompt")
    if system_prompt is None or turn_prompt is None:
        raise ValueError("YAML must contain system_prompt and turn_prompt")
    meta_data = data.get("meta", {})
    strategy_name = name or meta_data.get("name")
    if not strategy_name:
        raise ValueError("Strategy name must be provided either as argument or in YAML meta")
    description = meta_data.get("description")
    save_strategy(strategy_name, system_prompt, turn_prompt, description)
