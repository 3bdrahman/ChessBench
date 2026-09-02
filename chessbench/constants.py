"""Centralized constants and configuration defaults for chessbench.

This module replaces hardcoded magic numbers and default values scattered
across the codebase. All defaults should be defined here and imported
where needed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# =============================================================================
# HTTP / Network
# =============================================================================
DEFAULT_HTTP_TIMEOUT: float = 600.0
DEFAULT_HTTP_RETRIES: int = 3
DEFAULT_BACKOFF_BASE: float = 2.0
DEFAULT_MAX_BACKOFF: float = 15.0

# =============================================================================
# Reasoning Levels
# =============================================================================
REASONING_LEVELS: tuple[str, ...] = ("low", "mid", "high")
DEFAULT_REASONING_LEVEL: str = "high"
REASONING_MAX_TOKENS: dict[str, int] = {
    "low": 256,
    "mid": 1024,
    "high": 4096,
}

# =============================================================================
# LLM Provider Defaults
# =============================================================================
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_BENCHMARK_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int | None = None
DEFAULT_MAX_TOKENS_BENCHMARK: int | None = None
DEFAULT_SEED: int | None = 42

# Context windows (fallback when provider doesn't report)
DEFAULT_CONTEXT_WINDOW: int = 128_000
MIN_CONTEXT_WINDOW_FOR_CHESS: int = 256

# Model-specific context windows (override DEFAULT_CONTEXT_WINDOW when known)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI models
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "gpt-4o-2024-11-20": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4": 8_192,
    "gpt-4-0613": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-32k-0613": 32_768,
    "gpt-3.5-turbo": 16_384,
    "gpt-3.5-turbo-0125": 16_384,
    "gpt-3.5-turbo-1106": 16_384,
    "gpt-3.5-turbo-16k": 16_384,
    "gpt-3.5-turbo-instruct": 4_096,
    "o1-preview": 128_000,
    "o1-mini": 128_000,
    "o1": 200_000,
    "o3-mini": 200_000,
    # Anthropic models
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-sonnet-20240620": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-2.1": 200_000,
    "claude-2.0": 100_000,
    "claude-instant-1.2": 100_000,
    # Google models
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-flash-8b": 1_000_000,
    "gemini-1.0-pro": 32_768,
    "gemini-1.0-pro-vision": 12_288,
    # Groq models
    "llama3-70b-8192": 8_192,
    "llama3-8b-8192": 8_192,
    "mixtral-8x7b-32768": 32_768,
    "gemma-7b-it": 8_192,
    "gemma2-9b-it": 8_192,
    # DeepInfra models
    "meta-llama/Meta-Llama-3.1-405B-Instruct": 128_000,
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 128_000,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 128_000,
    "meta-llama/Llama-3.3-70B-Instruct": 128_000,
    "mistralai/Mistral-Large-2407": 128_000,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32_768,
    # Fireworks models
    "accounts/fireworks/models/llama-v3p1-405b-instruct": 128_000,
    "accounts/fireworks/models/llama-v3p1-70b-instruct": 128_000,
    "accounts/fireworks/models/llama-v3p1-8b-instruct": 128_000,
    "accounts/fireworks/models/mixtral-8x7b-instruct": 32_768,
    # Together models
    "meta-llama/Llama-3.1-405B-Instruct-Turbo": 128_000,
    "meta-llama/Llama-3.1-70B-Instruct-Turbo": 128_000,
    "meta-llama/Llama-3.1-8B-Instruct-Turbo": 128_000,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32_768,
    # NIM models
    "meta/llama-3.1-405b-instruct": 128_000,
    "meta/llama-3.1-70b-instruct": 128_000,
    "meta/llama-3.1-8b-instruct": 128_000,
    "mistralai/mixtral-8x7b-instruct-v0.1": 32_768,
}

# Model-specific temperature defaults (override DEFAULT_TEMPERATURE when known)
# Lower temperatures for reasoning models, higher for creative tasks
MODEL_TEMPERATURES: dict[str, float] = {
    # OpenAI models
    "gpt-4o": 0.0,
    "gpt-4o-mini": 0.0,
    "gpt-4o-2024-05-13": 0.0,
    "gpt-4o-2024-08-06": 0.0,
    "gpt-4o-2024-11-20": 0.0,
    "gpt-4.1": 0.0,
    "gpt-4.1-mini": 0.0,
    "gpt-4.1-nano": 0.0,
    "gpt-4-turbo": 0.0,
    "gpt-4-turbo-2024-04-09": 0.0,
    "gpt-4-turbo-preview": 0.0,
    "gpt-4": 0.0,
    "gpt-4-0613": 0.0,
    "gpt-4-32k": 0.0,
    "gpt-4-32k-0613": 0.0,
    "gpt-3.5-turbo": 0.1,
    "gpt-3.5-turbo-0125": 0.1,
    "gpt-3.5-turbo-1106": 0.1,
    "gpt-3.5-turbo-16k": 0.1,
    "gpt-3.5-turbo-instruct": 0.0,
    "o1-preview": 1.0,  # o1 models require temperature=1
    "o1-mini": 1.0,
    "o1": 1.0,
    "o3-mini": 1.0,
    # Anthropic models
    "claude-3-5-sonnet-20241022": 0.0,
    "claude-3-5-sonnet-20240620": 0.0,
    "claude-3-5-haiku-20241022": 0.0,
    "claude-3-opus-20240229": 0.0,
    "claude-3-sonnet-20240229": 0.0,
    "claude-3-haiku-20240307": 0.0,
    "claude-2.1": 0.0,
    "claude-2.0": 0.0,
    "claude-instant-1.2": 0.0,
    # Google models
    "gemini-1.5-pro": 0.0,
    "gemini-1.5-flash": 0.0,
    "gemini-1.5-flash-8b": 0.0,
    "gemini-1.0-pro": 0.0,
    "gemini-1.0-pro-vision": 0.0,
    # Groq models
    "llama3-70b-8192": 0.0,
    "llama3-8b-8192": 0.0,
    "mixtral-8x7b-32768": 0.0,
    "gemma-7b-it": 0.0,
    "gemma2-9b-it": 0.0,
    # DeepInfra models
    "meta-llama/Meta-Llama-3.1-405B-Instruct": 0.0,
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 0.0,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 0.0,
    "meta-llama/Llama-3.3-70B-Instruct": 0.0,
    "mistralai/Mistral-Large-2407": 0.0,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.0,
    # Fireworks models
    "accounts/fireworks/models/llama-v3p1-405b-instruct": 0.0,
    "accounts/fireworks/models/llama-v3p1-70b-instruct": 0.0,
    "accounts/fireworks/models/llama-v3p1-8b-instruct": 0.0,
    "accounts/fireworks/models/mixtral-8x7b-instruct": 0.0,
    # Together models
    "meta-llama/Llama-3.1-405B-Instruct-Turbo": 0.0,
    "meta-llama/Llama-3.1-70B-Instruct-Turbo": 0.0,
    "meta-llama/Llama-3.1-8B-Instruct-Turbo": 0.0,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.0,
    # NIM models
    "meta/llama-3.1-405b-instruct": 0.0,
    "meta/llama-3.1-70b-instruct": 0.0,
    "meta/llama-3.1-8b-instruct": 0.0,
    "mistralai/mixtral-8x7b-instruct-v0.1": 0.0,
}


def get_temperature(model_id: str) -> float:
    """Get temperature for a specific model, falling back to DEFAULT_TEMPERATURE."""
    model_lower = model_id.lower()
    for key, value in MODEL_TEMPERATURES.items():
        if key.lower() == model_lower:
            return value
    for key, value in MODEL_TEMPERATURES.items():
        if model_lower.startswith(key.lower()):
            return value
    return DEFAULT_TEMPERATURE

# =============================================================================
# Stockfish Engine Defaults
# =============================================================================
STOCKFISH_DEFAULT_DEPTH: int = 12
STOCKFISH_DEFAULT_THINK_TIME: float = 1.0
STOCKFISH_DEFAULT_THREADS: int = 1
STOCKFISH_DEFAULT_HASH_MB: int = 64
STOCKFISH_DEPTH_OPTIONS: tuple[int, ...] = (4, 8, 12, 16, 20)
STOCKFISH_SEARCH_PATHS: tuple[str, ...] = (
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/var/home/linuxbrew/.linuxbrew/bin/stockfish",
    "/home/linuxbrew/.linuxbrew/bin/stockfish",
    "/usr/games/stockfish",
    "C:/Program Files/Stockfish/stockfish.exe",
    "C:/Stockfish/stockfish.exe",
)
STOCKFISH_ENGINE_TIMEOUT_MARGIN: float = 2.0

# =============================================================================
# Game / Benchmark Defaults
# =============================================================================
DEFAULT_TIME_CONTROL_SECONDS_PER_MOVE: int = 30
DEFAULT_OPENING_BOOK: str = "eco_balanced"
DEFAULT_GAMES_PER_PAIRING: int = 10
DEFAULT_COLORS_MODE: str = "alternating"
DEFAULT_MAX_PARALLEL_GAMES: int = 1
DEFAULT_MOVE_TIMEOUT_SECONDS: int = 120
# ^ MUST exceed the worst-case move cycle inside get_move_with_result
# (every retry x (DEFAULT_HTTP_TIMEOUT + DEFAULT_MAX_BACKOFF)).
# If this falls below that ceiling, asyncio.wait_for cancels the move coroutine
# mid-retry — the game false-pauses with reason="timeout" and the move is never
# recorded even though the API key is valid and a request would have succeeded.
# Failsafe only: chess self-terminates (50-move rule, repetition, mate, stalemate)
# and the per-move timeout (DEFAULT_MOVE_TIMEOUT_SECONDS) bounds individual moves.
# This guard exists purely for pathological cases (e.g. a model that always returns
# a legal move but the engine somehow never declares a result). A normal game never
# approaches it. Non-fatal when it fires.
DEFAULT_GAME_TIMEOUT_SECONDS: int = 7200
DEFAULT_OUTPUT_DIR: str = "runs"

# =============================================================================
# Piece Values (Centipawns)
# =============================================================================
PIECE_VALUES_CP: dict[str, int] = {
    "PAWN": 100,
    "KNIGHT": 320,
    "BISHOP": 330,
    "ROOK": 500,
    "QUEEN": 900,
    "KING": 20_000,
}

PIECE_VALUES_MATERIAL: dict[str, int] = {
    "PAWN": 1,
    "KNIGHT": 3,
    "BISHOP": 3,
    "ROOK": 5,
    "QUEEN": 9,
    "KING": 0,
}

# =============================================================================
# Evaluation Weights
# =============================================================================
EVAL_WEIGHTS: dict[str, float] = {
    "capture_value": 1.0,
    "center_control": 0.8,
    "development": 0.7,
    "king_safety": 0.9,
    "pawn_structure": 0.6,
    "piece_activity": 0.75,
    "position_progress": 1.0,
}

# =============================================================================
# Position Analysis Thresholds
# =============================================================================
STAGNATION_THRESHOLD: int = 3
MATE_THREAT_SCORE: int = 10_000
UNDEFENDED_UNDER_ATTACK_SCORE: int = 200
VULNERABILITY_UNDEFENDED_SCORE: int = 50
VULNERABILITY_PINNED_SCORE: int = 100
KING_SAFETY_MULTIPLIER: int = 50
ISOLATED_PAWN_PENALTY: int = 20
UNDEFENDED_PIECE_PENALTY: int = 100
EXPOSED_PIECE_PENALTY: int = 50
DEVELOPMENT_BONUS: int = 10
CENTER_CONTROL_MULTIPLIER: int = 20
MATERIAL_BALANCE_MULTIPLIER: int = 100
PROGRESS_CENTER_BONUS: int = 50
PROGRESS_BACK_RANK_TO_CENTER_BONUS: int = 100

# =============================================================================
# Move Quality Thresholds (Centipawn Loss)
# =============================================================================
MOVE_QUALITY_THRESHOLDS: dict[str, int] = {
    "best": 0,
    "excellent": 10,
    "good": 50,
    "inaccuracy": 100,
    "mistake": 300,
    "blunder": 300,  # >= 300
}

# =============================================================================
# Glicko-2 Rating Constants
# =============================================================================
GLICKO2_DEFAULT_RATING: float = 1500.0
GLICKO2_DEFAULT_DEVIATION: float = 350.0
GLICKO2_DEFAULT_VOLATILITY: float = 0.06
GLICKO2_TAU: float = 0.5
GLICKO2_RATING_SCALE: float = 173.7178
GLICKO2_CONVERGENCE_TOLERANCE: float = 1e-6
GLICKO2_MAX_ITERATIONS: int = 100

# =============================================================================
# Prompt / Token Estimation
# =============================================================================
CHARS_PER_TOKEN_ESTIMATE: int = 4
DEFAULT_MAX_PROMPT_TOKENS: int = 2000
MIN_PROMPT_TOKENS_FOR_TRUNCATION: int = 50

# =============================================================================
# Thinking Analysis Keywords (loaded from external JSON file)
# =============================================================================
import json
from pathlib import Path

_THINKING_KEYWORDS_PATH = Path(__file__).parent / "data" / "thinking_keywords.json"

def _load_thinking_keywords() -> dict[str, list[str]]:
    """Load thinking analysis keywords from external JSON file."""
    try:
        with open(_THINKING_KEYWORDS_PATH) as f:
            return json.load(f)
    except Exception:
        # Fallback to minimal defaults if file not found
        return {
            "tactics": ["tactic", "tactics", "capture", "fork", "pin", "skewer"],
            "strategy": ["strategy", "plan", "planning", "development"],
            "time_pressure": ["time", "clock", "hurry", "rush"],
            "material": ["material", "piece", "exchange"],
            "positional": ["position", "control", "center"],
            "king_safety": ["king safety", "castle", "castling"],
            "structured_indicators": ["1.", "2.", "3.", "first", "then"],
        }

THINKING_KEYWORDS: dict[str, list[str]] = _load_thinking_keywords()

# =============================================================================
# Non-chat model tokens (for filtering)
# =============================================================================
NON_CHAT_TOKENS: tuple[str, ...] = (
    "embed", "embedding", "whisper", "tts", "dall-e", "dalle",
    "moderation", "clip", "rerank", "transcribe", "audio",
    "image-input", "image-gen", "imagen", "sd3", "sdxl", "flux",
    "sora", "realtime", "asr", "aqa", "guard",
    "bge-", "mxbai", "e5-", "gte-",
)

WEAK_FOR_CHESS_TOKENS: tuple[str, ...] = (
    "babbage", "davinci", "curie", "turbo-instruct",
)

FREE_TIER_PATTERN = re.compile(r"(:free|\bfree\b)", re.IGNORECASE)

# =============================================================================
# Move Parser Confidence Thresholds
# =============================================================================
MOVE_PARSE_CONFIDENCE: dict[str, float] = {
    "san_valid": 0.95,
    "uci_valid": 1.0,
    "uci_no_board": 0.9,
    "natural_language_exact": 0.8,
    "natural_language_ambiguous": 0.4,
    "target_square_only": 0.3,
    "fallback_uci": 0.6,
    "fallback_no_board": 0.5,
}

# =============================================================================
# Retry / Backoff
# =============================================================================
RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay": 2.0,
    "max_delay": 60.0,
    "exponential_base": 2.0,
}

# =============================================================================
# Logging / Output
# =============================================================================
LOG_DATE_FORMAT: str = "%Y.%m.%d"
PGN_EVENT_NAME: str = "Chess LLM Benchmark"
PGN_SITE: str = "Local"

# =============================================================================
# UI / Streamlit
# =============================================================================

_env_providers = os.environ.get("CHESSBENCH_HOSTED_PROVIDERS") or os.environ.get("CHESS_FIGHT_HOSTED_PROVIDERS")
HOSTED_PROVIDERS: tuple[str, ...] | None = tuple(_env_providers.split(",")) if _env_providers else None
DEFAULT_BOARD_SIZE: int = 600
DEFAULT_MOVE_DELAY: float = 0.1
DEFAULT_DEMO_DELAY: float = 0.5

# =============================================================================
# Reproducibility Verification
# =============================================================================
REPRODUCIBILITY_DEFAULTS = {
    "move_timing_tolerance_ms": 100,
    "token_tolerance": 5,
    "max_game_diffs": 10,
    "verification_games_per_pairing": 1,
    "verification_max_pairings": 2,
    "verification_time_control": 5,
    "verification_move_timeout": 30,
    "verification_game_timeout": 60,
}

# =============================================================================
# Rate Limiting
# =============================================================================
RATE_LIMIT_DEFAULTS = {
    "default_rpm": 60,
    "default_tpm": 100_000,
    "max_queue_time": 30.0,
    "cleanup_interval": 60.0,
}

# =============================================================================
# BenchmarkConfig Dataclass (to replace hardcoded defaults in runner.py)
# =============================================================================
@dataclass
class BenchmarkConfigDefaults:
    """Default values for BenchmarkConfig - single source of truth."""
    time_control_seconds_per_move: int = DEFAULT_TIME_CONTROL_SECONDS_PER_MOVE
    opening_book: str = DEFAULT_OPENING_BOOK
    games_per_pairing: int = DEFAULT_GAMES_PER_PAIRING
    colors: str = DEFAULT_COLORS_MODE
    temperature: float = DEFAULT_BENCHMARK_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS_BENCHMARK
    seed: int | None = DEFAULT_SEED
    max_parallel_games: int = DEFAULT_MAX_PARALLEL_GAMES
    move_timeout_seconds: int = DEFAULT_MOVE_TIMEOUT_SECONDS
    game_timeout_seconds: int = DEFAULT_GAME_TIMEOUT_SECONDS
    output_dir: str = DEFAULT_OUTPUT_DIR
    players: list[str] = field(default_factory=list)
    api_keys: dict[str, str] = field(default_factory=dict)
    run_name: str | None = None


# =============================================================================
# Helper functions
# =============================================================================
def get_piece_value_cp(piece_type: str) -> int:
    """Get centipawn value for piece type."""
    return PIECE_VALUES_CP.get(piece_type.upper(), 0)


def get_piece_value_material(piece_type: str) -> int:
    """Get material value for piece type."""
    return PIECE_VALUES_MATERIAL.get(piece_type.upper(), 0)


def get_context_window(model_id: str) -> int:
    """Get context window for a specific model, falling back to DEFAULT_CONTEXT_WINDOW."""
    model_lower = model_id.lower()
    # Try exact match first
    for key, value in MODEL_CONTEXT_WINDOWS.items():
        if key.lower() == model_lower:
            return value
    # Try prefix match (e.g., "gpt-4o" matches "gpt-4o-2024-05-13")
    for key, value in MODEL_CONTEXT_WINDOWS.items():
        if model_lower.startswith(key.lower()):
            return value
    return DEFAULT_CONTEXT_WINDOW
