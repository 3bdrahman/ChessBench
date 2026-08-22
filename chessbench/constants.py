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
DEFAULT_REASONING_LEVEL: str = "mid"
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
DEFAULT_MAX_PARALLEL_GAMES: int = 4
DEFAULT_MOVE_TIMEOUT_SECONDS: int = 450
# ^ MUST exceed the worst-case move cycle inside get_move_with_result
# (every retry × (DEFAULT_HTTP_TIMEOUT + DEFAULT_MAX_BACKOFF)).
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
# Thinking Analysis Keywords (moved from common_types.py for externalization)
# =============================================================================
THINKING_KEYWORDS: dict[str, list[str]] = {
    "tactics": [
        "tactic", "tactics", "capture", "fork", "pin", "skewer", "discovered",
        "checkmate", "mate", "combination", "sacrifice", "tactical", "threat",
        "attack", "defend", "counter", "intermezzo", "zwischenzug"
    ],
    "strategy": [
        "strategy", "strategic", "plan", "planning", "long-term", "outpost",
        "weakness", "control", "space", "development", "initiative", "prophylaxis",
        "pawn structure", "open file", "open diagonal", "bishop pair", "minority attack"
    ],
    "time_pressure": [
        "time", "clock", "hurry", "rush", "quick", "fast", "seconds", "minutes",
        "increment", "time trouble", "zeitnot", "low on time", "running out"
    ],
    "material": [
        "material", "pawn", "piece", "queen", "rook", "bishop", "knight", "king",
        "exchange", "advantage", "down", "up", "equal", "sacrifice", "win", "lose",
        "points", "value", "count"
    ],
    "positional": [
        "positional", "position", "square", "control", "center", "weak", "strong",
        "outpost", "backward", "isolated", "doubled", "passed", "blockade",
        "hole", "space", "cramped", "open", "closed"
    ],
    "king_safety": [
        "king safety", "castle", "castling", "king", "exposed", "shelter",
        "pawn shield", "attack on king", "king hunt", "mated", "checkmate"
    ],
    "structured_indicators": [
        "1.", "2.", "3.", "first", "second", "third", "then", "next",
        "because", "therefore", "thus", "so", "however", "but", "if", "then",
        "consider", "evaluate", "analyze", "compare", "option", "alternative"
    ],
}

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
    "disambiguation_resolved": 0.85,
    "piece_hint_resolved": 0.8,
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
