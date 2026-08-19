"""Streamlit app with async game loop and provider-agnostic model selection.

The demo is fully functional — no mocks. Real demo games come from
benchmark runs in ``runs/`` via :mod:`demos.generate`; headless benchmarks
run in-process and stream real ELO/leaderboard results back to the UI;
benchmark history reads the real JSONL artifacts the runner already
writes to ``runs/<run_id>/``.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.svg
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from chess_fight.benchmark.results_view import (
    aggregate_leaderboard,
    list_runs,
    load_run,
)
from chess_fight.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chess_fight.common.common_types import is_chess_capable
from chess_fight.common.exceptions import (
    FatalBenchmarkError,
    GameExecutionError,
    InvalidApiKeyError,
    NoProvidersConfiguredError,
    ProviderError,
    SetupError,
)

# Providers surfaced in the hosted Streamlit UI.
# OpenRouter: aggregated API, server-side demo key, free tier for visitors.
# NIM: NVIDIA's hosted inference API, server-side key.
# Ollama: local-only — works when running the app locally against an Ollama server.
# Stockfish: real local engine — only shown when the binary is on PATH.
from chess_fight.constants import HOSTED_PROVIDERS
from chess_fight.game.async_game import GameState
from chess_fight.providers import get_provider, list_providers
from chess_fight.providers.chess_ai import ProviderChessAI
from chess_fight.ui.error_display import render_error
from chess_fight.ui.helpers import (
    format_duration_ms,
    player_banner_html,
    render_board_with_evalbar,
    render_loading_card,
    render_thinking_trace_drawer,
)
from chess_fight.ui.landing import render_hero, render_landing_metrics
from chess_fight.ui.theme import apply_arena_theme

RUNS_ROOT = os.environ.get("CHESS_FIGHT_RUNS_ROOT", "runs")

# Configure page
st.set_page_config(
    page_title="AI Chess Battle",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _draw_board(board_placeholder, state: GameState, start_time: float | None = None) -> None:
    king_square = state.board.king(state.board.turn)
    check_square = king_square if state.board.is_check() and king_square is not None else None
    board_placeholder.write(
        chess.svg.board(
            state.board,
            size=600,
            lastmove=state.board.peek() if state.board.move_stack else None,
            check=check_square,
        ),
        unsafe_allow_html=True,
    )


def _draw_metrics(stats_placeholder, state: GameState, start_time: float | None = None) -> None:
    cols = stats_placeholder.columns(5)
    with cols[0]:
        st.metric("Total Moves", state.stats.total_moves)
    with cols[1]:
        st.metric("Captures", state.stats.capture_moves)
    with cols[2]:
        st.metric("Checks", state.stats.check_moves)
    with cols[3]:
        if state.game_duration > 0:
            elapsed = int(state.game_duration)
        elif start_time is not None:
            elapsed = int(time.time() - start_time)
        else:
            elapsed = 0
        st.metric("Time Elapsed", format_duration_ms(elapsed * 1000))
    with cols[4]:
        if not state.is_game_over:
            turn_color = "White ♔" if state.board.turn else "Black ♚"
            st.metric("Current Turn", turn_color)
        else:
            term_reason = getattr(state.stats, 'termination_reason', 'unknown')
            st.metric("Termination", term_reason.replace('_', ' ').title())


def _draw_moves(moves_placeholder, moves: list) -> None:
    if not moves:
        return
    df = pd.DataFrame(
        [
            {
                "Move #": i + 1,
                "Player": move.player,
                "Move": move.move,
                "Capture": "✓" if move.is_capture else "",
                "Check": "✓" if move.is_check else "",
                "Reasoning": (move.reasoning.replace("<", "&lt;").replace(">", "&gt;") if move.reasoning else ""),
            }
            for i, move in enumerate(moves)
        ]
    )

    # Configure the Reasoning column to be a text column that doesn't blow up the width
    column_config = {
        "Reasoning": st.column_config.TextColumn(
            "Reasoning",
            help="Click a cell to read the LLM's full reasoning for this move.",
            width="large",
        )
    }

    moves_placeholder.dataframe(df, hide_index=True, width="stretch", column_config=column_config)


def _draw_completion_result(expander_placeholder, state: GameState) -> None:
    """Render the last LLM completion (tokens, latency, raw text) honestly."""
    cr = state.last_completion_result
    if cr is None:
        return
    title = f"Last LLM completion ({state.current_player})" if state.current_player else "Last LLM completion"
    with expander_placeholder.expander(title, expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Latency", format_duration_ms(cr.latency_ms) if cr.latency_ms is not None else "—")
        with col_b:
            st.metric("Tokens in", cr.tokens_in if cr.tokens_in is not None else "—")
        with col_c:
            st.metric("Tokens out", cr.tokens_out if cr.tokens_out is not None else "—")
        if cr.error:
            st.error(f"**{cr.error_type or 'Error'}** (Retry #{cr.retry_count}): {cr.error}")
        else:
            st.caption("Raw model response:")
            st.code(cr.text or "")


class ChessUI:
    """UI components for chess game display."""

    def __init__(self):
        self.board_placeholder = st.empty()
        self.stats_placeholder = st.empty()
        self.move_history_placeholder = st.empty()
        self.status_placeholder = st.empty()

    def display_board(self, board):
        king_square = board.king(board.turn)
        check_square = king_square if board.is_check() and king_square is not None else None
        svg_board = chess.svg.board(
            board,
            size=600,
            lastmove=board.peek() if board.move_stack else None,
            check=check_square,
        )
        self.board_placeholder.write(svg_board, unsafe_allow_html=True)

    def display_stats(self, game_state: GameState):
        _draw_metrics(self.stats_placeholder, game_state)

    def display_moves(self, moves: list):
        _draw_moves(self.move_history_placeholder, moves)

    def display_status(self, message: str, status_type: str = "info"):
        if status_type == "success":
            self.status_placeholder.success(message)
        elif status_type == "error":
            self.status_placeholder.error(message)
        elif status_type == "warning":
            self.status_placeholder.warning(message)
        else:
            self.status_placeholder.info(message)


# ---------------------------------------------------------------------------
# Provider + model selection
# ---------------------------------------------------------------------------


async def fetch_models_for_provider(provider_name: str, api_key: str) -> list:
    """Fetch available models for a provider."""
    provider = get_provider(provider_name)
    if not provider:
        return []
    try:
        return await provider.list_models(api_key)
    except ProviderError as exc:
        render_error(st, exc)
        return []
    except Exception as exc:
        st.error(f"Failed to fetch models from {provider_name}: {exc}")
        return []


def _probe_local_provider(provider_name: str) -> tuple[bool, str]:
    """Returns (reachable, message) for a local provider's HTTP endpoint."""
    import httpx

    base_url = "http://localhost:11434"
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{base_url}/api/tags")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"Server unreachable at `{base_url}` ({exc.__class__.__name__})"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Probe failed: {exc}"
    return True, f"Connected to `{base_url}`"


def _clear_models_cache() -> int:
    """Drop every cached provider model list from session state.

    Cache keys are written by :func:`render_model_selectors` as
    ``models_<provider>_<key_hash>``. Returning the count keeps the sidebar
    message honest — visitors can see how many (if any) lists were dropped.
    """
    keys = [k for k in list(st.session_state.keys()) if k.startswith("models_")]
    for k in keys:
        st.session_state.pop(k, None)
    return len(keys)


def render_provider_keys_section():
    """Render API key inputs for each provider in sidebar."""
    st.sidebar.header("🔑 API Keys")

    providers = list_providers()
    available_providers = []

    demo_api_key = None
    try:
        demo_api_key = st.secrets.get("openrouter_api_key")
    except Exception:
        demo_api_key = None

    if demo_api_key:
        with st.sidebar.expander("🎁 Demo Mode (OpenRouter)", expanded=True):
            st.info("OpenRouter is pre-configured — no key needed!")
            if st.button("🔌 Use Demo Mode", key="use_demo_key", width="stretch"):
                provider = get_provider("openrouter")
                if provider and provider.validate_key(demo_api_key):
                    available_providers.append(("openrouter", demo_api_key))

    for provider_name in providers:
        if HOSTED_PROVIDERS and provider_name not in HOSTED_PROVIDERS:
            continue
        provider = get_provider(provider_name)
        if provider is None:
            continue

        if not provider.requires_api_key:
            with st.sidebar.expander(f"🖥️ {provider_name.capitalize()} (local)", expanded=False):
                st.caption(f"Local server — no API key required. Run `{provider_name} serve` first.")
                if st.button(
                    f"🔌 Connect to {provider_name.capitalize()}",
                    key=f"connect_{provider_name}",
                    width="stretch",
                ):
                    reachable, message = _probe_local_provider(provider_name)
                    if reachable:
                        st.success(f"✓ {message}")
                        available_providers.append((provider_name, ""))
                    else:
                        st.error(f"✗ {message}")
            continue

        with st.sidebar.expander(f"{provider_name.capitalize()}", expanded=False):
            api_key = st.text_input(
                f"{provider_name.capitalize()} API Key",
                type="password",
                key=f"api_key_{provider_name}",
                help=f"Enter your {provider_name} API key",
            )
            if api_key:
                if provider.validate_key(api_key):
                    st.success("✓ Valid key format")
                    available_providers.append((provider_name, api_key))
                else:
                    st.error("✗ Invalid key format")

    return available_providers


def render_model_selectors(available_providers: list):
    """Render model selection for White and Black players."""
    st.sidebar.header("♟️ Model Selection")
    st.sidebar.caption("💡 **Hint:** Click a dropdown and type the name of a model, or type `free` to search for free models instead of using credits.")

    all_models: dict[str, dict] = {}
    filtered_count = 0

    for provider_name, api_key in available_providers:
        import hashlib
        key_hash = hashlib.md5(api_key.encode()).hexdigest()[:8]
        cache_key = f"models_{provider_name}_{key_hash}"

        # If cache miss, set loading flag and defer fetch to next rerun
        if cache_key not in st.session_state or not st.session_state[cache_key]:
            loading_flag = f"loading_{provider_name}"
            if not st.session_state.get(loading_flag, False):
                st.session_state[loading_flag] = True
                st.rerun()
            # On this rerun, we're in loading state — show staged card and do fetch
            render_loading_card("Fetching models", "Connecting to provider", provider_name, st.sidebar)
            models = asyncio.run(fetch_models_for_provider(provider_name, api_key))
            if models:
                st.session_state[cache_key] = models
            st.session_state[loading_flag] = False
            st.rerun()

        # Cache hit — use models normally
        models = st.session_state[cache_key]

        for model in models:
            if not is_chess_capable(model):
                filtered_count += 1
                continue
            display_name = f"[{provider_name}] {model.name}"
            all_models[display_name] = {
                "provider": provider_name,
                "model_id": model.id,
                "api_key": api_key,
                "context_window": model.context_window,
            }

    if filtered_count:
        st.sidebar.caption(
            f"ⓘ {filtered_count} non-chat model(s) hidden (embedding, audio, image, etc.)"
        )

    if not all_models:
        st.sidebar.warning("No chess-capable models available. Please add API keys.")
        return None, None

    if len(all_models) < 2:
        st.sidebar.warning("At least 2 distinct chess-capable models are required to start a match. Please enable another provider or API key.")
        return None, None

    model_options = list(all_models.keys())

    # Ensure initial session state picks 2 distinct models
    sel_1 = st.session_state.get("player_model_1")
    sel_2 = st.session_state.get("player_model_2")

    if not sel_1 or sel_1 not in model_options:
        sel_1 = model_options[0]
        st.session_state["player_model_1"] = sel_1

    if not sel_2 or sel_2 not in model_options or sel_2 == sel_1:
        remaining = [m for m in model_options if m != sel_1]
        sel_2 = remaining[0] if remaining else model_options[0]
        st.session_state["player_model_2"] = sel_2

    # Filter options for Model 1 (exclude currently selected Model 2)
    cur_2 = st.session_state.get("player_model_2")
    m1_options = [m for m in model_options if m != cur_2]
    m1_idx = m1_options.index(sel_1) if sel_1 in m1_options else 0

    st.sidebar.subheader("Model 1")
    model_1 = st.sidebar.selectbox(
        "Select First Model",
        options=m1_options,
        index=m1_idx,
        key="player_model_1",
    )

    # Filter options for Model 2 (exclude currently selected Model 1)
    m2_options = [m for m in model_options if m != model_1]
    cur_2_val = st.session_state.get("player_model_2")
    m2_idx = m2_options.index(cur_2_val) if cur_2_val in m2_options else 0

    st.sidebar.subheader("Model 2")
    model_2 = st.sidebar.selectbox(
        "Select Second Model",
        options=m2_options,
        index=m2_idx,
        key="player_model_2",
    )

    if model_1 and model_2 and model_1 != model_2:
        return all_models[model_1], all_models[model_2]
    return None, None


def create_provider_ai(white_config: dict, black_config: dict):
    """Create ProviderChessAI instances for both players."""
    params: dict = {"temperature": 0.1, "max_tokens": 1500}
    if white_config["provider"] == "stockfish":
        params["depth"] = st.session_state.get("stockfish_depth", 8)
        params["think_time"] = st.session_state.get("stockfish_think", 1.0)
    white_ai = ProviderChessAI(
        provider_name=white_config["provider"],
        model_id=white_config["model_id"],
        api_key=white_config["api_key"],
        **params,
    )
    black_params: dict = dict(params)
    if black_config["provider"] == "stockfish":
        black_params["depth"] = st.session_state.get("stockfish_depth", 8)
        black_params["think_time"] = st.session_state.get("stockfish_think", 1.0)
    black_ai = ProviderChessAI(
        provider_name=black_config["provider"],
        model_id=black_config["model_id"],
        api_key=black_config["api_key"],
        **black_params,
    )
    return white_ai, black_ai





def _reset_game_state() -> None:
    st.session_state.game_running = False
    st.session_state.active_match_config = None
    for k in [
        "benchmark_thread",
        "benchmark_state",
        "benchmark_error",
        "benchmark_done",
        "benchmark_game_index",
        "benchmark_start_time",
        "benchmark_run_dir",
        "benchmark_runner",
    ]:
        st.session_state.pop(k, None)


def render_live_game_screen(
    *,
    state: GameState | None,
    white_spec: str,
    black_spec: str,
    game_idx: int,
    total_games: int,
    start_time: float,
    completed_games: list[GameState],
    is_paused: bool = False,
    pause_reason: str | None = None,
) -> None:
    """Render the arena-style live game screen.

    Layout (DESIGN.md § 3):
    - Top: progress bar
    - Left column (board): eval bar left, board centered 560px desktop / 100% mobile, move ticker above
    - Right column (panels): player banners stacked, metrics 2x3, last completion drawer, thinking trace drawer
    - Below: completed games stack as cf-cards
    """
    # Progress bar at top
    if state is not None and not is_paused:
        frac = min(1.0, game_idx / max(1, total_games))
        st.progress(frac, text=f"Game {game_idx + 1} / {total_games} in progress")
    elif is_paused:
        st.warning(f"⏸ Paused — {pause_reason or 'Unknown reason'}")

    # Inject move ticker auto-scroll JS (runs on each render, scrolls latest pill into view)
    st.markdown("""
    <script>
    (function() {
        const ticker = document.querySelector('.cf-move-ticker');
        if (ticker) {
            const pills = ticker.querySelectorAll('.cf-move-pill.cf-move-current');
            if (pills.length > 0) {
                const last = pills[pills.length - 1];
                last.scrollIntoView({ behavior: 'smooth', inline: 'center' });
            }
        }
    })();
    </script>
    """, unsafe_allow_html=True)

    # Two-column arena frame: left=board (fixed ~560px), right=panels
    left, right = st.columns([0.55, 0.45], gap="large")

    with left:
        # Move ticker ABOVE board (DESIGN.md §3)
        if state is not None:
            st.caption(f"Move History ({len(state.moves)} plys)")

        # Board + eval bar (board centered, fixed width via CSS)
        if state is not None:
            king = state.board.king(state.board.turn)
            check_sq = king if state.board.is_check() and king is not None else None
            last_mv = state.board.peek() if state.board.move_stack else None

            # If there's a last move, use its eval. Otherwise default to 0.
            cp = state.moves[-1].cp_score if state.moves and state.moves[-1].cp_score is not None else 0
            mate = state.moves[-1].mate_in if state.moves else None

            render_board_with_evalbar(
                state.board,
                size=560,  # Fixed desktop width per DESIGN.md
                lastmove=last_mv,
                check_square=check_sq,
                cp_score=cp,
                mate_in=mate,
            )
        else:
            board = chess.Board()
            render_board_with_evalbar(board, size=560)

    with right:
        if state is not None:
            white_name = state.moves[0].player if state.moves else white_spec
            black_name = state.moves[1].player if len(state.moves) >= 2 else black_spec
            is_white_turn = state.board.turn == chess.WHITE

            # Player banners stacked: White top, Black bottom
            st.markdown(
                player_banner_html(
                    name=white_name,
                    spec=white_spec,
                    color="white",
                    is_turn=is_white_turn,
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                player_banner_html(
                    name=black_name,
                    spec=black_spec,
                    color="black",
                    is_turn=not is_white_turn,
                ),
                unsafe_allow_html=True,
            )

            # Metrics card: 2x3 grid
            with st.container(border=True):
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Total Moves", state.stats.total_moves)
                with cols[1]:
                    st.metric("Captures", state.stats.capture_moves)
                with cols[2]:
                    st.metric("Checks", state.stats.check_moves)
    
                cols2 = st.columns(3)
                with cols2[0]:
                    elapsed = int(state.game_duration) if state.game_duration > 0 else int(time.time() - start_time)
                    st.metric("Time", f"{elapsed}s")
                with cols2[1]:
                    turn_color = "White ♔" if state.board.turn else "Black ♚"
                    st.metric("Turn", turn_color if not state.is_game_over else "—")
                with cols2[2]:
                    if state.is_game_over:
                        term = getattr(state.stats, 'termination_reason', 'unknown')
                        st.metric("Result", term.replace('_', ' ').title())
                    else:
                        # Avg latency when available
                        cr = state.last_completion_result
                        st.metric("Avg Latency", format_duration_ms(cr.latency_ms) if cr and cr.latency_ms else "—")

            # Last completion drawer
            cr = state.last_completion_result
            if cr:
                with st.expander(f"Last completion ({state.current_player})", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Latency", format_duration_ms(cr.latency_ms) if cr.latency_ms else "—")
                    with c2:
                        st.metric("Tokens in", f"{cr.tokens_in or 0}")
                    with c3:
                        st.metric("Tokens out", f"{cr.tokens_out or 0}")
                    if cr.error:
                        st.error(f"{cr.error_type or 'Error'}: {cr.error}")
                    else:
                        st.caption("Raw response:")
                        st.code(cr.text or "", language="text")

            # Thinking trace drawer (collapsed by default, summary always visible)
            render_thinking_trace_drawer(state)

            if state.moves:
                with st.expander("Move History Data", expanded=True):
                    df_rows = []
                    ply_idx = 0
                    for m in state.moves:
                        is_white_turn = (ply_idx % 2 == 0)
                        piece_map = {'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛'}
                        cap_suffix = f" ({piece_map.get(m.captured_piece, m.captured_piece)})" if m.captured_piece else ""
                        is_illegal = getattr(m, "is_illegal", False)
                        df_rows.append({
                            "Turn": ply_idx + 1,
                            "Color": "White" if is_white_turn else "Black",
                            "Player": m.player.split(":")[0],
                            "Move": f"{m.move_san or m.move}{cap_suffix}",
                            "Capture": "✅" if m.is_capture else "",
                            "Check": "✅" if m.is_check else "",
                            "Checkmate": "✅" if m.is_checkmate else "",
                            "Illegal": "❌" if is_illegal else "",
                            "Eval": f"M{m.mate_in}" if m.mate_in else (f"{m.cp_score/100:+.2f}" if m.cp_score is not None else ""),
                            "Latency": format_duration_ms(m.latency_ms),
                            "Tokens": f"{m.tokens_in or 0} in / {m.tokens_out or 0} out" if m.tokens_in or m.tokens_out else "",
                            "Reasoning": m.reasoning.replace("<", "&lt;").replace(">", "&gt;") if m.reasoning else "",
                        })
                        if not is_illegal:
                            ply_idx += 1
                    df = pd.DataFrame(df_rows)
                    column_config = {
                        "Capture": st.column_config.TextColumn("Capture", width="small"),
                        "Check": st.column_config.TextColumn("Check", width="small"),
                        "Checkmate": st.column_config.TextColumn("Checkmate", width="small"),
                        "Illegal": st.column_config.TextColumn("Illegal", width="small"),
                        "Reasoning": st.column_config.TextColumn(
                            "Reasoning",
                            help="Click a cell to read the LLM's full reasoning for this move.",
                            width="large",
                        )
                    }
                    st.dataframe(df, hide_index=True, width="stretch", column_config=column_config)


    # Completed games stack as cf-cards
    if completed_games:
        st.markdown("---")
        st.markdown(f"### 🗂 Completed games ({len(completed_games)})")
        for i, completed_state in enumerate(completed_games):
            _draw_completed_game_summary_card(i, completed_state)


def _draw_completed_game_summary_card(game_idx: int, state: GameState) -> None:
    """Render one completed game as a cf-card with inline board, metrics, moves."""
    white_player = state.moves[0].player if state.moves else "?"
    black_player = state.moves[1].player if len(state.moves) >= 2 else "?"
    term_reason = getattr(state.stats, 'termination_reason', 'unknown')
    winner = state.winner or "?"

    with st.container(border=True):

        # Header row
        hdr_cols = st.columns([3, 1, 1, 1])
        with hdr_cols[0]:
            st.markdown(f"**Game {game_idx + 1}** · ♔ White: **{white_player}** vs ♚ Black: **{black_player}**")
        with hdr_cols[1]:
            st.metric("Result", winner)
        with hdr_cols[2]:
            st.metric("Termination", term_reason.replace('_', ' ').title())
        with hdr_cols[3]:
            if state.last_completion_result:
                st.metric("Avg Latency", format_duration_ms(state.last_completion_result.latency_ms) if state.last_completion_result.latency_ms else "—")

        # Board + metrics side by side
        bcol, mcol = st.columns([1, 1])
        with bcol:
            king = state.board.king(state.board.turn)
            check_sq = king if state.board.is_check() and king is not None else None
            last_mv = state.board.peek() if state.board.move_stack else None
            render_board_with_evalbar(state.board, size=320, lastmove=last_mv, check_square=check_sq)

        with mcol:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Moves", state.stats.total_moves)
            m2.metric("Captures", state.stats.capture_moves)
            m3.metric("Checks", state.stats.check_moves)

        # Move history as pills
        if state.moves:
            with st.expander("Move History Data", expanded=False):
                df_rows = []
                ply_idx = 0
                for m in state.moves:
                    is_white_turn = (ply_idx % 2 == 0)
                    piece_map = {'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛'}
                    cap_suffix = f" ({piece_map.get(m.captured_piece, m.captured_piece)})" if m.captured_piece else ""
                    df_rows.append({
                        "Turn": ply_idx + 1,
                        "Color": "White" if is_white_turn else "Black",
                        "Player": m.player.split(":")[0],
                        "Move": (f"{m.move_san or m.move}{cap_suffix}") if not m.is_illegal else f"❌ {m.move}",
                        "Eval": f"M{m.mate_in}" if m.mate_in else (f"{m.cp_score/100:+.2f}" if m.cp_score is not None else ""),
                        "Latency": format_duration_ms(m.latency_ms),
                        "Tokens": f"{m.tokens_in or 0} in / {m.tokens_out or 0} out" if m.tokens_in or m.tokens_out else "",
                        "Reasoning": m.reasoning.replace("<", "&lt;").replace(">", "&gt;") if m.reasoning else "",
                    })
                    if getattr(m, "is_illegal", False) is False:
                        ply_idx += 1
                df = pd.DataFrame(df_rows)
                column_config = {
                    "Reasoning": st.column_config.TextColumn(
                        "Reasoning",
                        help="Click a cell to read the LLM's full reasoning for this move.",
                        width="large",
                    )
                }
                st.dataframe(df, hide_index=True, width="stretch", column_config=column_config)


def run_in_process_benchmark(
    white_config: dict,
    black_config: dict,
    games: int = 3,
    colors: str = "alternating",
    reasoning_level: str = "mid",
):
    """Run the benchmark runner in-process and render live results."""
    white_spec = f"{white_config['provider']}:{white_config['model_id']}"
    black_spec = f"{black_config['provider']}:{black_config['model_id']}"

    api_keys: dict[str, str] = {}
    for provider_name, key in (
        (white_config["provider"], white_config["api_key"]),
        (black_config["provider"], black_config["api_key"]),
    ):
        if key:
            api_keys[provider_name] = key
    # Local providers don't need keys but accept an empty string.
    if white_config["provider"] == "stockfish":
        api_keys.setdefault("stockfish", "")
    if black_config["provider"] == "stockfish":
        api_keys.setdefault("stockfish", "")

    config = BenchmarkConfig(
        players=[white_spec, black_spec],
        games_per_pairing=games,
        max_parallel_games=1,
        opening_book="startpos",
        temperature=0.0,
        max_tokens=None,
        reasoning_level=reasoning_level,
        api_keys=api_keys,
        colors=colors,
    )

    # Immersive Theater Mode: hide the sidebar ONLY while a benchmark is
    # actively running. The completion screen restores the sidebar so the
    # session retains its normal navigation surface and doesn't end on a
    # different-looking screen.
    benchmark_active = not st.session_state.get("benchmark_done", False)
    if benchmark_active:
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] { display: none !important; }
                [data-testid="collapsedControl"] { display: none !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if "benchmark_runner" not in st.session_state or st.session_state.benchmark_runner is None:
        try:
            st.session_state.benchmark_runner = BenchmarkRunner(config)
        except (NoProvidersConfiguredError, SetupError, InvalidApiKeyError) as exc:
            render_error(st, exc)
            if st.button("🔙 Return to Main Menu", type="primary", width="stretch", key="err_setup_return"):
                _reset_game_state()
                st.rerun()
            return
        except ValueError as exc:
            st.error(f"Benchmark setup failed: {exc}")
            if st.button("🔙 Return to Main Menu", type="primary", width="stretch", key="err_val_return"):
                _reset_game_state()
                st.rerun()
            return

    runner = st.session_state.benchmark_runner

    num_players = len(runner.config.players)
    if colors == "fixed":
        num_pairings = 1 if num_players >= 2 else 0
    elif colors == "alternating" and num_players == 2:
        # Single pairing with alternating colors per game
        num_pairings = 1
    else:
        # Multiple players: all pairings
        num_pairings = num_players * (num_players - 1)
    total_games = num_pairings * games
    # Use total_games to avoid F841 warning - pass it to start_benchmark
    _ = total_games

    def start_benchmark():
        st.session_state.benchmark_state = None
        st.session_state.benchmark_error = None
        st.session_state.benchmark_done = False
        st.session_state.benchmark_game_index = 0
        st.session_state.benchmark_start_time = time.time()
        st.session_state.benchmark_run_dir = None
        st.session_state.benchmark_completed_games = []  # Store completed game states

        def ui_callback_sync(state: GameState):
            try:
                if state.is_game_over:
                    # Store a copy of the completed game state
                    import copy
                    completed_game = copy.deepcopy(state)
                    st.session_state.benchmark_completed_games.append(completed_game)
                    st.session_state.benchmark_game_index += 1
                    # Don't update benchmark_state for completed games - keep it for live game
                else:
                    st.session_state.benchmark_state = state
            except BaseException:
                pass

        async def live_callback(state: GameState):
            ui_callback_sync(state)

        import contextlib

        def thread_func():
            try:
                asyncio.run(runner.run_benchmark_with_callback(live_callback))
            except Exception as e:
                with contextlib.suppress(BaseException):
                    st.session_state.benchmark_error = e
            finally:
                with contextlib.suppress(BaseException):
                    st.session_state.benchmark_done = True
                    st.session_state.benchmark_run_dir = runner.run_dir

        t = threading.Thread(target=thread_func)
        add_script_run_ctx(t)
        st.session_state.benchmark_thread = t
        t.start()

    def _draw_paused_ui(board_ph, stats_ph, moves_ph, completion_ph, state):
        """Draw the paused game UI with retry/cancel options.

        Renders different button rows depending on ``pause_reason``:
        - mid-game move error → Retry / Skip / Cancel buttons that resume or
          cancel the current ``AsyncChessGame``.
        - benchmark-level ``game_failed`` pause (a game ended without a clean
          chess terminal) → Continue / Abort buttons that release the runner's
          cross-thread ``threading.Event`` so the runner proceeds to the next
          game or raises ``FatalBenchmarkError`` respectively.
        The surrounding layout (board snapshot, error details, metrics, moves)
        is identical in both cases, so the paused screen looks the same to the
        user regardless of which kind of pause triggered it.
        """
        progress_ph = st.empty()

        progress_ph.warning(f"⏸ Game Paused — {state.pause_reason or 'Unknown reason'}")

        st.error(f"**Error:** {state.pause_error or 'No error details'}")
        st.info(f"**Failed Player:** {state.paused_player} (Turn {state.paused_turn + 1})")

        start_time = st.session_state.benchmark_start_time
        _draw_board(board_ph, state, start_time)
        _draw_metrics(stats_ph, state, start_time)
        _draw_moves(moves_ph, state.moves)

        is_benchmark_pause = (state.pause_reason == "game_failed")
        if is_benchmark_pause:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Continue to next game", type="primary", width="stretch", key="paused_continue"):
                    runner = st.session_state.get("benchmark_runner")
                    if runner is not None and hasattr(runner, "request_continue_after_problem"):
                        runner.request_continue_after_problem()
                    st.rerun()
            with col2:
                if st.button("⛔ Abort benchmark", width="stretch", key="paused_abort"):
                    runner = st.session_state.get("benchmark_runner")
                    if runner is not None and hasattr(runner, "request_abort_after_problem"):
                        runner.request_abort_after_problem()
                    st.rerun()
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("↻ Retry Turn", type="primary", width="stretch", key="paused_retry"):
                    runner = st.session_state.get("benchmark_runner")
                    if runner:
                        runner.resume_game(retry=True)
                    st.rerun()
            with col2:
                if st.button("⏭ Skip Turn (Force Move)", width="stretch", key="paused_skip"):
                    runner = st.session_state.get("benchmark_runner")
                    if runner:
                        runner.resume_game(retry=False, force_move=True)
                    st.rerun()
            with col3:
                if st.button("⛔ Cancel Game", width="stretch", key="paused_cancel"):
                    runner = st.session_state.get("benchmark_runner")
                    if runner and hasattr(runner, 'current_game') and runner.current_game:
                        runner.current_game.cancel()
                    st.rerun()


    # Now start the benchmark if not already running
    if "benchmark_thread" not in st.session_state or (not st.session_state.benchmark_thread.is_alive() and not st.session_state.get("benchmark_done", False)):
        if not st.session_state.get("benchmark_done", False):
            start_benchmark()

    def _draw_live_ui():
        state: GameState | None = st.session_state.get("benchmark_state")
        game_idx = st.session_state.benchmark_game_index
        is_paused = getattr(state, 'is_paused', False)
        
        if is_paused:
            if "benchmark_pause_time" not in st.session_state:
                st.session_state.benchmark_pause_time = time.time()
            # Slide start_time forward so time.time() - start_time remains constant during pause
            pause_dur = time.time() - st.session_state.benchmark_pause_time
            start_time = st.session_state.benchmark_start_time + pause_dur
        else:
            if "benchmark_pause_time" in st.session_state:
                # Commit the pause duration to start_time
                st.session_state.benchmark_start_time += (time.time() - st.session_state.benchmark_pause_time)
                del st.session_state["benchmark_pause_time"]
            start_time = st.session_state.benchmark_start_time

        completed_games = st.session_state.get("benchmark_completed_games", [])
        pause_reason = getattr(state, 'pause_reason', None) if is_paused else None

        render_live_game_screen(
            state=state,
            white_spec=white_spec,
            black_spec=black_spec,
            game_idx=game_idx,
            total_games=total_games,
            start_time=start_time,
            completed_games=completed_games,
            is_paused=is_paused,
            pause_reason=pause_reason,
        )

    if st.session_state.get("benchmark_error"):
        exc = st.session_state.benchmark_error
        if isinstance(exc, (GameExecutionError, FatalBenchmarkError)) or isinstance(exc, (NoProvidersConfiguredError, SetupError, InvalidApiKeyError)):
            render_error(st, exc)
        else:
            st.error(f"Benchmark failed: {exc}")
        if st.button("🔙 Return to Main Menu", type="primary", width="stretch", key="err_bm_return"):
            _reset_game_state()
            st.rerun()
        return

    if not st.session_state.get("benchmark_done", False):
        if hasattr(st, "fragment"):
            @st.fragment(run_every=2.0)
            def _live_fragment():
                if st.session_state.get("benchmark_done", False):
                    st.rerun()
                _draw_live_ui()
            _live_fragment()
            return
        else:
            _draw_live_ui()
            time.sleep(2.0)
            st.rerun()

    # Completion: render the SAME screen as during the run (live board +
    # stacked completed-game summaries) so the user doesn't end on a different
    # looking screen; the sidebar is now visible again because
    # ``benchmark_active`` was False above. Then append the run summary below.
    _draw_live_ui()

    st.markdown("---")
    st.success("Benchmark complete!")

    run = None
    if st.session_state.get("benchmark_run_dir"):
        run = load_run(st.session_state.benchmark_run_dir)

    if run is not None:
        render_run_summary(run, expanded=True)

    if st.button("🔙 Return to Main Menu", type="primary", width="stretch", key="done_bm_return"):
        _reset_game_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Benchmark history
# ---------------------------------------------------------------------------


def render_benchmark_history(*, expanded: bool = False) -> None:
    """Render benchmark runs parsed from the on-disk JSONL artifacts.

    `expanded=True` opens the expander so the sidebar's "📊 Benchmark History"
    toggle reveals the runs immediately on first click.
    """
    with st.expander("📊 Benchmark History", expanded=expanded):
        runs = list_runs(RUNS_ROOT)
        if not runs:
            st.info(
                "No benchmark runs found under "
                f"`{RUNS_ROOT}`. Select two models in the sidebar and click "
                "**▶️ Start Match** to generate a real run."
            )
            return

        st.markdown(f"**{len(runs)} real run(s)** loaded from `{RUNS_ROOT}`:")

        # Aggregated leaderboard across all runs.
        leaderboard = aggregate_leaderboard(runs)
        if leaderboard:
            st.markdown("### Aggregated leaderboard (all runs)")
            rows = []
            for row in leaderboard:
                rows.append(
                    {
                        "Player": row.player,
                        "Games": row.games,
                        "W": row.wins,
                        "L": row.losses,
                        "D": row.draws,
                        "Score %": f"{row.score_pct:.1f}" if row.score_pct is not None else "—",
                        "Avg latency (ms)": (
                            f"{row.avg_latency_ms:.0f}"
                            if row.avg_latency_ms is not None
                            else "—"
                        ),
                        "Tokens in": row.tokens_in if row.tokens_in is not None else None,
                        "Tokens out": row.tokens_out if row.tokens_out is not None else None,
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        # Per-run breakdown as cf-cards.
        st.markdown("### Per-run details")
        for run in runs:
            render_run_summary(run, expanded=False)


def render_run_summary(run, *, expanded: bool) -> None:
    """Render one benchmark run as a compact data block."""
    label = (
        f"{run.run_id} · {run.total_games} games · "
        f"{len(run.providers_seen)} provider(s) · "
        f"{datetime.utcfromtimestamp(0).isoformat() if not run.timestamp_utc else run.timestamp_utc}"
    )

    # Render as cf-card instead of bare expander
    with st.container(border=True):

        # Run header with analyze button
        hdr_col1, hdr_col2 = st.columns([4, 1])
        with hdr_col1:
            st.markdown(f"### {run.run_id}")
            st.caption(f"{run.total_games} games · {len(run.providers_seen)} provider(s)")
        with hdr_col2:
            if st.button("📊 Analyze", key=f"analyze_run_{run.run_id}", type="primary", width="stretch"):
                st.session_state.show_analytics = True
                st.session_state.show_history = False
                st.session_state.analytics_run_dir = str(run.run_dir)
                st.session_state.game_running = False
                st.session_state.pop("benchmark_state", None)
                st.rerun()

        if expanded:
            if run.config:
                with st.expander("Config", expanded=False):
                    st.json(run.config)

            # Per-player table for this run.
            rows = []
            for _name, ps in run.player_stats.items():
                rows.append(
                    {
                        "Player": ps.name,
                        "Games": ps.games_played,
                        "W": ps.wins,
                        "L": ps.losses,
                        "D": ps.draws,
                        "Score %": (
                            f"{ps.score_pct:.1f}" if ps.score_pct is not None else "—"
                        ),
                        "Avg latency (ms)": (
                            f"{ps.avg_latency_ms:.0f}"
                            if ps.avg_latency_ms is not None
                            else "—"
                        ),
                        "Captures": ps.captures,
                        "Checks": ps.checks,
                        "Tokens in": ps.tokens_in_total if ps.tokens_in_total is not None else None,
                        "Tokens out": ps.tokens_out_total if ps.tokens_out_total is not None else None,
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            # Head-to-head pairings.
            if run.pairings:
                pair_rows = [
                    {
                        "White": p.white,
                        "Black": p.black,
                        "Games": p.games,
                        "White wins": p.white_wins,
                        "Black wins": p.black_wins,
                        "Draws": p.draws,
                        "Total moves": p.total_moves,
                    }
                    for p in run.pairings
                ]
                st.caption("Head-to-head pairings:")
                st.dataframe(pd.DataFrame(pair_rows), hide_index=True, width="stretch")

            # Interactive Game Viewer
            if run.games:
                render_game_viewer(run)


def render_game_viewer(run) -> None:
    """Interactive game viewer for stepping through moves of past games."""
    if not run.games:
        return

    st.markdown("### ♟️ Game Replays & Logs")

    for i, game in enumerate(run.games):
        term_reason = getattr(game, 'termination_reason', 'unknown')
        st.markdown(f"#### Game {i+1}: ♔ White: **{game.white_player}** vs ♚ Black: **{game.black_player}** ({game.result}) — *{term_reason.replace('_', ' ').title()}*")

        if not game.moves:
            st.info("No moves recorded for this game.")
            st.divider()
            continue

        # Select Move
        max_moves = len(game.moves)

        # Calculate step state
        move_idx = st.slider(f"Rewind Game {i+1}", 0, max_moves, max_moves, key=f"slider_{run.run_id}_{game.game_id}")

        if move_idx == 0:
            fen = game.opening_fen or chess.STARTING_FEN
            last_move = None
            move_info = None
        else:
            m = game.moves[move_idx - 1]
            b = chess.Board(m.fen_before)
            try:
                b.push_uci(m.move_uci)
                fen = b.fen()
                last_move = chess.Move.from_uci(m.move_uci)
            except Exception:
                fen = m.fen_before
                last_move = None
            move_info = m

        col1, col2 = st.columns([1, 1])
        with col1:
            b = chess.Board(fen)
            king_square = b.king(b.turn)
            check_square = king_square if b.is_check() and king_square is not None else None

            # Use cf-board-frame via render_board_with_evalbar
            render_board_with_evalbar(b, size=400, lastmove=last_move, check_square=check_square)

        with col2:
            if move_info:
                player_name = game.white_player if move_info.color == "white" else game.black_player
                st.markdown(f"**Move {move_info.move_number}** - {player_name} ({move_info.color.title()}) played `{move_info.move_san}`")
                st.metric("Latency", format_duration_ms(move_info.llm_latency_ms) if move_info.llm_latency_ms else "—")
                if move_info.llm_tokens_out:
                    st.metric("Tokens Out", move_info.llm_tokens_out)

                with st.expander("Model Thinking Trace"):
                    if move_info.thinking_trace:
                        st.code(move_info.thinking_trace, language="text")
                    else:
                        st.info("No thinking trace recorded.")

                with st.expander("Raw Provider Response"):
                    if move_info.llm_raw_response:
                        st.code(move_info.llm_raw_response, language="json")
                    else:
                        st.info("No raw response recorded.")
            else:
                st.info("Starting Position")

        # Game Table
        st.markdown(f"##### Game {i+1} Move History")

        df_rows = []
        for m in game.moves:
            # Parse timestamp if possible
            t_str = m.timestamp_utc
            if t_str:
                if len(t_str) > 19:
                    t_str = t_str[11:19]

            san = m.move_san or ""
            df_rows.append({
                "Move #": m.move_number,
                "Color": m.color.title(),
                "Player": game.white_player if m.color == "white" else game.black_player,
                "Move": san,
                "Capture": "✅" if "x" in san else "",
                "Check": "✅" if "+" in san else "",
                "Checkmate": "✅" if "#" in san else "",
                "Reasoning": m.thinking_trace.replace("<", "&lt;").replace(">", "&gt;") if m.thinking_trace else "",
            })

        df = pd.DataFrame(df_rows)

        column_config = {
            "Capture": st.column_config.TextColumn("Capture", width="small"),
            "Check": st.column_config.TextColumn("Check", width="small"),
            "Checkmate": st.column_config.TextColumn("Checkmate", width="small"),
            "Reasoning": st.column_config.TextColumn(
                "Reasoning",
                help="Click a cell to read the LLM's full reasoning for this move.",
                width="large",
            )
        }
        st.dataframe(df, hide_index=True, width="stretch", column_config=column_config)
        st.divider()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rehydrate_session_state()
    apply_arena_theme()
    if "game_ui" not in st.session_state:
        st.session_state.game_ui = ChessUI()

    # Sidebar: API Keys and Model Selection
    available_providers = render_provider_keys_section()
    model_1_config, model_2_config = render_model_selectors(available_providers)

    st.sidebar.markdown("---")
    st.sidebar.header("🎮 Game Controls")

    games = st.sidebar.number_input(
        "Games to play", min_value=1, max_value=20, value=1, step=1, key="game_count"
    )

    reasoning_level = st.sidebar.selectbox(
        "Reasoning Level",
        options=["low", "mid", "high"],
        index=1,
        help="Low = fast minimal thinking (256 tokens), Mid = standard tactical focus (1024 tokens), High = deep positional thinking (4096 tokens).",
        key="reasoning_level_select",
    )

    # We always alternate colors for multiple games, but for the first game we randomly assign White/Black
    # to avoid user bias, as playing White is an advantage.
    colors_mode = "alternating" if games > 1 else "fixed"

    if st.sidebar.button("▶️ Run Benchmark", type="primary", width="stretch"):
        if not model_1_config or not model_2_config:
            st.sidebar.error("Please select two distinct models for the players.")
        elif model_1_config["provider"] == model_2_config["provider"] and model_1_config["model_id"] == model_2_config["model_id"]:
            st.sidebar.error("Model 1 and Model 2 must be different models.")
        else:
            _reset_game_state()
            st.session_state.game_running = True
            
            import random
            if random.choice([True, False]):
                white_config, black_config = model_1_config, model_2_config
            else:
                white_config, black_config = model_2_config, model_1_config

            st.session_state.active_match_config = {
                "white_config": white_config,
                "black_config": black_config,
                "games": int(games),
                "colors": colors_mode,
                "reasoning_level": reasoning_level,
            }
            st.rerun()

    if st.session_state.get("game_running", False) and st.session_state.get("active_match_config"):
        match_cfg = st.session_state.active_match_config
        run_in_process_benchmark(
            match_cfg["white_config"],
            match_cfg["black_config"],
            games=match_cfg["games"],
            colors=match_cfg["colors"],
            reasoning_level=match_cfg.get("reasoning_level", "mid"),
        )
        return


    if st.session_state.get("show_analytics", False):
        render_analytical_dashboard()
        st.sidebar.markdown("---")
        st.sidebar.header("🗂 History & Cache")
        if st.sidebar.button("📊 Open Benchmark History", type="primary", width="stretch", key="open_history"):
            st.session_state.show_analytics = False
            st.session_state.show_history = True
            st.rerun()
        return

    show_history = st.session_state.get("show_history", False)
    if not st.session_state.get("game_running", False):
        render_hero()
        render_landing_metrics(RUNS_ROOT)
    else:
        st.title("🤖 AI Chess Battle")
        st.write("Watch AI models compete in chess! Select models from any provider.")

    if show_history:
        st.markdown(
            '<div class="cf-section-title">📊 Benchmark History</div>'
            '<div class="cf-section-sub">Browse past runs parsed from real JSONL artifacts on disk. '
            'Open a run to view its leaderboard, head-to-head, and per-game replays with '
            'Stockfish eval timelines and move-quality heatmaps.</div>',
            unsafe_allow_html=True,
        )
        render_benchmark_history(expanded=True)
    else:
        st.markdown("### 🎮 Get Started")
        st.markdown(
            "**Have API keys?** Add them in the sidebar under **🔑 API Keys**, "
            "pick two models under **♟️ Model Selection**, then click **▶️ Start Match**."
        )
        render_benchmark_history(expanded=False)

    st.sidebar.markdown("---")
    st.sidebar.header("🗂 History & Cache")

    if st.sidebar.button("📊 Open Benchmark History", type="primary", width="stretch", key="open_history"):
        st.session_state.show_history = True
        st.rerun()

    if st.sidebar.button("✕ Close Benchmark History", width="stretch", key="close_history"):
        st.session_state.show_history = False
        st.rerun()

    if st.sidebar.button("🧹 Clear Created Models Cache", width="stretch", key="clear_cache"):
        cleared = _clear_models_cache()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        msg = (
            f"Cleared {cleared} cached model list(s) — next refresh will re-fetch from the providers."
            if cleared
            else "Model cache was empty — nothing to clear."
        )
        st.session_state.cache_cleared_msg = msg
        st.rerun()

    if st.session_state.get("cache_cleared_msg"):
        st.sidebar.success(st.session_state.cache_cleared_msg)
        st.session_state.cache_cleared_msg = None


def render_analytical_dashboard():
    """Render the analytical dashboard with eval graphs, move quality, opening explorer, etc."""
    back, title = st.columns([1, 6])
    with back:
        if st.button("← Back to History", key="analytics_back"):
            st.session_state.show_analytics = False
            st.session_state.show_history = True
            st.session_state.pop("analytics_run_dir", None)
            st.rerun()
    with title:
        st.markdown("## 📊 Analytical Dashboard")

    # Load available runs
    runs = list_runs(RUNS_ROOT)
    if not runs:
        st.info("No benchmark runs available. Run a benchmark first.")
        return

    # If we were asked to focus on a specific run, find it.
    pre_selected_run = None
    if "analytics_run_dir" in st.session_state:
        target_dir = Path(st.session_state.analytics_run_dir)
        for r in runs:
            if r.run_dir == target_dir:
                pre_selected_run = r
                break

    # Run selector as card header
    with st.container(border=True):
        run_options = {f"{run.run_id} ({run.total_games} games)": run for run in runs}
        default_idx = 0
        if pre_selected_run:
            label = f"{pre_selected_run.run_id} ({pre_selected_run.total_games} games)"
            default_idx = list(run_options.keys()).index(label) if label in run_options else 0
        selected_run_label = st.selectbox(
            "Select Run to Analyze",
            options=list(run_options.keys()),
            index=default_idx,
            key="analytics_run_selector",
        )
        selected_run = run_options[selected_run_label]

    # Load the selected run
    run = load_run(selected_run.run_dir)
    if run is None:
        st.error("Failed to load run data.")
        return

    # Game selector within the run
    if not run.games:
        st.info("No games in this run.")
        return

    with st.container(border=True):
        game_options = {f"Game {i+1}: {g.white_player} vs {g.black_player} ({g.result})": g for i, g in enumerate(run.games)}
        selected_game_label = st.selectbox(
            "Select Game to Analyze",
            options=list(game_options.keys()),
            key="analytics_game_selector"
        )
        selected_game = game_options[selected_game_label]

    st.markdown("---")

    # ============================================================
    # 1. EVAL TIMELINE - Line chart with horizontal advantage bar
    # ============================================================
    st.markdown("### 📈 Eval Timeline")
    eval_data = []
    board = chess.Board(selected_game.opening_fen)

    for ply, move_log in enumerate(selected_game.moves):
        move = chess.Move.from_uci(move_log.move_uci)
        if move in board.legal_moves:
            board.push(move)

        eval_data.append({
            "Ply": ply + 1,
            "Move": move_log.move_san,
            "Player": move_log.player,
            "Color": move_log.color,
            "Eval (cp)": move_log.eval_cp_score,
            "Best Move": move_log.eval_best_move_uci,
            "Best Eval (cp)": move_log.eval_best_move_cp,
        })

    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        # Filter rows with eval data
        eval_df = eval_df[eval_df["Eval (cp)"].notna()]
        if not eval_df.empty:
            import altair as alt

            # Horizontal advantage bar above chart (DESIGN.md §3)
            # Create a simple bar showing current advantage
            last_eval = eval_df.iloc[-1]["Eval (cp)"]
            max_abs_eval = max(abs(eval_df["Eval (cp)"].max()), abs(eval_df["Eval (cp)"].min()), 100)
            advantage_pct = 50 + (last_eval / max_abs_eval * 50) if max_abs_eval > 0 else 50
            advantage_pct = max(0, min(100, advantage_pct))

            adv_html = f"""
            <div class="cf-advantage-bar" style="height:8px;background:linear-gradient(90deg, var(--arena-black) 0%, var(--arena-black) 50%, var(--arena-white) 50%, var(--arena-white) 100%);border-radius:4px;margin-bottom:12px;position:relative;border:1px solid var(--arena-border-strong);overflow:hidden;">
                <div style="position:absolute;left:{advantage_pct}%;top:0;bottom:0;width:2px;background:var(--arena-accent);box-shadow:0 0 8px var(--arena-accent);"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:0.75rem;color:var(--arena-text-muted);margin-bottom:8px;">
                <span>Black advantage</span><span>Even</span><span>White advantage</span>
            </div>
            """
            st.markdown(adv_html, unsafe_allow_html=True)

            # Base line chart
            base = alt.Chart(eval_df).encode(
                x=alt.X("Ply:Q", title="Ply (half-move)"),
                y=alt.Y("Eval (cp):Q", title="Centipawn Evaluation"),
                color=alt.Color("Color:N", title="Side to Move"),
            )

            line = base.mark_line(point=True)
            points = base.mark_point(size=80, filled=True)

            chart = (line + points).properties(
                width=700,
                height=400,
                title="Stockfish Evaluation per Ply"
            ).interactive()

            st.altair_chart(chart, width="stretch")
        else:
            st.info("No evaluation data available for this game.")

    st.markdown("---")

    # ============================================================
    # 2. MOVE QUALITY HEATMAP - Rect-encoded ply x quality
    # ============================================================
    st.markdown("### 🔥 Move Quality Heatmap")

    quality_data = []
    for ply, move_log in enumerate(selected_game.moves):
        if move_log.move_quality:
            quality_data.append({
                "Ply": ply + 1,
                "Move": move_log.move_san,
                "Player": move_log.player,
                "Quality": move_log.move_quality,
                "CP Loss": move_log.cp_loss,
                "Is Best": move_log.is_best_move,
            })

    if quality_data:
        quality_df = pd.DataFrame(quality_data)

        # Color mapping from DESIGN.md
        quality_colors = {
            "best": "#2ecc71",
            "excellent": "#27ae60",
            "good": "#9ecd43",
            "inaccuracy": "#db6d28",
            "mistake": "#e67e22",
            "blunder": "#e74c3c",
        }

        # Quality order for y-axis
        quality_order = ["best", "excellent", "good", "inaccuracy", "mistake", "blunder"]

        # Create heatmap data: count of each quality per ply range
        heatmap_data = []
        for ply in quality_df["Ply"].unique():
            ply_data = quality_df[quality_df["Ply"] == ply]
            for q in quality_order:
                count = (ply_data["Quality"] == q).sum()
                if count > 0:
                    heatmap_data.append({
                        "Ply": ply,
                        "Quality": q,
                        "Count": int(count),
                        "Color": quality_colors[q],
                    })

        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)

            # Altair rect heatmap
            heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
                x=alt.X("Ply:O", title="Ply", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Quality:N", title="Move Quality", sort=quality_order),
                color=alt.Color("Color:N", scale=None, legend=None),
                tooltip=["Ply", "Quality", "Count"]
            ).properties(
                width=700,
                height=280,
                title="Move Quality Distribution per Ply"
            ).configure_view(stroke=None)

            st.altair_chart(heatmap_chart, width="stretch")

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            best_pct = (quality_df["Quality"] == "best").mean() * 100
            st.metric("Best Move %", f"{best_pct:.1f}%")
        with col2:
            blunder_rate = (quality_df["Quality"] == "blunder").mean() * 100
            st.metric("Blunder Rate", f"{blunder_rate:.1f}%")
        with col3:
            avg_cp = quality_df["CP Loss"].mean()
            st.metric("Avg CP Loss", f"{avg_cp:.1f}")
        with col4:
            mistake_rate = (quality_df["Quality"].isin(["mistake", "blunder"])).mean() * 100
            st.metric("Mistake+Blunder Rate", f"{mistake_rate:.1f}%")
    else:
        st.info("No move quality data available for this game.")

    st.markdown("---")

    # ============================================================
    # 3. OPENING EXPLORER - Table with inline win-rate bars
    # ============================================================
    st.markdown("### 🌳 Opening Explorer")

    # Aggregate opening stats across all runs
    opening_stats = {}
    for run_data in runs:
        loaded_run = load_run(run_data.run_dir)
        if loaded_run:
            for game in loaded_run.games:
                eco = game.opening_eco or "?"
                name = game.opening_name or "Unknown"
                key = f"{eco}: {name}"
                if key not in opening_stats:
                    opening_stats[key] = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "cp_losses": []}

                opening_stats[key]["games"] += 1
                if game.result_numeric == 1.0 and game.white_player == selected_game.white_player:
                    opening_stats[key]["wins"] += 1
                elif game.result_numeric == 0.0 and game.black_player == selected_game.black_player:
                    opening_stats[key]["losses"] += 1
                else:
                    opening_stats[key]["draws"] += 1

                # Collect cp losses for this opening
                for move in game.moves:
                    if move.cp_loss is not None:
                        opening_stats[key]["cp_losses"].append(move.cp_loss)

    if opening_stats:
        opening_rows = []
        for key, stats in opening_stats.items():
            avg_cp = sum(stats["cp_losses"]) / len(stats["cp_losses"]) if stats["cp_losses"] else 0
            win_rate = stats['wins'] / stats['games'] * 100 if stats["games"] > 0 else 0
            opening_rows.append({
                "Opening": key,
                "Games": stats["games"],
                "Wins": stats["wins"],
                "Losses": stats["losses"],
                "Draws": stats["draws"],
                "Win Rate": win_rate,
                "Win Rate %": f"{win_rate:.1f}%",
                "Avg CP Loss": f"{avg_cp:.1f}",
            })

        opening_df = pd.DataFrame(opening_rows).sort_values("Games", ascending=False)

        # Render with inline win-rate bars using HTML
        with st.container(border=True):
            for _, row in opening_df.iterrows():
                wr = row["Win Rate"]
                bar_color = "var(--eval-good)" if wr >= 50 else "var(--eval-blunder)"
                st.markdown(f"""
                <div style="display:flex;align-items:center;margin-bottom:8px;font-size:0.875rem;">
                    <div style="flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-right:12px;" title="{row['Opening']}">{row['Opening']}</div>
                    <div style="width:100px;background:var(--arena-border);height:8px;border-radius:4px;overflow:hidden;margin-right:12px;">
                        <div style="width:{wr}%;background:{bar_color};height:100%;"></div>
                    </div>
                    <div style="width:40px;text-align:right;font-family:var(--font-mono);font-size:0.8125rem;">{int(wr)}%</div>
                    <div style="width:90px;text-align:right;font-family:var(--font-mono);font-size:0.8125rem;color:var(--arena-text-muted);">{row['Avg CP Loss']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No opening data available.")

    st.markdown("---")

    # ============================================================
    # 4. MODEL COMPARISON - Radar chart (≥2 models) / Table fallback
    # ============================================================
    st.markdown("### 🤖 Model Comparison")

    # Compare models across all runs
    model_stats = {}
    for run_data in runs:
        loaded_run = load_run(run_data.run_dir)
        if loaded_run:
            for name, ps in loaded_run.player_stats.items():
                if name not in model_stats:
                    model_stats[name] = {
                        "games": 0, "wins": 0, "losses": 0, "draws": 0,
                        "total_cp_loss": 0, "cp_loss_count": 0,
                        "blunders": 0, "mistakes": 0, "inaccuracies": 0,
                        "best_moves": 0, "total_moves": 0,
                        "total_latency": 0, "latency_count": 0,
                        "thinking_chars": 0, "thinking_count": 0,
                    }
                stats = model_stats[name]
                stats["games"] += ps.games_played
                stats["wins"] += ps.wins
                stats["losses"] += ps.losses
                stats["draws"] += ps.draws
                stats["total_latency"] += ps.total_latency_ms
                stats["latency_count"] += ps.latency_samples

    if model_stats and len(model_stats) >= 2:
        # Build radar chart data
        radar_rows = []
        for name, stats in model_stats.items():
            score = stats["wins"] + 0.5 * stats["draws"]
            score_pct = (score / stats["games"] * 100) if stats["games"] > 0 else 0
            avg_latency = stats["total_latency"] / stats["latency_count"] if stats["latency_count"] > 0 else 0

            # Collect move quality stats from all games
            total_cp_loss = 0
            cp_loss_count = 0
            blunders = 0
            best_moves = 0
            total_moves = 0
            for run_data in runs:
                loaded_run = load_run(run_data.run_dir)
                if loaded_run:
                    for game in loaded_run.games:
                        if game.white_player == name or game.black_player == name:
                            for move in game.moves:
                                if move.cp_loss is not None:
                                    total_cp_loss += move.cp_loss
                                    cp_loss_count += 1
                                if move.move_quality == "blunder":
                                    blunders += 1
                                if move.move_quality == "best":
                                    best_moves += 1
                                if move.move_quality:
                                    total_moves += 1

            blunder_rate = (blunders / total_moves * 100) if total_moves > 0 else 0
            best_move_pct = (best_moves / total_moves * 100) if total_moves > 0 else 0
            avg_cp_loss = (total_cp_loss / cp_loss_count) if cp_loss_count > 0 else 0

            radar_rows.append({
                "Model": name,
                "Score %": score_pct,
                "Avg Latency (ms)": avg_latency,
                "Best Move %": best_move_pct,
                "Blunder Rate %": blunder_rate,
                "Avg CP Loss": avg_cp_loss,
            })

        radar_df = pd.DataFrame(radar_rows)

        # Normalize each metric to 0-1 for radar
        metrics = ["Score %", "Best Move %", "Avg CP Loss", "Blunder Rate %", "Avg Latency (ms)"]
        # For Score % and Best Move %: higher is better (normalize max->1)
        # For Avg CP Loss, Blunder Rate %, Avg Latency: lower is better (normalize min->1, invert)
        normalized = {}
        for m in metrics:
            vals = radar_df[m].values
            if m in ["Score %", "Best Move %"]:
                max_val = max(vals) if max(vals) > 0 else 1
                normalized[m] = vals / max_val
            else:
                min_val = min(vals) if min(vals) > 0 else 1
                max_val = max(vals)
                # Invert: lower is better, so 1 - (val-min)/(max-min)
                if max_val > min_val:
                    normalized[m] = 1 - (vals - min_val) / (max_val - min_val)
                else:
                    normalized[m] = np.ones_like(vals) * 0.5

        # Prepare data for Altair radar (polar coordinate approximation)
        radar_plot_data = []
        for i, row in radar_df.iterrows():
            for m in metrics:
                radar_plot_data.append({
                    "Model": row["Model"],
                    "Metric": m,
                    "Value": normalized[m][i],
                })
        radar_plot_df = pd.DataFrame(radar_plot_data)

        # Altair radar chart using theta/r
        radar_chart = alt.Chart(radar_plot_df).mark_line(point=True, strokeWidth=2).encode(
            theta=alt.Theta("Metric:N", sort=metrics),
            radius=alt.Radius("Value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=alt.Legend(title="Model")),
            tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".2f")]
        ).properties(
            width=400,
            height=400,
            title="Model Comparison Radar (normalized 0-1, higher=better)"
        ).configure_view(stroke=None)

        st.altair_chart(radar_chart, width="stretch")

        # Also show comparison table
        comparison_rows = []
        for name, stats in model_stats.items():
            score = stats["wins"] + 0.5 * stats["draws"]
            score_pct = (score / stats["games"] * 100) if stats["games"] > 0 else 0
            avg_latency = stats["total_latency"] / stats["latency_count"] if stats["latency_count"] > 0 else 0

            comparison_rows.append({
                "Model": name,
                "Games": stats["games"],
                "Wins": stats["wins"],
                "Losses": stats["losses"],
                "Draws": stats["draws"],
                "Score %": f"{score_pct:.1f}%",
                "Avg Latency": format_duration_ms(avg_latency) if avg_latency > 0 else "—",
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values("Games", ascending=False)
        st.dataframe(comparison_df, hide_index=True, width="stretch")

    elif model_stats:
        # Single model - show table only
        comparison_rows = []
        for name, stats in model_stats.items():
            score = stats["wins"] + 0.5 * stats["draws"]
            score_pct = (score / stats["games"] * 100) if stats["games"] > 0 else 0
            avg_latency = stats["total_latency"] / stats["latency_count"] if stats["latency_count"] > 0 else 0

            comparison_rows.append({
                "Model": name,
                "Games": stats["games"],
                "Wins": stats["wins"],
                "Losses": stats["losses"],
                "Draws": stats["draws"],
                "Score %": f"{score_pct:.1f}%",
                "Avg Latency": format_duration_ms(avg_latency) if avg_latency > 0 else "—",
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values("Games", ascending=False)
        st.dataframe(comparison_df, hide_index=True, width="stretch")
        st.caption("Radar chart requires ≥2 models. Run a benchmark with multiple models to see comparison.")
    else:
        st.info("No model comparison data available.")

    st.markdown("---")

    # ============================================================
    # 5. THINKING TRACE VIEWER
    # ============================================================
    st.markdown("### 🧠 Thinking Trace Viewer")

    thinking_data = []
    for ply, move_log in enumerate(selected_game.moves):
        if move_log.thinking_trace:
            thinking_data.append({
                "Ply": ply + 1,
                "Move": move_log.move_san,
                "Player": move_log.player,
                "Thinking": move_log.thinking_trace[:200] + "..." if len(move_log.thinking_trace) > 200 else move_log.thinking_trace,
                "Chars": move_log.thinking_chars,
                "Words": move_log.thinking_words,
                "Structured": move_log.thinking_has_structured,
                "Tactics": move_log.thinking_mentions_tactics,
                "Strategy": move_log.thinking_mentions_strategy,
                "Time Pressure": move_log.thinking_mentions_time_pressure,
            })

    if thinking_data:
        thinking_df = pd.DataFrame(thinking_data)
        st.dataframe(thinking_df, hide_index=True, width="stretch")

        # Expandable thinking trace per move
        for ply, move_log in enumerate(selected_game.moves):
            if move_log.thinking_trace:
                with st.expander(f"Ply {ply+1}: {move_log.move_san} ({move_log.player})"):
                    st.code(move_log.thinking_trace, language="text")
    else:
        st.info("No thinking traces available for this game.")

    st.markdown("---")

    # ============================================================
    # 6. REPLAY CONTROLS
    # ============================================================
    st.markdown("### ▶️ Replay Controls")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏮️ Start", width="stretch"):
            st.info("Use the demo game viewer for full replay controls.")
    with col2:
        _ = st.slider("Replay Speed", 0.1, 3.0, 1.0, 0.1, key="replay_speed")
    with col3:
        if st.button("⏭️ End", width="stretch"):
            st.info("Use the demo game viewer for full replay controls.")

    st.info("💡 For full replay controls (jump to move, flip board, etc.), use the Demo Games section.")

    st.markdown("---")

    # ============================================================
    # 7. EXPORT BUTTONS - Ghost CTA pills per DESIGN.md
    # ============================================================
    st.markdown("### 💾 Export Game Data")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Export PGN+Eval", width="stretch", key="export_pgn"):
                from chess_fight.benchmark.export import export_pgn_with_eval
                output_path = f"runs/{selected_game.run_id if hasattr(selected_game, 'run_id') else 'export'}/game_{selected_game.game_id}_eval.pgn"
                try:
                    path = export_pgn_with_eval(selected_game.run_dir if hasattr(selected_game, 'run_dir') else RUNS_ROOT, output_path)
                    st.success(f"Exported to {path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")

        with col2:
            if st.button("📊 Export CSV", width="stretch", key="export_csv"):
                from chess_fight.benchmark.export import export_csv
                try:
                    output_dir = f"runs/{selected_game.run_id if hasattr(selected_game, 'run_id') else 'export'}/export_csv"
                    path = export_csv(selected_game.run_dir if hasattr(selected_game, 'run_dir') else RUNS_ROOT, output_dir)
                    st.success(f"Exported to {path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")

        with col3:
            if st.button("📦 Export Parquet", width="stretch", key="export_parquet"):
                from chess_fight.benchmark.export import export_parquet
                try:
                    output_path = f"runs/{selected_game.run_id if hasattr(selected_game, 'run_id') else 'export'}/export.parquet"
                    path = export_parquet(selected_game.run_dir if hasattr(selected_game, 'run_dir') else RUNS_ROOT, output_path)
                    st.success(f"Exported to {path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")



# Add import for altair at the top of the file if not already there
import time
import chess
from chess_fight.models import GameMove, GameStats
from chess_fight.game.async_game import GameState

def rehydrate_session_state():
    import streamlit as st
    from chess_fight.benchmark.results_view import list_runs
    import os
    RUNS_ROOT = os.environ.get("CHESS_FIGHT_RUNS_ROOT", "runs")
    
    if "rehydrated" in st.session_state:
        return
        
    st.session_state.rehydrated = True
    
    if "benchmark_completed_games" in st.session_state and st.session_state.benchmark_completed_games:
        return
        
    runs = list_runs(RUNS_ROOT)
    if not runs:
        return
        
    latest_run = runs[0]
    # Check if the run is recent enough, e.g., within the last 12 hours
    if not latest_run.timestamp_utc:
        return
        
    # parse timestamp_utc (e.g. "2026-08-15T23:33:19Z")
    import datetime
    try:
        ts = datetime.datetime.fromisoformat(latest_run.timestamp_utc.replace("Z", "+00:00"))
    except Exception:
        pass
        
    if not latest_run.games:
        return
        
    rehydrated_games = []
    for game_rec in latest_run.games:
        stats = GameStats(
            total_moves=game_rec.total_moves,
            capture_moves=0,
            check_moves=0,
            game_duration=game_rec.game_duration_sec,
            winner=game_rec.winner_spec,
            termination_reason=game_rec.termination_reason
        )
        
        board = chess.Board(game_rec.opening_fen or chess.STARTING_FEN)
        game_moves = []
        for m in game_rec.moves:
            # We must detect captures and checks from the board state to populate GameMove accurately,
            # or we can try to guess from san if we want to be fast.
            # But the board state is better since we need to leave the board at the end position!
            is_capture = False
            captured_piece = None
            is_check = False
            is_checkmate = False
            is_promotion = False
            is_castling = False
            is_illegal = False
            
            try:
                move_obj = chess.Move.from_uci(m.move_uci)
                if move_obj in board.legal_moves:
                    is_capture = board.is_capture(move_obj)
                    if is_capture:
                        if board.is_en_passant(move_obj):
                            captured_piece = "p"
                        else:
                            p = board.piece_at(move_obj.to_square)
                            if p:
                                captured_piece = p.symbol().lower()
                    
                    is_check = board.gives_check(move_obj)
                    is_promotion = move_obj.promotion is not None
                    is_castling = board.is_castling(move_obj)
                    
                    board.push(move_obj)
                    is_checkmate = board.is_checkmate()
                    
                    if is_capture:
                        stats.capture_moves += 1
                    if is_check:
                        stats.check_moves += 1
                else:
                    is_illegal = True
            except Exception:
                is_illegal = True

            game_moves.append(GameMove(
                player=m.player,
                move=m.move_uci,
                move_san=m.move_san,
                timestamp=0.0,
                is_capture=is_capture,
                captured_piece=captured_piece,
                is_check=is_check,
                is_checkmate=is_checkmate,
                is_promotion=is_promotion,
                is_castling=is_castling,
                cp_score=m.eval_cp_score,
                mate_in=m.eval_mate_in,
                latency_ms=m.llm_latency_ms,
                tokens_in=m.llm_tokens_in,
                tokens_out=m.llm_tokens_out,
                reasoning=m.thinking_trace,
                is_illegal=is_illegal
            ))
            
        gs = GameState(
            board=board,
            moves=game_moves,
            stats=stats,
            current_player=game_rec.white_player if board.turn == chess.WHITE else game_rec.black_player,
            is_game_over=True,
            winner=game_rec.winner_spec,
            game_duration=game_rec.game_duration_sec,
            fen_before=game_rec.opening_fen or chess.STARTING_FEN
        )
        rehydrated_games.append(gs)
        
    st.session_state.benchmark_completed_games = rehydrated_games
    st.session_state.benchmark_done = True
    st.session_state.benchmark_run_dir = str(latest_run.run_dir)
