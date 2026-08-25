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
from pathlib import Path
from typing import Any

import chess
import chess.svg
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from chessbench.benchmark.results_view import (
    list_runs,
    load_run,
)
from chessbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chessbench.common.common_types import ChatMessage, is_chess_capable
from chessbench.common.exceptions import (
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
from chessbench.constants import HOSTED_PROVIDERS
from chessbench.game.async_game import GameState
from chessbench.models import GameMove, GameStats
from chessbench.prompts import validate_prompt_text
from chessbench.providers import get_provider, list_providers
from chessbench.providers.chess_ai import ProviderChessAI
from chessbench.ui.error_display import render_error
from chessbench.ui.helpers import (
    format_duration_ms,
    player_banner_html,
    render_board_with_evalbar,
    render_move_ticker_html,
    render_thinking_trace_drawer,
)
from chessbench.ui.landing import render_hero, render_landing_metrics
from chessbench.ui.prompt_workbench import can_launch_match, is_ab_eligible
from chessbench.ui.theme import apply_arena_theme

RUNS_ROOT = os.environ.get("CHESSBENCH_RUNS_ROOT") or os.environ.get(
    "CHESS_FIGHT_RUNS_ROOT", "runs"
)

# Configure page
st.set_page_config(
    page_title="ChessBench Arena",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _draw_board(
    board_placeholder, state: GameState, start_time: float | None = None
) -> None:
    king_square = state.board.king(state.board.turn)
    check_square = (
        king_square if state.board.is_check() and king_square is not None else None
    )
    board_placeholder.write(
        chess.svg.board(
            state.board,
            size=600,
            lastmove=state.board.peek() if state.board.move_stack else None,
            check=check_square,
        ),
        unsafe_allow_html=True,
    )


def _draw_metrics(
    stats_placeholder, state: GameState, start_time: float | None = None
) -> None:
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
            term_reason = getattr(state.stats, "termination_reason", "unknown")
            st.metric("Termination", term_reason.replace("_", " ").title())


def _draw_moves(moves_placeholder, moves: list) -> None:
    if not moves:
        return
    df = pd.DataFrame(
        [
            {
                "Move #": i + 1,
                "Player": move.player,
                "Move": move.move,
                "Capture": 1 if move.is_capture else 0,
                "Check": 1 if move.is_check else 0,
                "Reasoning": (
                    move.reasoning.replace("<", "&lt;").replace(">", "&gt;")
                    if move.reasoning
                    else ""
                ),
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

    moves_placeholder.dataframe(
        df, hide_index=True, width="stretch", column_config=column_config
    )


def _draw_completion_result(expander_placeholder, state: GameState) -> None:
    """Render the last LLM completion (tokens, latency, raw text) honestly."""
    cr = state.last_completion_result
    if cr is None:
        return
    title = (
        f"Last LLM completion ({state.current_player})"
        if state.current_player
        else "Last LLM completion"
    )
    with expander_placeholder.expander(title, expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Latency",
                format_duration_ms(cr.latency_ms) if cr.latency_ms is not None else "—",
            )
        with col_b:
            st.metric("Tokens in", cr.tokens_in if cr.tokens_in is not None else "—")
        with col_c:
            st.metric("Tokens out", cr.tokens_out if cr.tokens_out is not None else "—")
        if cr.error:
            st.error(
                f"**{cr.error_type or 'Error'}** (Retry #{cr.retry_count}): {cr.error}"
            )
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
        check_square = (
            king_square if board.is_check() and king_square is not None else None
        )
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


def _safe_async_run(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_list_models(provider_name: str, api_key: str) -> list[Any]:
    """Cache model list responses to accelerate Streamlit Cloud reruns."""
    provider = get_provider(provider_name)
    if not provider:
        return []
    try:
        return _safe_async_run(provider.list_models(api_key))  # type: ignore[no-any-return]
    except Exception:
        return []


async def fetch_models_for_provider(provider_name: str, api_key: str) -> list:
    """Fetch available models for a provider with caching."""
    provider = get_provider(provider_name)
    if not provider:
        return []
    try:
        models = _cached_list_models(provider_name, api_key)
        if models:
            return models
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


def _get_secret_or_env(provider_name: str) -> str | None:
    """Lookup API key from environment variable or Streamlit secrets."""
    env_key = os.getenv(f"{provider_name.upper()}_API_KEY")
    if env_key:
        return env_key
    try:
        if hasattr(st, "secrets"):
            return st.secrets.get(f"{provider_name.lower()}_api_key") or st.secrets.get(
                f"{provider_name.upper()}_API_KEY"
            )
    except Exception:
        pass
    return None


def _test_model_async(model_info: dict):
    """Test a model with a 1-token query to verify connectivity and measure latency."""
    import time

    provider_name = model_info.get("provider", "")
    model_id = model_info.get("model_id", "")
    api_key = model_info.get("api_key", "")

    provider = get_provider(provider_name)
    if not provider:
        st.sidebar.error(f"Provider {provider_name} not available.")
        return

    start_t = time.perf_counter()
    try:
        with st.spinner(f"Testing {model_id}..."):
            _ = asyncio.run(
                provider.complete(
                    api_key=api_key,
                    model=model_id,
                    messages=[
                        ChatMessage(role="system", content="Respond with 'OK'."),
                        ChatMessage(role="user", content="Test"),
                    ],
                    temperature=0.0,
                    max_tokens=5,
                )
            )
        elapsed_ms = int((time.perf_counter() - start_t) * 1000)
        if hasattr(st, "toast"):
            st.toast(f"✅ {model_id} connected in {elapsed_ms}ms!", icon="⚡")
        else:
            st.sidebar.success(f"✅ {model_id} connected in {elapsed_ms}ms!")
    except Exception as exc:
        st.sidebar.error(f"❌ {model_id} test failed: {exc}")


def render_sidebar_header_and_nav():
    """Render top brand header and segmented surface switcher in sidebar."""
    st.sidebar.markdown(
        '<div class="sb-brand-container">'
        '  <div class="sb-brand-title">♟️ CHESSBENCH</div>'
        '  <div class="sb-brand-tag">v1.0</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_provider_keys_section():
    """Render provider status badges in sidebar based on Streamlit Secrets / Env Vars."""
    providers = list_providers()
    if "nim" in providers:
        providers.remove("nim")
        providers.insert(0, "nim")
    available_providers = []
    status_list = []

    for provider_name in providers:
        if HOSTED_PROVIDERS and provider_name not in HOSTED_PROVIDERS:
            continue
        provider = get_provider(provider_name)
        if provider is None:
            continue

        if not provider.requires_api_key:
            reachable, _ = _probe_local_provider(provider_name)
            if reachable:
                available_providers.append((provider_name, ""))
                status_list.append((provider_name, True, "local active"))
            else:
                status_list.append((provider_name, False, "local offline"))
            continue

        secret_key = st.session_state.get(
            f"user_key_{provider_name}"
        ) or _get_secret_or_env(provider_name)
        env_var_name = f"{provider_name.upper()}_API_KEY"
        if secret_key and provider.validate_key(secret_key):
            available_providers.append((provider_name, secret_key))
            status_list.append((provider_name, True, "Active"))
        else:
            status_list.append((provider_name, False, f"Needs {env_var_name}"))

    active_count = len(available_providers)
    total_count = len(status_list)

    expander_title = f"🔑 Provider Keys ({active_count}/{total_count} Active)"
    with st.sidebar.expander(expander_title, expanded=False):
        st.caption(
            "API keys are loaded automatically from Streamlit Secrets or Environment Variables."
        )

        grid_html = '<div class="sb-provider-grid">'
        for pname, is_active, label in status_list:
            if is_active:
                grid_html += f'<div class="sb-provider-badge-active"><span>✓ {pname.capitalize()}</span><span style="font-size:0.64rem; opacity:0.85;">{label}</span></div>'
            else:
                grid_html += f'<div class="sb-provider-badge-inactive"><span>🔒 {pname.capitalize()}</span><span style="font-size:0.64rem; opacity:0.65;">{label}</span></div>'
        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

        inactive_keyed = [
            pname for pname, is_active, label in status_list
            if not is_active and label.startswith("Needs ")
        ]
        if inactive_keyed:
            st.caption(
                "Paste your own key to unlock a provider for this session. "
                "Keys are never stored or sent anywhere except the provider's API."
            )
            for pname in inactive_keyed:
                env_var_name = f"{pname.upper()}_API_KEY"
                st.text_input(
                    env_var_name,
                    type="password",
                    key=f"user_key_{pname}",
                    placeholder=f"Paste your {pname.capitalize()} key…",
                    on_change=lambda pn=pname: st.session_state.update({f"_key_validated_{pn}": False}) or st.rerun(),
                )

        with st.popover("📋 Streamlit Secrets Template"):
            st.markdown(
                "Copy the template below into **App Settings → Secrets** on Streamlit Cloud, "
                "or into `.streamlit/secrets.toml` locally:"
            )
            st.code(
                "# Streamlit Secrets Template\n"
                'OPENAI_API_KEY = "your_openai_key_here"\n'
                'ANTHROPIC_API_KEY = "your_anthropic_key_here"\n'
                'GOOGLE_API_KEY = "your_google_key_here"\n'
                'GROQ_API_KEY = "your_groq_key_here"\n'
                'NIM_API_KEY = "your_nvidia_nim_key_here"\n'
                'OPENROUTER_API_KEY = "your_openrouter_key_here"\n'
                'TOGETHER_API_KEY = "your_together_key_here"\n'
                'FIREWORKS_API_KEY = "your_fireworks_key_here"\n'
                'DEEPINFRA_API_KEY = "your_deepinfra_key_here"\n',
                language="toml",
            )

    return available_providers


def render_model_selectors(available_providers: list):
    """Render model selection for White and Black players."""
    st.sidebar.markdown(
        '<div style="font-weight:650; font-size:0.95rem; margin-bottom:8px; color:var(--arena-text);">⚔️ Player Matchup</div>',
        unsafe_allow_html=True,
    )

    all_models: dict[str, dict] = {}
    filtered_count = 0

    for provider_name, api_key in available_providers:
        with st.spinner(f"Fetching models from {provider_name}..."):
            models = asyncio.run(fetch_models_for_provider(provider_name, api_key))

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
        st.sidebar.caption(f"ⓘ {filtered_count} non-chat model(s) hidden")

    if not all_models:
        st.sidebar.warning("Couldn't fetch models")
        return None, None

    if len(all_models) < 2:
        st.sidebar.warning(
            "At least 2 distinct chess-capable models are required to start a match. Please configure another provider in Streamlit Secrets."
        )
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

    # Player 1 Card
    st.sidebar.markdown(
        '<div class="sb-player-card sb-player-card-white">'
        '  <div class="sb-player-header"><span>Player 1</span><span class="sb-player-badge-white">P1</span></div>',
        unsafe_allow_html=True,
    )
    col_m1, col_t1 = st.sidebar.columns([4, 1])
    model_1 = col_m1.selectbox(
        "Select First Model",
        options=m1_options,
        index=m1_idx,
        key="player_model_1",
        label_visibility="collapsed",
    )
    if col_t1.button(
        "🔬", key="test_model_1", help="Test model latency & connectivity"
    ):
        _test_model_async(all_models[model_1])

    m1_info = all_models[model_1]
    ctx1 = (
        f"{m1_info['context_window']//1024}k ctx"
        if m1_info.get("context_window")
        else "standard"
    )
    st.sidebar.markdown(
        f'<div class="sb-model-meta"><span class="sb-tag">{m1_info["provider"]}</span><span class="sb-tag">{ctx1}</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    def _swap_player_models():
        p1 = st.session_state.get("player_model_1")
        p2 = st.session_state.get("player_model_2")
        if p1 and p2:
            st.session_state["player_model_1"] = p2
            st.session_state["player_model_2"] = p1

    # Swap Button
    _, col_swap, _ = st.sidebar.columns([1, 3, 1])
    col_swap.button(
        "⇄ Swap White & Black",
        key="btn_swap_players",
        help="Swap White and Black model assignments",
        width="stretch",
        on_click=_swap_player_models,
    )

    # Player 2 Card
    m2_options = [m for m in model_options if m != model_1]
    cur_2_val = st.session_state.get("player_model_2")
    m2_idx = m2_options.index(cur_2_val) if cur_2_val in m2_options else 0

    st.sidebar.markdown(
        '<div class="sb-player-card sb-player-card-black">'
        '  <div class="sb-player-header"><span>Player 2</span><span class="sb-player-badge-black">P2</span></div>',
        unsafe_allow_html=True,
    )
    col_m2, col_t2 = st.sidebar.columns([4, 1])
    model_2 = col_m2.selectbox(
        "Select Second Model",
        options=m2_options,
        index=m2_idx,
        key="player_model_2",
        label_visibility="collapsed",
    )
    if col_t2.button(
        "🔬", key="test_model_2", help="Test model latency & connectivity"
    ):
        _test_model_async(all_models[model_2])

    m2_info = all_models[model_2]
    ctx2 = (
        f"{m2_info['context_window']//1024}k ctx"
        if m2_info.get("context_window")
        else "standard"
    )
    st.sidebar.markdown(
        f'<div class="sb-model-meta"><span class="sb-tag">{m2_info["provider"]}</span><span class="sb-tag">{ctx2}</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if model_1 and model_2 and model_1 != model_2:
        return all_models[model_1], all_models[model_2]
    return None, None


def render_prompt_management(model_1_config, model_2_config):
    """Delegate to workbench_ui module."""
    from chessbench.ui.workbench_ui import (
        render_prompt_management as workbench_render_prompt_management,
    )

    return workbench_render_prompt_management(model_1_config, model_2_config)


def create_provider_ai(white_config: dict, black_config: dict):
    """Create ProviderChessAI instances for both players."""
    params: dict = {"temperature": 0.1, "max_tokens": 1500}
    white_ai = ProviderChessAI(
        provider_name=white_config["provider"],
        model_id=white_config["model_id"],
        api_key=white_config["api_key"],
        **params,
    )
    black_params: dict = dict(params)
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
    # Handle paused state
    if is_paused and state is not None:
        st.warning(f"⏸ Paused — {pause_reason or 'Unknown reason'}")
        if getattr(state, "pause_error", None):
            st.error(f"**Error:** {state.pause_error}")
        if getattr(state, "paused_player", None):
            turn_info = getattr(state, "paused_turn", 0) + 1
            st.info(f"**Failed Player:** {state.paused_player} (Turn {turn_info})")

        is_benchmark_pause = pause_reason == "game_failed"
        if is_benchmark_pause:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "▶️ Continue to next game",
                    type="primary",
                    width="stretch",
                    key="paused_continue",
                ):
                    runner = st.session_state.get("benchmark_runner")
                    if runner is not None and hasattr(
                        runner, "request_continue_after_problem"
                    ):
                        runner.request_continue_after_problem()
                    st.rerun()
            with col2:
                if st.button("⛔ Abort benchmark", width="stretch", key="paused_abort"):
                    runner = st.session_state.get("benchmark_runner")
                    if runner is not None and hasattr(
                        runner, "request_abort_after_problem"
                    ):
                        runner.request_abort_after_problem()
                    st.rerun()
        else:
            can_rewind = len(state.moves) > 1 or (
                len(state.moves) == 1
                and not getattr(state.moves[0], "is_illegal", False)
            )
            if can_rewind:
                col_rewind, col1, col2, col3 = st.columns(4)
                with col_rewind:
                    if st.button(
                        "⏪ Rewind Turn", width="stretch", key="paused_rewind"
                    ):
                        runner = st.session_state.get("benchmark_runner")
                        if runner:
                            runner.resume_game(retry=False, rewind=True)
                        st.rerun()
            else:
                col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "↻ Retry Turn", type="primary", width="stretch", key="paused_retry"
                ):
                    runner = st.session_state.get("benchmark_runner")
                    if runner:
                        runner.resume_game(retry=True)
                    st.rerun()
            with col2:
                if st.button(
                    "⏭ Skip Turn (Force Move)", width="stretch", key="paused_skip"
                ):
                    runner = st.session_state.get("benchmark_runner")
                    if runner:
                        runner.resume_game(retry=False, force_move=True)
                    st.rerun()
            with col3:
                if st.button("⛔ Cancel Game", width="stretch", key="paused_cancel"):
                    runner = st.session_state.get("benchmark_runner")
                    if (
                        runner
                        and hasattr(runner, "current_game")
                        and runner.current_game
                    ):
                        runner.current_game.cancel()
                    st.rerun()

    # Inject move ticker auto-scroll JS (runs on each render, scrolls latest pill into view)
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    # Two-column arena frame: left=board, right=panels
    if state is not None:
        left, right = st.columns([0.55, 0.45], gap="large")

        with left:
            # Board + eval bar (board centered, fixed width via CSS)
            king = state.board.king(state.board.turn)
            check_sq = king if state.board.is_check() and king is not None else None
            last_mv = state.board.peek() if state.board.move_stack else None

            # If there's a last move, use its eval. Otherwise default to 0.
            cp = (
                state.moves[-1].cp_score
                if state.moves and state.moves[-1].cp_score is not None
                else 0
            )
            mate = state.moves[-1].mate_in if state.moves else None

            render_board_with_evalbar(
                state.board,
                size=560,  # Fixed desktop width per DESIGN.md
                lastmove=last_mv,
                check_square=check_sq,
                cp_score=cp,
                mate_in=mate,
            )

        with right:
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

            # Metrics card
            with st.container(border=True):
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Game", f"{game_idx + 1} / {total_games}")
                with cols[1]:
                    st.metric("Total Moves", state.stats.total_moves)
                with cols[2]:
                    st.metric("Captures", state.stats.capture_moves)
                with cols[3]:
                    st.metric("Checks", state.stats.check_moves)

                cols2 = st.columns(3)
                with cols2[0]:
                    elapsed = (
                        int(state.game_duration)
                        if state.game_duration > 0
                        else int(time.time() - start_time)
                    )
                    st.metric("Time", format_duration_ms(elapsed * 1000))
                with cols2[1]:
                    turn_color = "White ♔" if state.board.turn else "Black ♚"
                    st.metric("Turn", turn_color if not state.is_game_over else "—")
                with cols2[2]:
                    if state.is_game_over:
                        term = getattr(state.stats, "termination_reason", "unknown")
                        st.metric("Result", term.replace("_", " ").title())

            # Last completion drawer
            cr = state.last_completion_result
            if cr:
                with st.expander(
                    f"Last completion ({state.current_player})", expanded=False
                ):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(
                            "Latency",
                            format_duration_ms(cr.latency_ms) if cr.latency_ms else "—",
                        )
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

        # Render the move history dataframe full-width below the board and panels
        if state.moves:
            with st.expander("Move History Data", expanded=False):
                df_rows = []
                ply_idx = 0
                for m in state.moves:
                    is_white_turn = ply_idx % 2 == 0
                    piece_map = {"p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛"}
                    cap_suffix = (
                        f" ({piece_map.get(m.captured_piece, m.captured_piece)})"
                        if m.captured_piece
                        else ""
                    )
                    is_illegal = getattr(m, "is_illegal", False)
                    df_rows.append(
                        {
                            "Turn": ply_idx + 1,
                            "Color": "White" if is_white_turn else "Black",
                            "Player": m.player.split(":", 1)[-1],
                            "Move": f"{m.move_san or m.move}{cap_suffix}",
                            "Capture": 1 if m.is_capture else 0,
                            "Check": 1 if m.is_check else 0,
                            "Checkmate": 1 if m.is_checkmate else 0,
                            "Illegal": 1 if is_illegal else 0,
                            "Eval": (
                                f"M{m.mate_in}"
                                if m.mate_in
                                else (
                                    f"{m.cp_score/100:+.2f}"
                                    if m.cp_score is not None
                                    else ""
                                )
                            ),
                            "Latency": format_duration_ms(m.latency_ms),
                            "Tokens": (
                                f"{m.tokens_in or 0} in / {m.tokens_out or 0} out"
                                if m.tokens_in or m.tokens_out
                                else ""
                            ),
                            "Reasoning": (
                                m.reasoning.replace("<", "&lt;") if m.reasoning else ""
                            ),
                        }
                    )
                    if not is_illegal:
                        ply_idx += 1
                df = pd.DataFrame(df_rows)
                column_config = {
                    "Capture": st.column_config.NumberColumn(
                        "Capture", width="small", format="%d"
                    ),
                    "Check": st.column_config.NumberColumn(
                        "Check", width="small", format="%d"
                    ),
                    "Checkmate": st.column_config.NumberColumn(
                        "Checkmate", width="small", format="%d"
                    ),
                    "Illegal": st.column_config.NumberColumn(
                        "Illegal", width="small", format="%d"
                    ),
                    "Reasoning": st.column_config.TextColumn(
                        "Reasoning",
                        help="Click a cell to read the LLM's full reasoning for this move.",
                        width="large",
                    ),
                }
                st.dataframe(
                    df, hide_index=True, width="stretch", column_config=column_config
                )

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
    term_reason = getattr(state.stats, "termination_reason", "unknown")
    winner = state.winner or "?"

    with st.container(border=True):

        # Header row
        hdr_cols = st.columns([4, 1, 1])
        with hdr_cols[0]:
            st.markdown(
                f"**Game {game_idx + 1}** · ♔ White: **{white_player}** vs ♚ Black: **{black_player}**"
            )
        with hdr_cols[1]:
            st.metric("Result", winner)
        with hdr_cols[2]:
            st.metric("Termination", term_reason.replace("_", " ").title())

        # Board + metrics side by side
        bcol, mcol = st.columns([1, 1])
        with bcol:
            king = state.board.king(state.board.turn)
            check_sq = king if state.board.is_check() and king is not None else None
            last_mv = state.board.peek() if state.board.move_stack else None
            render_board_with_evalbar(
                state.board, size=320, lastmove=last_mv, check_square=check_sq
            )

        with mcol:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Moves", state.stats.total_moves)
            m2.metric("Captures", state.stats.capture_moves)
            m3.metric("Checks", state.stats.check_moves)

        # Move history as pills
        if state.moves:
            st.markdown(render_move_ticker_html(state.moves), unsafe_allow_html=True)
            with st.expander("Move History Data", expanded=False):
                df_rows = []
                ply_idx = 0
                for m in state.moves:
                    is_white_turn = ply_idx % 2 == 0
                    piece_map = {"p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛"}
                    cap_suffix = (
                        f" ({piece_map.get(m.captured_piece, m.captured_piece)})"
                        if m.captured_piece
                        else ""
                    )
                    is_illegal = getattr(m, "is_illegal", False)
                    df_rows.append(
                        {
                            "Turn": ply_idx + 1,
                            "Color": "White" if is_white_turn else "Black",
                            "Player": m.player.split(":", 1)[-1],
                            "Move": (
                                f"{m.move_san or m.move}{cap_suffix}"
                                if not is_illegal
                                else f"Illegal: {m.move}"
                            ),
                            "Capture": 1 if m.is_capture else 0,
                            "Check": 1 if m.is_check else 0,
                            "Checkmate": 1 if m.is_checkmate else 0,
                            "Illegal": 1 if is_illegal else 0,
                            "Eval": (
                                f"M{m.mate_in}"
                                if m.mate_in
                                else (
                                    f"{m.cp_score/100:+.2f}"
                                    if m.cp_score is not None
                                    else ""
                                )
                            ),
                            "Latency": format_duration_ms(m.latency_ms),
                            "Tokens": (
                                f"{m.tokens_in or 0} in / {m.tokens_out or 0} out"
                                if m.tokens_in or m.tokens_out
                                else ""
                            ),
                            "Reasoning": (
                                m.reasoning.replace("<", "&lt;").replace(">", "&gt;")
                                if m.reasoning
                                else ""
                            ),
                        }
                    )
                    if getattr(m, "is_illegal", False) is False:
                        ply_idx += 1
                df = pd.DataFrame(df_rows)
                column_config = {
                    "Capture": st.column_config.NumberColumn(
                        "Capture", width="small", format="%d"
                    ),
                    "Check": st.column_config.NumberColumn(
                        "Check", width="small", format="%d"
                    ),
                    "Checkmate": st.column_config.NumberColumn(
                        "Checkmate", width="small", format="%d"
                    ),
                    "Illegal": st.column_config.NumberColumn(
                        "Illegal", width="small", format="%d"
                    ),
                    "Reasoning": st.column_config.TextColumn(
                        "Reasoning",
                        help="Click a cell to read the LLM's full reasoning for this move.",
                        width="large",
                    ),
                }
                st.dataframe(
                    df, hide_index=True, width="stretch", column_config=column_config
                )


def run_in_process_benchmark(
    white_config: dict,
    black_config: dict,
    games: int = 3,
    colors: str = "alternating",
    reasoning_level: str = "mid",
    system_prompts: dict[str, str] | None = None,
    turn_prompts: dict[str, str] | None = None,
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
        system_prompt=system_prompts or {},
        turn_prompt=turn_prompts or {},
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
                [data-testid="stSidebar"] { display: none !important; width: 0 !important; }
                [data-testid="collapsedControl"] { display: none !important; }
                section.main > div.block-container { max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if (
        "benchmark_runner" not in st.session_state
        or st.session_state.benchmark_runner is None
    ):
        try:
            st.session_state.benchmark_runner = BenchmarkRunner(config)
        except (NoProvidersConfiguredError, SetupError, InvalidApiKeyError) as exc:
            render_error(st, exc)
            if st.button(
                "🔙 Return to Main Menu",
                type="primary",
                width="stretch",
                key="err_setup_return",
            ):
                _reset_game_state()
                st.rerun()
            return
        except ValueError as exc:
            st.error(f"Benchmark setup failed: {exc}")
            if st.button(
                "🔙 Return to Main Menu",
                type="primary",
                width="stretch",
                key="err_val_return",
            ):
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
                    # Clear the live state so it doesn't repeat in the UI
                    st.session_state.benchmark_state = None
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

    # Now start the benchmark if not already running
    if (
        "benchmark_thread" not in st.session_state
        or not st.session_state.benchmark_thread.is_alive()
    ) and not st.session_state.get("benchmark_done", False):
        start_benchmark()

    def _draw_live_ui():
        state: GameState | None = st.session_state.get("benchmark_state")
        game_idx = st.session_state.benchmark_game_index
        is_paused = getattr(state, "is_paused", False)

        if is_paused:
            if "benchmark_pause_time" not in st.session_state:
                st.session_state.benchmark_pause_time = time.time()
            # Slide start_time forward so time.time() - start_time remains constant during pause
            pause_dur = time.time() - st.session_state.benchmark_pause_time
            start_time = st.session_state.benchmark_start_time + pause_dur
        else:
            if "benchmark_pause_time" in st.session_state:
                # Commit the pause duration to start_time
                st.session_state.benchmark_start_time += (
                    time.time() - st.session_state.benchmark_pause_time
                )
                del st.session_state["benchmark_pause_time"]
            start_time = st.session_state.benchmark_start_time

        completed_games = st.session_state.get("benchmark_completed_games", [])
        pause_reason = getattr(state, "pause_reason", None) if is_paused else None

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
        error = st.session_state.benchmark_error
        if isinstance(
            error,
            (
                GameExecutionError,
                FatalBenchmarkError,
                NoProvidersConfiguredError,
                SetupError,
                InvalidApiKeyError,
            ),
        ):
            render_error(st, error)
        else:
            st.error(f"Benchmark failed: {error}")
        if st.button(
            "🔙 Return to Main Menu",
            type="primary",
            width="stretch",
            key="err_bm_return",
        ):
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

    if st.button(
        "🔙 Return to Main Menu", type="primary", width="stretch", key="done_bm_return"
    ):
        _reset_game_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Benchmark history
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def get_cached_runs(runs_root: str):
    return list_runs(runs_root)


def flush_benchmark_history(runs_root: str):
    import os
    import shutil

    if os.path.exists(runs_root):
        shutil.rmtree(runs_root)
    get_cached_runs.clear()


def render_benchmark_history(*, expanded: bool = False) -> None:
    """Render benchmark runs parsed from the on-disk JSONL artifacts.

    `expanded=True` opens the expander so the sidebar's "📊 Benchmark History"
    toggle reveals the runs immediately on first click.
    """
    with st.expander("📊 Benchmark History", expanded=expanded):
        render_landing_metrics(RUNS_ROOT)

        _col1, col2 = st.columns([4, 1])
        with col2:
            if st.button(
                "🗑️ Flush History",
                key="flush_history",
                type="secondary",
                width="stretch",
            ):
                flush_benchmark_history(RUNS_ROOT)
                st.rerun()

        runs = get_cached_runs(RUNS_ROOT)
        if not runs:
            st.info(
                "No benchmark runs found under "
                f"`{RUNS_ROOT}`. Select two models in the sidebar and click "
                "**▶️ Start Match** to generate a real run."
            )
            return

        st.markdown(f"**{len(runs)} real benchmark(s)** loaded from `{RUNS_ROOT}`:")
        # Per-run details
        st.markdown("### Previous Game Tables")
        for run in runs:
            render_run_summary(run, expanded=False)


def render_run_summary(run, *, expanded: bool) -> None:
    """Render one benchmark run as a compact data block."""
    # Render as cf-card instead of bare expander
    with st.container(border=True):

        # Run header with analyze button
        hdr_col1, hdr_col2 = st.columns([4, 1])
        with hdr_col1:
            st.markdown(f"### {run.run_id}")
            st.caption(
                f"{run.total_games} games · {len(run.providers_seen)} provider(s)"
            )
        with hdr_col2:
            if st.button(
                "📊 Analyze",
                key=f"analyze_run_{run.run_id}",
                type="primary",
                width="stretch",
            ):
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
                        "Tokens in": (
                            ps.tokens_in_total
                            if ps.tokens_in_total is not None
                            else None
                        ),
                        "Tokens out": (
                            ps.tokens_out_total
                            if ps.tokens_out_total is not None
                            else None
                        ),
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
        term_reason = getattr(game, "termination_reason", "unknown")
        st.markdown(
            f"#### Game {i+1}: ♔ White: **{game.white_player}** vs ♚ Black: **{game.black_player}** ({game.result}) — *{term_reason.replace('_', ' ').title()}*"
        )

        if not game.moves:
            st.info("No moves recorded for this game.")
            st.divider()
            continue

        # Select Move
        max_moves = len(game.moves)

        # Calculate step state
        move_idx = st.slider(
            f"Rewind Game {i+1}",
            0,
            max_moves,
            max_moves,
            key=f"slider_{run.run_id}_{game.game_id}",
        )

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
            check_square = (
                king_square if b.is_check() and king_square is not None else None
            )

            # Use cf-board-frame via render_board_with_evalbar
            render_board_with_evalbar(
                b, size=400, lastmove=last_move, check_square=check_square
            )

        with col2:
            if move_info:
                player_name = (
                    game.white_player
                    if move_info.color == "white"
                    else game.black_player
                )
                st.markdown(
                    f"**Move {move_info.move_number}** - {player_name} ({move_info.color.title()}) played `{move_info.move_san}`"
                )
                st.metric(
                    "Latency",
                    (
                        format_duration_ms(move_info.llm_latency_ms)
                        if move_info.llm_latency_ms
                        else "—"
                    ),
                )
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
            if t_str and len(t_str) > 19:
                t_str = t_str[11:19]

            san = m.move_san or ""
            df_rows.append(
                {
                    "Move #": m.move_number,
                    "Color": m.color.title(),
                    "Player": (
                        game.white_player if m.color == "white" else game.black_player
                    ),
                    "Move": san,
                    "Capture": 1 if "x" in san else 0,
                    "Check": 1 if "+" in san else 0,
                    "Checkmate": 1 if "#" in san else 0,
                    "Reasoning": (
                        m.thinking_trace.replace("<", "&lt;").replace(">", "&gt;")
                        if m.thinking_trace
                        else ""
                    ),
                }
            )

        df = pd.DataFrame(df_rows)

        column_config = {
            "Capture": st.column_config.NumberColumn(
                "Capture", width="small", format="%d"
            ),
            "Check": st.column_config.NumberColumn("Check", width="small", format="%d"),
            "Checkmate": st.column_config.NumberColumn(
                "Checkmate", width="small", format="%d"
            ),
            "Reasoning": st.column_config.TextColumn(
                "Reasoning",
                help="Click a cell to read the LLM's full reasoning for this move.",
                width="large",
            ),
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

    # Sidebar Header & Brand
    render_sidebar_header_and_nav()

    # Sidebar: Model Selection
    available_providers = render_provider_keys_section()
    model_1_config, model_2_config = render_model_selectors(available_providers)

    # Prompt Strategy Workbench
    system_prompts_by_spec, turn_prompts_by_spec, _prompts_by_color = (
        render_prompt_management(model_1_config, model_2_config)
    )

    # Game Controls Section
    st.sidebar.markdown(
        '<div style="font-weight:650; font-size:0.95rem; margin-top:12px; margin-bottom:8px; color:var(--arena-text);">🎮 Match Settings</div>',
        unsafe_allow_html=True,
    )

    games = st.sidebar.number_input(
        "Games to play", min_value=1, max_value=20, value=1, step=1, key="game_count"
    )
    # Reasoning level is selected once in the strategy workbench
    # (key="reasoning_level_selector"); read it here to avoid a duplicate key.
    reasoning_level = st.session_state.get("reasoning_level_selector", "high")

    colors_mode = "alternating" if games > 1 else "fixed"

    # Match Preview Summary Box
    if model_1_config and model_2_config:
        m1_name = model_1_config["model_id"].split("/")[-1]
        m2_name = model_2_config["model_id"].split("/")[-1]
        st.sidebar.markdown(
            '<div class="sb-match-summary">'
            '  <div class="sb-match-vs">'
            f'    <div class="sb-match-vs-model" title="{m1_name}">{m1_name}</div>'
            '    <div class="sb-match-vs-divider">VS</div>'
            f'    <div class="sb-match-vs-model" title="{m2_name}" style="text-align:right;">{m2_name}</div>'
            "  </div>"
            '  <div class="sb-match-details">'
            f'    <span>{games} Game{"s" if games > 1 else ""} ({colors_mode})</span>'
            f"    <span>Reasoning: <strong>{reasoning_level.upper()}</strong></span>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Same-model A/B hint
    same_model_hint = ""
    if model_1_config and model_2_config:
        if (
            model_1_config["provider"] == model_2_config["provider"]
            and model_1_config["model_id"] == model_2_config["model_id"]
        ):
            # Same model, check if A/B eligible
            sys_1 = st.session_state.get(
                f"sys_prompt_{model_1_config['provider']}:{model_1_config['model_id']}",
                "",
            )
            turn_1 = st.session_state.get(
                f"turn_prompt_{model_1_config['provider']}:{model_1_config['model_id']}",
                "",
            )
            sys_2 = st.session_state.get(
                f"sys_prompt_{model_2_config['provider']}:{model_2_config['model_id']}",
                "",
            )
            turn_2 = st.session_state.get(
                f"turn_prompt_{model_2_config['provider']}:{model_2_config['model_id']}",
                "",
            )
            v1 = validate_prompt_text(sys_1, turn_1)
            v2 = validate_prompt_text(sys_2, turn_2)
            if is_ab_eligible(
                model_1_config, model_2_config, sys_1, turn_1, sys_2, turn_2
            ):
                same_model_hint = (
                    "💡 A/B mode enabled: same model with different prompts"
                )
            else:
                same_model_hint = (
                    "⚠️ Same model and same strategy — prompts must differ for A/B mode"
                )
        else:
            same_model_hint = ""
    if same_model_hint:
        st.sidebar.caption(same_model_hint)

    if st.sidebar.button("⚔️ Launch AI Arena Match", type="primary", width="stretch"):
        if not model_1_config or not model_2_config:
            st.sidebar.error("Please select two distinct models for the players.")
        elif (
            model_1_config["provider"] == model_2_config["provider"]
            and model_1_config["model_id"] == model_2_config["model_id"]
        ):
            # Same model: check A/B eligibility
            sys_1 = st.session_state.get(
                f"sys_prompt_{model_1_config['provider']}:{model_1_config['model_id']}",
                "",
            )
            turn_1 = st.session_state.get(
                f"turn_prompt_{model_1_config['provider']}:{model_1_config['model_id']}",
                "",
            )
            sys_2 = st.session_state.get(
                f"sys_prompt_{model_2_config['provider']}:{model_2_config['model_id']}",
                "",
            )
            turn_2 = st.session_state.get(
                f"turn_prompt_{model_2_config['provider']}:{model_2_config['model_id']}",
                "",
            )
            v1 = validate_prompt_text(sys_1, turn_1)
            v2 = validate_prompt_text(sys_2, turn_2)
            if not is_ab_eligible(
                model_1_config, model_2_config, sys_1, turn_1, sys_2, turn_2
            ):
                st.sidebar.error(
                    "Model 1 and Model 2 are the same and have identical prompts. Please differentiate prompts for A/B mode."
                )
                return
        # Launch hard-gate: validate both players' prompts
        sys_1 = st.session_state.get(
            f"sys_prompt_{model_1_config['provider']}:{model_1_config['model_id']}", ""
        )
        turn_1 = st.session_state.get(
            f"turn_prompt_{model_1_config['provider']}:{model_1_config['model_id']}", ""
        )
        sys_2 = st.session_state.get(
            f"sys_prompt_{model_2_config['provider']}:{model_2_config['model_id']}", ""
        )
        turn_2 = st.session_state.get(
            f"turn_prompt_{model_2_config['provider']}:{model_2_config['model_id']}", ""
        )
        v1 = validate_prompt_text(sys_1, turn_1)
        v2 = validate_prompt_text(sys_2, turn_2)
        can_launch, error_msg = can_launch_match(v1, v2)
        if not can_launch:
            st.sidebar.error(error_msg)
            return

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
            "system_prompts": system_prompts_by_spec,
            "turn_prompts": turn_prompts_by_spec,
        }
        st.rerun()

    if st.session_state.get("game_running", False) and st.session_state.get(
        "active_match_config"
    ):
        config = st.session_state.active_match_config
        run_in_process_benchmark(
            white_config=config["white_config"],
            black_config=config["black_config"],
            games=config["games"],
            colors=config["colors"],
            reasoning_level=config["reasoning_level"],
            system_prompts=config.get("system_prompts"),
            turn_prompts=config.get("turn_prompts"),
        )
        return

    if st.session_state.get("show_analytics", False):
        render_analytical_dashboard()
        st.sidebar.markdown("---")
        if st.sidebar.button(
            "📊 Open Benchmark History",
            type="primary",
            width="stretch",
            key="open_history",
        ):
            st.session_state.show_analytics = False
            st.session_state.show_history = True
            st.rerun()
        return

    show_history = st.session_state.get("show_history", False)
    if not st.session_state.get("game_running", False):
        render_hero()
    else:
        st.title("🤖 AI Chess Battle")
        st.write("Watch AI models compete in chess! Select models from any provider.")

    if show_history:
        st.markdown(
            '<div class="cf-section-title">📊 Benchmark History</div>'
            '<div class="cf-section-sub">Browse past runs parsed from real JSONL artifacts on disk. '
            "Open a run to view its leaderboard, head-to-head, and per-game replays with "
            "Stockfish eval timelines and move-quality heatmaps.</div>",
            unsafe_allow_html=True,
        )
        render_benchmark_history(expanded=True)
    else:
        st.markdown("### 🎮 Get Started")
        st.markdown(
            "API keys are loaded automatically via **Streamlit Secrets**. "
            "Select two models under **⚔️ Player Matchup** in the sidebar, then click **⚔️ Launch AI Arena Match**."
        )

    st.sidebar.markdown("---")

    if not show_history:
        if st.sidebar.button(
            "📊 Open Benchmark History",
            type="primary",
            width="stretch",
            key="open_history",
        ):
            st.session_state.show_history = True
            st.rerun()
    else:
        if st.sidebar.button("🏠 Back to Home", width="stretch", key="close_history"):
            st.session_state.show_history = False
            st.rerun()


def render_analytical_dashboard():
    """Render the research-grade analytical dashboard with multi-model matrix, phase analytics, and deep dive charts."""
    back, title = st.columns([1, 6])
    with back:
        if st.button("← Back to History", key="analytics_back"):
            st.session_state.show_analytics = False
            st.session_state.show_history = True
            st.session_state.pop("analytics_run_dir", None)
            st.rerun()
    with title:
        st.markdown("## 📊 Research Analytics Dashboard")

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

    # Run selector
    with st.container(border=True):
        run_options = {f"{run.run_id} ({run.total_games} games)": run for run in runs}
        default_idx = 0
        if pre_selected_run:
            label = f"{pre_selected_run.run_id} ({pre_selected_run.total_games} games)"
            default_idx = (
                list(run_options.keys()).index(label) if label in run_options else 0
            )
        selected_run_label = st.selectbox(
            "Select Run to Analyze",
            options=list(run_options.keys()),
            index=default_idx,
            key="analytics_run_selector",
        )
        selected_run = run_options[selected_run_label]

    # Load full run data
    run = load_run(selected_run.run_dir)
    if run is None or not run.games:
        st.error("Failed to load benchmark run data.")
        return

    import altair as alt

    # 4 Main Research Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏆 Model Matrix & Leaderboard",
            "⚔️ Game Deep Dive & Eval Graph",
            "♟️ Game Phase Analytics",
            "🧠 Reasoning & Resilience",
        ]
    )

    # ============================================================
    # TAB 1: MODEL MATRIX & LEADERBOARD
    # ============================================================
    with tab1:
        st.markdown("### 🏆 Comparative Model Performance Matrix")
        from chessbench.benchmark.results_view import (
            compute_model_performance_matrix,
            compute_phase_breakdown,
            compute_retry_resilience,
            compute_thinking_quality_correlation,
        )

        matrix = compute_model_performance_matrix(run)
        if matrix:
            matrix_df = pd.DataFrame(matrix)
            st.dataframe(
                matrix_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Accuracy %": st.column_config.NumberColumn(
                        "Accuracy %", format="%.1f%%"
                    ),
                    "Win Rate %": st.column_config.NumberColumn(
                        "Win Rate %", format="%.1f%%"
                    ),
                    "Blunder Rate %": st.column_config.NumberColumn(
                        "Blunder Rate %", format="%.1f%%"
                    ),
                    "Illegal/Retry Rate %": st.column_config.NumberColumn(
                        "Illegal/Retry Rate %", format="%.1f%%"
                    ),
                    "ACPL": st.column_config.NumberColumn(
                        "ACPL (Lower is Better)", format="%.1f"
                    ),
                },
            )

            # Comparative Altair Bar Chart
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                chart_acpl = (
                    alt.Chart(matrix_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Player:N", title="Model", sort="-y"),
                        y=alt.Y("ACPL:Q", title="Average Centipawn Loss (ACPL)"),
                        color=alt.Color("Player:N", legend=None),
                        tooltip=["Player", "ACPL", "Accuracy %"],
                    )
                    .properties(title="Centipawn Loss per Model (Lower = Better)")
                )
                st.altair_chart(chart_acpl, width="stretch")

            with col_c2:
                chart_acc = (
                    alt.Chart(matrix_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Player:N", title="Model", sort="-y"),
                        y=alt.Y("Accuracy %:Q", title="CAPS Accuracy %"),
                        color=alt.Color("Player:N", legend=None),
                        tooltip=["Player", "Accuracy %", "Blunder Rate %"],
                    )
                    .properties(title="Overall Move Accuracy % per Model")
                )
                st.altair_chart(chart_acc, width="stretch")

    # ============================================================
    # TAB 2: GAME DEEP DIVE & EVAL GRAPH
    # ============================================================
    with tab2:
        game_options = {
            f"Game {i+1}: {g.white_player} vs {g.black_player} ({g.result})": g
            for i, g in enumerate(run.games)
        }
        selected_game_label = st.selectbox(
            "Select Game to Analyze",
            options=list(game_options.keys()),
            key="analytics_tab2_game_selector",
        )
        selected_game = game_options[selected_game_label]

        st.markdown("### 📈 Stockfish Centipawn Advantage Timeline")
        eval_data = []
        board = chess.Board(selected_game.opening_fen)

        for ply, move_log in enumerate(selected_game.moves):
            move = chess.Move.from_uci(move_log.move_uci) if move_log.move_uci else None
            if move and move in board.legal_moves:
                board.push(move)

            eval_data.append(
                {
                    "Ply": ply + 1,
                    "Move": move_log.move_san,
                    "Player": move_log.player,
                    "Color": move_log.color,
                    "Eval (cp)": move_log.eval_cp_score,
                    "Best Move": move_log.eval_best_move_uci,
                    "Best Eval (cp)": move_log.eval_best_move_cp,
                    "Quality": move_log.move_quality or "unknown",
                    "Phase": move_log.game_phase or "middlegame",
                    "Material Imbalance": move_log.material_imbalance or 0,
                }
            )

        if eval_data:
            eval_df = pd.DataFrame(eval_data)
            eval_df_valid = eval_df[eval_df["Eval (cp)"].notna()]
            if not eval_df_valid.empty:
                last_eval = eval_df_valid.iloc[-1]["Eval (cp)"]
                max_abs_eval = max(
                    abs(eval_df_valid["Eval (cp)"].max()),
                    abs(eval_df_valid["Eval (cp)"].min()),
                    100,
                )
                advantage_pct = max(0, min(100, 50 + (last_eval / max_abs_eval * 50)))

                adv_html = f"""
                <div class="cf-advantage-bar" style="height:8px;background:linear-gradient(90deg, #1e2330 0%, #1e2330 50%, #f0b421 50%, #f0b421 100%);border-radius:4px;margin-bottom:8px;position:relative;overflow:hidden;">
                    <div style="position:absolute;left:{advantage_pct}%;top:0;bottom:0;width:3px;background:#ffffff;box-shadow:0 0 8px #ffffff;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:0.75rem;color:var(--arena-text-muted);margin-bottom:12px;">
                    <span>♚ Black Advantage</span><span>Even</span><span>♔ White Advantage</span>
                </div>
                """
                st.markdown(adv_html, unsafe_allow_html=True)

                # Line + points chart
                base = alt.Chart(eval_df_valid).encode(
                    x=alt.X("Ply:Q", title="Ply (Half-Move)"),
                    y=alt.Y(
                        "Eval (cp):Q", title="Centipawn Score (+ = White, - = Black)"
                    ),
                    color=alt.Color("Player:N", title="Player"),
                )
                line = base.mark_line(strokeWidth=2)
                points = base.mark_point(size=70, filled=True).encode(
                    tooltip=["Ply", "Move", "Player", "Eval (cp)", "Quality", "Phase"]
                )
                chart = (line + points).properties(width=700, height=360).interactive()
                st.altair_chart(chart, width="stretch")

                # Material Trajectory Chart
                st.markdown("### ⚖️ Material Balance Trajectory")
                mat_chart = (
                    alt.Chart(eval_df)
                    .mark_area(opacity=0.4)
                    .encode(
                        x=alt.X("Ply:Q", title="Ply"),
                        y=alt.Y(
                            "Material Imbalance:Q",
                            title="Material Balance (White - Black)",
                        ),
                        color=alt.value("#f0b421"),
                        tooltip=["Ply", "Move", "Material Imbalance"],
                    )
                    .properties(height=180)
                    .interactive()
                )
                st.altair_chart(mat_chart, width="stretch")

        # Move History Table
        with st.expander("📄 Full Move History Data", expanded=False):
            move_table_rows = []
            for ply, m in enumerate(selected_game.moves):
                move_table_rows.append(
                    {
                        "Ply": ply + 1,
                        "Player": m.player,
                        "Move": m.move_san,
                        "Eval": (
                            f"{m.eval_cp_score/100:+.2f}"
                            if m.eval_cp_score is not None
                            else "—"
                        ),
                        "Quality": m.move_quality or "—",
                        "Phase": m.game_phase or "—",
                        "Latency (ms)": m.llm_latency_ms,
                        "Retries": m.validation_retries + m.illegal_attempts_count,
                        "Reasoning": m.thinking_trace or "",
                    }
                )
            st.dataframe(
                pd.DataFrame(move_table_rows), hide_index=True, width="stretch"
            )

    # ============================================================
    # TAB 3: GAME PHASE ANALYTICS
    # ============================================================
    with tab3:
        st.markdown("### ♟️ Model Performance by Board Phase")
        st.caption(
            "Breakdown of Average Centipawn Loss (ACPL) and Blunder % across Opening, Middlegame, and Endgame."
        )

        phase_results = compute_phase_breakdown(run)
        if phase_results:
            p_rows = []
            for player, metrics in phase_results.items():
                p_rows.append(
                    {
                        "Player": player,
                        "Opening ACPL": metrics.get("opening_acpl"),
                        "Opening Blunder %": metrics.get("opening_blunder_pct"),
                        "Middlegame ACPL": metrics.get("middlegame_acpl"),
                        "Middlegame Blunder %": metrics.get("middlegame_blunder_pct"),
                        "Endgame ACPL": metrics.get("endgame_acpl"),
                        "Endgame Blunder %": metrics.get("endgame_blunder_pct"),
                    }
                )
            phase_df = pd.DataFrame(p_rows)
            st.dataframe(phase_df, hide_index=True, width="stretch")

            # Board Complexity Scatter Plot
            st.markdown("### 🌀 Board Complexity vs Centipawn Loss")
            complexity_data = []
            for g in run.games:
                for m in g.moves:
                    if m.position_complexity is not None and m.cp_loss is not None:
                        complexity_data.append(
                            {
                                "Player": m.player,
                                "Legal Candidates": m.position_complexity,
                                "Centipawn Loss": m.cp_loss,
                                "Move": m.move_san,
                            }
                        )
            if complexity_data:
                comp_df = pd.DataFrame(complexity_data)
                comp_chart = (
                    alt.Chart(comp_df)
                    .mark_circle(size=60, opacity=0.6)
                    .encode(
                        x=alt.X(
                            "Legal Candidates:Q",
                            title="Position Complexity (Number of Legal Moves)",
                        ),
                        y=alt.Y("Centipawn Loss:Q", title="Centipawn Loss"),
                        color=alt.Color("Player:N"),
                        tooltip=[
                            "Player",
                            "Move",
                            "Legal Candidates",
                            "Centipawn Loss",
                        ],
                    )
                    .properties(height=350)
                    .interactive()
                )
                st.altair_chart(comp_chart, width="stretch")

    # ============================================================
    # TAB 4: REASONING & RESILIENCE ANALYTICS
    # ============================================================
    with tab4:
        st.markdown("### 🧠 Reasoning Depth vs Move Quality")
        st.caption(
            "Analyzing whether longer thinking traces lead to lower centipawn loss."
        )

        thinking_points = compute_thinking_quality_correlation(run)
        if thinking_points:
            think_df = pd.DataFrame(thinking_points)
            think_chart = (
                alt.Chart(think_df)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("Thinking Words:Q", title="Thinking Trace Word Count"),
                    y=alt.Y("Centipawn Loss:Q", title="Centipawn Loss"),
                    color=alt.Color("Player:N"),
                    tooltip=[
                        "Player",
                        "Move",
                        "Thinking Words",
                        "Centipawn Loss",
                        "Quality",
                    ],
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(think_chart, width="stretch")
        else:
            st.info("No thinking trace word counts recorded for this run.")

        st.markdown("---")
        st.markdown("### 🛡️ Retry Resilience & Error Recovery")
        resilience = compute_retry_resilience(run)
        if resilience:
            res_rows = [{"Player": k, **v} for k, v in resilience.items()]
            st.dataframe(pd.DataFrame(res_rows), hide_index=True, width="stretch")

        st.markdown("---")
        st.markdown("### 💾 Export Benchmark Artifacts")
        with st.container(border=True):
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                if st.button(
                    "📄 Export PGN + Eval", width="stretch", key="export_tab4_pgn"
                ):
                    from chessbench.benchmark.export import export_pgn_with_eval

                    try:
                        path = export_pgn_with_eval(
                            run.run_dir, run.run_dir / "export_eval.pgn"
                        )
                        st.success(f"Exported to {path}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            with col_exp2:
                if st.button("📊 Export CSV", width="stretch", key="export_tab4_csv"):
                    from chessbench.benchmark.export import export_csv

                    try:
                        path = export_csv(run.run_dir, run.run_dir / "export_csv")
                        st.success(f"Exported to {path}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            with col_exp3:
                if st.button(
                    "📦 Export Parquet", width="stretch", key="export_tab4_parquet"
                ):
                    from chessbench.benchmark.export import export_parquet

                    try:
                        path = export_parquet(
                            run.run_dir, run.run_dir / "export.parquet"
                        )
                        st.success(f"Exported to {path}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")


def rehydrate_session_state():
    import os

    import streamlit as st

    from chessbench.benchmark.results_view import list_runs

    RUNS_ROOT = os.environ.get("CHESS_FIGHT_RUNS_ROOT", "runs")

    if "rehydrated" in st.session_state:
        return

    st.session_state.rehydrated = True

    if (
        "benchmark_completed_games" in st.session_state
        and st.session_state.benchmark_completed_games
    ):
        return

    runs = list_runs(RUNS_ROOT)
    if not runs:
        return

    latest_run = runs[0]
    # Check if the run is recent enough, e.g., within the last 12 hours
    if not latest_run.timestamp_utc:
        return

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
            termination_reason=game_rec.termination_reason,
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

            game_moves.append(
                GameMove(
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
                    is_illegal=is_illegal,
                )
            )

        gs = GameState(
            board=board,
            moves=game_moves,
            stats=stats,
            current_player=(
                game_rec.white_player
                if board.turn == chess.WHITE
                else game_rec.black_player
            ),
            is_game_over=True,
            winner=game_rec.winner_spec,
            game_duration=game_rec.game_duration_sec,
            fen_before=game_rec.opening_fen or chess.STARTING_FEN,
        )
        rehydrated_games.append(gs)

    st.session_state.benchmark_completed_games = rehydrated_games
    st.session_state.benchmark_done = True
    st.session_state.benchmark_run_dir = str(latest_run.run_dir)
