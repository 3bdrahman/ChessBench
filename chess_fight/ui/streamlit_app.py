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
import time
from datetime import datetime

import chess.svg
import pandas as pd
import streamlit as st

from chess_fight.benchmark.results_view import (
    aggregate_leaderboard,
    list_runs,
    load_run,
)
from chess_fight.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chess_fight.common.common_types import is_chess_capable, is_free_tier
from chess_fight.common.exceptions import (
    GameExecutionError,
    InvalidApiKeyError,
    MoveValidationError,
    NoProvidersConfiguredError,
    ProviderError,
    SetupError,
)
from chess_fight.game.async_game import AsyncChessGame, GameState
from chess_fight.providers import get_provider, list_providers
from chess_fight.providers.chess_ai import ProviderChessAI
from chess_fight.ui.error_display import render_error

# Providers surfaced in the hosted Streamlit UI.
# OpenRouter: aggregated API, server-side demo key, free tier for visitors.
# NIM: NVIDIA's hosted inference API, server-side key.
# Ollama: local-only — works when running the app locally against an Ollama server.
# Stockfish: real local engine — only shown when the binary is on PATH.
# (Anthropic, Google, Groq, OpenAI can be hidden to keep the demo UX focused by setting
# the CHESS_FIGHT_HOSTED_PROVIDERS env var to "openrouter,nim,ollama,stockfish".)
_env_providers = os.environ.get("CHESS_FIGHT_HOSTED_PROVIDERS")
HOSTED_PROVIDERS: tuple[str, ...] | None = tuple(_env_providers.split(",")) if _env_providers else None

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
        st.metric("Time Elapsed", f"{elapsed}s")
    with cols[4]:
        if state.current_player:
            st.metric("Current Turn", state.current_player)


def _draw_moves(moves_placeholder, moves: list) -> None:
    if not moves:
        return
    df = pd.DataFrame(
        [
            {
                "Move #": i + 1,
                "Player": move.player,
                "Move": move.move,
                "Time": datetime.fromtimestamp(move.timestamp).strftime("%H:%M:%S"),
                "Capture": "✓" if move.is_capture else "",
                "Check": "✓" if move.is_check else "",
            }
            for i, move in enumerate(moves)
        ]
    )
    moves_placeholder.dataframe(df, hide_index=True, use_container_width=True)


def _draw_completion_result(expander_placeholder, state: GameState) -> None:
    """Render the last LLM completion (tokens, latency, raw text) honestly."""
    cr = state.last_completion_result
    if cr is None:
        return
    with expander_placeholder.expander("Last LLM completion", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Latency (ms)", cr.latency_ms if cr.latency_ms is not None else "—")
        with col_b:
            st.metric("Tokens in", cr.tokens_in if cr.tokens_in is not None else "—")
        with col_c:
            st.metric("Tokens out", cr.tokens_out if cr.tokens_out is not None else "—")
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
            if st.button("🔌 Use Demo Mode", key="use_demo_key", use_container_width=True):
                provider = get_provider("openrouter")
                if provider and provider.validate_key(demo_api_key):
                    available_providers.append(("openrouter", demo_api_key))

    for provider_name in providers:
        if HOSTED_PROVIDERS and provider_name not in HOSTED_PROVIDERS:
            continue
        provider = get_provider(provider_name)
        if provider is None:
            continue

        if provider_name == "stockfish":
            render_stockfish_section(provider, available_providers)  # type: ignore[arg-type]
            continue

        if not provider.requires_api_key:
            with st.sidebar.expander(f"🖥️ {provider_name.capitalize()} (local)", expanded=False):
                st.caption(f"Local server — no API key required. Run `{provider_name} serve` first.")
                if st.button(
                    f"🔌 Connect to {provider_name.capitalize()}",
                    key=f"connect_{provider_name}",
                    use_container_width=True,
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


def render_stockfish_section(provider: object, available_providers: list) -> None:
    """Honest Stockfish enrollment — never offer it if the binary is absent."""
    with st.sidebar.expander("🤖 Stockfish (local)", expanded=False):
        binary = provider.find_binary()
        if binary is None:
            st.warning(
                "No Stockfish binary found on PATH. Install it from "
                "[stockfishchess.org/download](https://stockfishchess.org/download/) "
                "and either add it to PATH or set the `STOCKFISH_PATH` environment "
                "variable. This provider is a *real* local engine — no API key, "
                "no mock fallback."
            )
            return
        st.success(f"✓ Found Stockfish at `{binary}`")
        depth = st.slider("Search depth", 1, 20, 8, key="stockfish_depth")
        think = st.slider("Think time per move (s)", 0.1, 5.0, 1.0, 0.1, key="stockfish_think")
        if st.button("🔌 Enable Stockfish", key="enable_stockfish", use_container_width=True):
            available_providers.append(("stockfish", ""))
            st.session_state["stockfish_depth"] = depth
            st.session_state["stockfish_think"] = think


def render_model_selectors(available_providers: list):
    """Render model selection for White and Black players."""
    st.sidebar.header("♟️ Model Selection")

    all_models: dict[str, dict] = {}
    filtered_count = 0
    for provider_name, api_key in available_providers:
        cache_key = f"models_{provider_name}"
        if cache_key not in st.session_state:
            with st.spinner(f"Fetching {provider_name} models..."):
                models = asyncio.run(fetch_models_for_provider(provider_name, api_key))
                st.session_state[cache_key] = models
        else:
            models = st.session_state[cache_key]

        for model in models:
            if not is_chess_capable(model):
                filtered_count += 1
                continue
            free_marker = " ★free" if is_free_tier(model) else ""
            display_name = f"[{provider_name}] {model.name}{free_marker}"
            all_models[display_name] = {
                "provider": provider_name,
                "model_id": model.id,
                "api_key": api_key,
                "context_window": model.context_window,
                "is_free": is_free_tier(model),
            }

    if filtered_count:
        st.sidebar.caption(
            f"ⓘ {filtered_count} non-chat model(s) hidden (embedding, audio, image, etc.)"
        )

    if not all_models:
        st.sidebar.warning("No chess-capable models available. Please add API keys.")
        return None, None

    model_options = list(all_models.keys())

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.subheader("White ♔")
        try:
            white_model = st.selectbox(
                "Select Model",
                options=model_options,
                key="player_white_model",
                index=0 if model_options else None,
            )
        except Exception as e:
            import traceback
            st.error(f"Error in white_model: {e}\n\n```python\n{traceback.format_exc()}\n```")
            white_model = model_options[0] if model_options else None
    with col2:
        st.subheader("Black ♚")
        try:
            black_model = st.selectbox(
                "Select Model",
                options=model_options,
                key="player_black_model",
                index=1 if len(model_options) > 1 else 0,
            )
        except Exception as e:
            import traceback
            st.error(f"Error in black_model: {e}\n\n```python\n{traceback.format_exc()}\n```")
            black_model = model_options[0] if model_options else None

    if white_model and black_model:
        return all_models[white_model], all_models[black_model]
    return None, None


def create_provider_ai(white_config: dict, black_config: dict):
    """Create ProviderChessAI instances for both players."""
    params: dict = {"temperature": 0.1, "max_tokens": 100}
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


# ---------------------------------------------------------------------------
# Live game
# ---------------------------------------------------------------------------


async def run_async_game(white_ai, black_ai, ui_callback, delay=0.1):
    """Run the async game loop."""
    game = AsyncChessGame(white_ai, black_ai)
    return await game.play_game(ui_callback, delay)


# ---------------------------------------------------------------------------
# In-process benchmark
# ---------------------------------------------------------------------------


def run_in_process_benchmark(white_config: dict, black_config: dict, games: int = 3):
    """Run the benchmark runner in-process and render live results.

    Replaces the subprocess stub. Uses BenchmarkRunner directly so
    progress, ELO, and per-game logs are surfaced in real time — no
    stdout scraping, no fabricated leaderboard numbers.
    """
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
        max_tokens=100,
        api_keys=api_keys,
    )

    progress_placeholder = st.empty()
    board_placeholder = st.empty()
    stats_placeholder = st.empty()
    moves_placeholder = st.empty()
    completion_placeholder = st.empty()
    status_placeholder = st.empty()

    progress_placeholder.info(
        f"Running in-process benchmark: {white_spec} vs {black_spec} ({games} games)..."
    )

    runner: BenchmarkRunner | None = None
    try:
        runner = BenchmarkRunner(config)
    except (NoProvidersConfiguredError, SetupError, InvalidApiKeyError) as exc:
        render_error(st, exc)
        return
    except ValueError as exc:
        st.error(f"Benchmark setup failed: {exc}")
        return

    game_index = {"value": 0}
    total_games = len(runner.players) * (len(runner.players) - 1) * games
    progress_bar = progress_placeholder.progress(0.0, text=f"Game 0 / {total_games}")

    async def live_callback(state: GameState):
        if state.is_game_over:
            game_index["value"] += 1
            frac = min(1.0, game_index["value"] / max(1, total_games))
            progress_bar.progress(
                frac, text=f"Game {game_index['value']} / {total_games} complete"
            )
        _draw_board(board_placeholder, state, runner.start_time if runner else None)
        _draw_metrics(stats_placeholder, state, runner.start_time if runner else None)
        _draw_moves(moves_placeholder, state.moves)
        _draw_completion_result(completion_placeholder, state)

    async def run_benchmark():
        await runner.run_benchmark_with_callback(live_callback)

    try:
        asyncio.run(run_benchmark())
    except GameExecutionError as exc:
        render_error(st, exc)
        return
    except (NoProvidersConfiguredError, SetupError, InvalidApiKeyError) as exc:
        render_error(st, exc)
        return
    except Exception as exc:
        st.error(f"Benchmark failed: {exc}")
        return
    finally:
        progress_placeholder.empty()

    status_placeholder.success("Benchmark complete!")
    st.balloons()

    # Show real ELO leaderboard + per-pairing results from the run we just did.
    run = load_run(runner.run_dir)
    if run is not None:
        render_run_summary(run, expanded=True)


# ---------------------------------------------------------------------------
# Benchmark history
# ---------------------------------------------------------------------------


def render_benchmark_history() -> None:
    """Render benchmark runs parsed from the on-disk JSONL artifacts."""
    with st.expander("📊 Benchmark History", expanded=False):
        runs = list_runs(RUNS_ROOT)
        if not runs:
            st.info(
                "No benchmark runs found under "
                f"`{RUNS_ROOT}`. Pin two models in the sidebar and click "
                "**Run Quick Benchmark** to generate a real run."
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
                        "Tokens in": row.tokens_in or "—",
                        "Tokens out": row.tokens_out or "—",
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Per-run breakdown.
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
    with st.expander(label, expanded=expanded):
        if run.config:
            st.caption("Config:")
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
                    "Tokens in": ps.tokens_in_total or "—",
                    "Tokens out": ps.tokens_out_total or "—",
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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
            st.dataframe(pd.DataFrame(pair_rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Demo games section — REAL replays from benchmark runs / on-disk demos
# ---------------------------------------------------------------------------


def render_demo_games_section() -> None:
    """Render the demo-game picker using the real demos package."""
    try:
        from demos import ReplayEngine, list_demo_games
    except ImportError:
        st.markdown(
            "Demos module unavailable. Ensure `demos/` is on the import path."
        )
        return

    # Always re-derive from real sources — no caching of synthetic state.
    demo_games = list_demo_games(runs_root=RUNS_ROOT, auto_generate=True)
    if not demo_games:
        st.markdown(
            "**No real demo games yet.** Run a benchmark (sidebar → **Pin & Benchmark**) "
            "or play a Stockfish vs Stockfish game with the local engine to populate "
            "`demos/games/` with real PGNs. No synthetic replays are shown."
        )
        return

    st.markdown(
        f"**🎮 {len(demo_games)} real demo game(s)** — sourced from "
        f"`{RUNS_ROOT}/` benchmark artifacts and `demos/games/`."
    )

    game_labels = {g.filename: g.display_label() for g in demo_games}
    selected_demo = st.selectbox(
        "Choose a demo game to watch:",
        options=list(game_labels.keys()),
        format_func=lambda k: game_labels[k],
        key="demo_game_selector_main",
    )
    demo_delay = st.slider(
        "Replay Speed (seconds per move)", 0.1, 2.0, 0.5, 0.1, key="demo_delay_main"
    )

    if st.button(
        "▶️ Watch Demo Game", type="secondary", use_container_width=True, key="watch_demo_main"
    ):
        board_placeholder = st.empty()
        stats_placeholder = st.empty()
        moves_placeholder = st.empty()
        status_placeholder = st.empty()

        st.session_state.game_running = True
        st.session_state.demo_start_time = time.time()

        async def demo_ui_update(state: GameState):
            _draw_board(board_placeholder, state, st.session_state.get("demo_start_time"))
            _draw_metrics(stats_placeholder, state, st.session_state.get("demo_start_time"))
            _draw_moves(moves_placeholder, state.moves)
            if state.is_game_over:
                status_placeholder.success(f"Game Over! Winner: {state.winner}")
                st.balloons()

        engine = ReplayEngine(selected_demo)
        try:
            asyncio.run(engine.replay(ui_callback=demo_ui_update, delay=demo_delay))
        finally:
            st.session_state.game_running = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if "game_ui" not in st.session_state:
        st.session_state.game_ui = ChessUI()

    st.title("🤖 AI Chess Battle")
    st.write("Watch AI models compete in chess! Select models from any provider.")

    # Hero: demo games + benchmark history — both real, no API key needed.
    if "game_running" not in st.session_state or not st.session_state.game_running:
        st.markdown("### 🎮 Get Started")
        render_demo_games_section()
        st.markdown(
            "**Have API keys?** Add them in the sidebar under **🔑 API Keys**, "
            "pick two models under **♟️ Model Selection**, then click **▶️ Start New Game**."
        )

    render_benchmark_history()

    # Sidebar: API Keys and Model Selection
    available_providers = render_provider_keys_section()
    white_config, black_config = render_model_selectors(available_providers)

    # Pin & Benchmark section
    st.sidebar.markdown("---")
    st.sidebar.header("📌 Pin & Benchmark")
    st.sidebar.caption("Run a real in-process benchmark")

    if "pinned_white" not in st.session_state:
        st.session_state.pinned_white = None
    if "pinned_black" not in st.session_state:
        st.session_state.pinned_black = None

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📌 Pin White", disabled=white_config is None, use_container_width=True):
            st.session_state.pinned_white = white_config
            st.sidebar.success(
                f"Pinned: {white_config['provider']}:{white_config['model_id']}"
            )
    with col2:
        if st.button("📌 Pin Black", disabled=black_config is None, use_container_width=True):
            st.session_state.pinned_black = black_config
            st.sidebar.success(
                f"Pinned: {black_config['provider']}:{black_config['model_id']}"
            )

    if st.session_state.pinned_white:
        st.sidebar.info(
            f"♔ White: {st.session_state.pinned_white['provider']}:"
            f"{st.session_state.pinned_white['model_id']}"
        )
    if st.session_state.pinned_black:
        st.sidebar.info(
            f"♚ Black: {st.session_state.pinned_black['provider']}:"
            f"{st.session_state.pinned_black['model_id']}"
        )

    if st.session_state.pinned_white and st.session_state.pinned_black:
        games = st.sidebar.number_input(
            "Games per pairing", min_value=1, max_value=20, value=3, step=1, key="qb_games"
        )
        if st.sidebar.button(
            f"🏁 Run Quick Benchmark ({games} games)",
            type="secondary",
            use_container_width=True,
        ):
            run_in_process_benchmark(
                st.session_state.pinned_white,
                st.session_state.pinned_black,
                games=int(games),
            )

    # Live game controls
    st.sidebar.header("🎮 Game Controls")
    delay = st.sidebar.slider("Move Delay (seconds)", 0.0, 2.0, 0.5, 0.1)

    if st.sidebar.button("▶️ Start New Game", type="primary", use_container_width=True):
        if not white_config or not black_config:
            st.error("Please select models for both players.")
            return

        try:
            white_ai, black_ai = create_provider_ai(white_config, black_config)
            st.success(
                f"Game started: {white_config['provider']}:{white_config['model_id']} (White) "
                f"vs {black_config['provider']}:{black_config['model_id']} (Black)"
            )

            board_placeholder = st.empty()
            stats_placeholder = st.empty()
            moves_placeholder = st.empty()
            completion_placeholder = st.empty()
            status_placeholder = st.empty()

            st.session_state.game_running = True
            st.session_state.game_start_time = time.time()

            async def ui_update(state: GameState):
                _draw_board(board_placeholder, state, st.session_state.game_start_time)
                _draw_metrics(stats_placeholder, state, st.session_state.game_start_time)
                _draw_moves(moves_placeholder, state.moves)
                _draw_completion_result(completion_placeholder, state)
                if state.is_game_over:
                    status_placeholder.success(f"Game Over! Winner: {state.winner}")
                    st.balloons()

            asyncio.run(run_async_game(white_ai, black_ai, ui_update, delay))

        except ProviderError as exc:
            render_error(st, exc)
        except MoveValidationError as exc:
            render_error(st, exc)
        except Exception as exc:
            st.error(f"An error occurred: {exc!s}")
            st.error("Please check your API keys and model configurations.")
        finally:
            st.session_state.game_running = False

    # Analytical Dashboard
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Analytical Dashboard")

    if st.sidebar.button("📊 Show Analytical Dashboard", use_container_width=True):
        st.session_state.show_analytics = not st.session_state.get("show_analytics", False)

    if st.session_state.get("show_analytics", False):
        render_analytical_dashboard()


def render_analytical_dashboard():
    """Render the analytical dashboard with eval graphs, move quality, opening explorer, etc."""
    st.markdown("## 📊 Analytical Dashboard")

    # Load available runs
    runs = list_runs(RUNS_ROOT)
    if not runs:
        st.info("No benchmark runs available. Run a benchmark first.")
        return

    # Run selector
    run_options = {f"{run.run_id} ({run.total_games} games)": run for run in runs}
    selected_run_label = st.selectbox(
        "Select Run to Analyze",
        options=list(run_options.keys()),
        key="analytics_run_selector"
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

    game_options = {f"Game {i+1}: {g.white_player} vs {g.black_player} ({g.result})": g for i, g in enumerate(run.games)}
    selected_game_label = st.selectbox(
        "Select Game to Analyze",
        options=list(game_options.keys()),
        key="analytics_game_selector"
    )
    selected_game = game_options[selected_game_label]

    st.markdown("---")

    # ============================================================
    # 1. EVAL GRAPH - Line chart of centipawn eval per ply
    # ============================================================
    st.markdown("### 📈 Eval Graph")
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
            # Create line chart
            import altair as alt

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

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No evaluation data available for this game.")

    st.markdown("---")

    # ============================================================
    # 2. MOVE QUALITY TIMELINE - Bar chart per move
    # ============================================================
    st.markdown("### 🎯 Move Quality Timeline")

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

        # Color mapping
        quality_colors = {
            "best": "#2ecc71",
            "excellent": "#27ae60",
            "good": "#f1c40f",
            "inaccuracy": "#f39c12",
            "mistake": "#e67e22",
            "blunder": "#e74c3c",
        }

        # Create bar chart
        chart = alt.Chart(quality_df).mark_bar().encode(
            x=alt.X("Ply:O", title="Ply"),
            y=alt.Y("CP Loss:Q", title="Centipawn Loss"),
            color=alt.Color("Quality:N", scale=alt.Scale(
                domain=list(quality_colors.keys()),
                range=list(quality_colors.values())
            )),
            tooltip=["Ply", "Move", "Player", "Quality", "CP Loss", "Is Best"]
        ).properties(
            width=700,
            height=350,
            title="Move Quality per Ply (CP Loss)"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

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
    # 3. OPENING EXPLORER - Tree view of played openings
    # ============================================================
    st.markdown("### 🌳 Opening Explorer")

    # Aggregate opening stats across all runs
    opening_stats = {}
    for run in list_runs(RUNS_ROOT):
        run_data = load_run(run.run_dir)
        if run_data:
            for game in run_data.games:
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
            opening_rows.append({
                "Opening": key,
                "Games": stats["games"],
                "Wins": stats["wins"],
                "Losses": stats["losses"],
                "Draws": stats["draws"],
                "Win Rate": f"{stats['wins']/stats['games']*100:.1f}%" if stats["games"] > 0 else "0%",
                "Avg CP Loss": f"{avg_cp:.1f}",
            })

        opening_df = pd.DataFrame(opening_rows).sort_values("Games", ascending=False)
        st.dataframe(opening_df, hide_index=True, use_container_width=True)
    else:
        st.info("No opening data available.")

    st.markdown("---")

    # ============================================================
    # 4. MODEL COMPARISON DASHBOARD
    # ============================================================
    st.markdown("### 🤖 Model Comparison Dashboard")

    # Compare models across all runs
    model_stats = {}
    for run in runs:
        for name, ps in run.player_stats.items():
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
            # Note: detailed move quality stats would need to be aggregated from moves

    if model_stats:
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
                "Avg Latency (ms)": f"{avg_latency:.0f}" if avg_latency > 0 else "—",
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values("Games", ascending=False)
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
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
        st.dataframe(thinking_df, hide_index=True, use_container_width=True)

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
        if st.button("⏮️ Start", use_container_width=True):
            st.info("Use the demo game viewer for full replay controls.")
    with col2:
        _ = st.slider("Replay Speed", 0.1, 3.0, 1.0, 0.1, key="replay_speed")
    with col3:
        if st.button("⏭️ End", use_container_width=True):
            st.info("Use the demo game viewer for full replay controls.")

    st.info("💡 For full replay controls (jump to move, flip board, etc.), use the Demo Games section.")

    st.markdown("---")

    # ============================================================
    # 7. EXPORT BUTTONS
    # ============================================================
    st.markdown("### 💾 Export Game Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Export PGN+Eval", use_container_width=True):
            from chess_fight.benchmark.export import export_pgn_with_eval
            output_path = f"runs/{selected_game.run_id if hasattr(selected_game, 'run_id') else 'export'}/game_{selected_game.game_id}_eval.pgn"
            try:
                path = export_pgn_with_eval(selected_game.run_dir if hasattr(selected_game, 'run_dir') else RUNS_ROOT, output_path)
                st.success(f"Exported to {path}")
            except Exception as e:
                st.error(f"Export failed: {e}")

    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            from chess_fight.benchmark.export import export_csv
            try:
                output_dir = f"runs/{selected_game.run_id if hasattr(selected_game, 'run_id') else 'export'}/export_csv"
                path = export_csv(selected_game.run_dir if hasattr(selected_game, 'run_dir') else RUNS_ROOT, output_dir)
                st.success(f"Exported to {path}")
            except Exception as e:
                st.error(f"Export failed: {e}")

    with col3:
        if st.button("📦 Export Parquet", use_container_width=True):
            from chess_fight.benchmark.export import export_parquet
            try:
                output_path = f"runs/{selected_game.run_id if hasattr(selected_game, 'run_id') else 'export'}/export.parquet"
                path = export_parquet(selected_game.run_dir if hasattr(selected_game, 'run_dir') else RUNS_ROOT, output_path)
                st.success(f"Exported to {path}")
            except Exception as e:
                st.error(f"Export failed: {e}")


# Add import for altair at the top of the file if not already there
