"""Streamlit app with async game loop and provider-agnostic model selection."""

import asyncio
import os
import sys
import time
from datetime import datetime

import chess.svg
import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.async_game import AsyncChessGame, GameState
from game.sync_game import ChessGame
from models import AnthropicChessAI, ChessAI, LlamaChessAI, ModelType, OpenAIChessAI
from providers import get_provider, list_providers
from providers.chess_ai import ProviderChessAI

# Configure page
st.set_page_config(
    page_title="AI Chess Battle",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ChessUI:
    """UI components for chess game display."""

    def __init__(self):
        self.board_placeholder = st.empty()
        self.stats_placeholder = st.empty()
        self.move_history_placeholder = st.empty()
        self.status_placeholder = st.empty()

    def display_board(self, board: chess.Board):
        svg_board = chess.svg.board(
            board,
            size=600,
            lastmove=board.peek() if board.move_stack else None,
            check=board.king(board.turn) if board.is_check() else None
        )
        self.board_placeholder.write(svg_board, unsafe_allow_html=True)

    def display_stats(self, game_state: GameState):
        cols = st.columns(5)
        with cols[0]:
            st.metric("Total Moves", game_state.stats.total_moves)
        with cols[1]:
            st.metric("Captures", game_state.stats.capture_moves)
        with cols[2]:
            st.metric("Checks", game_state.stats.check_moves)
        with cols[3]:
            elapsed = int(game_state.game_duration) if game_state.game_duration > 0 else int(time.time() - st.session_state.get('game_start_time', time.time()))
            st.metric("Time Elapsed", f"{elapsed}s")
        with cols[4]:
            if game_state.current_player:
                st.metric("Current Turn", game_state.current_player)

    def display_moves(self, moves: list):
        if not moves:
            return

        df = pd.DataFrame([{
            "Move #": i + 1,
            "Player": move.player,
            "Move": move.move,
            "Time": datetime.fromtimestamp(move.timestamp).strftime('%H:%M:%S'),
            "Capture": "✓" if move.is_capture else "",
            "Check": "✓" if move.is_check else ""
        } for i, move in enumerate(moves)])

        self.move_history_placeholder.dataframe(df, hide_index=True, use_container_width=True)

    def display_status(self, message: str, status_type: str = "info"):
        if status_type == "success":
            self.status_placeholder.success(message)
        elif status_type == "error":
            self.status_placeholder.error(message)
        elif status_type == "warning":
            self.status_placeholder.warning(message)
        else:
            self.status_placeholder.info(message)


def create_legacy_ai_player(model_type: ModelType) -> ChessAI:
    """Create legacy AI player for backward compatibility."""
    if model_type == ModelType.CHATGPT_4O:
        return OpenAIChessAI(model_type)
    elif model_type == ModelType.CLAUDE_SONNET:
        return AnthropicChessAI(model_type)
    else:
        return LlamaChessAI(model_type)


async def fetch_models_for_provider(provider_name: str, api_key: str) -> list:
    """Fetch available models for a provider."""
    provider = get_provider(provider_name)
    if not provider:
        return []

    try:
        models = await provider.list_models(api_key)
        return models
    except Exception as e:
        st.error(f"Failed to fetch models from {provider_name}: {e}")
        return []


def render_provider_keys_section():
    """Render API key inputs for each provider in sidebar."""
    st.sidebar.header("🔑 API Keys")

    providers = list_providers()
    available_providers = []

    # Auto-detect server-side demo key from st.secrets (Streamlit Cloud)
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
        with st.sidebar.expander(f"{provider_name.capitalize()}", expanded=False):
            api_key = st.text_input(
                f"{provider_name.capitalize()} API Key",
                type="password",
                key=f"api_key_{provider_name}",
                help=f"Enter your {provider_name} API key"
            )

            if api_key:
                provider = get_provider(provider_name)
                if provider and provider.validate_key(api_key):
                    st.success("✓ Valid key format")
                    available_providers.append((provider_name, api_key))
                else:
                    st.error("✗ Invalid key format")

    return available_providers


def render_model_selectors(available_providers: list):
    """Render model selection for White and Black players."""
    st.sidebar.header("♟️ Model Selection")

    # Collect all models from all providers
    all_models = {}
    for provider_name, api_key in available_providers:
        # Use cached models from session state if available
        cache_key = f"models_{provider_name}"
        if cache_key not in st.session_state:
            with st.spinner(f"Fetching {provider_name} models..."):
                models = asyncio.run(fetch_models_for_provider(provider_name, api_key))
                st.session_state[cache_key] = models
        else:
            models = st.session_state[cache_key]

        for model in models:
            display_name = f"[{provider_name}] {model.name}"
            all_models[display_name] = {
                "provider": provider_name,
                "model_id": model.id,
                "api_key": api_key,
                "context_window": model.context_window,
            }

    if not all_models:
        st.sidebar.warning("No models available. Please add API keys.")
        return None, None

    # Model selectors
    model_options = list(all_models.keys())

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.subheader("White ♔")
        white_model = st.selectbox(
            "Select Model",
            options=model_options,
            key="white_model",
            index=0 if model_options else None
        )

    with col2:
        st.subheader("Black ♚")
        black_model = st.selectbox(
            "Select Model",
            options=model_options,
            key="black_model",
            index=1 if len(model_options) > 1 else 0
        )

    if white_model and black_model:
        return all_models[white_model], all_models[black_model]
    return None, None


def create_provider_ai(white_config: dict, black_config: dict):
    """Create ProviderChessAI instances for both players."""
    white_ai = ProviderChessAI(
        provider_name=white_config["provider"],
        model_id=white_config["model_id"],
        api_key=white_config["api_key"],
        temperature=0.1,
        max_tokens=100
    )
    black_ai = ProviderChessAI(
        provider_name=black_config["provider"],
        model_id=black_config["model_id"],
        api_key=black_config["api_key"],
        temperature=0.1,
        max_tokens=100
    )
    return white_ai, black_ai


async def run_async_game(white_ai, black_ai, ui_callback, delay=0.1):
    """Run the async game loop."""
    game = AsyncChessGame(white_ai, black_ai)
    return await game.play_game(ui_callback, delay)


def main():
    # Initialize UI
    if 'game_ui' not in st.session_state:
        st.session_state.game_ui = ChessUI()

    ui = st.session_state.game_ui

    # Title
    st.title("🤖 AI Chess Battle")
    st.write("Watch AI models compete in chess! Select models from any provider.")

    # Lazy import demos package (built separately, may not exist in dev)
    _demo_available = False
    try:
        from demos import ReplayEngine, list_demo_games
        _demo_available = True
    except ImportError:
        pass

    # Hero section — inviting when no game is running
    if 'game_running' not in st.session_state or not st.session_state.game_running:
        st.markdown(
            """### 🎮 Get Started
            **No API key?** Use **📺 Watch Demo Game** in the sidebar to replay a recorded match — no setup needed!

            **Have API keys?** Add them in the sidebar under **🔑 API Keys**, pick two models under **♟️ Model Selection**, then click **▶️ Start New Game**.
            """
        )

    # Sidebar: API Keys and Model Selection
    available_providers = render_provider_keys_section()
    white_config, black_config = render_model_selectors(available_providers)

    # Game controls
    st.sidebar.header("🎮 Game Controls")

    game_mode = st.sidebar.radio(
        "Game Mode",
        options=["Async (Live Updates)", "Legacy (Blocking)"],
        index=0
    )

    delay = st.sidebar.slider("Move Delay (seconds)", 0.0, 2.0, 0.5, 0.1)

    # Start game button
    if st.sidebar.button("▶️ Start New Game", type="primary", use_container_width=True):
        if not white_config or not black_config:
            st.error("Please select models for both players.")
            return

        try:
            # Initialize game
            if game_mode == "Async (Live Updates)":
                # Use new provider-agnostic AI
                white_ai, black_ai = create_provider_ai(white_config, black_config)

                st.success(f"Game started: {white_config['provider']}:{white_config['model_id']} (White) vs {black_config['provider']}:{black_config['model_id']} (Black)")

                # Create placeholders for live updates
                board_placeholder = st.empty()
                stats_placeholder = st.empty()
                moves_placeholder = st.empty()
                status_placeholder = st.empty()

                # Track game state
                st.session_state.game_running = True
                st.session_state.game_start_time = time.time()

                # Run async game
                async def ui_update(state: GameState):
                    board_placeholder.write(
                        chess.svg.board(
                            state.board,
                            size=600,
                            lastmove=state.board.peek() if state.board.move_stack else None,
                            check=state.board.king(state.board.turn) if state.board.is_check() else None
                        ),
                        unsafe_allow_html=True
                    )

                    # Update stats
                    cols = stats_placeholder.columns(5)
                    with cols[0]:
                        st.metric("Total Moves", state.stats.total_moves)
                    with cols[1]:
                        st.metric("Captures", state.stats.capture_moves)
                    with cols[2]:
                        st.metric("Checks", state.stats.check_moves)
                    with cols[3]:
                        elapsed = int(state.game_duration) if state.game_duration > 0 else int(time.time() - st.session_state.game_start_time)
                        st.metric("Time Elapsed", f"{elapsed}s")
                    with cols[4]:
                        if state.current_player:
                            st.metric("Current Turn", state.current_player)

                    # Update moves
                    if state.moves:
                        df = pd.DataFrame([{
                            "Move #": i + 1,
                            "Player": move.player,
                            "Move": move.move,
                            "Time": datetime.fromtimestamp(move.timestamp).strftime('%H:%M:%S'),
                            "Capture": "✓" if move.is_capture else "",
                            "Check": "✓" if move.is_check else ""
                        } for i, move in enumerate(state.moves)])
                        moves_placeholder.dataframe(df, hide_index=True, use_container_width=True)

                    if state.is_game_over:
                        status_placeholder.success(f"Game Over! Winner: {state.winner}")
                        st.balloons()

                # Run the async game
                asyncio.run(run_async_game(white_ai, black_ai, ui_update, delay))

            else:
                # Legacy blocking mode
                player1 = create_legacy_ai_player(ModelType(white_config["model_id"]))
                player2 = create_legacy_ai_player(ModelType(black_config["model_id"]))
                game = ChessGame(player1, player2)

                st.success(f"Game started: {player1.name} (White) vs {player2.name} (Black)")

                while not game.is_game_over:
                    current_player = player1 if len(game.moves) % 2 == 0 else player2
                    with st.spinner(f"Thinking... {current_player.name}'s turn"):
                        move = game.play_move()
                        if move:
                            ui.display_board(game.board)
                            ui.display_stats(game)
                            ui.display_moves(game.moves)
                            time.sleep(delay)

                st.balloons()
                st.success(f"Game Over! Winner: {game.stats.winner}")
                ui.display_stats(game)
                ui.display_moves(game.moves)

        except Exception as e:
            st.error(f"An error occurred: {e!s}")
            st.error("Please check your API keys and model configurations.")
        finally:
            st.session_state.game_running = False

    # Watch Demo Game section
    if _demo_available:
        st.sidebar.markdown("---")
        st.sidebar.header("📺 Watch Demo Game")
        demo_games = list_demo_games()
        if demo_games:
            game_labels = {
                g["filename"]: (
                    f"{g['white']} vs {g['black']} "
                    f"({g['move_count']} moves, {g['result']}) — {g['opening']}"
                )
                for g in demo_games
            }
            selected_demo = st.sidebar.selectbox(
                "Choose game",
                options=list(game_labels.keys()),
                format_func=lambda k: game_labels[k],
                key="demo_game_selector",
            )
            demo_delay = st.sidebar.slider(
                "Replay Speed (s)", 0.1, 2.0, 0.5, 0.1,
                key="demo_delay",
            )
            if st.sidebar.button(
                "▶️ Watch Demo", type="secondary", use_container_width=True,
                key="watch_demo_btn",
            ):
                board_placeholder = st.empty()
                stats_placeholder = st.empty()
                moves_placeholder = st.empty()
                status_placeholder = st.empty()

                st.session_state.game_running = True
                st.session_state.demo_start_time = time.time()

                async def demo_ui_update(state: GameState):
                    board_placeholder.write(
                        chess.svg.board(
                            state.board, size=600,
                            lastmove=state.board.peek() if state.board.move_stack else None,
                            check=state.board.king(state.board.turn) if state.board.is_check() else None,
                        ),
                        unsafe_allow_html=True,
                    )
                    cols = stats_placeholder.columns(5)
                    with cols[0]:
                        st.metric("Total Moves", state.stats.total_moves)
                    with cols[1]:
                        st.metric("Captures", state.stats.capture_moves)
                    with cols[2]:
                        st.metric("Checks", state.stats.check_moves)
                    with cols[3]:
                        elapsed = int(state.game_duration) if state.game_duration > 0 else int(time.time() - st.session_state.get("demo_start_time", time.time()))
                        st.metric("Time Elapsed", f"{elapsed}s")
                    with cols[4]:
                        if state.current_player:
                            st.metric("Current Turn", state.current_player)
                    if state.moves:
                        df = pd.DataFrame([
                            {
                                "Move #": i + 1,
                                "Player": move.player,
                                "Move": move.move,
                                "Time": datetime.fromtimestamp(move.timestamp).strftime("%H:%M:%S"),
                                "Capture": "✓" if move.is_capture else "",
                                "Check": "✓" if move.is_check else "",
                            }
                            for i, move in enumerate(state.moves)
                        ])
                        moves_placeholder.dataframe(df, hide_index=True, use_container_width=True)
                    if state.is_game_over:
                        status_placeholder.success(f"Game Over! Winner: {state.winner}")
                        st.balloons()

                engine = ReplayEngine(selected_demo)
                try:
                    asyncio.run(engine.replay(ui_callback=demo_ui_update, delay=demo_delay))
                finally:
                    st.session_state.game_running = False

    # Display current board if game exists
    if 'game' in st.session_state:
        ui.display_board(st.session_state.game.board)
        ui.display_stats(st.session_state.game)
        ui.display_moves(st.session_state.game.moves)

    # Info section
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        **AI Chess Battle** - Provider-agnostic benchmark for LLM chess capability.

        **Features:**
        - Multi-provider support (OpenAI, Anthropic, Google, NIM, OpenRouter, Ollama, Groq)
        - Async live updates
        - Cross-provider battles
        - Secure client-side API keys
        - Watch recorded demo games (no API key needed)

        **Providers:**
        - **OpenAI**: GPT-4o, o1 models
        - **Anthropic**: Claude 3.5 Sonnet/Haiku/Opus
        - **Google**: Gemini models
        - **NVIDIA NIM**: Self-hosted/cloud models
        - **OpenRouter**: 100+ models via one API (demo default)
        - **Groq**: Fast open-source models
        - **Ollama**: Local Llama/Qwen models
        """)


if __name__ == "__main__":
    main()
