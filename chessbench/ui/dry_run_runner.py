"""Dry-run execution machinery: candidate collection and background LLM-vs-Stockfish scoring."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import chess
import streamlit as st

from chessbench.providers import get_provider
from chessbench.providers.chess_ai import ProviderChessAI

from .dry_run_scoring import (
    DryRunCell,
    compute_centipawn_loss,
    parse_fen_lines,
)
from .strategy_store import list_strategies, load_strategy

# Dry-run safety caps to prevent unbounded LLM calls on public deployment
MAX_FEN_POSITIONS = 50
MAX_STRATEGIES = 20
MAX_LLM_CALLS_PER_SESSION = 500


def _start_dry_run(white_config: dict[str, Any] | None, black_config: dict[str, Any] | None) -> None:
    """Start the dry-run in a background thread."""
    if white_config is None and black_config is None:
        st.error("Please configure a model in the sidebar first.")
        return
    fen_text = st.session_state.get("dry_run_fen_input", "")
    boards, errors = parse_fen_lines(fen_text)
    if errors:
        error_msgs = "\n".join(f"Line {line}: {msg}" for line, msg in errors)
        st.error(f"Invalid FEN positions:\n{error_msgs}")
        return
    if not boards:
        st.error("No valid FEN positions provided.")
        return

    # Enforce caps on FEN positions
    if len(boards) > MAX_FEN_POSITIONS:
        st.error(f"Too many FEN positions ({len(boards)}). Maximum allowed: {MAX_FEN_POSITIONS}.")
        return

    candidates: list[tuple[str, str, str]] = []
    if white_config is not None:
        m1_spec = f"{white_config['provider']}:{white_config['model_id']}"
        sys_prompt_1 = st.session_state.get(f"sys_prompt_{m1_spec}", "")
        turn_prompt_1 = st.session_state.get(f"turn_prompt_{m1_spec}", "")
        if sys_prompt_1 and turn_prompt_1:
            candidates.append(("P1 Editor (White)", sys_prompt_1, turn_prompt_1))
    if black_config is not None:
        m2_spec = f"{black_config['provider']}:{black_config['model_id']}"
        sys_prompt_2 = st.session_state.get(f"sys_prompt_{m2_spec}", "")
        turn_prompt_2 = st.session_state.get(f"turn_prompt_{m2_spec}", "")
        if sys_prompt_2 and turn_prompt_2:
            candidates.append(("P2 Editor (Black)", sys_prompt_2, turn_prompt_2))
    strategies = list_strategies()
    if len(strategies) > MAX_STRATEGIES:
        st.error(f"Too many strategies saved ({len(strategies)}). Maximum allowed: {MAX_STRATEGIES}. Please remove some from the Prompt Strategy Workbench.")
        return
    for meta in strategies:
        try:
            sys_p, turn_p = load_strategy(meta.name)
            candidates.append((f"Strategy: {meta.name}", sys_p, turn_p))
        except Exception:
            pass
    seen: set[tuple[str, str]] = set()
    unique_candidates: list[tuple[str, str, str]] = []
    for label, sys_p, turn_p in candidates:
        key = (sys_p, turn_p)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((label, sys_p, turn_p))
    if not unique_candidates:
        st.error("No candidate strategies found.")
        return

    total_calls = len(unique_candidates) * len(boards)
    if total_calls > MAX_LLM_CALLS_PER_SESSION:
        st.error(f"Dry-run would require {total_calls} LLM calls, exceeding session limit of {MAX_LLM_CALLS_PER_SESSION}. Reduce FEN positions or strategies.")
        return
    st.session_state["dry_run_candidate_prompts"] = list(unique_candidates)

    st.session_state["dry_run_running"] = True
    st.session_state["dry_run_results"] = None
    st.session_state["dry_run_error"] = None
    st.session_state["dry_run_progress"] = {
        "total": len(unique_candidates) * len(boards),
        "completed": 0,
        "current_candidate": "",
        "current_position": 0,
    }
    st.session_state["dry_run_cells"] = []

    def _worker():
        try:
            config = white_config or black_config
            if config is None:
                raise RuntimeError("No valid configuration provided")
            asyncio.run(
                _run_dry_run_async(
                    config,
                    unique_candidates,
                    boards,
                )
            )
        except Exception as e:  # pragma: no cover - defensive
            st.session_state["dry_run_error"] = str(e)
        finally:
            st.session_state["dry_run_running"] = False

    thread = threading.Thread(target=_worker)
    thread.start()


async def _run_dry_run_async(
    white_config: dict[str, Any],
    candidates: list[tuple[str, str, str]],
    boards: list[chess.Board],
) -> None:
    """Run the dry-run asynchronously, updating session state with progress."""
    provider_name = white_config["provider"]
    model_id = white_config["model_id"]
    api_key = white_config["api_key"]
    provider = get_provider(provider_name)
    if provider is None:
        raise RuntimeError(f"Provider {provider_name} not available")
    ai = ProviderChessAI(
        provider_name=provider_name,
        model_id=model_id,
        api_key=api_key,
        temperature=0.1,
        max_tokens=1500,
    )
    from chessbench.benchmark.evaluator import StockfishEvaluator

    evaluator = StockfishEvaluator()

    all_cells: list[DryRunCell] = []
    completed = 0
    for label, _sys_prompt, _turn_prompt in candidates:
        st.session_state["dry_run_progress"]["current_candidate"] = label
        st.session_state["dry_run_progress"]["current_position"] = 0
        for idx, board in enumerate(boards):
            st.session_state["dry_run_progress"]["current_position"] = idx + 1
            fen = board.fen()
            try:
                move_uci, completion_result = await ai.get_move_with_result(fen)
                if move_uci not in [m.uci() for m in board.legal_moves]:
                    raise ValueError(f"Illegal move: {move_uci}")
                cp_loss = compute_centipawn_loss(fen, move_uci, evaluator)
                cell = DryRunCell(
                    fen=fen,
                    move_uci=move_uci,
                    cp_loss=cp_loss,
                    latency_ms=completion_result.latency_ms,
                    tokens_in=completion_result.tokens_in,
                    tokens_out=completion_result.tokens_out,
                    error=None,
                )
            except Exception as e:  # pragma: no cover - defensive
                error_msg = str(e)
                if hasattr(e, "__cause__") and e.__cause__:
                    error_msg = f"{error_msg}: {e.__cause__}"
                cell = DryRunCell(
                    fen=fen,
                    move_uci="",
                    cp_loss=None,
                    latency_ms=None,
                    tokens_in=None,
                    tokens_out=None,
                    error=error_msg,
                )
            all_cells.append(cell)
            completed += 1
            st.session_state["dry_run_progress"]["completed"] = completed
            await asyncio.sleep(0.01)

    st.session_state["dry_run_cells"] = all_cells
    st.session_state["dry_run_candidate_labels"] = [label for label, _, _ in candidates]
    st.session_state["dry_run_num_positions"] = len(boards)
