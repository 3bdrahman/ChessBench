"""Dry-run UI component for comparing prompt strategies."""
from __future__ import annotations

from typing import Any

import chess
import pandas as pd
import streamlit as st

from chessbench.ui.helpers import format_duration_ms

from .dry_run_runner import _start_dry_run
from .dry_run_scoring import DryRunCell, aggregate_results, format_cp_loss
from .strategy_store import save_strategy

# Constants
SAMPLE_FENS = [
    chess.STARTING_FEN,  # Start Position
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",  # Middlegame (1.e4 c5 2.Nf3 d6 3.d4 cxd4)
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK1R w KQkq - 0 5",  # Tactical (Légal trap)
    "8/2p5/3p4/KP5r/1R3p2/8/5P2/6K1 b - - 0 1",  # Rook Endgame
]
SAMPLE_FEN_NAMES = [
    "Start Position",
    "Middlegame (1.e4 c5 2.Nf3 d6 3.d4 cxd4)",
    "Tactical (Légal trap)",
    "Rook Endgame",
]











def render_dry_run(white_config: dict[str, Any] | None, black_config: dict[str, Any] | None) -> None:
    """Render the Strategy Dry-Run expander in the sidebar."""
    with st.sidebar.expander("🧪 Strategy Dry-Run", expanded=False):
        # FEN input
        st.markdown("**📋 Positions (FEN, one per line)**")
        st.text_area(
            "FEN Positions",
            height=120,
            placeholder="Enter one FEN per line...",
            key="dry_run_fen_input",
        )
        col_load, col_run = st.columns([1, 1])
        with col_load:
            if st.button("Load sample suite", key="dry_run_load_sample"):
                # Load the 4 sample positions
                sample_text = "\n".join(SAMPLE_FENS)
                st.session_state["dry_run_fen_input"] = sample_text
                st.rerun()
        with col_run:
            run_disabled = True
            run_help: str | None = "Configure a model in the sidebar first"
            # Check if we have a valid white player config (at least one of them)
            if white_config is not None or black_config is not None:
                # Additionally, check if the provider API key is valid (already done in _get_white_player_config)
                run_disabled = False
                run_help = None
            st.button(
                "Run Dry-Run",
                key="dry_run_run",
                disabled=run_disabled,
                help=run_help,
                on_click=lambda: _start_dry_run(white_config, black_config),
            )

        # Show progress and results if dry-run has been run
        if st.session_state.get("dry_run_running", False):
            _render_dry_run_progress()
        elif st.session_state.get("dry_run_results") is not None:
            _render_dry_run_results()


def _render_dry_run_progress() -> None:
    """Render progress bar and live counter."""
    progress = st.session_state.get("dry_run_progress", {})
    total = progress.get("total", 0)
    completed = progress.get("completed", 0)
    current_candidate = progress.get("current_candidate", "")
    current_position = progress.get("current_position", 0)
    num_positions = st.session_state.get("dry_run_num_positions", 0)
    if total > 0:
        pct = completed / total
        st.progress(pct, text=f"Completed {completed}/{total} positions")
    else:
        st.progress(0, text="No positions to run")
    st.caption(f"Current candidate: {current_candidate} (position {current_position}/{num_positions})")


def _render_dry_run_results() -> None:
    """Render the results table and drill-down."""
    cells: list[DryRunCell] = st.session_state.get("dry_run_cells", [])
    candidate_labels: list[str] = st.session_state.get("dry_run_candidate_labels", [])
    num_positions: int = st.session_state.get("dry_run_num_positions", 0)
    if not cells or not candidate_labels or num_positions == 0:
        st.info("No dry-run results available.")
        return

    # Split cells per candidate
    results_by_candidate: dict[str, list[DryRunCell]] = {}
    start = 0
    for label in candidate_labels:
        end = start + num_positions
        results_by_candidate[label] = cells[start:end]
        start = end

    # Compute aggregates for each candidate
    summary_rows: list[dict[str, Any]] = []
    for label, cand_cells in results_by_candidate.items():
        agg = aggregate_results(cand_cells)
        summary_rows.append(
            {
                "Candidate": label,
                "Avg CP Loss": agg["avg_cp_loss"],
                "Accuracy (≤50cp)": agg["accuracy"],
                "Illegal/Format Errors": agg["illegal_count"],
                "Mean Latency (ms)": agg["mean_latency_ms"],
                "Total Tokens": agg["total_tokens"],
            }
        )

    if not summary_rows:
        st.info("No valid results to display.")
        return

    # Create DataFrame
    df = pd.DataFrame(summary_rows)
    # Replace None with NaN for better display
    df = df.fillna(value=pd.NA)
    # Determine winner: lowest avg CP loss (lower is better)
    # If all NaN, then no winner
    if "Avg CP Loss" in df.columns and not df["Avg CP Loss"].isna().all():
        winner_idx = df["Avg CP Loss"].idxmin()
    else:
        winner_idx = None

    # Configure column display
    column_config = {
        "Candidate": st.column_config.TextColumn("Candidate", width="medium"),
        "Avg CP Loss": st.column_config.NumberColumn(
            "Avg CP Loss", format="%.1f", help="Average centipawn loss (lower is better)"
        ),
        "Accuracy (≤50cp)": st.column_config.NumberColumn(
            "Accuracy (≤50cp)", format="%.1f", help="Percentage of moves within 50 cp of best"
        ),
        "Illegal/Format Errors": st.column_config.NumberColumn(
            "Illegal/Format Errors", help="Number of illegal moves, format errors, or provider errors"
        ),
        "Mean Latency (ms)": st.column_config.NumberColumn(
            "Mean Latency (ms)", format="%.0f", help="Average latency in milliseconds"
        ),
        "Total Tokens": st.column_config.NumberColumn(
            "Total Tokens", help="Sum of input and output tokens"
        ),
    }

    # Highlight winner row
    def _highlight_winner(s: pd.Series) -> list[str]:
        if winner_idx is not None and s.name == winner_idx:
            return ["background-color: #d4edda"] * len(s)
        return [""] * len(s)

    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config=column_config,
    )
    # Apply highlighting (requires Styler, but we can't use styler with st.dataframe directly)
    # Instead, we'll just note the winner below.
    if winner_idx is not None:
        winner_label = df.loc[winner_idx, "Candidate"]
        st.caption(f"🏆 Winner: **{winner_label}** (lowest average centipawn loss)")

    # Drill-down expanders
    st.markdown("---")
    st.markdown("**🔍 Per-Position Drill-Down**")
    for label, cand_cells in results_by_candidate.items():
        with st.expander(f"Candidate: {label}", expanded=False):
            for idx, cell in enumerate(cand_cells):
                st.markdown(f"**Position {idx+1}**")
                st.code(cell.fen, language="fen")
                if cell.error:
                    st.error(f"Error: {cell.error}")
                else:
                    st.markdown(f"Move: `{cell.move_uci}`")
                    st.markdown(f"Centipawn Loss: {format_cp_loss(cell.cp_loss)}")
                    st.markdown(f"Latency: {format_duration_ms(cell.latency_ms) if cell.latency_ms else '—'}")
                    st.markdown(f"Tokens: {cell.tokens_in} in / {cell.tokens_out} out")
                st.markdown("---")

# Promote Winner button
    if winner_idx is not None:
        winner_label = df.loc[winner_idx, "Candidate"]
        # Find the system and turn prompts for the winner
        candidate_prompts = st.session_state.get("dry_run_candidate_prompts", [])
        winner_sys_p = None
        winner_turn_p = None
        for _label, sys_p, turn_p in candidate_prompts:
            if _label == winner_label:
                winner_sys_p = sys_p
                winner_turn_p = turn_p
                break
        if winner_sys_p is not None:
            with st.form(key="promote_winner_form"):
                new_name = st.text_input(
                    "Enter a name for the winning strategy",
                    value=winner_label,
                    key="promote_winner_name"
                )
                submitted = st.form_submit_button("Promote to Strategy")
                if submitted and new_name:
                    # Save the strategy
                    if winner_sys_p is not None and winner_turn_p is not None:
                        try:
                            save_strategy(new_name, winner_sys_p, winner_turn_p)
                            st.success(f"Strategy '{new_name}' saved!")
                        except Exception as e:
                            st.error(f"Failed to save strategy: {e}")
                    else:
                        st.error("Winner's prompts not found.")
        else:
            st.error("Could not find the winner's prompts.")
