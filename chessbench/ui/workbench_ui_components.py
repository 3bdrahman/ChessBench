"""
Preview and diff components for the Prompt Workbench.
"""
from __future__ import annotations

from difflib import unified_diff

import chess
import streamlit as st

from chessbench.prompts import (
    build_prompt_context,
)


def render_preview_popover(
    prompt_sys: str,
    prompt_turn: str,
    reasoning_level: str,
    label: str,
    rendered_tokens_estimate: int | None = None,
) -> None:
    """Render preview popover for a player's prompts."""
    # Position picker
    st.markdown("**Select Position for Preview**")
    position_options = {
        "Start Position": chess.STARTING_FEN,
        "Middlegame (1.e4 c5 2.Nf3 d6 3.d4 cxd4)": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",
        "Tactical (Légal trap)": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK1R w KQkq - 0 5",
        "Rook Endgame": "8/2p5/3p4/KP5r/1R3p2/8/5P2/6K1 b - - 0 1",
    }
    selected_position_name = st.selectbox(
        "Position",
        options=list(position_options.keys()),
        index=0,
        key=f"preview_position_select_{label}",
    )
    selected_fen = position_options[selected_position_name]

    # Build context and render using simplified prompts API
    try:
        board = chess.Board(selected_fen)
        ctx = build_prompt_context(board, "white")

        reasoning_directives = {
            "low": "Be concise. Reasoning under 30 words.",
            "mid": "Concise strategic & tactical reasoning (under 150 words).",
            "high": "Deep step-by-step tactical calculation and candidate move evaluation.",
        }
        reasoning = reasoning_directives.get(reasoning_level, reasoning_directives["high"])

        system_content = prompt_sys.format(**ctx) + f"\n\n[REASONING LEVEL: {reasoning_level.upper()}]\n{reasoning}"
        user_content = prompt_turn.format(**ctx)

        st.markdown("**[SYSTEM]**")
        st.code(system_content, language="text")
        st.markdown("**[USER]**")
        st.code(user_content, language="text")
        if rendered_tokens_estimate is not None:
            st.caption(f"~{rendered_tokens_estimate} tokens rendered")
    except Exception as e:
        st.error(f"Preview generation failed: {e}")


def render_prompt_diff(
    sys_1: str,
    turn_1: str,
    sys_2: str,
    turn_2: str,
    reasoning_level: str,
) -> None:
    """Render P1↔P2 DIFF button and diff output."""
    if st.button("🔍 P1↔P2 DIFF", key="btn_prompt_diff"):
        # Render final prompts with reasoning directive for both players
        try:
            board = chess.Board(chess.STARTING_FEN)
            ctx = build_prompt_context(board, "white")

            user_content_1 = turn_1.format(**ctx)
            user_content_2 = turn_2.format(**ctx)

            user_msg1 = user_content_1
            user_msg2 = user_content_2

            # Compute unified diff
            diff_lines = unified_diff(
                user_msg1.splitlines(keepends=True),
                user_msg2.splitlines(keepends=True),
                fromfile="P1 Prompt",
                tofile="P2 Prompt",
            )
            diff_text = "".join(diff_lines)
            if diff_text:
                st.code(diff_text, language="diff")
            else:
                st.info("No differences found.")
        except Exception as e:
            st.error(f"Diff generation failed: {e}")
