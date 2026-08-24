"""
Preview and diff components for the Prompt Workbench.
"""
from __future__ import annotations

from difflib import unified_diff

import chess
import streamlit as st

from chessbench.models.chess_ai import get_reasoning_directive
from chessbench.prompts import (
    create_safe_prompt_template,
)
from chessbench.prompts.sample_context import build_sample_context


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

    # Build context and render
    try:
        tmpl, _ = create_safe_prompt_template(prompt_sys, prompt_turn)
        sample_ctx = build_sample_context(
            chess.Board(selected_fen),
            move_history=[],
            reasoning_level=reasoning_level,
            stagnation_threshold=3,
            prompt_template=tmpl,
        )
        msgs = tmpl.render_messages(
            sample_ctx,
            system_suffix=get_reasoning_directive(reasoning_level),
        )
        for msg in msgs:
            st.markdown(f"**[{msg.role.upper()}]**")
            st.code(msg.content, language="text")
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
            tmpl1, _ = create_safe_prompt_template(sys_1, turn_1)
            tmpl2, _ = create_safe_prompt_template(sys_2, turn_2)
            # Use a neutral board for diff (starting position)
            sample_ctx = build_sample_context(
                chess.Board(chess.STARTING_FEN),
                move_history=[],
                reasoning_level=reasoning_level,
                stagnation_threshold=3,
                prompt_template=None,  # We'll use the template's own variables
            )
            # Render messages including system suffix (reasoning directive)
            msgs1 = tmpl1.render_messages(
                sample_ctx,
                system_suffix=get_reasoning_directive(reasoning_level),
            )
            msgs2 = tmpl2.render_messages(
                sample_ctx,
                system_suffix=get_reasoning_directive(reasoning_level),
            )
            # Extract the user message content (last message) for diff
            user_msg1 = msgs1[-1].content if msgs1 else ""
            user_msg2 = msgs2[-1].content if msgs2 else ""
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
