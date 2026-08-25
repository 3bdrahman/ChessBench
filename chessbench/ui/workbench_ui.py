"""
Prompt Workbench UI module for Streamlit chess-LLM benchmark app.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from chessbench.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TURN_PROMPT,
    validate_prompt_text,
)
from chessbench.ui.prompt_workbench import compute_budget_state
from chessbench.ui.workbench_ui_components import (
    render_preview_popover,
    render_prompt_diff,
)
from chessbench.ui.workbench_ui_components_strategy import (
    render_strategy_store,
    render_variable_reference,
)


def render_prompt_management(
    model_1_config: dict[str, Any] | None,
    model_2_config: dict[str, Any] | None,
) -> tuple[
    dict[str, str],  # system_prompts_by_spec
    dict[str, str],  # turn_prompts_by_spec
    dict[str, dict[str, str]],  # prompts_by_color: {"system": {spec: text}, "turn": {spec: text}}
]:
    """Render prompt management UI and return prompt dictionaries.

    Returns:
        system_prompts_by_spec: dict mapping model spec to system prompt text
        turn_prompts_by_spec: dict mapping model spec to turn prompt text
        prompts_by_color: dict with keys "system" and "turn", each mapping color to prompt text
    """
    if not model_1_config or not model_2_config:
        return {}, {}, {"system": {}, "turn": {}}

    m1_spec = f"{model_1_config['provider']}:{model_1_config['model_id']}"
    m2_spec = f"{model_2_config['provider']}:{model_2_config['model_id']}"

    # Initialize defaults in session state if missing
    if f"sys_prompt_{m1_spec}" not in st.session_state:
        st.session_state[f"sys_prompt_{m1_spec}"] = DEFAULT_SYSTEM_PROMPT
    if f"turn_prompt_{m1_spec}" not in st.session_state:
        st.session_state[f"turn_prompt_{m1_spec}"] = DEFAULT_TURN_PROMPT
    if f"sys_prompt_{m2_spec}" not in st.session_state:
        st.session_state[f"sys_prompt_{m2_spec}"] = DEFAULT_SYSTEM_PROMPT
    if f"turn_prompt_{m2_spec}" not in st.session_state:
        st.session_state[f"turn_prompt_{m2_spec}"] = DEFAULT_TURN_PROMPT

    with st.sidebar.expander("⚡ Prompt Strategy Workbench", expanded=False):
        # Reasoning level selector
        st.markdown("**🧠 Reasoning Level**")
        reasoning_level = st.radio(
            "Select reasoning level",
            options=["low", "mid", "high"],
            index=2,  # default to high
            horizontal=True,
            key="reasoning_level_selector",
            help="Affects preview and directive suffix",
        )

        # Player tabs
        tab_p1, tab_p2 = st.tabs(["♔ P1 (White)", "♚ P2 (Black)"])

        # We'll track active tab via session state for variable insertion
        def set_active_p1():
            st.session_state.active_prompt_tab = "P1"
        def set_active_p2():
            st.session_state.active_prompt_tab = "P2"

        # --- Player 1 ---
        with tab_p1:
            st.caption(f"Strategy for `{model_1_config['model_id'][:20]}`")
            sys_1 = st.text_area(
                "System Prompt (P1)",
                value=st.session_state[f"sys_prompt_{m1_spec}"],
                height=80,
                key=f"ui_sys_1_{m1_spec}",
                on_change=set_active_p1,
            )
            turn_1 = st.text_area(
                "Turn Prompt (P1)",
                value=st.session_state[f"turn_prompt_{m1_spec}"],
                height=140,
                key=f"ui_turn_1_{m1_spec}",
                on_change=set_active_p1,
            )

            # Validation and budget gauge
            v1 = validate_prompt_text(sys_1, turn_1)
            if not v1.is_valid:
                st.error("❌ P1 Prompt Invalid:\n" + "\n".join(f"- {e}" for e in v1.errors))
            else:
                st.caption(f"✅ Budget: ~`{v1.estimated_tokens}` tokens")

            # Budget gauge
            context_window = model_1_config.get('context_window')
            budget_state = compute_budget_state(v1.rendered_tokens_estimate, context_window)
            budget_colors = {
                "ok": "green",
                "warning": "orange",
                "error": "red",
                "standard": "gray",
            }
            budget_label = {
                "ok": "OK",
                "warning": "WARNING",
                "error": "ERROR",
                "standard": "STANDARD",
            }
            st.markdown(
                f'<div style="background-color:{budget_colors[budget_state]}; padding:4px; border-radius:4px; text-align:center; color:white; font-weight:bold;">'
                f'Budget: {budget_label[budget_state]}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Reset button
            col_res1, col_prev1 = st.columns([1, 1])
            if col_res1.button("🔄 Reset P1", key=f"reset_p1_{m1_spec}"):
                st.session_state[f"sys_prompt_{m1_spec}"] = DEFAULT_SYSTEM_PROMPT
                st.session_state[f"turn_prompt_{m1_spec}"] = DEFAULT_TURN_PROMPT
                st.rerun()

            # Preview button
            with col_prev1.popover("👁️ Preview Output"):
                render_preview_popover(sys_1, turn_1, reasoning_level, "P1", v1.rendered_tokens_estimate)

        # --- Player 2 ---
        with tab_p2:
            st.caption(f"Strategy for `{model_2_config['model_id'][:20]}`")
            sys_2 = st.text_area(
                "System Prompt (P2)",
                value=st.session_state[f"sys_prompt_{m2_spec}"],
                height=80,
                key=f"ui_sys_2_{m2_spec}",
                on_change=set_active_p2,
            )
            turn_2 = st.text_area(
                "Turn Prompt (P2)",
                value=st.session_state[f"turn_prompt_{m2_spec}"],
                height=140,
                key=f"ui_turn_2_{m2_spec}",
                on_change=set_active_p2,
            )

            v2 = validate_prompt_text(sys_2, turn_2)
            if not v2.is_valid:
                st.error("❌ P2 Prompt Invalid:\n" + "\n".join(f"- {e}" for e in v2.errors))
            else:
                st.caption(f"✅ Budget: ~`{v2.estimated_tokens}` tokens")

            # Budget gauge
            context_window = model_2_config.get('context_window')
            budget_state = compute_budget_state(v2.rendered_tokens_estimate, context_window)
            budget_colors = {
                "ok": "green",
                "warning": "orange",
                "error": "red",
                "standard": "gray",
            }
            budget_label = {
                "ok": "OK",
                "warning": "WARNING",
                "error": "ERROR",
                "standard": "STANDARD",
            }
            st.markdown(
                f'<div style="background-color:{budget_colors[budget_state]}; padding:4px; border-radius:4px; text-align:center; color:white; font-weight:bold;">'
                f'Budget: {budget_label[budget_state]}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Reset button
            col_res2, col_prev2 = st.columns([1, 1])
            if col_res2.button("🔄 Reset P2", key=f"reset_p2_{m2_spec}"):
                st.session_state[f"sys_prompt_{m2_spec}"] = DEFAULT_SYSTEM_PROMPT
                st.session_state[f"turn_prompt_{m2_spec}"] = DEFAULT_TURN_PROMPT
                st.rerun()

            # Preview button
            with col_prev2.popover("👁️ Preview Output"):
                render_preview_popover(sys_2, turn_2, reasoning_level, "P2", v2.rendered_tokens_estimate)

        # P1↔P2 DIFF button
        st.markdown("---")
        render_prompt_diff(sys_1, turn_1, sys_2, turn_2, reasoning_level)

    # Save edits back to session state to persist them
    st.session_state[f"sys_prompt_{m1_spec}"] = sys_1
    st.session_state[f"turn_prompt_{m1_spec}"] = turn_1
    st.session_state[f"sys_prompt_{m2_spec}"] = sys_2
    st.session_state[f"turn_prompt_{m2_spec}"] = turn_2

    # Return the new contract
    system_prompts_by_spec = {m1_spec: sys_1, m2_spec: sys_2}
    turn_prompts_by_spec = {m1_spec: turn_1, m2_spec: turn_2}
    prompts_by_color = {
        "system": {"white": sys_1, "black": sys_2},  # Assuming white is P1, black is P2; but note: color assignment may swap
        "turn": {"white": turn_1, "black": turn_2},
    }
    # Note: The actual color assignment (which player gets white/black) is handled in main() with random swap.
    # We return the prompts by spec and also by color assuming P1 is white and P2 is black.
    # The caller (main) should swap the color mapping if the players are swapped.
    # However, the instruction says: "Prompts always flow via by_color dicts (guarantees correct assignment when specs collide)."
    # We'll return the color mapping based on the spec order (m1_spec -> white, m2_spec -> black) and let main handle swapping.
    return system_prompts_by_spec, turn_prompts_by_spec, prompts_by_color
