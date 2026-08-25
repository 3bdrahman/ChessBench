"""
Strategy store and variable reference components for the Prompt Workbench.
"""
from __future__ import annotations

import chess
import streamlit as st

from chessbench.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TURN_PROMPT,
    KNOWN_VARIABLES,
    create_safe_prompt_template,
)
from chessbench.prompts.sample_context import build_sample_context
from chessbench.ui.strategy_store import (
    delete_strategy,
    export_yaml,
    import_yaml,
    list_strategies,
    load_strategy,
    save_strategy,
)


def render_strategy_store(
    m1_spec: str,
    m2_spec: str,
) -> None:
    """Render strategy store UI components."""
    st.markdown("**💾 Strategy Store**")
    col_strat1, _col_strat2 = st.columns([3, 2])
    saved_strategies = list_strategies()
    strategy_options = ["-- New Strategy --"] + [s.name for s in saved_strategies]
    selected_strategy = col_strat1.selectbox(
        "Load Saved Strategy",
        options=strategy_options,
        index=0,
        key=f"strategy_selector_{m1_spec}_{m2_spec}",
        label_visibility="collapsed",
    )
    if selected_strategy != "-- New Strategy --":
        try:
            sys_p, turn_p = load_strategy(selected_strategy)
            st.session_state[f"sys_prompt_{m1_spec}"] = sys_p
            st.session_state[f"turn_prompt_{m1_spec}"] = turn_p
            st.session_state[f"sys_prompt_{m2_spec}"] = sys_p
            st.session_state[f"turn_prompt_{m2_spec}"] = turn_p
            st.success(f"Loaded strategy '{selected_strategy}'")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load strategy: {e}")

    # Save current strategy
    col_save1, col_save2, col_save3 = st.columns([2, 2, 1])
    strategy_name = col_save1.text_input(
        "Save current as…",
        value="",
        key=f"strategy_name_input_{m1_spec}_{m2_spec}",
        placeholder="Enter strategy name",
    )
    overwrite = col_save2.checkbox("Overwrite if exists", value=False, key=f"overwrite_checkbox_{m1_spec}_{m2_spec}")
    if col_save3.button("💾 Save", key=f"btn_save_strategy_{m1_spec}_{m2_spec}"):
        if not strategy_name:
            st.error("Strategy name cannot be empty")
        else:
            existing = {meta.name for meta in list_strategies()}
            if strategy_name in existing and not overwrite:
                st.error(f"Strategy '{strategy_name}' already exists — enable Overwrite to replace it")
            else:
                try:
                    sys_1 = st.session_state[f"sys_prompt_{m1_spec}"]
                    turn_1 = st.session_state[f"turn_prompt_{m1_spec}"]
                    # Single shared strategy: both players are saved as one entry (P1's texts).
                    save_strategy(strategy_name, sys_1, turn_1, f"Strategy for {m1_spec}")
                    st.success(f"Strategy '{strategy_name}' saved")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save strategy: {e}")

    # Delete strategy
    if saved_strategies:
        col_del1, col_del2 = st.columns([3, 1])
        del_strategy = col_del1.selectbox(
            "Delete Strategy",
            options=["-- Select --", *[s.name for s in saved_strategies]],
            index=0,
            key=f"delete_strategy_select_{m1_spec}_{m2_spec}",
            label_visibility="collapsed",
        )
        if col_del2.button("🗑️ Delete", key=f"btn_delete_strategy_{m1_spec}_{m2_spec}") and del_strategy != "-- Select --":
            try:
                delete_strategy(del_strategy)
                st.success(f"Strategy '{del_strategy}' deleted")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete strategy: {e}")

    # Export/Import
    col_exp1, col_exp2 = st.columns([2, 2])
    if saved_strategies:
        exp_strategy = col_exp1.selectbox(
            "Export Strategy",
            options=["-- Select --", *[s.name for s in saved_strategies]],
            index=0,
            key=f"export_strategy_select_{m1_spec}_{m2_spec}",
            label_visibility="collapsed",
        )
        if col_exp2.button("📥 Export YAML", key=f"btn_export_yaml_{m1_spec}_{m2_spec}") and exp_strategy != "-- Select --":
            try:
                yaml_text = export_yaml(exp_strategy)
                st.download_button(
                    label="💾 Download YAML",
                    data=yaml_text,
                    file_name=f"{exp_strategy}.yaml",
                    mime="text/yaml",
                    key=f"dl_yaml_{m1_spec}_{m2_spec}",
                )
            except Exception as e:
                st.error(f"Failed to export strategy: {e}")

    uploaded_file = st.file_uploader(
        "Import Strategy YAML",
        type=["yaml", "yml"],
        key=f"strategy_file_uploader_{m1_spec}_{m2_spec}",
        help="Upload a YAML file to import a strategy",
    )
    if uploaded_file is not None:
        try:
            yaml_text = uploaded_file.read().decode("utf-8")
            # Ask for name or use from file
            import_name = st.text_input(
                "Strategy name (optional)",
                value="",
                key=f"import_strategy_name_{m1_spec}_{m2_spec}",
                help="If empty, uses name from YAML file",
            )
            if st.button("📤 Import", key=f"btn_import_strategy_{m1_spec}_{m2_spec}"):
                import_yaml(yaml_text, import_name if import_name else None)
                st.success("Strategy imported")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to import strategy: {e}")


def render_variable_reference(m1_spec: str, m2_spec: str) -> None:
    """Render variable reference expander."""
    st.markdown(
        '<div style="margin: 8px 0 4px 0; font-size: 0.72rem; color: var(--arena-text-muted); font-weight:600;">KEY VARIABLES</div>',
        unsafe_allow_html=True,
    )
    # Show mandatory category chips
    mandatory_color = "{color}"
    mandatory_board = "{fen}"
    mandatory_move = "{legal_moves_uci}"
    st.markdown(
        f'<span class="sb-var-chip">{mandatory_color}</span><span class="sb-var-chip">{mandatory_board}</span><span class="sb-var-chip">{mandatory_move}</span>'
        '<span class="sb-var-chip-info">→ <a href="#variable-reference" style="color:var(--arena-link); text-decoration:none;">Reference</a></span>',
        unsafe_allow_html=True,
    )

    # Variable reference expander
    with st.expander("📚 Variable Reference", expanded=False):
        # Build sample context once per session
        if "variable_sample_context" not in st.session_state:
            sample_board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4")  # Middlegame after 1.e4 c5 2.Nf3 d6 3.d4 cxd4
            try:
                # Create a dummy template to get all variables
                dummy_template, _ = create_safe_prompt_template(DEFAULT_SYSTEM_PROMPT, DEFAULT_TURN_PROMPT)
                st.session_state.variable_sample_context = build_sample_context(
                    sample_board,
                    move_history=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQK1R b KQkq - 0 1"],
                    reasoning_level="high",
                    stagnation_threshold=3,
                    prompt_template=dummy_template,
                )
            except Exception:
                st.session_state.variable_sample_context = {}

        # Category filter
        categories = ["Mandatory Player Side", "Mandatory Board State", "Mandatory Legal Moves", "Rich Context"]
        selected_category = st.selectbox(
            "Filter by category",
            options=["All", *categories],
            index=0,
            key=f"variable_category_filter_{m1_spec}_{m2_spec}",
        )

        # Display variables
        filtered_vars = {}
        for var, meta in KNOWN_VARIABLES.items():
            if selected_category == "All" or meta["category"] == selected_category:
                filtered_vars[var] = meta

        # Create a table
        if filtered_vars:
            # Prepare data for display
            table_data = []
            for var, meta in filtered_vars.items():
                sample_val = st.session_state.variable_sample_context.get(var, "[sample]")
                table_data.append({
                    "Variable": f"{{{var}}}",
                    "Category": meta["category"],
                    "Description": meta["description"],
                    "Sample Value": str(sample_val)[:50] + ("..." if len(str(sample_val)) > 50 else ""),
                })
            
            # Display the dataframe once
            st.dataframe(
                table_data,
                hide_index=True,
                width="stretch",
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Sample Value": st.column_config.TextColumn("Sample Value", width="medium"),
                },
            )
            
            # Insert buttons (only once, outside the loop)
            st.markdown("**Insert variable into active tab's turn prompt:**")
            col_var1, col_var2 = st.columns([3, 1])
            insert_var = col_var1.selectbox(
                "Select variable to insert",
                options=["-- Select --", *list(filtered_vars.keys())],
                index=0,
                key=f"insert_variable_select_{m1_spec}_{m2_spec}",
            )
            if col_var2.button("➕ Insert", key=f"btn_insert_variable_{m1_spec}_{m2_spec}") and insert_var != "-- Select --":  # noqa: RUF001
                # Determine active tab (we need to know which tab is active)
                # Since we can't easily get the active tab from st.tabs, we'll use a session state to track
                # For simplicity, we'll insert into the last active tab stored in session state
                # We'll set this in the tab callbacks below
                active_tab = st.session_state.get("active_prompt_tab", "P1")
                if active_tab == "P1":
                    current_turn = st.session_state[f"turn_prompt_{m1_spec}"]
                    st.session_state[f"turn_prompt_{m1_spec}"] = current_turn + "{" + insert_var + "}"
                else:
                    current_turn = st.session_state[f"turn_prompt_{m2_spec}"]
                    st.session_state[f"turn_prompt_{m2_spec}"] = current_turn + "{" + insert_var + "}"
                st.rerun()
        else:
            st.info("No variables match the selected filter.")
