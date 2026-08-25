"""
Simplified variable reference for the Prompt Workbench.
"""
from __future__ import annotations

import streamlit as st


def render_variable_reference(m1_spec: str, m2_spec: str) -> None:
    """Render simplified variable reference - just the 4 mandatory variables."""
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
        '<span class="sb-var-chip-info">→ {ascii_board}</span>',
        unsafe_allow_html=True,
    )

    with st.expander("📚 Variable Reference", expanded=False):
        st.markdown("""
| Variable | Description |
|----------|-------------|
| `{color}` | Player color: `white` or `black` |
| `{fen}` | Full FEN string of current position |
| `{ascii_board}` | ASCII board diagram |
| `{legal_moves_uci}` | Space-separated list of legal moves in UCI notation |
""")


# Stub for compatibility - strategy store removed
def render_strategy_store(m1_spec: str, m2_spec: str) -> None:
    """Strategy store feature removed in simplified version."""

