"""Chessboard.js Streamlit component wrapper."""

from pathlib import Path
from typing import Any

import chess
import streamlit as st
import streamlit.components.v1 as components

# Get the component directory
_COMPONENT_DIR = Path(__file__).parent / "chessboard"

# Declare the component
_chessboard_component = components.declare_component(
    "chessboard",
    path=str(_COMPONENT_DIR)
)


def chessboard(
    fen: str = chess.STARTING_FEN,
    orientation: str = "white",
    last_move: str | None = None,
    check_square: str | None = None,
    legal_moves: list[str] | None = None,
    show_coordinates: bool = True,
    draggable: bool = True,
    animate: bool = True,
    on_move: callable | None = None,
    key: str | None = None
) -> dict[str, Any]:
    """
    Render an interactive chessboard using chessboard.js.
    
    Args:
        fen: FEN string for the position
        orientation: "white" or "black" - board orientation
        last_move: UCI move string to highlight (e.g., "e2e4")
        check_square: Square in check (e.g., "e8")
        legal_moves: List of legal move UCI strings to highlight
        show_coordinates: Show rank/file coordinates
        draggable: Allow piece dragging
        animate: Animate piece movements
        on_move: Callback function(move_uci, source, target, piece) -> bool
        key: Unique key for the component
    
    Returns:
        Dict with component state including position, lastMove, etc.
    """

    # Prepare config
    config = {
        "position": fen,
        "orientation": orientation,
        "showCoordinates": show_coordinates,
        "draggable": draggable,
        "animate": animate,
        "pieceTheme": "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
    }

    # Call the component
    component_value = _chessboard_component(
        config=config,
        lastMove=last_move,
        checkSquare=check_square,
        legalMoves=legal_moves or [],
        clearHighlights=False,
        key=key,
        default={"position": fen, "lastMove": last_move}
    )

    return component_value or {}


def render_chessboard_js():
    """Inject the chessboard.js library and component JS."""
    # This should be called once in the app
    st.components.v1.html("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css" />
    """, height=0)


if __name__ == "__main__":
    import chess
    import streamlit as st

    st.title("Chessboard.js Demo")

    # Render chessboard.js library
    render_chessboard_js()

    # Demo board
    board = chess.Board()

    col1, col2 = st.columns([2, 1])

    with col1:
        result = chessboard(
            fen=board.fen(),
            orientation="white",
            last_move="e2e4" if board.move_stack else None,
            check_square=board.king(board.turn) if board.is_check() else None,
            legal_moves=[m.uci() for m in board.legal_moves[:5]],
            key="demo_board"
        )
        st.write("Component result:", result)

    with col2:
        st.write("**Controls**")
        if st.button("Make Move e2e4"):
            board.push_uci("e2e4")
            st.rerun()

        if st.button("Flip Board"):
            st.rerun()

        st.write("**Position:**")
        st.code(board.fen())
