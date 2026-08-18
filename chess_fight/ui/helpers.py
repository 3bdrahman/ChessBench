"""Reusable UI helpers shared across the Streamlit app.

Provides:
- HTML/CSS composite widgets (cards, pills, eval bar, board frame, skeletons)
- Aggregated metric rows pulled from real run data (no fabrication)
- Move-quality badges rendered as semantic pills (text + color, never color alone)

Every helper routes through the design tokens in :mod:`chess_fight.ui.theme`
so the entire app stays on-brand.
"""

from __future__ import annotations

from typing import Any

import chess
import chess.svg
import streamlit as st

from chess_fight.benchmark.results_view import (
    aggregate_leaderboard,
    list_runs,
)

QUALITY_HUES: dict[str, str] = {
    "best": "best",
    "excellent": "excellent",
    "good": "good",
    "inaccuracy": "inaccuracy",
    "mistake": "mistake",
    "blunder": "blunder",
}


def quality_pill_html(quality: str | None) -> str:
    """Render a move-quality pill as inline HTML.

    Always carries the quality word (color alone is not accessible). Returns
    an empty span for unknown qualities so callers can compose without
    conditional branches.
    """
    if not quality or quality not in QUALITY_HUES:
        return ""
    cls = QUALITY_HUES[quality]
    label = quality.upper()
    return f'<span class="cf-quality-pill cf-pill-{cls}">{label}</span>'


def render_html_card(klass: str, title: str | None, body_html: str) -> None:
    """Render an arbitrary HTML card with optional title header."""
    title_html = f'<div style="font-weight:600;font-size:0.95rem;margin-bottom:8px;color:var(--arena-text)">{title}</div>' if title else ""
    st.markdown(f'<div class="{klass}">{title_html}{body_html}</div>', unsafe_allow_html=True)


def framed_board_html(
    board: chess.Board,
    *,
    size: int = 600,
    lastmove: chess.Move | None = None,
    check_square: int | None = None,
    flipped: bool = False,
) -> str:
    """Render a chess board SVG wrapped in the arena frame + optional eval bar slot.

    The frame is a single .cf-board-frame wrapper. Callers can wrap this in
    a flex layout (with an eval bar) — see render_board_with_evalbar.
    """
    svg = chess.svg.board(
        board,
        size=size,
        lastmove=lastmove,
        check=check_square,
        flipped=flipped,
    )
    return f'<div class="cf-board-frame">{svg}</div>'


def render_board_with_evalbar(
    board: chess.Board,
    *,
    size: int = 480,
    lastmove: chess.Move | None = None,
    check_square: int | None = None,
    cp_score: int | None = None,
    mate_in: int | None = None,
    flipped: bool = False,
) -> None:
    """Render the board + a thin eval bar to its left.

    Eval bar semantics (DESIGN.md § 2):
    - White advantage fills the bar white (fill from top).
    - Black advantage fills the bar black (fill from bottom).
    - A mate score fills solid white with a "M{N}" glyph.
    - The fill height animates on transition via --dur-med --ease-out.
    """
    if mate_in is not None and mate_in > 0:
        black_pct = 0.0
        eval_label = f"M{mate_in}"
    elif mate_in is not None and mate_in < 0:
        black_pct = 100.0
        eval_label = f"-M{abs(mate_in)}"
    elif cp_score is None:
        black_pct = 50.0
        eval_label = ""
    else:
        # Map cp_score (typically -1000..1000) to 0..100 black fill.
        # Use a sigmoid-ish squash so big advantages saturate near 100/0.
        clamped = max(-2000, min(2000, cp_score))
        # Logistic squash: pct = 100 / (1 + exp(cp/350))
        import math
        white_pct = 100.0 / (1.0 + math.exp(-clamped / 350.0))
        black_pct = 100.0 - white_pct
        eval_label = f"{'+' if cp_score >= 0 else ''}{cp_score / 100:.2f}"

    html = f"""
    <div style="display:flex;gap:8px;align-items:stretch;justify-content:center">
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px">
            <div class="cf-evalbar" aria-label="Stockfish advantage: {eval_label or 'even'}">
                <div class="cf-evalbar-fill-black" style="transform:scaleY({black_pct/100.0})"></div>
            </div>
            <span style="font-family:var(--font-mono);font-size:0.7rem;color:var(--arena-text-muted);font-variant-numeric:tabular-nums">{eval_label}</span>
        </div>
        {framed_board_html(board, size=size, lastmove=lastmove, check_square=check_square, flipped=flipped)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_skeleton_rows(n: int = 3) -> None:
    """Render `n` shimmering skeleton rows for a loading state."""
    rows = "".join('<div class="cf-skeleton"></div>' for _ in range(n))
    st.markdown(f'<div style="padding:8px 0">{rows}</div>', unsafe_allow_html=True)


def render_loading_card(title: str, stage: str, provider: str | None = None, container: Any = None) -> None:
    """Render a staged loading card with skeleton rows.

    `stage` is the current status line displayed above the skeleton. The card
    is the anchored element on the page while a provider call is in flight —
    it never collapses silently to "No models available" when the call goes slow.

    `container` is the Streamlit container to render into (e.g., `st.sidebar`).
    Defaults to the main `st` module.
    """
    if container is None:
        import streamlit as st
        container = st

    provider_label = f" · {provider.capitalize()}" if provider else ""
    body = f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
            <span class="cf-turn-dot"></span>
            <span style="color:var(--arena-text);font-weight:600;font-size:0.875rem">{title}</span>
        </div>
        <div style="color:var(--arena-text-muted);font-size:0.8125rem;margin-bottom:14px">{stage}{provider_label}</div>
    """
    rows = "".join('<div class="cf-skeleton"></div>' for _ in range(3))
    container.markdown(
        f'<div class="cf-card cf-card-compact">{body}<div>{rows}</div></div>',
        unsafe_allow_html=True,
    )


def render_move_ticker(moves: list[Any], current_ply: int | None = None) -> None:
    """Render the horizontal move ticker with quality pills.

    `moves` are expected to have `.move_san`, `.is_capture`, `.is_check`,
    and optionally `.move_quality` (for completed games with full data).
    """
    if not moves:
        return

    pills = []
    for i, m in enumerate(moves):
        san = getattr(m, "move_san", None) or getattr(m, "move", "?")
        capture = "x" if getattr(m, "is_capture", False) else ""
        check = "+" if getattr(m, "is_check", False) else ""
        label = f"{capture}{san}{check}"

        # Quality pill if available (from completed games with full JSONL data)
        quality = getattr(m, "move_quality", None)
        q_html = quality_pill_html(quality)

        # Current ply highlight
        active = " cf-move-current" if current_ply is not None and i == current_ply else ""
        pills.append(
            f'<span class="cf-move-pill{active}">{label}{q_html}</span>'
        )

    html = f"""
    <div class="cf-move-ticker" role="list" aria-label="Move history">
        {''.join(pills)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_thinking_trace_drawer(state: Any) -> None:
    """Render the collapsible thinking trace drawer under the live board.

    Shows a one-line summary when collapsed, full trace when expanded.
    """
    cr = getattr(state, "last_completion_result", None)
    if not cr or not cr.text:
        return

    thinking = cr.text
    char_count = len(thinking)
    word_count = len(thinking.split())
    # Check for structured reasoning indicators
    structured = any(w in thinking.lower() for w in ("1.", "2.", "first", "then", "because", "therefore"))

    summary = f"Thinking… {char_count} chars · {word_count} words · {'structured' if structured else 'freeform'}"

    with st.expander(summary, expanded=False):
        st.code(thinking, language="text")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Latency", f"{cr.latency_ms or 0} ms")
        with cols[1]:
            st.metric("Tokens in", f"{cr.tokens_in or 0}")
        with cols[2]:
            st.metric("Tokens out", f"{cr.tokens_out or 0}")


def headline_metric_row(runs_root: str = "runs") -> None:
    """Render the 4-up headline metric strip on the landing page.

    Numbers are aggregated across all real runs under `runs_root`. When the
    directory is empty (no runs yet), each metric reads "—" honestly.
    """
    runs = list_runs(runs_root)
    if not runs:
        cards = [
            ("Runs", "—"),
            ("Games played", "—"),
            ("Moves recorded", "—"),
            ("Models benchmarked", "—"),
        ]
    else:
        total_runs = len(runs)
        total_games = sum(r.total_games for r in runs)
        total_moves = sum(r.total_moves for r in runs)
        leaderboard = aggregate_leaderboard(runs)
        cards = [
            ("Runs", f"{total_runs}"),
            ("Games played", f"{total_games}"),
            ("Moves recorded", f"{total_moves}"),
            ("Models benchmarked", f"{len(leaderboard)}"),
        ]

    import textwrap
    cells = "".join(
        textwrap.dedent(f"""
            <div class="cf-metric-card" style="flex:1;min-width:140px;display:flex;flex-direction:column;gap:8px;padding:24px;background:linear-gradient(145deg, var(--arena-bg-elevated) 0%, rgba(20,24,35,0.8) 100%);border:1px solid var(--arena-border);border-radius:var(--r-lg);box-shadow:0 8px 32px rgba(0,0,0,0.3);backdrop-filter:blur(10px);">
                <div style="font-size:2.25rem;font-weight:750;letter-spacing:-0.02em;color:var(--arena-text);font-variant-numeric:tabular-nums;line-height:1.1;text-shadow:0 0 16px rgba(255,255,255,0.1)">{value}</div>
                <div style="font-size:0.8rem;color:var(--arena-accent);text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">{label}</div>
            </div>
        """)
        for label, value in cards
    )
    
    st.markdown(
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin:32px auto;max-width:860px">{cells}</div>',
        unsafe_allow_html=True,
    )


def player_banner_html(
    *,
    name: str,
    spec: str | None,
    color: str,
    is_turn: bool = False,
    is_winner: bool = False,
) -> str:
    """Render a player banner card (avatar + name + spec + turn indicator)."""
    active_cls = " active" if is_turn else ""
    avatar_cls = "cf-avatar-white" if color == "white" else "cf-avatar-black"
    glyph = "♔" if color == "white" else "♚"
    spec_html = spec or "&nbsp;"
    turn_dot = '<span class="cf-turn-dot"></span>' if is_turn else ""
    result_mark = ""
    if is_winner:
        res_str = "1 — 0" if color == "white" else "0 — 1"
        result_mark = f'<span style="margin-left:auto;font-family:var(--font-mono);font-size:0.75rem;color:var(--arena-good);font-weight:700">{res_str}</span>'
    return f"""
    <div class="cf-player{active_cls}">
        {turn_dot}
        <div class="cf-player-avatar {avatar_cls}">{glyph}</div>
        <div class="cf-player-meta">
            <div class="cf-player-name">{name}</div>
            <div class="cf-player-spec">{spec_html}</div>
        </div>
        {result_mark}
    </div>
    """


def demo_badge_html() -> str:
    """Render the demo-run marker badge."""
    return '<span class="cf-demo-badge">★ Demo Run</span>'


def is_demo_run_summary(summary: Any) -> bool:
    """Return True if a run's parsed summary says is_demo=True."""
    if summary is None:
        return False
    config = getattr(summary, "config", None) or {}
    if isinstance(config, dict) and config.get("is_demo"):
        return True
    # summary.json also stores is_demo at root for demo runs
    import json
    from pathlib import Path
    if isinstance(getattr(summary, "run_dir", None), Path):
        try:
            sp = summary.run_dir / "summary.json"
            if sp.is_file():
                data = json.loads(sp.read_text(encoding="utf-8"))
                return bool(data.get("is_demo"))
        except (OSError, json.JSONDecodeError):
            return False
    return False


__all__ = [
    "demo_badge_html",
    "framed_board_html",
    "headline_metric_row",
    "is_demo_run_summary",
    "player_banner_html",
    "quality_pill_html",
    "render_board_with_evalbar",
    "render_html_card",
    "render_loading_card",
    "render_skeleton_rows",
]
