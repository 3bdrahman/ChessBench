"""Landing page hero for the Streamlit app.

Renders the immersive hero from DESIGN.md § 3 (Landing):
1. Centered animated chess board with 3D parallax tilt
2. Display title + subtitle value proposition
3. Provider/feature chip row
4. Headline metric strip pulled from real run aggregates (no fabrication)

The hero is purely visual — it never blocks the sidebar or the model
selectors. Selecting two models and clicking Start Match still does what
it always did; the hero just reframes that surface with intent.
"""

from __future__ import annotations

import streamlit as st

from chess_fight.ui.helpers import (
    headline_metric_row,
    render_html_card,
)


def render_hero() -> None:
    """Render the arena hero on the main page."""
    import textwrap
    hero_html = textwrap.dedent("""
        <section class="cf-hero">
            <div class="cf-hero-title">AI Chess Battle</div>
            <p class="cf-hero-sub">
                Watch frontier language models play real chess — move-by-move, with
                Stockfish-evaluated quality scoring, thinking traces, and live ELO.
                A benchmarking arena, not a demo reel.
            </p>
            <div class="cf-chip-row">
                <span class="cf-chip"><strong>8</strong>&nbsp;providers</span>
                <span class="cf-chip"><strong>Stockfish 18</strong>&nbsp;ground-truth evals</span>
                <span class="cf-chip"><strong>Glicko-2</strong>&nbsp;Bayesian ELO</span>
                <span class="cf-chip"><strong>295 ECO</strong>&nbsp;opening positions</span>
                <span class="cf-chip"><strong>Async</strong>&nbsp;live game loop</span>
            </div>
        </section>
    """)
    st.markdown(hero_html, unsafe_allow_html=True)


def render_landing_metrics(runs_root: str = "runs") -> None:
    """Render the headline metric strip below the hero.

    Pulls real numbers from `runs/` via the existing `list_runs` reader. When
    no runs exist, each card honestly reads "—" instead of fabricating.
    """
    headline_metric_row(runs_root)


def card(
    title: str,
    body_html: str,
    *,
    compact: bool = False,
) -> None:
    """Render an arena card.

    Wraps content in the `.cf-card` container from the theme so every card
    across the app shares the same surface, border, padding, and depth.
    """
    klass = "cf-card cf-card-compact" if compact else "cf-card"
    render_html_card(klass, title, body_html)


__all__ = ["card", "render_hero", "render_landing_metrics"]
