"""Arena theme injection for the Streamlit app.

Injects the design tokens from DESIGN.md as CSS custom properties and applies
them to the Streamlit surface (background, sidebar, cards, typography, focus
ring). The tokens stay scoped to the app container so we never fight
Streamlit's own theme in surprising ways.

All visual identities route through here — every other UI module consumes
the same tokens, so changing a value here re-skins the entire app.
"""

from __future__ import annotations

import streamlit as st

_ARENA_CSS = """
:root {
    --arena-bg: #0a0c14;
    --arena-bg-elevated: #11141f;
    --arena-bg-inset: #070810;
    --arena-border: rgba(255,255,255,0.08);
    --arena-border-strong: rgba(255,255,255,0.14);
    --arena-text: #e8eaf2;
    --arena-text-muted: #8a90a5;
    --arena-text-faint: #545a6e;
    --arena-accent: #f0b421;
    --arena-accent-hot: #ff7a45;
    --arena-white: #f5f0e1;
    --arena-black: #3a2d20;
    --arena-good: #3fb950;
    --arena-blunder: #f85149;
    --arena-mistake: #d29922;
    --arena-inaccuracy: #db6d28;

    --q-best: #2ecc71;
    --q-excellent: #27ae60;
    --q-good: #9ecd43;
    --q-inaccuracy: #db6d28;
    --q-mistake: #e67e22;
    --q-blunder: #e74c3c;

    --r-pill: 999px;
    --r-sm: 6px;
    --r-md: 10px;
    --r-lg: 16px;

    --shadow-card: 0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(0,0,0,0.4);
    --shadow-board: 0 0 0 1px rgba(255,255,255,0.06), 0 24px 64px rgba(0,0,0,0.6);
    --glow-accent: 0 0 24px rgba(240,180,33,0.35);

    --font-mono: "JetBrains Mono", "Fira Code", ui-monospace, monospace;

    --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
    --dur-fast: 120ms;
    --dur-med: 260ms;
    --dur-slow: 520ms;
    --dur-arena: 7s;
}

/* App canvas — deep dark with cool tint */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--arena-bg) !important;
    background-image:
        radial-gradient(circle at 20% 0%, rgba(240,180,33,0.04) 0%, transparent 45%),
        radial-gradient(circle at 80% 100%, rgba(79,184,255,0.03) 0%, transparent 50%);
    background-attachment: fixed;
}

/* Ambient arena pulse on the background — slow, low amplitude */
@keyframes cf-arena-pulse {
    0%, 100% { opacity: 0.85; }
    50% { opacity: 1; }
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(80% 60% at 50% 0%, rgba(240,180,33,0.06) 0%, transparent 60%),
        radial-gradient(60% 50% at 50% 100%, rgba(110,80,255,0.04) 0%, transparent 60%);
    animation: cf-arena-pulse var(--dur-arena) var(--ease-out) infinite;
    z-index: 0;
}

/* Sidebar — elevated dark surface */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
    background: var(--arena-bg-elevated) !important;
    border-right: 1px solid var(--arena-border) !important;
}
[data-testid="stSidebar"] * { color: var(--arena-text); }
[data-testid="stSidebar"] .stHeading {
    font-weight: 600;
    letter-spacing: -0.01em;
    border-bottom: 1px solid var(--arena-border);
    padding-bottom: 8px;
    margin-bottom: 8px;
}
[data-testid="stSidebar"] hr {
    border-color: var(--arena-border);
    margin: 16px 0;
}

/* Typography upgrades */
.stApp, .stMarkdown, .stMarkdown p, .stText {
    color: var(--arena-text) !important;
    font-feature-settings: "cv05", "ss01", "tnum";
}

/* Reduce whitespace at the top of the sidebar */
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0rem !important;
}
[data-testid="stSidebarNav"] {
    padding-top: 0 !important;
}
/* Reduce whitespace at the top of main content */
.block-container {
    padding-top: 3rem !important;
}

h1, h2, h3, .stHeading {
    letter-spacing: -0.01em;
    color: var(--arena-text);
}
h1 { font-weight: 650; letter-spacing: -0.02em; }
code, .stCodeBlock, pre {
    font-family: var(--font-mono) !important;
}

/* Cards — strip Streamlit default container chrome and apply arena card surface */
.element-container:has(> .cf-card-host) {
    background: transparent !important;
    border: none !important;
}
.cf-card {
    background: var(--arena-bg-elevated);
    border: 1px solid var(--arena-border) !important;
    border-radius: var(--r-md) !important;
    padding: 24px !important;
    box-shadow: var(--shadow-card) !important;
}
.cf-card-compact { padding: 16px; }
.cf-card:hover { border-color: var(--arena-border-strong) !important; }

/* Metric cards — typographic emphasis, tabular nums */
[data-testid="stMetric"] {
    background: var(--arena-bg-elevated);
    border: 1px solid var(--arena-border);
    border-radius: var(--r-md);
    padding: 12px 16px !important;
    box-shadow: var(--shadow-card);
}
[data-testid="stMetricValue"] {
    color: var(--arena-text) !important;
    font-variant-numeric: tabular-nums;
    font-weight: 650;
    letter-spacing: -0.01em;
}
[data-testid="stMetricLabel"] {
    color: var(--arena-text-muted) !important;
    font-size: 0.8125rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
[data-testid="stMetricDelta"] { color: var(--arena-accent) !important; }

/* Buttons — amber primary, ghost default */
.stButton > button {
    border-radius: var(--r-sm) !important;
    font-weight: 600 !important;
    letter-spacing: 0.005em;
    transition: transform var(--dur-fast) var(--ease-out), border-color var(--dur-fast), background var(--dur-fast) !important;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"], .stButton > button[data-kind="primary"] {
    background: linear-gradient(180deg, #f0b421 0%, #d99a14 100%) !important;
    border-color: #d99a14 !important;
    color: #0a0c14 !important;
    box-shadow: 0 0 0 1px rgba(240,180,33,0.4), 0 8px 20px rgba(240,180,33,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: var(--glow-accent), 0 0 0 1px rgba(240,180,33,0.6) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--arena-bg-elevated) !important;
    border: 1px solid var(--arena-border) !important;
    color: var(--arena-text) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--arena-border-strong) !important;
    background: var(--arena-bg-elevated) !important;
}
.stButton > button:focus-visible {
    outline: 2px solid rgba(240,180,33,0.6) !important;
    outline-offset: 2px;
}

/* Inputs */
[data-testid="stTextInput input"], [data-testid="stSelectbox"], .stSelectbox, .stNumberInput input {
    background: var(--arena-bg-inset) !important;
    border: 1px solid var(--arena-border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--arena-text) !important;
}
[data-testid="stTextInput input"]:focus, [data-testid="stSelectbox"]:focus-within, .stNumberInput input:focus {
    border-color: var(--arena-accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(240,180,33,0.2) !important;
}

/* Expanders — restyle as cards */
[data-testid="stExpander"] {
    background: var(--arena-bg-elevated) !important;
    border: 1px solid var(--arena-border) !important;
    border-radius: var(--r-md) !important;
    overflow: hidden;
}
[data-testid="stExpander"] > details > summary {
    color: var(--arena-text) !important;
    font-weight: 600 !important;
    padding: 14px 16px !important;
}
[data-testid="stExpander"] > details > summary:hover { background: rgba(255,255,255,0.02); }
[data-testid="stExpander"] > details[open] > summary {
    border-bottom: 1px solid var(--arena-border);
}

/* Dataframe tables — slim, dark */
[data-testid="stDataFrame"] {
    border: 1px solid var(--arena-border) !important;
    border-radius: var(--r-md) !important;
    overflow: hidden;
    background: var(--arena-bg-elevated) !important;
}
[data-testid="stDataFrame"] * { background: transparent !important; }

/* Progress bars */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--arena-accent) 0%, var(--arena-accent-hot) 100%) !important;
}

/* SVG board — give it a dimensional frame */
.cf-board-frame {
    background: var(--arena-bg-inset);
    padding: 12px;
    border-radius: var(--r-lg);
    border: 1px solid var(--arena-border);
    box-shadow: var(--shadow-board);
    display: inline-block;
}
.cf-board-frame svg { display: block; border-radius: 8px; }

/* Quality pills */
.cf-quality-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: var(--r-pill);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    border: 1px solid;
}
.cf-pill-best { color: var(--q-best); background: rgba(46,204,113,0.14); border-color: rgba(46,204,113,0.45); }
.cf-pill-excellent { color: var(--q-excellent); background: rgba(39,174,96,0.14); border-color: rgba(39,174,96,0.45); }
.cf-pill-good { color: var(--q-good); background: rgba(158,205,67,0.14); border-color: rgba(158,205,67,0.45); }
.cf-pill-inaccuracy { color: var(--q-inaccuracy); background: rgba(219,109,40,0.14); border-color: rgba(219,109,40,0.45); }
.cf-pill-mistake { color: var(--q-mistake); background: rgba(230,126,34,0.14); border-color: rgba(230,126,34,0.45); }
.cf-pill-blunder { color: var(--q-blunder); background: rgba(231,76,60,0.14); border-color: rgba(231,76,60,0.5); }

/* Player banner */
.cf-player {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    border-radius: var(--r-md);
    background: var(--arena-bg-elevated);
    border: 1px solid var(--arena-border);
}
.cf-player.active {
    border-color: var(--arena-accent);
    box-shadow: 0 0 0 1px rgba(240,180,33,0.4), 0 4px 18px rgba(240,180,33,0.15);
}
.cf-player-avatar {
    width: 36px;
    height: 36px;
    border-radius: var(--r-pill);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.cf-avatar-white { background: rgba(245,240,225,0.18); color: var(--arena-text); border: 1px solid rgba(245,240,225,0.5); }
.cf-avatar-black { background: rgba(58,45,32,0.45); color: var(--arena-text); border: 1px solid rgba(255,255,255,0.2); }
.cf-player-meta { display: flex; flex-direction: column; min-width: 0; }
.cf-player-name { font-weight: 600; color: var(--arena-text); font-size: 0.875rem; }
.cf-player-spec { font-family: var(--font-mono); font-size: 0.75rem; color: var(--arena-text-muted); }
.cf-turn-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--arena-accent); margin-right: 8px;
    box-shadow: 0 0 12px rgba(240,180,33,0.6);
    animation: cf-dot-pulse 1.4s var(--ease-out) infinite;
}
@keyframes cf-dot-pulse { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(0.85); opacity: 0.6; } }

/* Skeleton loading shimmer */
.cf-skeleton {
    height: 36px;
    border-radius: var(--r-sm);
    background: linear-gradient(90deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.10) 40%, rgba(255,255,255,0.04) 80%);
    background-size: 240% 100%;
    animation: cf-shimmer 1.4s linear infinite;
    margin-bottom: 8px;
}
@keyframes cf-shimmer { from { background-position: 200% 50%; } to { background-position: -200% 50%; } }

/* Hero */
.cf-hero {
    position: relative;
    padding: 64px 24px 80px;
    text-align: center;
    background: radial-gradient(circle at center, rgba(240, 180, 33, 0.05) 0%, transparent 60%);
    border-radius: var(--r-lg);
    margin-bottom: 32px;
}
.cf-hero-title {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.05;
    margin: 0 auto 16px;
    background: linear-gradient(135deg, #f5f0e1 0%, #f0b421 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 8px 32px rgba(240, 180, 33, 0.25);
    animation: cf-title-float 4s ease-in-out infinite alternate;
}
@keyframes cf-title-float {
    0% { transform: translateY(0); }
    100% { transform: translateY(-6px); }
}
.cf-hero-sub {
    font-size: 1.15rem;
    color: var(--arena-text-muted);
    max-width: 680px;
    margin: 0 auto 40px;
    line-height: 1.6;
    text-align: center;
}
.cf-hero-board {
    display: flex;
    justify-content: center;
    margin: 40px 0;
    perspective: 1200px;
}
.cf-hero-board svg {
    transform: rotateX(8deg);
    filter: drop-shadow(0 24px 48px rgba(0,0,0,0.5));
    transition: transform var(--dur-med) var(--ease-out);
}
.cf-chip-row { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin: 32px 0; }
.cf-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: var(--r-pill);
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--arena-border);
    color: var(--arena-text-muted);
    font-size: 0.85rem;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: transform var(--dur-med) var(--ease-out), border-color var(--dur-med), background var(--dur-med), box-shadow var(--dur-med);
}
.cf-chip:hover {
    transform: translateY(-3px);
    border-color: var(--arena-accent);
    background: rgba(240, 180, 33, 0.08);
    color: var(--arena-text);
    box-shadow: 0 6px 16px rgba(240, 180, 33, 0.15);
}
.cf-chip strong { color: var(--arena-text); font-weight: 700; }

/* Metric Cards */
.cf-metric-card {
    transition: transform var(--dur-med) var(--ease-out), border-color var(--dur-med), box-shadow var(--dur-med);
}
.cf-metric-card:hover {
    transform: translateY(-4px);
    border-color: var(--arena-accent);
    box-shadow: 0 12px 40px rgba(240, 180, 33, 0.2);
}

/* Section divider */
.cf-section {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px;
}
.cf-section-title {
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
    color: var(--arena-text);
}
.cf-section-sub {
    color: var(--arena-text-muted);
    margin-bottom: 24px;
    font-size: 0.95rem;
}

/* Demo marker badge */
.cf-demo-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 10px;
    border-radius: var(--r-pill);
    background: linear-gradient(90deg, rgba(240,180,33,0.18), rgba(255,122,69,0.18));
    border: 1px solid rgba(240,180,33,0.4);
    color: var(--arena-accent);
    font-size: 0.6875rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Eval bar — paired with the board */
.cf-evalbar {
    width: 14px;
    height: 480px;
    background: var(--arena-white);
    border-radius: var(--r-sm);
    position: relative;
    overflow: hidden;
    border: 1px solid var(--arena-border-strong);
    display: flex;
    align-items: flex-start;
}
.cf-evalbar-fill-black {
    width: 100%;
    background: var(--arena-black);
    transform-origin: top;
    transition: transform var(--dur-med) var(--ease-out);
}

/* Move ticker — horizontal scroller of move pills */
.cf-move-ticker {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding: 8px 4px 4px;
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
}
.cf-move-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: var(--r-pill);
    background: var(--arena-bg-elevated);
    border: 1px solid var(--arena-border);
    color: var(--arena-text);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    white-space: nowrap;
    scroll-snap-align: start;
    transition: border-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast);
}
.cf-move-pill:hover { border-color: var(--arena-border-strong); }
.cf-move-pill.cf-move-current {
    border-color: var(--arena-accent);
    box-shadow: 0 0 0 2px rgba(240,180,33,0.3);
}

/* Thinking trace drawer */
.cf-thinking-drawer summary {
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    color: var(--arena-text-muted);
    cursor: pointer;
}
.cf-thinking-drawer pre {
    background: var(--arena-bg-inset);
    border: 1px solid var(--arena-border);
    border-radius: var(--r-sm);
    padding: 12px;
    max-height: 300px;
    overflow: auto;
    font-size: 0.8125rem;
    line-height: 1.5;
}

/* Accessibility — keyboard focus */
:focus-visible {
    outline: 2px solid rgba(240,180,33,0.6);
    outline-offset: 2px;
    border-radius: var(--r-sm);
}

/* Reduce motion preference — no animation churn */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Mobile (<640px) — board scales, panels stack */
@media (max-width: 640px) {
    .cf-hero-title { font-size: 2rem; }
    .cf-board-frame { padding: 6px; }
    .cf-evalbar { height: 70vw; }
}

/* Run summary card */
.cf-run-summary {
    margin-bottom: 16px;
}
.cf-run-summary h3 { margin-top: 0; margin-bottom: 4px; }

/* Metrics grid for live game screen */
.cf-metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

/* Advantage bar for eval timeline */
.cf-advantage-bar {
    height: 8px;
    background: linear-gradient(90deg, var(--arena-black) 0%, var(--arena-black) 50%, var(--arena-white) 50%, var(--arena-white) 100%);
    border-radius: var(--r-sm);
    margin-bottom: 12px;
    position: relative;
    border: 1px solid var(--arena-border-strong);
    overflow: hidden;
}
.cf-advantage-bar::after {
    content: "";
    position: absolute;
    left: var(--advantage-pct, 50%);
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--arena-accent);
    box-shadow: 0 0 8px var(--arena-accent);
}

/* Export buttons as ghost CTA pills */
.cf-export-buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.cf-export-buttons .stButton > button {
    background: var(--arena-bg-elevated) !important;
    border: 1px solid var(--arena-border) !important;
    color: var(--arena-text-muted) !important;
}
.cf-export-buttons .stButton > button:hover {
    border-color: var(--arena-accent) !important;
    color: var(--arena-accent) !important;
    background: rgba(240,180,33,0.1) !important;
}

/* Heatmap cell styling for Altair charts */
.cf-heatmap-cell {
    transition: transform var(--dur-fast) var(--ease-out);
}
.cf-heatmap-cell:hover {
    transform: scale(1.05);
}

/* Radar chart container */
.cf-radar-chart {
    width: 100%;
    max-width: 500px;
    margin: 0 auto;
}

/* Win-rate inline bar container */
.cf-winrate-bar {
    height: 20px;
    background: var(--arena-bg-inset);
    border-radius: var(--r-sm);
    overflow: hidden;
    position: relative;
}
.cf-winrate-bar-fill {
    height: 100%;
    border-radius: var(--r-sm);
    transition: width var(--dur-med) var(--ease-out);
}

/* Move ticker auto-scroll helper */
.cf-move-ticker {
    scroll-behavior: smooth;
}
"""


def apply_arena_theme() -> None:
    """Inject the arena theme CSS into the current Streamlit run.

    Idempotent: Streamlit's st.markdown html is short-circuited on each rerun,
    so this can be called once per script run without polluting the page.
    """
    st.markdown(f"<style>{_ARENA_CSS}</style>", unsafe_allow_html=True)


__all__ = ["apply_arena_theme"]
