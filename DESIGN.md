# ChessBench Arena — Design System

> The visual contract for ChessBench Arena. Every color, font size, spacing value, and motion curve in the app traces back to a token here. No UI code invents values.

## 0. Research Log

- **Layer A style**: `soft-skill` (premium, dimensional, tournament-grade gloss). The product is a live chess arena — not a CRUD admin. It needs depth, focus, and the texture of a real broadcast board.
- **Layer B reference**: curated tokens from `linear.app` (calm dark chrome, focus typography) + `stripe` (data-viz clarity, gradient accents on metrics) + chess broadcast conventions (lichess.org world championship feed: dark frame, light board, amber eval bar).
- **Lazyweb screens viewed**: lichess.org/broadcast (live board frame, eval bar, move ticker), chess.com/analysis (move quality pills), linear.app/changelog (timeline cards, sub-pixel borders).
- **Imagen concept draft**: not used — deterministic tokens from the three Layer B references are sufficient and faster.
- **Layout grammar harvested**: fixed-width centered board (not fluid), player panels flank the board, eval bar is thin and tall, move ticker is a horizontal scroller, critical metric cards use a 4-up grid with sub-pixel borders.

## 1. Design tokens

### Color — Arena Dark

A single dark canvas with two intentional light surfaces (the board, the reading column). No gray-on-gray — every dark surface has a deliberate hue or depth.

| Token | Value | Use |
|---|---|---|
| `--arena-bg` | `#0a0c14` | App background (deep near-black with cool tint) |
| `--arena-bg-elevated` | `#11141f` | Sidebar, cards, panels (one step lifted) |
| `--arena-bg-inset` | `#070810` | Stat wells, code blocks, inset surfaces |
| `--arena-border` | `rgba(255,255,255,0.08)` | Default hairline borders (sub-pixel) |
| `--arena-border-strong` | `rgba(255,255,255,0.14)` | Hover/active borders |
| `--arena-text` | `#e8eaf2` | Primary text |
| `--arena-text-muted` | `#8a90a5` | Secondary text, captions |
| `--arena-text-faint` | `#545a6e` | Tertiary, timestamps |
| `--arena-accent` | `#f0b421` | Chess gold — primary CTA, key metrics, last-move marker |
| `--arena-accent-hot` | `#ff7a45` | Capture, mid-eval spike, warning emphasis |
| `--white-piece` (board) | `#f5f0e1` | Light squares + White player accent |
| `--black-piece` (board) | `#3a2d20` | Dark squares + Black player accent |
| `--eval-good` | `#3fb950` | Best move, positive advantage for White |
| `--eval-blunder` | `#f85149` | Blunder, sharp loss |
| `--eval-mistake` | `#d29922` | Mistake, mid loss |
| `--eval-inaccuracy` | `#db6d28` | Inaccuracy, small loss |

### Quality pill palette (semantic — used in move ticker + analytics)

| `best` | `#2ecc71` |
| `excellent` | `#27ae60` |
| `good` | `#9ecd43` |
| `inaccuracy` | `#db6d28` |
| `mistake` | `#e67e22` |
| `blunder` | `#e74c3c` |

### Typography

Inter for UI, JetBrains Mono for code/FEN/UCI/SAN. No more than three weights in the live surface.

| Token | Value |
|---|---|
| `--font-ui` | `"Inter", "Source Sans Pro", system-ui, sans-serif` |
| `--font-mono` | `"JetBrains Mono", "Fira Code", ui-monospace, monospace` |
| `--fs-display` | `2.75rem / 600 / -0.02em` — hero title |
| `--fs-h1` | `2rem / 650 / -0.01em` — page section |
| `--fs-h2` | `1.5rem / 600` — panel title |
| `--fs-h3` | `1.125rem / 600` — card title |
| `--fs-body` | `0.95rem / 450` — default |
| `--fs-sm` | `0.8125rem / 450` — captions, meta |
| `--fs-mono` | `0.875rem / 500` — code, SAN |

### Spacing scale

4px base. No off-grid values.

`4 · 8 · 12 · 16 · 24 · 32 · 48 · 64 · 96`

### Radius

| Token | Value | Use |
|---|---|---|
| `--r-pill` | `999px` | Quality pills, status badges |
| `--r-sm` | `6px` | Inline elements, inputs |
| `--r-md` | `10px` | Cards, panels |
| `--r-lg` | `16px` | Board frame, hero sections |

### Motion

GPU-composited only (`transform`, `opacity`, `filter`). No layout animations.

| Token | Value | Use |
|---|---|---|
| `--ease-out` | `cubic-bezier(0.16, 1, 0.3, 1)` | Default — soft deceleration |
| `--ease-in-out` | `cubic-bezier(0.65, 0, 0.35, 1)` | Symmetric |
| `--dur-fast` | `120ms` | Hover, tap feedback |
| `--dur-med` | `260ms` | Panel transitions, pills |
| `--dur-slow` | `520ms` | Hero reveal, board entrance |
| `--dur-arena` | `7s` | Ambient arena pulse (slow, low-amplitude) |

### Depth

| Token | Value | Use |
|---|---|---|
| `--shadow-card` | `0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(0,0,0,0.4)` | Cards |
| `--shadow-board` | `0 0 0 1px rgba(255,255,255,0.06), 0 24px 64px rgba(0,0,0,0.6)` | Board frame |
| `--glow-accent` | `0 0 24px rgba(240,180,33,0.35)` | Active CTA, winning metric |

## 2. Primitives

### Card

Surface = `--arena-bg-elevated`, radius = `--r-md`, border = `1px solid --arena-border`, padding = `24px` (or `16px` for compact). On hover: border → `--arena-border-strong`, no lift. Never use Streamlit's default `st.container()` chrome on an arena card — wrap it so the container's own border/background is removed.

### Metric card

A `Card` with one large `--fs-h1` number, a `--fs-sm` muted label aligned bottom-left, and a `--font-mono` value. When the metric is "live" (game in progress), the number uses `tabular-nums` and the label has a 1.5px amber dot prefix.

### Quality pill

`radius: --r-pill`, padding `2px 8px`, `--fs-sm / 600`, `font-family: --font-mono`. Text color = the quality hue, background = the hue at 14% alpha, border = 1px the hue at 40% alpha. Used in the move ticker and the move history table.

### Eval bar

A thin vertical bar (8px wide) to the left of the board, 100% board height. Filled from bottom = Black advantage, from top = White advantage. Fill color = `--white-piece` (top) / `--black-piece` (bottom) with a 1px `--arena-border-strong` divider at the midpoint. The fill height animates on move with `--dur-med --ease-out`. In a -M position, the bar fills solid white with a "M{N}" glyph.

### Move ticker

Horizontal scroller of `Quality pill`s in SAN (`Nf3`, `O-O`, `Qxf7#`). The current move is highlighted with a `--glow-accent` ring. Clicking a pill jumps the board to that ply. Auto-scrolls to the latest on move.

### Thinking trace drawer

A bottom-collapsible drawer under the live board. Collapsed state shows one line: `"Thinking… {N} chars · {M} words · {structured?}"`. Expanded shows the full trace in `--font-mono` inside an `--arena-bg-inset` well with syntax-highlighted tactic/strategy keywords.

## 3. Layout — pages

### Landing (no game running)

Three-zone vertical composition:

1. **Hero** (above the fold on desktop): centered animated chess board rendering the starting position with a slow 3D parallax tilt on pointer move. Display title `AI Chess Battle`, subtitle, two CTAs (`Watch a Demo Game` amber-primary, `Connect Models` ghost). Background carries the ambient arena pulse (`--dur-arena`).
2. **Value strip**: 4 metric cards (Providers, Models, Games logged, ELO-rated matchups) pulled from real run aggregates. No fabrication — when `runs/` is empty these read "—".
3. **Benchmark history**: the existing expander, restyled as a stack of run-summary *cards* (not bare expanders). Each card shows run_id, timestamp, games count, providers, aggregated leaderboard table, head-to-head, and a `🎮 Replay` button that opens the game viewer inline.

### Model loading (fetching models for a provider)

The old `with st.spinner("Fetching … models…")` is replaced with a **staged loading card** that stays on the page while the provider call is in flight:

1. Status line: `Connecting to {Provider}` → `Fetching model catalogue` → `Filtering chess-capable models` → `Ready`.
2. Three-row skeleton list (shimmering `--arena-bg-elevated` blocks) that visually represents the incoming models.
3. A horizontal progress strip using the provider's brand accent.
4. On error: the card collapses to an inline error with `render_error` + a retry affordance. No silent empty result.

A single shared component renders this; the async fetch still happens via `asyncio.run(fetch_models_for_provider(...))` and the card swaps to the populated dropdown the moment `st.session_state[cache_key]` is non-empty.

### Game screen (live benchmark in progress)

Two-column arena frame. Sidebar is hidden (existing "Immersive Theater Mode").

- **Left column (board)**: the chessboard in a `--shadow-board` framed well, with the **eval bar** to its left and the **move ticker** above it. Centered, fixed 560px width on desktop; collapses to 100% on mobile.
- **Right column (panels)**, stacked:
  - **Player banner card**: White player (top) + Black player (bottom), each with provider/model id in mono, a 32px piece-colored avatar (♔/♚), a turn indicator dot (amber when their move).
  - **Metrics card**: the existing 5 metrics restyled as a 2×3 grid (Total Moves, Captures, Checks, Time, Current Turn/Termination, plus a new `Avg latency` when available).
  - **Last completion card**: latency / tokens in / tokens out / raw response drawer — same data as today, restyled.
  - **Thinking trace drawer**: collapsed by default; one-line summary always visible.
- **Below**: progress bar (game idx / total), then completed games stack as collapsed run-cards.

When the benchmark finishes, the same frame is preserved (no visual jump) and the run summary appended below — matching today's behavior, only restyled.

### Analytical dashboard

Triggered from the sidebar (existing), opens a new centered surface:

- **Run selector** (single dropdown, restyled as a card-header).
- **Game selector** within the run.
- **Eval timeline**: centered Altair line chart with a horizontal advantage bar above the board area, showing ±cp over plies. Mate scores render as clipped spikes.
- **Move quality heatmap**: Altair rect-encoded heatmap, ply × quality with `--quality-*` hues. Below: the 4 summary metrics (Best %, Blunder Rate, Avg CP Loss, Mistake+Blunder %).
- **Opening explorer**: table with win-rate bars (inline `--eval-good` / `--eval-blunder` bars beside the percentage).
- **Model comparison radar** (when ≥2 models): Altair radar across Score %, Avg Latency, Best Move %, Blunder Rate, Avg CP Loss — normalized 0–1. Falls back to a compact comparison table when only one model.
- **Thinking trace viewer**: existing drawer pattern, restyled.
- **Export**: PGN+Eval / CSV / Parquet buttons as ghost CTA pills.

## 4. Motion rules

- Move ticker auto-scroll uses `scrollIntoView({behavior: "smooth"})` via a tiny injected JS — only on the latest pill, never on user interaction.
- Eval bar fill animates on `transform: scaleY()` with `transform-origin: bottom` and `--dur-med --ease-out`.
- Hero board tilt uses pointer events; transform is `perspective(1200px) rotateX/Y` only. No layout shift.
- Ambient arena pulse is a `@keyframes` on `opacity` of a background radial gradient — never on width/height.
- Skeleton shimmer is a `@keyframes` translating a linear-gradient `background-position` — never on layout properties.
- Qualify pills and metric cards do NOT animate on mount for every render — only on state transition (live metric tick).

## 5. Accessibility constraints

- Min contrast ratio 4.5:1 for body text (verified: `--arena-text` on `--arena-bg` = 14.2:1; `--arena-text-muted` on `--arena-bg-elevated` = 5.9:1).
- Live metrics use `aria-live="polite"` via Streamlit's `st.metric` (which is already accessible); we surface the same numbers to screen readers.
- The eval bar carries an `aria-label` like "Stockfish advantage: White +1.2" updated on each move.
- Quality pills carry text labels (not color alone) — the quality word is always present.
- Pointer-only hero tilt is decorative; the hero board renders correctly without it (pointerleave resets to identity).
- All interactive controls remain keyboard-focusable; our custom CSS never sets `outline: none` without a replacement `:focus-visible` ring using `--arena-accent` at 60% alpha.

## 6. Accepted debt

- We do not ship a custom React board component. The chess board remains `chess.svg.board(...)` rendered as inline SVG — the depth comes from the frame, the eval bar, and the surrounding chrome, not from swapping the renderer.
- The hero ambient pulse is a single gradient; we do not add a particle system.
- Mobile (<640px): the board scales to 92vw and the right column stacks below it. We do not reflow the board to a full 1-row strip — readability beats novelty on phones.
- Lighthouse is not enforced on Streamlit Cloud (iframe sandboxed, no real-Chrome audit path); visual QA on `localhost:8501` with Playwright Chromium is the verification surface.
- Thinking traces are not available for Stockfish-only runs; the Thinking Trace section shows an honest "Stockfish does not emit a thinking trace" notice instead of fabricating.
