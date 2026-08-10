# Development Brainstorm & Architecture Notes

**Status**: Planning document — no implementation yet
**North Star**: A production-grade, provider-agnostic benchmark for evaluating LLM chess capability that produces statistically valid, reproducible results and a compelling live demo.

---

## 1. Goal & North Star

Build the reference tool for "how well does model X play chess?" — usable by researchers, model providers, and hobbyists. The demo must be a **live streaming battle** (white vs black, user picks models from any connected provider) that runs reliably and looks professional.

**What "benchmark-quality" means here**:
- Reproducible: same model + same seed + same opening = same game (within temperature)
- Statistically valid: enough games, proper time controls, opening diversity, ELO estimation
- Provider-agnostic: user brings keys; we fetch available models dynamically
- Observable: full replay, token usage, reasoning traces, per-move timing
- Deployable: one-click to Streamlit Cloud / HF Spaces / Docker

---

## 2. Provider-Agnostic Architecture (Spine of the System)

### 2.1 Provider Abstraction Layer

```python
# providers/base.py
class ModelProvider(ABC):
    name: str # "openai", "anthropic", "google", "nim"
    requires_api_key: bool

    @abstractmethod
    async def list_models(self, api_key: str) -> list[ModelInfo]: ...

    @abstractmethod
    async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult: ...

    @abstractmethod
    def validate_key(self, api_key: str) -> bool: ...
```

```python
# providers/registry.py
PROVIDER_REGISTRY: dict[str, type[ModelProvider]] = {}
def register_provider(cls: type[ModelProvider]): ...
def get_provider(name: str) -> ModelProvider: ...
def list_providers() -> list[str]: ...
```

**Concrete providers to implement** (priority order):
| Provider | Models endpoint | Auth | Notes |
|----------|-----------------|------|-------|
| **OpenAI** | `GET /v1/models` | Bearer | Filter to `gpt-*`, `o1-*`; exclude embeddings/audio |
| **Anthropic** | `GET /v1/models` | x-api-key | Only `claude-*`; note max_tokens required |
| **Google (Vertex/Gemini API)** | `GET /v1/models` | Bearer / ADC | Filter `gemini-*`; handle safety settings |
| **NVIDIA NIM** | `GET /v1/models` | Bearer | OpenAI-compatible; self-hosted or cloud |
| **OpenRouter** | `GET /v1/models` | Bearer | Aggregator — useful fallback for 100+ models |
| **Ollama (local)** | `GET /api/tags` | none | For local Llama/Qwen/etc; optional |

### 2.2 Model Registry & Selection UI

- User enters API keys in sidebar (one per provider) → keys stored in `st.session_state`, never persisted
- On key entry: background fetch `list_models()` → cache for session
- Model selector: grouped by provider, shows context window, pricing tier, capabilities
- White/Black can each pick from **any** connected provider (cross-provider battles enabled)

### 2.3 Credential Isolation & Security

- Keys never leave browser session (Streamlit session state)
- Backend calls proxied through user's browser via `st.experimental_fragment` or direct fetch
- Option: user runs locally with `.env` for full privacy
- **No server-side key storage** — critical for trust

### 2.4 Unified ChessAI Interface

```python
# providers/chess_ai.py
class ProviderChessAI(ChessAI):
    def __init__(self, provider: ModelProvider, model_id: str, api_key: str, **params):
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.params = params # temperature, max_tokens, etc.

    async def _get_move_from_model(self, fen: str) -> str:
        prompt = self._create_prompt(fen)
        result = await self.provider.complete(self.api_key, self.model_id, [
            {"role": "user", "content": prompt}
        ], **self.params)
        return extract_move(result.text) # robust extraction (see §3)
```

**Breaking change**: `ChessAI.get_move()` becomes async. Game loop must be async (§4).

---

## 3. Critical Defects Discovered (Must Fix Before Anything Else)

| # | File:Line | Defect | Impact |
|---|-----------|--------|--------|
| **D1** | `models.py:816` | `OpenAIChessAI._get_move_from_model` returns raw `OpenAIChatCompletion` object, not `.choices[0].message.content` | **OpenAI completely broken** — every move falls back to `legal_moves[0]` (alphabetically first legal move) |
| **D2** | `models.py:728-733` | `_validate_move` strips trailing chars matching prefixes (e.g., `"e"` from `"move:"`) → corrupts UCI like `e2e4` → `e2e` | Silent move corruption; validation fails → fallback |
| **D3** | `models.py:189-195` | `_analyze_position_repetition` adds current FEN to `position_history` as side effect, but `repetitions` counts against `current_fen` which was never appended yet → always 0 | Stagnation detection never fires; repetition penalty dead |
| **D4** | `models.py:474-480` | `_analyze_defense` mutates `board.turn` directly to analyze opponent; if exception raised mid-loop, board state corrupted | Silent board corruption → invalid threat analysis |
| **D5** | `models.py:697-718` | `_create_prompt` calls `.format()` with 17 kwargs; template string only has ~10 placeholders (missing: `material_tension`, `position_dynamism`, `development_score`, `defense_analysis`, `vulnerability_analysis`, `ascii_board`) | `KeyError` at runtime — prompt rendering fails |
| **D6** | `ui.py:92-100` | Sync `while not game.is_game_over:` blocks Streamlit main thread with `time.sleep(0.5)` | UI freezes; no incremental updates; can't cancel |
| **D7** | `game.py:15-39` | `play_move()` raises on illegal move; no retry/fallback inside game loop (retries are in `get_move`) | Single model hiccup crashes entire game |

**P0 Fix Order**: D1 → D5 → D2 → D3 → D4 → D7 → D6

---

## 4. Prompt Quality & Move Output Reliability

### 4.1 Prompt Architecture (Current → Target)

| Aspect | Current | Target |
|--------|---------|--------|
| Structure | Single giant template with 17 sections | Composable: `system_prompt + position_context + move_list + instructions` |
| Move format | "UCI only" in template | Enforced by parser + few-shot examples |
| Reasoning | None exposed | Optional `<thinking>` block (strip before parse) |
| Temperature | Hardcoded 0.1 | Per-model configurable; benchmark mode = 0 |
| Context window | Unbounded (full prompt every move) | Sliding window: last N moves + current position |

### 4.2 Robust Move Extraction

```python
# providers/move_parser.py
def extract_move(text: str) -> str | None:
    """Extract UCI move from LLM output. Returns None if not found."""
    # 1. Strip <thinking>...</thinking> blocks
    # 2. Find UCI pattern: [a-h][1-8][a-h][1-8][qrbn]?
    # 3. Validate against legal moves (passed separately)
    # 4. Fallback: "I will play e2e4" → regex capture
    # 5. Return None if ambiguous/multiple
```

**Validation contract**: Parser + validator must be unit-tested against 100+ real LLM outputs (curated corpus).

### 4.3 Few-Shot Examples in Prompt

Add 3-5 examples of valid responses to the system prompt:

```
Example 1:
<thinking>White has a winning capture on d5...</thinking>
e4d5

Example 2:
<thinking>Developing knight to f3 controls center...</thinking>
g1f3
```

Reduces format errors dramatically.

---

## 5. Benchmark Rigor & Statistical Validity

### 5.1 Game Configuration Schema

```python
@dataclass
class BenchmarkConfig:
    # Game rules
    time_control: TimeControl = TimeControl(seconds_per_move=30) # or Fischer/increment
    opening_book: OpeningBook = OpeningBook.EC00_EC99_100 # 100 positions from ECO
    games_per_pairing: int = 10 # min for statistical significance
    colors: ColorAssignment = ColorAssignment.ALTERNATING # each model plays both colors

    # Model params (benchmark mode)
    temperature: float = 0.0
    max_tokens: int = 100
    seed: int | None = 42 # for reproducibility

    # Concurrency
    max_parallel_games: int = 4
```

### 5.2 Opening Diversity

- Don't always start from initial position
- Use standardized opening suite: 100 positions from ECO A00–E99 (2-3 moves deep)
- Each pairing plays each opening once as White, once as Black
- Track result per opening → opening-specific win rates

### 5.3 ELO Estimation

- Implement **Bayesian ELO** (or Glicko-2) over all completed games
- Input: game results (1/0.5/0) + opening + color
- Output: rating ± uncertainty per model
- Publish rating table with confidence intervals

### 5.4 Statistical Reporting

Per benchmark run, generate:
- Cross-table (model vs model: W/D/L, games played)
- ELO ratings with 95% CI
- Opening-specific performance
- Time-per-move distributions
- Token usage per model
- Error rates (invalid moves, timeouts, API failures)

### 5.5 Reproducibility Artifacts

Every benchmark run produces:
- `run.jsonl` — one line per game: config, moves, result, timestamps, tokens
- `run.pgn` — all games in PGN with metadata tags
- `run_report.html` — interactive report (Plotly/Altair)

---

## 6. Observability & Replay

### 6.1 Structured Logging (per move)

```json
{
  "game_id": "uuid",
  "move_number": 12,
  "player": "gpt-4o",
  "color": "white",
  "fen_before": "rnbqkbnr/...",
  "move_uci": "e2e4",
  "move_san": "e4",
  "llm_latency_ms": 842,
  "llm_tokens_in": 1847,
  "llm_tokens_out": 12,
  "llm_raw_response": "...",
  "thinking_trace": "...", // if model supports it
  "prompt_hash": "sha256",
  "validation_retries": 0,
  "timestamp_utc": "2026-08-01T18:22:13.441Z"
}
```

### 6.2 Live Replay Component

- Streamlit component: PGN viewer with slider, auto-play, flip board
- Highlight: blunders (Stockfish eval drop > 2.0), brilliant moves
- Show LLM reasoning trace per move (collapsible)

### 6.3 LLM-as-Judge (Optional but High-Value)

- Second LLM evaluates quality of reasoning trace (not just move)
- Rubric: tactical awareness, positional understanding, calculation depth
- Correlate judge score with game result

---

## 7. Demo UX & Live Streaming

### 7.1 Async Game Loop (Fixes D6)

```python
# game/async_game.py
class AsyncChessGame:
    async def play_game(self, white: ProviderChessAI, black: ProviderChessAI,
                        ui_callback: Callable[[GameState], Awaitable[None]]) -> GameResult:
        while not self.board.is_game_over():
            current = white if self.board.turn == chess.WHITE else black
            move = await current.get_move(self.board.fen())
            self.board.push_uci(move)
            await ui_callback(self.get_state())
            await asyncio.sleep(0.1) # yield to UI
        return self.result()
```

### 7.2 Streamlit Architecture

```
ui.py (entry)
├── sidebar: provider keys → model selectors (white/black)
├── controls: start/pause/resume/stop, speed slider
├── board: chess.svg or chessboard.js (via streamlit-component)
├── move_log: table with expandable reasoning
├── stats: live ELO estimate, material, eval bar (optional Stockfish)
└── replay: "Replay this game" button → loads run.jsonl
```

### 7.3 Visual Polish

- Use `chessboard.js` via custom component (better than SVG: drag pieces, animations, coordinates)
- Color schemes: light/dark mode
- Move animations (slide pieces)
- Last-move highlight, check flash
- Responsive layout (mobile-friendly for demo sharing)

---

## 8. Testing & Validation

### 8.1 Unit Tests (pytest)

| Module | Tests |
|--------|-------|
| `move_parser.py` | 100+ curated LLM outputs → correct UCI or None |
| `providers/*` | Mock HTTP → correct request/response handling |
| `prompt_builder.py` | Template renders without KeyError; includes all sections |
| `elo.py` | Known game sets → expected ratings ± tolerance |

### 8.2 Integration Tests

- **Provider parity**: Same prompt → OpenAI/Anthropic/Google/NIM all return valid UCI
- **Game loop**: 100 games without crash (mock providers)
- **Benchmark**: Full run with 4 models × 10 games each → produces report

### 8.3 Regression Corpus

- `tests/fixtures/llm_outputs/*.txt` — real outputs from each provider
- `tests/fixtures/games/*.pgn` — known-good games for replay testing
- CI runs parser against corpus on every PR

### 8.4 Lint/Type/Format

- `ruff check .`, `ruff format .`, `mypy --strict`
- Pre-commit hooks enforced

---

## 9. Distribution & Deployment

### 9.1 Streamlit Cloud (One-Click Demo)

- `requirements.txt` + `.streamlit/config.toml` + `streamlit_app.py`
- User forks repo → connects to Streamlit Cloud → deploys in 2 min
- Secrets: none (keys entered in UI)

### 9.2 Hugging Face Spaces

- `Dockerfile` + `README.md` + `app.py` (Gradio alternative or Streamlit)
- GPU not needed (inference is remote)

### 9.3 Docker (Self-Host)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
```

### 9.4 CLI Entry Point (for Benchmark Runs)

```bash
# Run headless benchmark
python -m chess_fight.benchmark \
  --config benchmark.yaml \
  --output runs/2026-08-01/
```

---

## 10. Stretch / Differentiators (Post-MVP)

| Feature | Why It Matters |
|---------|----------------|
| **Tournament Mode** | Round-robin / Swiss pairing; live bracket UI |
| **Side-by-Side Reasoning** | Show both models' `<thinking>` for same position |
| **Stockfish Eval Overlay** | Real-time eval bar; blunder detection |
| **Lichess Integration** | "Import to Lichess" button; analysis board link |
| **Custom Prompt Templates** | User uploads Jinja2 template for prompt engineering A/B |
| **Model Cost Tracking** | $ per game, $ per ELO point gained |
| **Multi-Game Dashboard** | Aggregate view across 100s of games; trend lines |
| **Export to W&B / MLflow** | Experiment tracking integration |

---

## 11. Prioritized Roadmap

### P0 — Must Have Before Any Demo (2-3 days)

| Task | Est. | Notes |
|------|------|-------|
| Fix D1–D7 (critical defects) | 4h | Blocks everything |
| Provider abstraction + OpenAI/Anthropic/Google/NIM/Ollama | 8h | Core requirement |
| Async game loop + Streamlit callback architecture | 4h | Fixes UI freeze |
| Move parser with test corpus (50+ samples per provider) | 4h | Reliability |
| Model selector UI (sidebar, grouped by provider) | 3h | User-facing |
| One-click Streamlit Cloud deploy config | 1h | Demo distribution |

### P1 — Benchmark Quality (1-2 weeks)

| Task | Est. | Notes |
|------|------|-------|
| Opening book (100 ECO positions) | 4h | Statistical validity |
| Bayesian ELO / Glicko-2 implementation | 8h | Rating system |
| Benchmark config + headless runner + report gen | 12h | Reproducible runs |
| Structured logging (JSONL) + replay component | 8h | Observability |
| Unit/integration test suite + CI | 8h | Confidence |

### P2 — Polish & Differentiators (Ongoing)

| Task | Est. | Notes |
|------|------|-------|
| chessboard.js component (animations, drag) | 8h | Visual quality |
| LLM-as-judge reasoning evaluation | 12h | Unique insight |
| Tournament mode + bracket UI | 16h | Demo wow factor |
| Lichess import/analysis integration | 4h | Community bridge |
| Cost tracking dashboard | 4h | Practical value |

---

## 12. Open Questions for User

1. **Benchmark vs Demo priority**: Is the live demo (Streamlit) the primary deliverable, or is the headless benchmark runner equally important? Affects P0/P1 sequencing.

2. **Stockfish integration**: Do you want local Stockfish for eval overlay / blunder detection? Requires binary bundling or `python-chess` engine protocol.

3. **Local model support**: Ollama is in the provider list — any others? (vLLM, TGI, llama.cpp server)

4. **Prompt engineering surface**: Should users be able to edit prompt templates in the UI (A/B testing), or is the prompt fixed per benchmark?

5. **Authentication model**: Pure client-side keys (current plan) vs optional server-side encrypted storage for teams?

6. **Time controls**: Per-move seconds (simple) vs Fischer/increment (tournament standard)? Affects async loop design.

7. **Data persistence**: SQLite for game history? Postgres? Just JSONL files?

8. **Multi-user**: Is this single-user demo or multi-tenant? Affects session isolation, deployment.

---

## Appendix: File Map (Current → Target)

```
chess_fight/
├── config.py → config.py (env loading only)
├── models.py → SPLIT INTO:
│   ├── chess_ai.py # Base ChessAI + prompt builder + move parser
│   ├── game_state.py # GameMove, GameStats, position analysis
│   └── evaluation.py # Heuristics (capture, development, king safety, etc.)
├── game.py → game/
│   ├── __init__.py
│   ├── sync_game.py # Legacy (delete after migration)
│   ├── async_game.py # New async loop
│   └── benchmark.py # Headless runner + ELO + reports
├── ui.py → ui/
│   ├── __init__.py
│   ├── streamlit_app.py # Entry point
│   ├── components/
│   │   ├── board.py # chessboard.js wrapper
│   │   ├── move_log.py
│   │   ├── stats.py
│   │   └── replay.py
│   └── providers_ui.py # Key entry + model selector
├── providers/ # NEW
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── openai.py
│   ├── anthropic.py
│   ├── google.py
│   ├── nim.py
│   ├── openrouter.py
│   ├── ollama.py
│   └── chess_ai.py # ProviderChessAI wrapper
├── move_parser.py # NEW - robust UCI extraction + tests
├── elo.py # NEW - Bayesian ELO / Glicko-2
├── openings.py # NEW - ECO opening book
├── logging.py # NEW - structured JSONL logger
├── tests/
│   ├── test_move_parser.py
│   ├── test_providers.py
│   ├── test_prompt.py
│   ├── test_elo.py
│   └── fixtures/
│       ├── llm_outputs/
│       └── games/
├── requirements.txt
├── pyproject.toml # ruff, mypy, pytest config
├── .streamlit/config.toml
├── streamlit_app.py # shim → ui.streamlit_app:main
├── Dockerfile
└── README.md # Updated with architecture diagram
```

---

**End of Brainstorm**

This document is the planning artifact. Next step: user confirms priorities → Plan Agent creates executable work plan → implementation begins.
