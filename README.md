# AI Chess Battle

> Watch AI models compete in chess! Select models from any provider and see them battle in real-time with live board updates, move tracking, and game statistics.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_v2.svg)](https://share.streamlit.io/deploy?repository=3bdrahman/chess_fight&branch=main&mainModule=streamlit_app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-194%20passed-brightgreen.svg)](./tests/)
[![Mypy](https://img.shields.io/badge/mypy-strict-success.svg)](./pyproject.toml)
[![Ruff](https://img.shields.io/badge/ruff-linted-success.svg)](./pyproject.toml)

**Try the live demo →** [chess-fight.streamlit.app](https://chess-fight.streamlit.app)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No API keys? Try the **Watch Demo Game** feature in the sidebar — replay recorded chess matches with full stats and animation. Or enable the **Stockfish** provider (local engine, no API key needed) to play against a real chess engine immediately.

## Quick Demo

Three ways to experience the app:

- **Live demo (no setup)**: Open [chess-fight.streamlit.app](https://chess-fight.streamlit.app) — pre-configured with OpenRouter free tier, works instantly in browser
- **No API key (local)**: Run locally and use the built-in **Watch Demo Game** feature to replay recorded matches with full stats and animation
- **Stockfish (local)**: Install Stockfish and enable it in the sidebar — play against a real grandmaster-level engine with no API key
- **With API keys**: Configure provider keys in the sidebar and watch live LLM vs LLM battles with real-time board updates

## Key Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google, NVIDIA NIM, OpenRouter, Groq, Ollama, **Stockfish (local engine)**
- **Async Live Updates**: Watch moves unfold in real-time with animated board
- **Cross-Provider Battles**: Pit any model against any other
- **Watch Demo Games**: Replay recorded matches instantly — no API key required. Demo games are **auto-generated from real benchmark runs** — no synthetic replays.
- **Benchmark History**: Browse past benchmark runs with ELO leaderboards, win/loss stats, and per-move token usage.
- **In-Process Benchmarking**: Run quick benchmarks directly in the UI with live progress, ELO leaderboard, and per-move completion details.
- **Comprehensive Statistics**: Capture tracking, check frequency, game duration, move history
- **Bayesian ELO Ratings**: Glicko-2 powered rating system with confidence intervals
- **Headless Benchmark Runner**: Run reproducible tournaments from the CLI

## Watch Demo Games (No API Key Required)

Demo games are **auto-generated from real benchmark runs** — each replay shows a genuine LLM vs LLM (or Stockfish vs Stockfish) match with real moves, tokens, and timing. No synthetic replays.

Run a benchmark first (sidebar → **Pin & Benchmark** → **Run Quick Benchmark**), then visit **Watch Demo Game** to replay the games.

## Providers

The hosted Streamlit demo exposes three providers (OpenRouter, NVIDIA NIM, Ollama). The full provider set (Anthropic, Google, Groq, OpenAI) remains available in the registry for the headless benchmark runner and self-hosted Streamlit deployments.

| Provider | Key Format | Notes |
|---|---|---|
| **OpenRouter** | `sk-or-v1-...` | 100+ models, free tier available (demo default) |
| **NVIDIA NIM** | NIM key | Hosted Llama / Qwen / Mistral models |
| **Ollama** | None | Local Llama/Qwen models (run `ollama serve` first) |
| **Stockfish** | None | **Local grandmaster engine** — install from [stockfishchess.org](https://stockfishchess.org/download/) |
| Anthropic | `sk-ant-...` | Self-host only — add `anthropic_api_key` and edit `HOSTED_PROVIDERS` |
| Google | Google AI key | Self-host only — Gemini models |
| Groq | `gsk_...` | Self-host only — fast inference, generous free tier |
| OpenAI | `sk-...` | Self-host only — GPT-4o, o1, etc. |

## Headless Benchmark

Run reproducible tournaments from the CLI:

```bash
python -m chess_fight.benchmark.runner \
  --players openai:gpt-4o anthropic:claude-3-5-sonnet-20241022 \
  --games 10 \
  --parallel 4
```

Configure via `benchmark.yaml` — supports ECO opening books (295 positions), Glicko-2 ratings, and detailed per-move logging.

### In-Process Benchmark (UI)

For quick experiments, pin two models in the sidebar and click **Run Quick Benchmark**. The UI streams live progress, displays the ELO leaderboard, and shows per-move completion details (tokens, latency, raw response) — all in-process, no subprocess overhead.

### Benchmark History

Visit the **Benchmark History** expander in the sidebar to browse past runs with ELO leaderboards, win/loss/draw stats, per-pairing crosstabs, and per-move token/latency data — all parsed from real JSONL artifacts.

## Project Structure

```
chess_fight/
├── chess_fight/
│   ├── providers/       # Provider-agnostic LLM abstraction layer (+ Stockfish local engine)
│   ├── game/            # Async and sync game engines
│   ├── models/          # Chess AI base class + position evaluation
│   ├── benchmark/       # Headless benchmark runner + ELO + openings + results viewer
│   └── ui/              # Streamlit app + chessboard component
├── demos/               # Auto-generated demo games from real benchmark runs
├── tests/               # 194 tests covering all modules
└── streamlit_app.py     # App entry point
```

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for Streamlit Cloud deployment instructions.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

## Portfolio Showcase

**Tech Stack:** Python 3.11+, Streamlit, AsyncIO, chess library, Plotly/Pandas

**Architecture Highlights:**
- Provider-agnostic abstraction layer supporting **8 LLM backends** (including Stockfish local engine)
- Async game engine with real-time Streamlit UI updates (NO polling)
- Glicko-2 Bayesian ELO rating system with comprehensive statistics
- 295 ECO opening positions for benchmark reproducibility
- Professional CI/CD pipeline (ruff + mypy + pytest + Docker)
- **Zero mocks** — all providers, engines, and benchmarks exercise real code paths

**Why it matters:** Demonstrates system design (provider abstraction, async game loop, concurrent benchmark runner, real-time UI), not just "making API calls." The architecture handles 8 disparate LLM/engine APIs through a unified interface.

## License

MIT — see [LICENSE](./LICENSE).