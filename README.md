# AI Chess Battle

> Watch AI models compete in chess! Select models from any provider and see them battle in real-time with live board updates, move tracking, and game statistics.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_v2.svg)](https://share.streamlit.io/deploy?repository=3bdrahman/chess_fight&branch=main&mainModule=streamlit_app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-126%20passed-brightgreen.svg)](./tests/)
[![Mypy](https://img.shields.io/badge/mypy-strict-success.svg)](./pyproject.toml)
[![Ruff](https://img.shields.io/badge/ruff-linted-success.svg)](./pyproject.toml)

**Try the live demo →** [share.streamlit.io](https://share.streamlit.io/deploy?repository=3bdrahman/chess_fight&branch=main&mainModule=streamlit_app.py)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No API keys? Try the **Watch Demo Game** feature in the sidebar — replay recorded chess matches with full stats and animation.

## Key Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google, NVIDIA NIM, OpenRouter, Groq, and Ollama
- **Async Live Updates**: Watch moves unfold in real-time with animated board
- **Cross-Provider Battles**: Pit any model against any other
- **Watch Demo Games**: Replay recorded matches instantly — no API key required
- **Comprehensive Statistics**: Capture tracking, check frequency, game duration, move history
- **Bayesian ELO Ratings**: Glicko-2 powered rating system with confidence intervals
- **Headless Benchmark Runner**: Run reproducible tournaments from the CLI

## Watch Demo Games (No API Key Required)

Built-in demo games let you see the full UI in action instantly:

| Opening | Result |
|---|---|
| **Italian Game** | Tactical battle, White victory |
| **Queen's Gambit** | Positional struggle, Black counter-attack |
| **English Opening** | Long endgame draw |
| **Reti Opening** | Sharp attacking play |
| **Sicilian Defense** | Black mates |

## Providers

| Provider | Key Format | Notes |
|---|---|---|
| **OpenRouter** | `sk-or-v1-...` | 100+ models, free tier available (demo default) |
| **Groq** | `gsk_...` | Fast inference, generous free tier |
| **OpenAI** | `sk-...` | GPT-4o, o1, etc. |
| **Anthropic** | `sk-ant-...` | Claude 3.5 Sonnet/Haiku/Opus |
| **Google** | via Google AI | Gemini models |
| **NVIDIA NIM** | NIM key | Self-hosted/cloud models |
| **Ollama** | None | Local Llama/Qwen models |

## Headless Benchmark

Run reproducible tournaments from the CLI:

```bash
python -m chess_fight.benchmark.runner \
  --players openai:gpt-4o anthropic:claude-3-5-sonnet-20241022 \
  --games 10 \
  --parallel 4
```

Configure via `benchmark.yaml` — supports ECO opening books (295 positions), Glicko-2 ratings, and detailed per-move logging.

## Project Structure

```
chess_fight/
├── chess_fight/
│   ├── providers/       # Provider-agnostic LLM abstraction layer
│   ├── game/            # Async and sync game engines
│   ├── models/          # Chess AI base class + position evaluation
│   ├── benchmark/       # Headless benchmark runner + ELO + openings
│   └── ui/              # Streamlit app + chessboard component
├── demos/               # Recorded games for replay
├── tests/               # 126 tests covering all modules
└── streamlit_app.py     # App entry point
```

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for Streamlit Cloud deployment instructions.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — see [LICENSE](./LICENSE).
