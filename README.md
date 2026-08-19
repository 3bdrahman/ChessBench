# ChessBench — LLM Critical Thinking Benchmark

> **The most rigorous, reproducible framework for evaluating LLM reasoning capabilities through chess.** ChessBench transforms chess into a structured reasoning benchmark: every move requires tactical calculation, strategic planning, positional understanding, and time management — the same cognitive skills that define "critical thinking" in LLMs.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_v2.svg)](https://share.streamlit.io/deploy?repository=3bdrahman/chess_fight&branch=main&mainModule=streamlit_app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-194%20passed-brightgreen.svg)](./tests/)
[![Mypy](https://img.shields.io/badge/mypy-strict-success.svg)](./pyproject.toml)
[![Ruff](https://img.shields.io/badge/ruff-linted-success.svg)](./pyproject.toml)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](./Dockerfile)

**Try the live demo →** [chess-fight.streamlit.app](https://chess-fight.streamlit.app)

---

## Why Chess for LLM Benchmarking?

Chess is the **ideal reasoning benchmark** because it demands:

| Cognitive Skill | Chess Manifestation | Measured By |
|---|---|---|
| **Tactical Calculation** | Finding forcing sequences, mates, tactics | Move quality vs. Stockfish (centipawn loss) |
| **Strategic Planning** | Long-term positional improvements | Plan coherence across 10+ moves |
| **Positional Understanding** | Pawn structures, weak squares, outposts | Evaluation accuracy in quiet positions |
| **Time Management** | Allocating thinking time under clock | Latency vs. quality tradeoffs |
| **Error Recovery** | Responding to opponent surprises | Resilience after suboptimal moves |
| **Multi-step Reasoning** | Candidate move evaluation trees | Thinking trace structure & depth |

Unlike static benchmarks (MMLU, GSM8K, HumanEval), chess provides **infinite unique positions**, **ground-truth evaluation** (Stockfish), and **interactive adversarial pressure** — models can't memorize answers.

---

## Quick Start

### Option 1: Web Demo (No Setup)
Visit **[chess-fight.streamlit.app](https://chess-fight.streamlit.app)** — pre-configured with OpenRouter free tier, works instantly in browser.

### Option 2: Local CLI (Headless Benchmarking)
```bash
# Install
pip install -e .

# Run a benchmark: GPT-4o vs Claude 3.5 Sonnet, 10 games each
chessbench run \
  --players openai:gpt-4o anthropic:claude-3-5-sonnet-20241022 \
  --games 10 \
  --parallel 4 \
  --output runs/my_benchmark

# View results
chessbench report runs/my_benchmark --format html --open
```

### Option 3: Local Streamlit UI
```bash
pip install -e .
streamlit run streamlit_app.py
```

### Option 4: Docker (Reproducible Environments)
```bash
docker build -t chessbench .
docker run -it --rm -v $(pwd)/runs:/app/runs -e OPENAI_API_KEY chessbench \
  run --players openai:gpt-4o anthropic:claude-3-5-sonnet-20241022 --games 10
```

---

## Core Capabilities

### 🎯 **Multi-Provider Model Evaluation**
Test any model from any provider through a unified interface:
- **OpenAI** (GPT-4o, o1, o3-mini, GPT-4.1)
- **Anthropic** (Claude 3.5/3.7 Sonnet, Opus, Haiku)
- **Google** (Gemini 1.5/2.0 Pro, Flash)
- **OpenRouter** (100+ models, free tier available)
- **NVIDIA NIM** (Llama, Qwen, Mistral hosted)
- **Groq** (Fast inference, generous free tier)
- **Ollama** (Local Llama/Qwen/Mistral — no API key)
- **Stockfish** (Local grandmaster engine — ground truth)

### 🏆 **Bayesian ELO Ratings (Glicko-2)**
Statistically rigorous ratings with confidence intervals:
- Proper rating periods (not incremental updates)
- 95% confidence intervals on every rating
- Head-to-head crosstabs with win/draw/loss breakdown
- Rating convergence diagnostics

### 📊 **Per-Move Analysis**
Every move captured with:
- **Centipawn loss** vs. Stockfish best move
- **Move quality classification** (Best/Excellent/Good/Inaccuracy/Mistake/Blunder)
- **Thinking trace analysis** (structured reasoning, tactical/strategic keywords)
- **Token usage & latency** per move
- **Full PGN export** with Stockfish evaluation annotations

### 🔬 **Adversarial Evaluation Mode**
Calibrate model strength against Stockfish at controlled depths:
```
Model vs Stockfish Depth 8  →  45% score  →  ~1400 ELO equivalent
Model vs Stockfish Depth 12 →  30% score  →  ~1800 ELO equivalent
Model vs Stockfish Depth 16 →  15% score  →  ~2200 ELO equivalent
```
Produces **equivalent Stockfish depth** — a human-interpretable strength metric.

### ✅ **Reproducibility Verification**
- Config hashing (SHA-256) for exact run reproduction
- Git commit, Python version, dependency versions recorded
- Behavioral verification: re-run subset of games, compare move sequences
- CI/CD integration for regression detection

### 📈 **Rich Export Formats**
- **Parquet** — columnar analytics (DuckDB, Polars, Pandas)
- **CSV** — spreadsheet compatibility
- **PGN + Eval** — Stockfish annotations for chess tools (ChessBase, Lichess)
- **HTML Report** — publication-ready benchmark reports
- **JSON** — programmatic access

---

## Installation

### Requirements
- Python 3.11+
- Stockfish binary (for evaluation): `apt install stockfish` / `brew install stockfish` / [Download](https://stockfishchess.org/download/)

### Development Install
```bash
git clone https://github.com/yourorg/chessbench
cd chessbench
pip install -e ".[dev]"

# Verify
pytest tests/ -v
chessbench --help
```

---

## CLI Reference

### `chessbench run` — Execute Benchmarks
```bash
# Basic tournament
chessbench run --players openai:gpt-4o anthropic:claude-3-5-sonnet --games 20

# Full configuration
chessbench run \
  --players openai:gpt-4o anthropic:claude-3-5-sonnet google:gemini-1.5-pro \
  --games 10 \
  --parallel 4 \
  --opening-book eco_balanced \
  --time-control 30 \
  --temperature 0.0 \
  --reasoning-level high \
  --max-tokens 2000 \
  --output runs/tournament_001 \
  --name "frontier_models_2024"

# From config file
chessbench run --config benchmark.yaml
```

### `chessbench evaluate` — Adversarial Strength Calibration
```bash
# Test model against Stockfish at multiple depths
chessbench evaluate --model openai:gpt-4o --depths 8,12,16,20 --games 4
```

### `chessbench report` — Generate Reports
```bash
# HTML report (opens in browser)
chessbench report runs/my_run --format html --open

# PDF report (requires weasyprint)
chessbench report runs/my_run --format pdf --output report.pdf

# Export data for analysis
chessbench report runs/my_run --format parquet --output analysis/
```

### `chessbench verify` — Reproducibility Check
```bash
# Quick config hash verification
chessbench verify runs/my_run

# Full behavioral verification (requires API keys)
chessbench verify runs/my_run --full-behavioral
```

### `chessbench history` — Browse Past Runs
```bash
chessbench history --runs-dir runs/
```

### `chessbench models` — List Available Models
```bash
chessbench models --provider openai
chessbench models --provider openrouter --filter free
```

---

## Configuration

### Benchmark Config (`benchmark.yaml`)
```yaml
# Game rules
time_control_seconds_per_move: 30
opening_book: "eco_balanced"  # eco_balanced | eco_all | startpos
games_per_pairing: 10
colors: "alternating"         # alternating | fixed

# Model parameters (benchmark mode = deterministic)
temperature: 0.0
max_tokens: 100
seed: 42
reasoning_level: "mid"        # low | mid | high

# Concurrency
max_parallel_games: 4

# Timeouts
move_timeout_seconds: 120
game_timeout_seconds: 7200

# Players (provider:model_id)
players:
  - "openai:gpt-4o"
  - "anthropic:claude-3-5-sonnet-20241022"
  - "google:gemini-1.5-pro"
  - "openrouter:anthropic/claude-3.5-sonnet"

# API keys (loaded from env: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
api_keys: {}

# Output
output_dir: "runs"
run_name: null  # auto-generated from timestamp
```

### Environment Variables
```bash
# Required for API providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="sk-or-v1-..."
export NIM_API_KEY="..."
export GROQ_API_KEY="gsk_..."

# Optional: Custom Stockfish path
export STOCKFISH_PATH="/usr/local/bin/stockfish"

# Optional: Runs directory
export CHESS_FIGHT_RUNS_ROOT="/data/benchmarks"

# Optional: Hosted providers for Streamlit Cloud
export CHESS_FIGHT_HOSTED_PROVIDERS="openrouter,nim,ollama"
```

---

## Benchmark Suites

### Pre-Configured Suites (in `configs/`)
```bash
# Frontier models comparison
chessbench run --config configs/frontier_models.yaml

# Open-weight models (local via Ollama)
chessbench run --config configs/open_weights.yaml

# Reasoning models (o1, o3, Claude 3.7 thinking)
chessbench run --config configs/reasoning_models.yaml

# Budget models (free tier / cheap API)
chessbench run --config configs/budget_models.yaml

# Critical thinking stress test (complex positions only)
chessbench run --config configs/critical_thinking.yaml
```

### Custom Suite Definition
```yaml
# configs/my_suite.yaml
name: "My Custom Suite"
description: "Evaluate models on endgame technique"
benchmark:
  time_control_seconds_per_move: 60
  opening_book: "eco_balanced"
  games_per_pairing: 5
  reasoning_level: "high"
  max_tokens: 4000
players:
  - "openai:o3-mini"
  - "anthropic:claude-3-7-sonnet-20250219"
  - "openrouter:deepseek/deepseek-r1"
positions:
  - type: "endgame"
    theme: "king_pawn_vs_king"
    count: 10
  - type: "endgame"
    theme: "rook_vs_pawn"
    count: 10
```

---

## Understanding Results

### Leaderboard Output
```
=== FINAL LEADERBOARD ===
  openai:gpt-4o:           1623.4 ± 42.1 (95% CI: 1541-1706)
  anthropic:claude-3.5:    1587.2 ± 38.7 (95% CI: 1511-1663)
  google:gemini-1.5-pro:   1534.1 ± 45.3 (95% CI: 1445-1623)
  openrouter:claude-3.5:   1578.9 ± 41.2 (95% CI: 1498-1660)
```

### Move Quality Distribution
| Model | Best | Excellent | Good | Inaccuracy | Mistake | Blunder |
|-------|------|-----------|------|------------|---------|---------|
| GPT-4o | 23% | 31% | 28% | 12% | 4% | 2% |
| Claude-3.5 | 19% | 34% | 30% | 11% | 5% | 1% |

### Thinking Quality Metrics
- **Structured reasoning %** — moves with numbered steps, explicit evaluation
- **Tactical awareness** — mentions forks, pins, discovered attacks
- **Strategic depth** — references pawn structure, outposts, prophylaxis
- **Time awareness** — acknowledges clock pressure

---

## Architecture

```
chessbench/
├── chess_fight/
│   ├── providers/          # Provider-agnostic LLM abstraction (8 backends)
│   ├── game/               # Async game engine with real-time updates
│   ├── models/             # Chess AI base + rich position evaluation
│   ├── benchmark/          # Headless runner + Glicko-2 + openings + export
│   └── ui/                 # Streamlit app (demo + in-process benchmarking)
├── configs/                # Pre-built benchmark suites
├── tests/                  # 194 tests (unit + integration)
├── scripts/                # CI/CD automation scripts
└── streamlit_app.py        # Web demo entry point
```

### Key Design Decisions
- **Zero mocks** — all providers, engines, benchmarks exercise real code paths
- **Provider abstraction** — swap models without changing benchmark logic
- **Async throughout** — concurrent games, non-blocking UI, rate-limit aware
- **Demand-driven prompts** — only compute position features the template needs
- **Reproducibility first** — config hashing, seeded randomness, version pinning

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for:
- Development setup
- Code style (ruff + mypy strict)
- Testing guidelines
- Adding new providers
- Benchmark suite design principles

---

## Citation

If you use ChessBench in research, please cite:
```bibtex
@software{chessbench2024,
  title = {ChessBench: LLM Critical Thinking Benchmark via Chess},
  author = {ChessBench Contributors},
  year = {2024},
  url = {https://github.com/yourorg/chessbench}
}
```

---

## License

MIT — see [LICENSE](./LICENSE).

---

## Acknowledgments

- **Stockfish** — the ground-truth evaluator that makes this rigorous
- **python-chess** — rock-solid chess library
- **Streamlit** — beautiful interactive demos
- **Glicko-2** — Mark Glickman's rating system
- All open-source model providers for API access