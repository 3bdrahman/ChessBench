# ChessBench — LLM Reasoning Benchmark via Chess

Evaluate Large Language Models (LLMs) on tactical calculation, strategic planning, and position evaluation through chess matches.

---

## ⚡ Quick Start

### 1. Web App
Access the live benchmark app directly at **[https://chessbench.streamlit.app/](https://chessbench.streamlit.app/)**

Or run locally:
```bash
pip install -e .
streamlit run streamlit_app.py
```

### 2. Command Line (CLI)
```bash
git clone https://github.com/3bdrahman/chessbench.git
cd chessbench
pip install -e .

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run a match between GPT-4o and Claude 3.5 Sonnet
chessbench run --players openai:gpt-4o anthropic:claude-3-5-sonnet-20241022 --games 5
```

---

## 🎯 Features

- **Multi-Provider Support**: Benchmark models across OpenAI, Anthropic, Google Gemini, OpenRouter, Groq, Together, Fireworks, DeepInfra, and local Stockfish.
- **Move Quality Evaluation**: Real-time evaluation using Stockfish engine (centipawn loss, move quality classification).
- **Ratings & Leaderboard**: Glicko-2 Bayesian ratings with head-to-head stats.
- **Analytics & History**: Interactive replay dashboard with Stockfish evaluation timelines, move quality heatmaps, and PGN/CSV exports.
- **Prompt Customization**: Customizable system and turn prompts with live validation.

---

## 💻 CLI Commands

| Command | Description |
|---|---|
| `chessbench run` | Run a benchmark match between models |
| `chessbench evaluate` | Test an LLM against Stockfish engine levels |
| `chessbench report` | Generate HTML, CSV, PGN, or Parquet reports |
| `chessbench history` | Browse past benchmark runs |
| `chessbench models` | List available models across providers |

---

## ⚙️ Configuration & Environment

Set provider keys as environment variables or in `.streamlit/secrets.toml`:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export GROQ_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

---

## 🧪 Testing

```bash
pytest
```

---

## 📜 License

MIT License. See [LICENSE](./LICENSE) for details.
