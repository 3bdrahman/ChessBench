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

- **Provider-Agnostic Support**: Benchmark OpenAI, Anthropic, Google Gemini, OpenRouter, Groq, Together, Fireworks, DeepInfra, local Stockfish, or any OpenAI-compatible Chat Completions endpoint.
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

### Any OpenAI-compatible endpoint

In the web app, expand **Provider Keys**, then:

1. Enter a provider name, base URL, and optional API key.
2. Add the provider; it appears in both player provider selectors.
3. Select it for a player. ChessBench probes `GET /models` at that point and
   fills the model selector with the response.
4. If the server does not implement `GET /models`, provide comma-separated
   fallback model IDs when adding it.

Providers added in the web app live only for the current browser session.

For CLI or server-managed deployments, configure the same adapter with
environment variables:

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY=""  # Optional for local servers
export CHESSBENCH_ALLOW_PRIVATE_ENDPOINTS="true"  # Required for localhost UI targets

# Only needed when the server does not implement GET /models
export OPENAI_COMPATIBLE_MODELS="model-a,model-b"

chessbench run \
  --players openai_compatible:model-a openai_compatible:model-b \
  --games 5
```

The base URL is used exactly as configured, so include the endpoint's API
prefix (usually `/v1`). Advanced endpoints can use JSON values in
`OPENAI_COMPATIBLE_HEADERS`, `OPENAI_COMPATIBLE_QUERY`, and
`OPENAI_COMPATIBLE_EXTRA_BODY`. Tool calling is opt-in with
`OPENAI_COMPATIBLE_SUPPORTS_TOOLS=true`; servers that use
`max_completion_tokens` can set
`OPENAI_COMPATIBLE_MAX_TOKENS_FIELD=max_completion_tokens`.

On hosted deployments, private/loopback targets are blocked to prevent
server-side request forgery. Set `CHESSBENCH_PUBLIC_DEPLOYMENT=true` when the
hosted-provider filter is not configured. A trusted private deployment can opt
in with `CHESSBENCH_ALLOW_PRIVATE_ENDPOINTS=true`. Administrators can further
restrict session-added endpoints with the exact-host allowlist
`OPENAI_COMPATIBLE_ALLOWED_HOSTS`.

---

## 🧪 Testing

```bash
pytest
```

---

## 📜 License

MIT License. See [LICENSE](./LICENSE) for details.
