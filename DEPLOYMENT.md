# ChessBench Deployment Guide

ChessBench supports flexible deployment options ranging from instant zero-setup browser demos to headless CLI evaluation, Docker containers, cloud VMs, and Kubernetes clusters.

## Deployment Overview

| Method | Best For | Setup Time | Prerequisites |
|---|---|---|---|
| **Streamlit Cloud** | Live web app demo & interactive battles | ⚡ 2 mins | GitHub Account |
| **Docker** | Containerized reproducible benchmarks | 🚀 5 mins | Docker / Docker Compose |
| **Cloud VM (Ubuntu/Debian)** | Persistent benchmarking & self-hosted server | 🛠️ 10 mins | Python 3.11+, Stockfish |
| **Kubernetes** | Enterprise scaling & multi-tenant runs | ☸️ Advanced | K8s Cluster, Helm/kubectl |

---

## Streamlit Cloud (Recommended for Web Demos)

### 1. Repository Setup
Push your repository to GitHub:
```bash
git push origin main
```

### 2. Connect to Streamlit Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io).
2. Click **New app**, select `3bdrahman/chessbench`, branch `main`, and set main file to `streamlit_app.py`.
3. Configure environment secrets in **Settings → Secrets**:

```toml
# Streamlit secrets.toml
OPENROUTER_API_KEY = "sk-or-v1-..."  # Shared key for demo visitors
CHESS_FIGHT_HOSTED_PROVIDERS = "openrouter,nim"
```

> [!TIP]
> `CHESS_FIGHT_HOSTED_PROVIDERS` controls which provider dropdowns are pre-enabled for demo users without requiring them to supply an API key.

---

## Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t chessbench:latest .
```

### 2. Run Headless Benchmarks via Container
Pass API keys via environment variables and mount a host volume for persistent run logs:

```bash
docker run -it --rm \
  -v $(pwd)/runs:/app/runs \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  -e GOOGLE_API_KEY \
  -e OPENROUTER_API_KEY \
  chessbench:latest \
  run --players openai:gpt-4o anthropic:claude-3-5-sonnet-20241022 --games 10
```

### 3. Run Streamlit UI in Docker
```bash
docker run -d --name chessbench-ui \
  -p 8501:8501 \
  -v $(pwd)/runs:/app/runs \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  chessbench:latest \
  streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

### 4. Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  chessbench:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./runs:/app/runs
      - ./configs:/app/configs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - TOGETHER_API_KEY=${TOGETHER_API_KEY}
      - FIREWORKS_API_KEY=${FIREWORKS_API_KEY}
      - DEEPINFRA_API_KEY=${DEEPINFRA_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - NIM_API_KEY=${NIM_API_KEY}
      - CHESS_FIGHT_HOSTED_PROVIDERS=openrouter,nim
      - CHESS_FIGHT_RUNS_ROOT=/app/runs
    restart: unless-stopped
    resources:
      limits:
        memory: 4G
        cpus: '2'
```

Start container stack:
```bash
docker compose up -d
```

---

## Cloud VM / VPS Deployment

### 1. System Installation (Ubuntu 22.04+ / Debian 12+)
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv stockfish git

git clone https://github.com/3bdrahman/chessbench.git
cd chessbench

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Systemd Service (Streamlit UI)
Create `/etc/systemd/system/chessbench.service`:

```ini
[Unit]
Description=ChessBench Streamlit Arena Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/chessbench
Environment=PATH=/opt/chessbench/.venv/bin
EnvironmentFile=/opt/chessbench/.env
ExecStart=/opt/chessbench/.venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl enable --now chessbench
```

---

## Environment Variables Reference

| Variable | Required For | Description |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI models | Access GPT-4o, o1, o3-mini |
| `ANTHROPIC_API_KEY` | Anthropic models | Access Claude 3.5 Sonnet, Claude 3.7 |
| `GOOGLE_API_KEY` | Google models | Access Gemini 1.5 Pro, Gemini 2.0 Flash |
| `OPENROUTER_API_KEY` | OpenRouter models | Access 100+ open & proprietary models |
| `NIM_API_KEY` | NVIDIA NIM | Access hosted Llama, Qwen, Mistral models |
| `GROQ_API_KEY` | Groq models | Access high-speed Llama & Mixtral models |
| `TOGETHER_API_KEY` | Together AI | Access open-source model endpoints |
| `FIREWORKS_API_KEY` | Fireworks AI | Access fast open-weights inference |
| `DEEPINFRA_API_KEY` | DeepInfra | Access serverless open-source models |
| `STOCKFISH_PATH` | Stockfish evaluation | Custom binary path (defaults to system `stockfish`) |
| `CHESS_FIGHT_RUNS_ROOT` | Run storage | Root directory for benchmark run logs (default: `runs/`) |

---

## CI/CD Pipeline Integration (GitHub Actions)

Run automated weekly benchmarks using [.github/workflows/benchmark.yml](file:///var/home/usef/coding/chessbench/.github/workflows/benchmark.yml):

```yaml
name: Scheduled Benchmark

on:
  schedule:
    - cron: '0 2 * * 0'
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: sudo apt-get update && sudo apt-get install -y stockfish
      - run: pip install -e ".[dev]"
      - name: Run Benchmark Tournament
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          chessbench run --players openai:gpt-4o anthropic:claude-3-5-sonnet-20241022 --games 10
```