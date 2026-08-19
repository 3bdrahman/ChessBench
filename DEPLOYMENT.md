# Deployment Guide

ChessBench can be deployed in multiple ways depending on your needs.

## Deployment Options

| Method | Use Case | Complexity |
|--------|----------|------------|
| **Streamlit Cloud** | Public demo, sharing results | ⭐ Easiest |
| **Docker** | Reproducible environments, CI/CD | ⭐⭐ Medium |
| **VPS/Cloud VM** | Full control, persistent runs | ⭐⭐ Medium |
| **Kubernetes** | Scaling, multi-user | ⭐⭐⭐ Advanced |

---

## Streamlit Cloud (Recommended for Demos)

### 1. Push to GitHub
```bash
git push origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Main file: `streamlit_app.py`
4. Add secrets in **Settings → Secrets**:

```toml
# .streamlit/secrets.toml (Streamlit Cloud UI)
openrouter_api_key = "sk-or-v1-..."  # Demo key for visitors
CHESS_FIGHT_HOSTED_PROVIDERS = "openrouter,nim,ollama"
```

### 3. Configure Hosted Providers
The demo exposes only `openrouter`, `nim`, `ollama` by default (set via `CHESS_FIGHT_HOSTED_PROVIDERS`). Visitors use the shared OpenRouter key; power users add their own keys in the sidebar.

---

## Docker Deployment

### Build Image
```bash
docker build -t chessbench:latest .
```

### Run Benchmark (Headless)
```bash
# With API keys from environment
docker run -it --rm \
  -v $(pwd)/runs:/app/runs \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  -e GOOGLE_API_KEY \
  -e OPENROUTER_API_KEY \
  chessbench:latest \
  run --players openai:gpt-4o anthropic:claude-3-5-sonnet --games 10
```

### Run Streamlit UI
```bash
docker run -d --rm \
  -p 8501:8501 \
  -v $(pwd)/runs:/app/runs \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  -e CHESS_FIGHT_HOSTED_PROVIDERS=openrouter,nim,ollama \
  chessbench:latest \
  streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

### Docker Compose (Production)
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
      - CHESS_FIGHT_HOSTED_PROVIDERS=openrouter,nim,ollama
      - CHESS_FIGHT_RUNS_ROOT=/app/runs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'

  # Optional: PostgreSQL for run metadata (future)
  # db:
  #   image: postgres:16
  #   volumes:
  #     - pgdata:/var/lib/postgresql/data

# volumes:
#   pgdata:
```

```bash
docker compose up -d
```

---

## VPS / Cloud VM Deployment

### Requirements
- Ubuntu 22.04+ / Debian 12+
- Python 3.11+
- Stockfish: `apt install stockfish`
- 2+ GB RAM, 2+ vCPUs
- 10+ GB disk for runs

### Setup
```bash
# System packages
sudo apt update && sudo apt install -y python3.11 python3.11-venv stockfish git

# Clone
git clone https://github.com/yourorg/chessbench
cd chessbench

# Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Environment
cp .env.example .env  # Edit with your API keys
```

### Systemd Service (Streamlit UI)
```ini
# /etc/systemd/system/chessbench.service
[Unit]
Description=ChessBench Streamlit UI
After=network.target

[Service]
Type=simple
User=chessbench
WorkingDirectory=/opt/chessbench
Environment=PATH=/opt/chessbench/.venv/bin
EnvironmentFile=/opt/chessbench/.env
ExecStart=/opt/chessbench/.venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now chessbench
sudo systemctl status chessbench
```

### Nginx Reverse Proxy (HTTPS)
```nginx
# /etc/nginx/sites-available/chessbench
server {
    listen 80;
    server_name chessbench.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/chessbench /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
# Then: certbot --nginx -d chessbench.yourdomain.com
```

---

## Kubernetes Deployment

### Namespace & ConfigMap
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: chessbench
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chessbench-config
  namespace: chessbench
data:
  CHESS_FIGHT_HOSTED_PROVIDERS: "openrouter,nim,ollama"
  CHESS_FIGHT_RUNS_ROOT: "/data/runs"
```

### Secret (API Keys)
```bash
kubectl create secret generic chessbench-secrets \
  --from-literal=OPENAI_API_KEY="sk-..." \
  --from-literal=ANTHROPIC_API_KEY="sk-ant-..." \
  --from-literal=OPENROUTER_API_KEY="sk-or-..." \
  -n chessbench
```

### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chessbench
  namespace: chessbench
spec:
  replicas: 2
  selector:
    matchLabels:
      app: chessbench
  template:
    metadata:
      labels:
        app: chessbench
    spec:
      containers:
      - name: chessbench
        image: yourregistry/chessbench:latest
        ports:
        - containerPort: 8501
        envFrom:
        - configMapRef:
            name: chessbench-config
        - secretRef:
            name: chessbench-secrets
        volumeMounts:
        - name: runs
          mountPath: /data/runs
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: runs
        persistentVolumeClaim:
          claimName: chessbench-runs-pvc
```

### Service & Ingress
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: chessbench
  namespace: chessbench
spec:
  selector:
    app: chessbench
  ports:
  - port: 80
    targetPort: 8501
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chessbench
  namespace: chessbench
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  tls:
  - hosts:
    - chessbench.yourdomain.com
    secretName: chessbench-tls
  rules:
  - host: chessbench.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: chessbench
            port:
              number: 80
```

### Persistent Volume
```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chessbench-runs-pvc
  namespace: chessbench
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

---

## CI/CD Pipeline (GitHub Actions)

### Automated Benchmarking on Schedule
```yaml
# .github/workflows/benchmark.yml
name: Scheduled Benchmark

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly Sunday 2 AM UTC
  workflow_dispatch:
    inputs:
      config:
        description: 'Benchmark config to run'
        required: true
        default: 'frontier_models'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Stockfish
        run: sudo apt-get update && sudo apt-get install -y stockfish

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run benchmark
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          chessbench suite --run ${{ github.event.inputs.config }}

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: runs/
          retention-days: 30

      - name: Generate report
        run: |
          latest_run=$(ls -td runs/*/ | head -1)
          chessbench report "$latest_run" --format html --output report.html

      - name: Deploy report to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./report.html
          destination_dir: benchmarks/${{ github.event.inputs.config }}/${{ github.run_id }}
```

### PR Validation
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: sudo apt-get update && sudo apt-get install -y stockfish
      - run: pip install -e ".[dev]"
      - run: ruff check chess_fight/
      - run: mypy chess_fight/
      - run: pytest tests/ --cov=chess_fight --cov-fail-under=80

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          push: false
          tags: chessbench:test
```

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For OpenAI models | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic models | Anthropic API key |
| `GOOGLE_API_KEY` | For Google models | Google AI Studio key |
| `OPENROUTER_API_KEY` | For OpenRouter | OpenRouter key (free tier available) |
| `NIM_API_KEY` | For NVIDIA NIM | NVIDIA NIM API key |
| `GROQ_API_KEY` | For Groq | Groq API key |
| `STOCKFISH_PATH` | Optional | Custom Stockfish binary path |
| `CHESS_FIGHT_RUNS_ROOT` | Optional | Base directory for run artifacts |
| `CHESS_FIGHT_HOSTED_PROVIDERS` | Streamlit only | Comma-separated providers for hosted demo |

---

## Monitoring & Maintenance

### Log Rotation
```bash
# /etc/logrotate.d/chessbench
/var/log/chessbench/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
}
```

### Health Check Endpoint
```bash
# In Streamlit, add to streamlit_app.py:
# st.set_page_config(..., menu_items={
#     'Get Help': 'https://github.com/yourorg/chessbench/issues',
#     'Report a bug': 'https://github.com/yourorg/chessbench/issues',
# })
```

### Backup Runs
```bash
# Daily backup to S3/GCS
aws s3 sync runs/ s3://your-bucket/chessbench/runs/ --delete
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Stockfish not found | `apt install stockfish` or set `STOCKFISH_PATH` |
| API rate limits | Reduce `max_parallel_games`, increase `move_timeout_seconds` |
| Out of memory | Reduce `max_parallel_games`, limit `max_tokens` |
| Streamlit won't load | Check port 8501, firewall, `server.address=0.0.0.0` |
| Docker permission denied | Add user to docker group: `sudo usermod -aG docker $USER` |

---

## Security Considerations

- **Never commit API keys** — use environment variables / secrets
- **Rotate keys regularly** — especially for shared demo keys
- **Limit hosted providers** — `CHESS_FIGHT_HOSTED_PROVIDERS` restricts what visitors see
- **Run benchmarks in isolated networks** — no egress to unknown domains
- **Audit dependencies** — `pip-audit` in CI