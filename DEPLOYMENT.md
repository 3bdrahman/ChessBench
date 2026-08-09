# Deployment Guide — chess_fight

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_v2.svg)](https://share.streamlit.io/deploy?repository=3bdrahman/chess_fight&branch=main&mainModule=streamlit_app.py)

---

## Live Demo

**Try it now →** [https://chess-fight.streamlit.app](https://chess-fight.streamlit.app)

Pre-configured with OpenRouter free tier — no API keys needed, works instantly in your browser.

---

## Two Ways to Try

1. **Hosted demo** — no setup needed, just open [chess-fight.streamlit.app](https://chess-fight.streamlit.app) and play.
2. **Self-host on Streamlit Cloud** — deploy your own instance in ~5 minutes.

---

## Self-Host on Streamlit Cloud

### Step 1 — Fork the repo

Fork this repository to your own GitHub account.

### Step 2 — Open Streamlit Cloud

Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

### Step 3 — Connect your fork

Click **"New app"**, then select your forked repo, the `main` branch, and `streamlit_app.py` as the main module.

### Step 4 — Set secrets

Before the first deploy, add your API key(s) in the Streamlit Cloud dashboard:

1. In your app's settings page, open the **Secrets** tab.
2. Paste the contents of `.streamlit/secrets.toml.example` (below) and replace the placeholder values with your real keys.
3. Click **Save**.

```toml
# .streamlit/secrets.toml.example
openrouter_api_key = "sk-or-v1-your_openrouter_key_here"
# openai_api_key = "sk-..."
# anthropic_api_key = "sk-ant-..."
# groq_api_key = "gsk-..."
```

> **Tip:** You only need `openrouter_api_key` for the default demo (OpenRouter's free tier with rate limits). Add the others if you want extra provider options.

### Step 5 — Deploy

Click **"Deploy"**. Streamlit Cloud will install dependencies from `requirements.txt` and start your app. The first deploy takes ~1–2 minutes; subsequent pushes to `main` redeploy automatically.

---

## Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Free Tier Limits

| Resource | Free tier limit |
|---|---|
| Streamlit Cloud | 1 app per workspace, 1 GB RAM, always-on |
| OpenRouter | 200 free req/day with rate-limited free models — perfect for demos |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on deploy | `requirements.txt` missing a dependency | Add the missing package and push |
| App shows "no API key" error | Secrets not set in dashboard | Add `openrouter_api_key` in the Secrets tab |
| App is slow / OOM kills | Free tier RAM limit (1 GB) | Reduce concurrent games or switch to a smaller model |
| Deploy stuck on "Installing dependencies" | Large dependency install | Normal on first deploy; wait 2–3 minutes |
| CORS / XHR errors in browser | Custom domain or proxy in front of Streamlit | Disable the proxy or set `enableCORS = true` in `.streamlit/config.toml` |
