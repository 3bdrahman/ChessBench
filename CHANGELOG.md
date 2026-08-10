# Changelog

All notable changes to AI Chess Battle are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-07 — Portfolio Polish Release

### Added
- **Loading spinners** during model fetches and game execution for a smoother UX.
- **Friendly error messages + retry UX** for provider failures (rate limits, network,
  auth) — Streamlit session no longer crashes on transient API errors.
- **Impressive landing page** with value proposition, three feature highlights, and
  a prominent "Try Demo Game" call-to-action for first-time visitors.
- **Community files** for portfolio polish: `SECURITY.md`, GitHub issue templates
  (bug + feature), pull-request template, and `.dockerignore`.
- **`Makefile`** with common dev commands: `install`, `test`, `lint`, `fmt`,
  `clean`, `run`, `docker-build`, `docker-run`, `check`.
- **`DEVELOPMENT.md`** (renamed from `OPTIMIZATIONS.md`) for architecture notes
  without confusing portfolio visitors.

### Changed
- **Provider error handling**: All 7 providers (`openai`, `anthropic`, `google`,
  `groq`, `nim`, `openrouter`, `ollama`) now wrap API calls in `try/except`,
  returning graceful `CompletionResult` errors instead of crashing the app.
- **Provider timeouts**: All OpenAI-compatible providers now use a 30s timeout;
  Ollama uses 60s for local inference.
- **Context windows**: Updated `groq` and `nim` from 8192 to 131072 (most models
  support 128k+); `openrouter` aligned to 131072 for flagship models.
- **Config module**: `config.py` now loads all 7 provider env vars (was: 2) so
  `dotenv`-based setup covers every supported provider out of the box.
- **Demo Mode card**: Now scrolls into view automatically with a prominent CTA
  button for visitors without an API key.

### Fixed
- **Google provider**: Assistant messages were silently dropped, breaking
  multi-turn conversation context — now preserved as proper `ContentDict` entries.
- **Anthropic provider**: Removed fragile `from anthropic._types import NOT_GIVEN`
  import; replaced with a plain `None` sentinel.
- **Legacy `ChessAI` classes**: `OpenAIChessAI`, `AnthropicChessAI`, `LlamaChessAI`
  no longer import optional SDKs (`ollama`, `Anthropic`, `OpenAI`) at module top —
  the package now imports cleanly even when those optional dependencies are missing.
- **Legacy async methods**: Sync SDK calls in legacy `_get_move_from_model` methods
  now run in an executor via `loop.run_in_executor`, so they no longer block the
  event loop when used as async.
- **Duplicate `_validate_move`**: Removed; now delegates to `move_parser.validate_move`.
- **Dead code**: Removed unused `chess_fight/providers/base.py` (was a thin re-export).

### Security
- New `SECURITY.md` documents the project's security model (no key persistence,
  user-facing keys only) and vulnerability disclosure process.

## [0.1.0] - 2026-08-06 — Portfolio / Demo-Ready Polish

### Added
- Multi-provider LLM chess benchmark across **OpenAI, Anthropic, Google,
  NVIDIA NIM, OpenRouter, Groq, and Ollama**.
- Async live-updating Streamlit UI with animated SVG board, per-move
  statistics (captures, checks, time elapsed), and full PGN move history.
- **Watch Demo Game** replay feature — five recorded matches ship in
  `demos/games/`, so the entire UI works offline with no API key.
- Headless benchmark runner (`python -m chess_fight.benchmark.runner`)
  with parallel execution, ECO opening books (295 positions), and
  Glicko-2 ELO rating updates.
- Cross-provider battles: any model vs any other model.
- Free-tier auto-detection for OpenRouter `:free` and similar models.
- Comprehensive test suite: **166 tests** covering move parsing,
  providers, ELO, capabilities, integration, and the demo replay engine.
- Documentation: `README.md`, `DEPLOYMENT.md`, `CONTRIBUTING.md`, plus
  inline module docstrings.

### Security
- API keys are never logged or persisted server-side; each provider
  uses format validation only (no key exfiltration paths).
