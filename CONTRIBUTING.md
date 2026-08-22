# Contributing to ChessBench

Thank you for contributing! ChessBench is a rigorous LLM benchmarking platform — we maintain high standards for code quality, reproducibility, and scientific integrity.

> [!IMPORTANT]
> **Quality Gates**: All pull requests must pass `ruff check`, `mypy`, and the complete `pytest` test suite (295 tests passing).

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/3bdrahman/chessbench.git
cd chessbench

# Install editable package with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run the test suite (295 tests)
PYTHONPATH=. pytest tests/ -v

# Run static type checking
mypy chessbench/

# Run code linting and formatting checks
ruff check chessbench/
```

---

## Development Standards

### Code Quality (Enforced in CI)

| Tool | Config File | Purpose | Command |
|---|---|---|---|
| **pytest** | `pyproject.toml` | Unit & integration tests | `pytest tests/ -v` |
| **ruff** | `pyproject.toml` | Linting & formatting | `ruff check chessbench/` |
| **mypy** | `pyproject.toml` | Static type checking | `mypy chessbench/` |

> [!NOTE]
> All three tools must execute cleanly with zero errors before merging.

### Type Hints Guidelines

- Use standard Python 3.11+ type annotations everywhere (`list[str]`, `dict[str, int]`, `tuple[int, ...]`).
- Use union syntax `X | None` instead of `Optional[X]`.
- Avoid untyped `Any` — use strict protocols, generics, or `object`.
- Always run `mypy chessbench/` before creating a pull request.

### Testing Guidelines

```bash
# Run all unit and integration tests
PYTHONPATH=. pytest tests/ -v

# Run specific module tests
pytest tests/test_elo.py -v

# Run tests with coverage output
pytest tests/ --cov=chessbench --cov-report=term-missing

# Integration tests only (require live API keys in environment)
pytest tests/test_integration.py -v
```

> [!TIP]
> **Zero-Mock Philosophy**: Test real code paths wherever feasible. We prefer executing actual logic over relying on extensive mock structures.

---

## Architecture Guidelines

### Provider Abstraction

Adding a new provider backend? Implement `ModelProvider` in `chessbench/providers/your_provider.py`:

```python
from chessbench.common.common_types import ChatMessage, CompletionResult, ModelInfo
from chessbench.providers.registry import register_provider

@register_provider
class YourProvider(ModelProvider):
    name = "yourprovider"
    requires_api_key = True

    async def list_models(self, api_key: str) -> list[ModelInfo]: ...
    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResult: ...
    def validate_key(self, api_key: str) -> bool: ...
    async def validate_model(self, api_key: str, model_id: str) -> tuple[bool, str]: ...
```

Then register module `"yourprovider"` in `_PROVIDER_MODULES` inside [`chessbench/providers/__init__.py`](file:///var/home/usef/coding/chessbench/chessbench/providers/__init__.py).

Existing supported providers include:
`anthropic`, `deepinfra`, `fireworks`, `google`, `groq`, `nim`, `openai`, `openrouter`, `together`.

### Benchmark Core & Analytics

New benchmark components live under [`chessbench/benchmark/`](file:///var/home/usef/coding/chessbench/chessbench/benchmark/):
- [`runner.py`](file:///var/home/usef/coding/chessbench/chessbench/benchmark/runner.py) — Core async tournament execution engine
- [`adversarial.py`](file:///var/home/usef/coding/chessbench/chessbench/benchmark/adversarial.py) — LLM vs Stockfish calibration module
- [`evaluator.py`](file:///var/home/usef/coding/chessbench/chessbench/benchmark/evaluator.py) — Ground-truth Stockfish move evaluation
- [`statistics.py`](file:///var/home/usef/coding/chessbench/chessbench/benchmark/statistics.py) — Glicko-2 ratings & statistical metrics
- [`export.py`](file:///var/home/usef/coding/chessbench/chessbench/benchmark/export.py) — Multi-format dataset exporters (Parquet, CSV, PGN, HTML, JSON)

---

## Pull Request Checklist

- [ ] `ruff check chessbench/` passes without warnings.
- [ ] `mypy chessbench/` passes without errors.
- [ ] `PYTHONPATH=. pytest tests/` passes all 295 unit and integration tests.
- [ ] New functionality includes corresponding unit test coverage.
- [ ] Documentation (`README.md`, docstrings, `DEPLOYMENT.md`) is up to date.

---

## License

By contributing to ChessBench, you agree that your contributions will be licensed under the MIT License.