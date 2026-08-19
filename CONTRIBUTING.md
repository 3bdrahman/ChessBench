# Contributing to ChessBench

Thank you for contributing! ChessBench is a serious LLM benchmarking platform — we maintain high standards for code quality, reproducibility, and scientific rigor.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourorg/chessbench
cd chessbench

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run type checking
mypy chess_fight/

# Run linting
ruff check chess_fight/
```

## Development Standards

### Code Quality (Enforced by CI)

| Tool | Config | Purpose |
|------|--------|---------|
| **ruff** | `pyproject.toml` | Linting, formatting, import sorting |
| **mypy** | `pyproject.toml` | Static type checking (strict-ish) |
| **pytest** | `pyproject.toml` | Unit + integration tests |

**All three must pass before merging.**

### Type Hints

- Use type hints everywhere (functions, methods, variables)
- Prefer `list[str]` over `List[str]` (Python 3.11+)
- Use `| None` over `Optional[]`
- Avoid `Any` — use `object` or proper generics
- Run `mypy chess_fight/` before committing

### Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_elo.py -v

# With coverage
pytest tests/ --cov=chess_fight --cov-report=term-missing

# Integration tests only (require API keys)
pytest tests/test_integration.py -v
```

**Test Requirements:**
- New features need tests
- Bug fixes need regression tests
- Aim for >90% coverage on new code
- Zero-mock philosophy: test real code paths

### Code Style

- **Line length**: 100 chars
- **Quotes**: Double quotes
- **Imports**: Sorted (ruff handles this)
- **No print statements** in library code (use `logging`)
- **Async/await** throughout (no blocking calls in async functions)

## Architecture Guidelines

### Provider Abstraction

Adding a new provider? Implement `ModelProvider` in `chess_fight/providers/your_provider.py`:

```python
@register_provider
class YourProvider(ModelProvider):
    name = "yourprovider"
    requires_api_key = True

    async def list_models(self, api_key: str) -> list[ModelInfo]: ...
    async def complete(self, messages, **kwargs) -> CompletionResult: ...
    def validate_key(self, api_key: str) -> bool: ...
    async def validate_model(self, api_key: str, model_id: str) -> tuple[bool, str]: ...
```

Then register in `chess_fight/providers/__init__.py` `_PROVIDER_MODULES`.

### Benchmark Extensions

New benchmark modes go in `chess_fight/benchmark/`:
- `runner.py` — core tournament logic
- `adversarial.py` — LLM vs Stockfish calibration
- `evaluator.py` — Stockfish evaluation
- `statistics.py` — statistical analysis
- `export.py` — output formats

### Position Evaluation

Rich chess analysis in `chess_fight/models/evaluation.py`:
- `PositionEvaluator` — 50+ position features
- Used by prompt templates for demand-driven context

## Benchmark Design Principles

When adding benchmark suites or modifying evaluation:

1. **Reproducibility First** — Every run must be reproducible via config hash
2. **Statistical Rigor** — Glicko-2 with proper rating periods, confidence intervals
3. **Ground Truth** — Stockfish evaluation on every move
4. **No Cherry-Picking** — All games counted (clean terminations only for ELO)
5. **Transparent Failure** — Failed games logged, not hidden

## Adding a Benchmark Suite

1. Create `configs/your_suite.yaml` following existing patterns
2. Document in README.md suite table
3. Test: `chessbench suite --run your_suite`
4. Verify results are scientifically meaningful

## Pull Request Checklist

- [ ] `ruff check chess_fight/` passes
- [ ] `mypy chess_fight/` passes
- [ ] `pytest tests/` passes
- [ ] New code has tests
- [ ] Types are correct (no `type: ignore` without justification)
- [ ] Documentation updated (README, docstrings)
- [ ] No breaking changes without version bump discussion

## Release Process

1. Version bump in `pyproject.toml`
2. Changelog entry
3. Tag release: `git tag v0.x.x`
4. GitHub Actions publishes to PyPI

## Getting Help

- **Issues**: Bug reports, feature requests
- **Discussions**: Design questions, benchmark methodology
- **Discord/Slack**: Real-time chat (link in repo description)

## License

By contributing, you agree your contributions will be licensed under MIT.