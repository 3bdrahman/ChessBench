# Contributing to AI Chess Battle

Thank you for your interest in contributing! This document outlines the process for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/chess_fight/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Screenshots if applicable

### Suggesting Features

1. Check existing issues for similar requests
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and motivation
   - Possible implementation approach

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
4. **Run the test suite**: `pytest tests/ -v`
5. **Run linting**: `ruff check . && mypy --config-file pyproject.toml .`
6. **Commit your changes**: Use clear, descriptive commit messages
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Create a Pull Request**

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/chess_fight.git
cd chess_fight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
ruff check .
mypy --config-file pyproject.toml .
```

## Code Style

- **Formatter**: Ruff (configured in `pyproject.toml`)
- **Type checker**: MyPy (configured in `pyproject.toml`)
- **Line length**: 100 characters
- **Import sorting**: Handled by Ruff (isort rules)

Run `ruff check . --fix` to auto-fix most style issues.

## Testing

- Write tests for all new functionality
- Place tests in the `tests/` directory
- Use pytest with asyncio support for async tests
- Aim for high test coverage

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=chess_fight --cov=benchmark --cov=providers --cov=game --cov=ui --cov-report=html
```

## Architecture Overview

```
chess_fight/
├── benchmark/       # Headless benchmark runner, ELO, openings, logging
├── common/          # Shared types and base classes
├── demos/           # Demo game replay functionality
├── game/            # Chess game logic (async/sync)
├── models/          # Chess AI implementations and evaluation
├── providers/       # LLM provider abstractions
├── ui/              # Streamlit web interface
└── tests/           # Test suite
```

## Pull Request Guidelines

1. **One feature/fix per PR** - Keep changes focused
2. **Clear title and description** - Explain what and why
3. **Link related issues** - Use "Fixes #123" or "Closes #123"
4. **Tests pass** - All CI checks must pass
5. **Documentation updated** - If you change user-facing features

## Release Process

Releases are managed by maintainers. Version numbers follow [Semantic Versioning](https://semver.org/).

## Questions?

Feel free to open an issue for any questions about contributing!

---

Thank you for contributing to AI Chess Battle! 🎮♟️