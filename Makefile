.PHONY: install test lint fmt clean run docker-build docker-run check

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=chess_fight --cov-report=term-missing

lint:
	ruff check .
	mypy --config-file pyproject.toml .

fmt:
	ruff check . --fix
	ruff format .

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info/ runs/

run:
	streamlit run streamlit_app.py

docker-build:
	docker build -t chess_fight:latest .

docker-run:
	docker run --rm -d -p 8501:8501 --name chess_fight chess_fight:latest

check: lint test
