# Dockerfile for ChessBench
# Multi-stage build for minimal production image

# =============================================================================
# Build Stage
# =============================================================================
FROM python:3.11-slim AS builder

# Install Python dependencies
WORKDIR /app
COPY pyproject.toml requirements-lock.txt README.md ./
COPY chessbench/ ./chessbench/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-lock.txt && \
    pip install --no-cache-dir --no-deps .

# =============================================================================
# Runtime Stage
# =============================================================================
FROM python:3.11-slim AS runtime

# Install the only required runtime system dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r chessbench && useradd -r -g chessbench chessbench

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy runtime entry points and configuration. The installed package remains
# root-owned under site-packages; only the run-output directory is writable.
WORKDIR /app
COPY streamlit_app.py ./
COPY benchmark.yaml ./
COPY README.md ./
COPY LICENSE ./

# Create the only writable application directory.
RUN mkdir -p /app/runs && chown chessbench:chessbench /app/runs

# Switch to non-root user
USER chessbench

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHESS_FIGHT_RUNS_ROOT=/app/runs \
    STOCKFISH_PATH=/usr/games/stockfish

# Default command shows help
ENTRYPOINT ["chessbench"]
CMD ["--help"]

# Health check (for container orchestration)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD chessbench --version || exit 1

# Labels
LABEL org.opencontainers.image.title="ChessBench" \
      org.opencontainers.image.description="LLM Critical Thinking Benchmark via Chess" \
      org.opencontainers.image.vendor="ChessBench Contributors" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/3bdrahman/chessbench"
