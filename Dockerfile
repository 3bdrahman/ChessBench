# Dockerfile for ChessBench
# Multi-stage build for minimal production image

# =============================================================================
# Build Stage
# =============================================================================
FROM python:3.11-slim AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY pyproject.toml README.md ./
COPY chess_fight/ ./chess_fight/
COPY configs/ ./configs/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[dev]"

# =============================================================================
# Runtime Stage
# =============================================================================
FROM python:3.11-slim AS runtime

# Install runtime dependencies (Stockfish, minimal system deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r chessbench && useradd -r -g chessbench chessbench

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
WORKDIR /app
COPY --from=builder /app/chess_fight ./chess_fight
COPY --from=builder /app/configs ./configs
COPY streamlit_app.py ./
COPY benchmark.yaml ./
COPY README.md ./
COPY LICENSE ./

# Create runs directory with correct permissions
RUN mkdir -p /app/runs && chown -R chessbench:chessbench /app

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
      org.opencontainers.image.source="https://github.com/yourorg/chessbench"