"""Centralized configuration loaded from environment variables.

Loads all provider API keys at import time. Providers and the Streamlit UI
should reference these constants instead of reading os.getenv directly so
that configuration stays consistent across the codebase.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# LLM provider API keys
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
NIM_API_KEY: str | None = os.getenv("NIM_API_KEY")
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

# Local provider base URLs (no API key required)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
NIM_BASE_URL: str = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
