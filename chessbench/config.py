"""Centralized configuration loaded from environment variables.

Loads provider API keys at runtime from environment variables. Providers and the Streamlit UI
reference these helpers and constants so that configuration stays consistent.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_provider_key(provider_name: str) -> str | None:
    """Retrieve API key for provider from environment variables."""
    env_var = f"{provider_name.upper()}_API_KEY"
    return os.getenv(env_var)


# LLM provider API keys
OPENAI_API_KEY: str | None = get_provider_key("openai")
ANTHROPIC_API_KEY: str | None = get_provider_key("anthropic")
GOOGLE_API_KEY: str | None = get_provider_key("google")
GROQ_API_KEY: str | None = get_provider_key("groq")
NIM_API_KEY: str | None = get_provider_key("nim")
OPENROUTER_API_KEY: str | None = get_provider_key("openrouter")
TOGETHER_API_KEY: str | None = get_provider_key("together")
FIREWORKS_API_KEY: str | None = get_provider_key("fireworks")
DEEPINFRA_API_KEY: str | None = get_provider_key("deepinfra")

# Local provider base URLs (no API key required)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
NIM_BASE_URL: str = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
