"""Centralized configuration loaded from environment variables and Streamlit secrets.

Loads all provider API keys at runtime. Providers and the Streamlit UI
reference these helpers and constants so that configuration stays consistent
across CLI, Docker, and hosted Streamlit Cloud environments.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_provider_key(provider_name: str) -> str | None:
    """Retrieve API key for provider from env vars or Streamlit secrets."""
    env_var = f"{provider_name.upper()}_API_KEY"
    key = os.getenv(env_var)
    if key:
        return key

    # Streamlit secrets fallback for hosted Streamlit Cloud
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            secret_key_lower = f"{provider_name.lower()}_api_key"
            secret_key_upper = env_var
            return st.secrets.get(secret_key_lower) or st.secrets.get(secret_key_upper)
    except Exception:
        pass

    return None


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
