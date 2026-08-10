"""UI error display helpers for Streamlit.

Maps typed :mod:`chess_fight.common.exceptions` to user-friendly
Streamlit messages with actionable guidance.
"""

from __future__ import annotations

from typing import Any

from chess_fight.common.exceptions import (
    AuthenticationError,
    ConnectionError,
    GameExecutionError,
    InvalidApiKeyError,
    ModelNotFoundError,
    MoveValidationError,
    NetworkError,
    NoProvidersConfiguredError,
    ProviderAPIError,
    ProviderError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    SetupError,
    TimeoutError,
)

_PROVIDER_DASHBOARDS: dict[str, str] = {
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "google": "https://aistudio.google.com/apikey",
    "groq": "https://console.groq.com/keys",
    "openrouter": "https://openrouter.ai/settings/keys",
    "nim": "https://build.nvidia.com/",
}


def _dashboard_url(provider: str) -> str:
    return _PROVIDER_DASHBOARDS.get(provider, "https://example.com/")


def render_error(st: Any, exc: BaseException) -> None:
    """Render a user-friendly error with actionable guidance.

    ``st`` is the Streamlit module (or a duck-typed mock for tests).
    """
    if isinstance(exc, NoProvidersConfiguredError):
        st.error("**No providers configured** — add API keys in the sidebar or enable Stockfish.")
        st.info("Tip: Enable Stockfish for instant local play (no API key required).")
        return

    if isinstance(exc, InvalidApiKeyError):
        provider_label = exc.provider.capitalize()
        st.error(f"**Invalid API key for {provider_label}**")
        st.caption(
            f"Expected prefix: `{exc.expected_prefix}…` — got `{exc.got_prefix}…`. "
            "Check your key at the provider dashboard."
        )
        st.markdown(f"[Open {provider_label} dashboard]({_dashboard_url(exc.provider)})")
        return

    if isinstance(exc, AuthenticationError):
        st.error(f"**Authentication failed** for {exc.provider}: {exc.detail}")
        st.caption("Your key may be expired or revoked. Generate a new one at the provider dashboard.")
        st.markdown(f"[Open {exc.provider} dashboard]({_dashboard_url(exc.provider)})")
        return

    if isinstance(exc, ModelNotFoundError):
        st.error(f"**Model not found**: `{exc.model_id}` on {exc.provider}")
        if exc.available_models:
            st.caption("Available models: " + ", ".join(exc.available_models[:5]))
            if len(exc.available_models) > 5:
                st.caption(f"  (+{len(exc.available_models) - 5} more)")
        else:
            st.caption("Check the provider's model list and try again.")
        return

    if isinstance(exc, RateLimitError):
        msg = f"Rate limited by **{exc.provider}**"
        if exc.retry_after:
            msg += f" — retry in {exc.retry_after:.0f}s"
        st.warning(msg)
        return

    if isinstance(exc, TimeoutError):
        st.warning(
            f"{exc.provider} timed out ({exc.timeout_seconds}s). "
            "The model may be slow or the service is under load. Try again."
        )
        return

    if isinstance(exc, ConnectionError):
        host_part = f" at `{exc.host}`" if exc.host else ""
        st.error(f"Cannot reach **{exc.provider}**{host_part}. Check your network or the provider's status page.")
        return

    if isinstance(exc, ProviderUnavailableError):
        st.error(f"**{exc.provider} is unavailable**: {exc}")
        return

    if isinstance(exc, QuotaExceededError):
        st.error(f"**Quota exceeded for {exc.provider}**: {exc.detail}")
        st.caption("Top up your account or switch providers.")
        return

    if isinstance(exc, GameExecutionError):
        st.error(f"**Benchmark aborted**: {exc}")
        if exc.cause is not None:
            with st.expander("Show underlying error"):
                st.code(str(exc.cause))
        return

    if isinstance(exc, SetupError):
        st.error(f"**Setup error**: {exc}")
        return

    if isinstance(exc, ProviderAPIError):
        st.error(f"**Provider error** ({exc.provider}, HTTP {exc.status_code}): {exc.detail}")
        return

    if isinstance(exc, ProviderError):
        st.error(f"**{exc.provider} error**: {exc}")
        return

    if isinstance(exc, NetworkError):
        host_part = f" at `{exc.host}`" if exc.host else ""
        st.error(f"Network error reaching **{exc.provider}**{host_part}.")
        return

    if isinstance(exc, MoveValidationError):
        st.warning(f"Model returned an unparseable move: {exc}")
        return

    # Fallback
    st.error(f"Unexpected error: {exc}")


__all__ = ["render_error"]
