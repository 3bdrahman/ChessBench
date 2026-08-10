"""Tests for the centralized model capability filters."""

from chess_fight.common.common_types import (
    CAP_CHESS,
    ModelInfo,
    is_chess_capable,
    is_free_tier,
)


def _info(id_: str, *, context: int | None = 8192, capabilities: list[str] | None = None) -> ModelInfo:
    return ModelInfo(
        id=id_,
        name=id_,
        provider="test",
        context_window=context,
        capabilities=list(capabilities or []),
    )


class TestIsChessCapable:
    def test_cap_flag_always_passes(self):
        """Models that explicitly advertise the chess capability pass through."""
        info = _info("anything-weird", capabilities=[CAP_CHESS])
        assert is_chess_capable(info) is True

    def test_chat_model_passes(self):
        info = _info("gpt-4o", context=128000)
        assert is_chess_capable(info) is True

    def test_claude_passes(self):
        info = _info("claude-3-5-sonnet-20241022", context=200000)
        assert is_chess_capable(info) is True

    def test_llama_passes(self):
        info = _info("meta/llama-3.1-70b-instruct", context=8192)
        assert is_chess_capable(info) is True

    def test_gemini_passes(self):
        info = _info("gemini-1.5-pro", context=1000000)
        assert is_chess_capable(info) is True

    def test_groq_free_passes(self):
        info = _info("llama-3.3-70b-versatile", context=8192)
        assert is_chess_capable(info) is True


class TestIsChessIncapable:
    def test_openai_embedding_rejected(self):
        assert is_chess_capable(_info("text-embedding-3-large")) is False

    def test_openai_whisper_rejected(self):
        assert is_chess_capable(_info("whisper-1")) is False

    def test_openai_tts_rejected(self):
        assert is_chess_capable(_info("tts-1")) is False

    def test_openai_dalle_rejected(self):
        assert is_chess_capable(_info("dall-e-3")) is False

    def test_openai_moderation_rejected(self):
        assert is_chess_capable(_info("omni-moderation-latest")) is False

    def test_openai_babbage_rejected(self):
        assert is_chess_capable(_info("babbage-002")) is False

    def test_openai_davinci_rejected(self):
        assert is_chess_capable(_info("davinci-002")) is False

    def test_openai_gpt35_instruct_rejected(self):
        """gpt-3.5-turbo-instruct is a completion model, not chat — reject."""
        assert is_chess_capable(_info("gpt-3.5-turbo-instruct")) is False

    def test_openrouter_image_rejected(self):
        assert is_chess_capable(_info("stability-ai/sdxl")) is False

    def test_nim_embed_rejected(self):
        assert is_chess_capable(_info("nvidia/nv-embedqa-e5-v5")) is False

    def test_nim_asr_rejected(self):
        assert is_chess_capable(_info("nvidia/parakeet-ctc-1.1b-asr")) is False

    def test_nim_clip_rejected(self):
        assert is_chess_capable(_info("nvidia/nvclip")) is False

    def test_gemini_imagen_rejected(self):
        assert is_chess_capable(_info("imagen-3.0-generate-002")) is False

    def test_gemini_aqa_rejected(self):
        assert is_chess_capable(_info("aqa-experimental")) is False

    def test_gemini_text_embedding_rejected(self):
        assert is_chess_capable(_info("text-embedding-004")) is False

    def test_groq_whisper_rejected(self):
        assert is_chess_capable(_info("whisper-large-v3")) is False

    def test_groq_guard_rejected(self):
        assert is_chess_capable(_info("llama-guard-3-8b")) is False

    def test_ollama_nomic_embed_rejected(self):
        assert is_chess_capable(_info("nomic-embed-text")) is False

    def test_ollama_mxbai_embed_rejected(self):
        assert is_chess_capable(_info("mxbai-embed-large")) is False

    def test_ollama_bge_rejected(self):
        assert is_chess_capable(_info("bge-large")) is False

    def test_zero_context_rejected(self):
        assert is_chess_capable(_info("some-chat-model", context=128)) is False

    def test_very_small_context_rejected(self):
        assert is_chess_capable(_info("some-chat-model", context=200)) is False

    def test_unknown_context_passes(self):
        """If context window is unknown (None), don't reject — only reject known-bad values."""
        assert is_chess_capable(_info("some-chat-model", context=None)) is True


class TestIsFreeTier:
    def test_pricing_tier_free(self):
        assert is_free_tier(_info("anything", capabilities=[CAP_CHESS])) is False  # default

    def test_explicit_free_pricing(self):
        info = _info("llama-3.3-70b-versatile")
        info.pricing_tier = "free"
        assert is_free_tier(info) is True

    def test_openrouter_free_suffix(self):
        assert is_free_tier(_info("meta-llama/llama-3.3-70b-instruct:free")) is True

    def test_paid_model(self):
        assert is_free_tier(_info("gpt-4o")) is False
