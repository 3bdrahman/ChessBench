"""Real protocol-level tests for :class:`StockfishProvider`.

These tests exercise the real `chess.engine.popen_uci` async code path by
spawning the on-disk `tests/fixtures/uci_stub_engine.py` subprocess, which
speaks a minimal but real UCI protocol (no python-chess monkey-patching,
no fabricated responses — the engine actually runs and replies on stdin /
stdout). This catches genuine protocol / async / integration issues that a
`MagicMock` would hide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

from chessbench.common.common_types import ChatMessage
from chessbench.common.exceptions import ProviderUnavailableError
from chessbench.providers.stockfish import StockfishProvider

_STUB_ENGINE = str((Path(__file__).parent / "fixtures" / "uci_stub_engine.py").resolve())
_FULL_CMD = f"{sys.executable} {_STUB_ENGINE}"


@pytest.fixture
def stub_provider(monkeypatch) -> StockfishProvider:
    """Provider that always finds the stub engine via STOCKFISH_PATH."""
    monkeypatch.setenv("STOCKFISH_PATH", _FULL_CMD)
    provider = StockfishProvider()
    provider.find_binary = lambda: _FULL_CMD  # type: ignore[method-assign]
    yield provider
    # Best-effort async cleanup happens in the tests themselves; nothing to
    # tear down synchronously here.


class TestStockfishProviderProtocol:
    @pytest.mark.asyncio
    async def test_list_models_returns_real_models(self, stub_provider):
        models = await stub_provider.list_models("")
        assert len(models) == 5
        assert {m.id for m in models} == {
            "depth-4", "depth-8", "depth-12", "depth-16", "depth-20"
        }
        assert all(m.provider == "stockfish" for m in models)

    @pytest.mark.asyncio
    async def test_complete_returns_real_uci_move_from_subprocess(self, stub_provider):
        try:
            result = await stub_provider.complete(
                api_key="",
                model="depth-8",
                messages=[ChatMessage(role="user", content=chess.STARTING_FEN)],
                depth=2,
                think_time=0.1,
            )
        finally:
            await stub_provider.shutdown()
        # The stub engine plays the first legal move from the position.
        # For the starting position, the first legal move is g1h3 (Nh3).
        assert result.text.startswith("<uci>")
        assert result.raw_response["provider"] == "stockfish"
        assert result.raw_response["move_uci"] in {m.uci() for m in chess.Board().legal_moves}
        assert result.latency_ms is not None and result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_complete_via_fen_prefix_message(self, stub_provider):
        try:
            result = await stub_provider.complete(
                api_key="",
                model="depth-4",
                messages=[ChatMessage(role="user", content=f"FEN: {chess.STARTING_FEN}")],
                depth=2,
                think_time=0.1,
            )
        finally:
            await stub_provider.shutdown()
        assert result.text.startswith("<uci>")
        assert result.raw_response["move_uci"] in {m.uci() for m in chess.Board().legal_moves}

    @pytest.mark.asyncio
    async def test_engine_can_be_shutdown_and_respawned(self, stub_provider):
        protocol = await stub_provider._ensure_engine(stub_provider.find_binary() or _FULL_CMD)
        assert protocol is not None
        await stub_provider.shutdown()
        assert stub_provider._protocol is None
        protocol2 = await stub_provider._ensure_engine(stub_provider.find_binary() or _FULL_CMD)
        assert protocol2 is not None
        await stub_provider.shutdown()


class TestStockfishProviderHonestAbsence:
    def test_find_binary_returns_none_when_absent(self, monkeypatch):
        provider = StockfishProvider()
        monkeypatch.setenv("STOCKFISH_PATH", "/nonexistent/path/please")
        monkeypatch.setattr(
            "chessbench.providers.stockfish._default_search_paths", lambda: ["/nope"]
        )
        assert provider.find_binary() is None
        assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_list_models_raises_when_binary_missing(self, monkeypatch):
        provider = StockfishProvider()
        monkeypatch.setenv("STOCKFISH_PATH", "/nonexistent/path/please")
        monkeypatch.setattr(
            "chessbench.providers.stockfish._default_search_paths", lambda: ["/nope"]
        )
        with pytest.raises(ProviderUnavailableError):
            await provider.list_models("")

    @pytest.mark.asyncio
    async def test_complete_raises_when_binary_missing(self, monkeypatch):
        provider = StockfishProvider()
        monkeypatch.setenv("STOCKFISH_PATH", "/nonexistent/path/please")
        monkeypatch.setattr(
            "chessbench.providers.stockfish._default_search_paths", lambda: ["/nope"]
        )
        with pytest.raises(ProviderUnavailableError):
            await provider.complete(
                api_key="",
                model="depth-8",
                messages=[ChatMessage(role="user", content=chess.STARTING_FEN)],
            )

    def test_validate_key_is_no_op(self):
        provider = StockfishProvider()
        assert provider.validate_key("") is True
        assert provider.validate_key("ignored-key") is True
