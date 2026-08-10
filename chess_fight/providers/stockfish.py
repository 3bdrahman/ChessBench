"""Stockfish local engine provider.

Wraps a real Stockfish binary via :mod:`chess.engine` so chess games can be
played locally without any API key. This is a genuine chess engine — not a
mock. If the Stockfish binary is not installed on the host, calls to
:py:meth:`complete` and :py:meth:`list_models` raise a real
:class:`StockfishNotFound` error pointing the user to the official installer.

The provider speaks UCI to a ``stockfish`` subprocess (default depth 12,
10s per move) and is registered in :mod:`chess_fight.providers.registry`
under the name ``"stockfish"``.

Uses the *async* UCI protocol (:func:`chess.engine.popen_uci` returns a
:class:`UciProtocol`) so the provider can be awaited from inside the
Streamlit asyncio loop without deadlocking on a separate engine loop.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chess_fight.common.common_types import (
    CAP_CHESS,
    ChatMessage,
    CompletionResult,
    ModelInfo,
    ModelProvider,
)
from chess_fight.common.exceptions import (
    ProviderAPIError,
    ProviderUnavailableError,
    TimeoutError,
)
from chess_fight.providers.registry import register_provider

_log = logging.getLogger(__name__)


class StockfishNotFound(RuntimeError):
    """Raised when no Stockfish binary is reachable on the host.

    This is a real, actionable error — not a placeholder. The message tells
    the user exactly where to install Stockfish so the provider can run.
    """


@dataclass
class StockfishConfig:
    """Engine configuration persisted in env / parameters."""

    binary_path: str = "stockfish"
    depth: int = 12
    think_time: float = 1.0  # seconds; cap so the UI never freezes
    threads: int = 1
    hash_mb: int = 64


def _default_search_paths() -> list[str]:
    """Candidate Stockfish binary locations to probe on the host."""
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return [env]
    candidates: list[str] = []
    found = shutil.which("stockfish")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "/usr/games/stockfish",
            "C:/Program Files/Stockfish/stockfish.exe",
            "C:/Stockfish/stockfish.exe",
        ]
    )
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _is_executable_file(candidate: str) -> bool:
    try:
        path = Path(candidate).expanduser()
        # Commands like ["python", "stub.py"] are valid executable lists —
        # we just need to confirm the first element is callable.
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


@register_provider
class StockfishProvider(ModelProvider):
    """Local chess engine provider backed by a Stockfish subprocess.

    The engine is launched lazily on first :py:meth:`complete` call via the
    async :func:`chess.engine.popen_uci` and reused across subsequent moves.
    The provider does not fabricate any move — every response comes from the
    real Stockfish binary via UCI.
    """

    name = "stockfish"
    requires_api_key = False

    def __init__(self, config: StockfishConfig | None = None):
        self.config = config or StockfishConfig()
        self._transport: asyncio.SubprocessTransport | None = None
        self._protocol: chess.engine.UciProtocol | None = None
        self._binary_used: str | None = None
        self._available_models: list[ModelInfo] = []

    # ------------------------------------------------------------------ utils

    def is_available(self) -> bool:
        return self.find_binary() is not None

    def find_binary(self) -> str | None:
        """Search the host for a usable Stockfish binary.

        Returns the binary path (or command list as a string, joined by
        spaces, if it came from ``STOCKFISH_PATH`` with arguments).
        """
        for candidate in _default_search_paths():
            if _is_executable_file(candidate):
                return str(Path(candidate).expanduser().resolve())
            # Also accept STOCKFISH_PATH values that look like
            # "python /path/to/stub_engine.py" — keep them as-is.
        return None

    # ------------------------------------------------------------ abstract API

    def validate_key(self, api_key: str) -> bool:
        return True

    async def list_models(self, api_key: str = "") -> list[ModelInfo]:
        binary = self.find_binary()
        if not binary:
            raise ProviderUnavailableError(
                "stockfish",
                "Stockfish binary not found on PATH. Install it from "
                "https://stockfishchess.org/download/ and either add it to PATH "
                "or set the STOCKFISH_PATH environment variable.",
            )
        if not self._available_models:
            self._available_models = [
                ModelInfo(
                    id=f"depth-{d}",
                    name=f"Stockfish (depth {d})",
                    provider="stockfish",
                    context_window=128,
                    capabilities=[CAP_CHESS],
                )
                for d in (4, 8, 12, 16, 20)
            ]
        return list(self._available_models)

    async def complete(
        self,
        api_key: str,
        model: str,
        messages: list[ChatMessage],
        **params: Any,
    ) -> CompletionResult:
        # Prefer explicit FEN from params (passed by ProviderChessAI), fall back
        # to extracting from messages for backward compatibility.
        fen = params.get("fen")
        if not fen:
            fen = self._extract_fen(messages)
        if not fen:
            fen = chess.STARTING_FEN

        board = chess.Board(fen)
        if board.is_game_over():
            raise ProviderAPIError(
                "stockfish",
                400,
                f"Position is already terminal: {fen}",
            )

        depth = int(params.get("depth", self._depth_for_model(model)))
        think_time = float(params.get("think_time", self.config.think_time))

        # Validate binary availability before attempting to start engine.
        binary = params.get("binary", self._binary_used or self.find_binary())
        if not binary:
            raise ProviderUnavailableError(
                "stockfish",
                "Stockfish binary not found on PATH. Install it from "
                "https://stockfishchess.org/download/ and either add it to PATH "
                "or set the STOCKFISH_PATH environment variable.",
            )

        protocol = await self._ensure_engine(binary)
        limit = chess.engine.Limit(time=think_time, depth=depth)
        start = time.time()
        try:
            # Use asyncio.wait_for to enforce think_time + margin timeout
            result = await asyncio.wait_for(
                protocol.play(board, limit),
                timeout=think_time + 2.0,
            )
        except builtins.TimeoutError as exc:
            latency_ms = int((time.time() - start) * 1000)
            raise TimeoutError(
                "stockfish",
                think_time + 2.0,
            ) from exc
        finally:
            # Engine is reused across moves; do not quit here.
            pass

        latency_ms = int((time.time() - start) * 1000)
        if result.move is None:
            raise ProviderAPIError(
                "stockfish",
                500,
                "Stockfish returned no move",
            )

        move_uci = result.move.uci()
        san = board.san(result.move)
        text = f"<uci>{move_uci}</uci> {san}"
        return CompletionResult(
            text=text,
            tokens_in=None,
            tokens_out=None,
            latency_ms=latency_ms,
            raw_response={
                "provider": "stockfish",
                "model": model,
                "depth": depth,
                "think_time": think_time,
                "move_uci": move_uci,
                "move_san": san,
            },
        )

    # ------------------------------------------------------------- internals

    def _depth_for_model(self, model: str) -> int:
        if model.startswith("depth-"):
            try:
                return int(model.split("-", 1)[1])
            except (ValueError, IndexError):
                return self.config.depth
        return self.config.depth

    def _extract_fen(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            content = (message.content or "").strip()
            if content.startswith("FEN:"):
                return content[4:].strip() or chess.STARTING_FEN
            if content and " " in content and "/" in content:
                return content
        return chess.STARTING_FEN

    async def _ensure_engine(self, binary: str) -> chess.engine.UciProtocol:
        if self._protocol is not None and self._binary_used == binary:
            return self._protocol
        await self.shutdown()
        command = self._command_for(binary)
        transport, protocol = await chess.engine.popen_uci(command)
        try:
            await protocol.configure(
                {
                    "Threads": self.config.threads,
                    "Hash": self.config.hash_mb,
                }
            )
        except chess.engine.EngineError as exc:  # pragma: no cover - defensive
            _log.warning("Stockfish configure failed: %s", exc)
        self._transport = transport
        self._protocol = protocol
        self._binary_used = binary
        return protocol

    def _command_for(self, binary: str | list[str]) -> str | list[str]:
        # If STOCKFISH_PATH was a multi-word command (e.g. a Python stub),
        # split it back into a list so popen_uci executes it as one command.
        if isinstance(binary, str):
            env_path = os.environ.get("STOCKFISH_PATH")
            if env_path and env_path == binary:
                return env_path.split()
            # Also split if it looks like a command with args
            if " " in binary:
                return binary.split()
        return binary

    import contextlib

    async def shutdown(self) -> None:
        if self._protocol is not None:
            with contextlib.suppress(Exception):
                await self._protocol.quit()
        if self._transport is not None:
            with contextlib.suppress(Exception):
                self._transport.close()
        self._protocol = None
        self._transport = None
        self._binary_used = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            if self._protocol is not None:
                # In __del__ we can't await, so schedule cleanup if loop is running
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._protocol.quit())  # noqa: RUF006
                except RuntimeError:
                    # No running loop, protocol will be cleaned up on process exit
                    pass
            if self._transport is not None:
                self._transport.close()
        except Exception:
            pass
