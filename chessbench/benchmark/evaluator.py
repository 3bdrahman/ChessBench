"""Stockfish evaluator for ground-truth position analysis."""



import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import chess
import chess.engine

_log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a Stockfish evaluation."""
    cp_score: int | None = None
    mate_in: int | None = None
    best_move_uci: str | None = None
    best_move_cp: int | None = None
    top3_moves: list[dict[str, Any]] | None = None
    depth: int | None = None
    time_ms: int | None = None


class StockfishEvaluator:
    """Async Stockfish evaluator for position analysis."""

    def __init__(
        self,
        binary_path: str = "stockfish",
        depth: int = 18,
        time_ms: int = 100,
        threads: int = 1,
        hash_mb: int = 64,
        multipv: int = 3,
    ):
        self.binary_path = binary_path
        self.depth = depth
        self.time_ms = time_ms
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = max(1, multipv)
        self._transport: Any = None
        self._engine: Any = None
        self._available = False
        self._check_binary()

    def _check_binary(self) -> bool:
        """Check if Stockfish binary is available."""
        try:
            import shutil
            if shutil.which(self.binary_path):
                self._available = True
                return True
            # Try common paths
            common_paths = [
                "/usr/bin/stockfish",
                "/usr/local/bin/stockfish",
                "/opt/homebrew/bin/stockfish",
                "/usr/games/stockfish",
            ]
            for path in common_paths:
                if shutil.which(path):
                    self.binary_path = path
                    self._available = True
                    return True
        except Exception:
            pass
        self._available = False
        return False

    async def start(self) -> bool:
        """Start the Stockfish engine."""
        if not self._available:
            _log.warning("Stockfish binary not found at %s", self.binary_path)
            return False

        try:
            self._transport, self._engine = await chess.engine.popen_uci(self.binary_path)
            if self._engine is not None:
                await self._engine.configure({
                    "Threads": self.threads,
                    "Hash": self.hash_mb,
                })
            return True
        except Exception as e:
            _log.warning("Failed to start Stockfish: %s", e)
            self._available = False
            return False

    async def stop(self) -> None:
        """Stop the Stockfish engine."""
        if self._engine:
            with contextlib.suppress(Exception):
                await self._engine.quit()
            self._engine = None
        if self._transport:
            with contextlib.suppress(Exception):
                self._transport.close()
            self._transport = None

    async def __aenter__(self) -> StockfishEvaluator:
        await self.start()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.stop()

    async def evaluate(self, board: chess.Board) -> EvaluationResult | None:
        """
        Evaluate a position with Stockfish.

        Returns EvaluationResult with cp_score, best_move, top3_moves, etc.
        Returns None if engine is not available.
        """
        if not self._available or not self._engine:
            return None

        start_time = time.time()

        try:
            # Use depth-limited analysis with MultiPV
            limit = chess.engine.Limit(depth=self.depth)
            info = await self._engine.analyse(board, limit, multipv=self.multipv)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # When multipv > 1, info is a list of dicts. We use the principal variation for the main score.
            pv_info = info[0] if isinstance(info, list) and info else (info if isinstance(info, dict) else {})

            # Extract score
            score = pv_info.get("score")
            cp_score = None
            mate_in = None
            if score:
                if score.is_mate():
                    mate_in = score.white().mate()
                else:
                    cp_score = score.white().score()

            # Extract best move
            pv = pv_info.get("pv")
            best_move = pv[0] if pv else None
            best_move_uci = best_move.uci() if best_move else None

            # Extract best move score
            best_move_cp = None
            if best_move and "score" in pv_info:
                best_move_cp = pv_info["score"].white().score()

            # Extract top 3 moves from MultiPV
            top3_moves: list[dict[str, Any]] = []
            if isinstance(info, list):
                # MultiPV returns a list of info dicts, one per PV line
                for pv_info in info[:self.multipv]:
                    pv = pv_info.get("pv", [])
                    if pv:
                        move = pv[0]
                        score = pv_info.get("score")
                        cp = None
                        if score and not score.is_mate():
                            cp = score.white().score()
                        top3_moves.append({
                            "uci": move.uci(),
                            "san": board.san(move),
                            "cp": cp,
                        })
            else:
                # Fallback to single PV
                pv = info.get("pv", [])
                for move in pv[:3]:
                    if move:
                        top3_moves.append({
                            "uci": move.uci(),
                            "san": board.san(move),
                            "cp": None,
                        })

            return EvaluationResult(
                cp_score=cp_score,
                mate_in=mate_in,
                best_move_uci=best_move_uci,
                best_move_cp=best_move_cp,
                top3_moves=top3_moves,
                depth=self.depth,
                time_ms=elapsed_ms,
            )

        except Exception as e:
            _log.warning("Stockfish evaluation failed: %s", e)
            return None
