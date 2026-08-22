"""Structured logging for benchmark games."""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import chess


class MoveQuality(Enum):
    """Move quality classification based on centipawn loss."""
    BEST = "best"           # 0 cp loss
    EXCELLENT = "excellent" # < 10 cp loss
    GOOD = "good"           # < 50 cp loss
    INACCURACY = "inaccuracy"  # < 100 cp loss
    MISTAKE = "mistake"     # < 300 cp loss
    BLUNDER = "blunder"     # >= 300 cp loss


@dataclass
class MoveLogEntry:
    """Single move log entry."""
    game_id: str
    move_number: int
    player: str
    color: str
    fen_before: str
    move_uci: str
    move_san: str
    llm_latency_ms: int
    llm_tokens_in: int | None
    llm_tokens_out: int | None
    llm_raw_response: str
    thinking_trace: str | None
    prompt_hash: str
    validation_retries: int
    timestamp_utc: str
    # Stockfish evaluation fields
    eval_cp_score: int | None = None
    eval_mate_in: int | None = None
    eval_best_move_uci: str | None = None
    eval_best_move_cp: int | None = None
    eval_top3_moves: list[dict[str, Any]] | None = None
    eval_depth: int | None = None
    eval_time_ms: int | None = None
    # Move quality metrics
    cp_loss: int | None = None
    move_quality: str | None = None
    is_best_move: bool = False
    # Thinking trace analysis
    thinking_chars: int | None = None
    thinking_words: int | None = None
    thinking_has_structured: bool | None = None
    thinking_mentions_tactics: bool | None = None
    thinking_mentions_strategy: bool | None = None
    thinking_mentions_time_pressure: bool | None = None
    thinking_mentions_material: bool | None = None
    thinking_mentions_positional: bool | None = None
    thinking_mentions_king_safety: bool | None = None


@dataclass
class GameLogEntry:
    """Complete game log entry."""
    game_id: str
    white_player: str
    black_player: str
    white_provider: str
    black_provider: str
    opening_eco: str | None
    opening_name: str | None
    opening_fen: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    result_numeric: float  # 1.0, 0.0, 0.5
    moves: list[MoveLogEntry]
    total_moves: int
    game_duration_sec: float
    timestamp_utc: str
    config: dict[str, Any]
    termination_reason: str = "unknown"


class BenchmarkLogger:
    """Structured JSONL logger for benchmark runs."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.game_log_path = self.run_dir / "games.jsonl"
        self.move_log_path = self.run_dir / "moves.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.pgn_path = self.run_dir / "games.pgn"
        self.error_log_path = self.run_dir / "errors.jsonl"

        self.current_game_id: str = ""
        self.current_game_moves: list[MoveLogEntry] = []
        self.current_game_info: dict[str, Any] = {}
        self.run_config: dict[str, Any] = {}
        self.games_completed: list[GameLogEntry] = []

    def start_run(self, config: dict[str, Any]) -> None:
        """Start a new benchmark run."""
        self.run_config = config
        # Clear previous logs
        for path in [self.game_log_path, self.move_log_path, self.pgn_path, self.error_log_path]:
            if path.exists():
                path.unlink()

    def start_game(self, white_player: str, black_player: str,
                   white_provider: str, black_provider: str,
                   opening_eco: str | None = None,
                   opening_name: str | None = None,
                   opening_fen: str | None = None) -> None:
        """Start logging a new game."""
        self.current_game_id = str(uuid.uuid4())[:8]
        self.current_game_moves = []
        self.current_game_info = {
            'white_player': white_player,
            'black_player': black_player,
            'white_provider': white_provider,
            'black_provider': black_provider,
            'opening_eco': opening_eco,
            'opening_name': opening_name,
            'opening_fen': opening_fen or chess.STARTING_FEN,
        }

    def log_move(self, move_number: int, player: str, color: str,
                 fen_before: str, move_uci: str, move_san: str,
                 llm_latency_ms: int, llm_tokens_in: int | None,
                 llm_tokens_out: int | None, llm_raw_response: str,
                 thinking_trace: str | None, prompt_hash: str,
                 validation_retries: int,
                 eval_cp_score: int | None = None,
                 eval_mate_in: int | None = None,
                 eval_best_move_uci: str | None = None,
                 eval_best_move_cp: int | None = None,
                 eval_top3_moves: list[dict[str, Any]] | None = None,
                 eval_depth: int | None = None,
                 eval_time_ms: int | None = None) -> None:
        """Log a single move."""
        # Calculate move quality metrics from Stockfish evaluation
        cp_loss = None
        move_quality = None
        is_best_move = False

        if eval_cp_score is not None and eval_best_move_cp is not None:
            # cp_loss = best_move_cp - move_cp (positive means loss)
            cp_loss = eval_best_move_cp - eval_cp_score
            is_best_move = cp_loss == 0

            # Classify move quality based on cp_loss
            if cp_loss == 0:
                move_quality = MoveQuality.BEST.value
            elif cp_loss < 10:
                move_quality = MoveQuality.EXCELLENT.value
            elif cp_loss < 50:
                move_quality = MoveQuality.GOOD.value
            elif cp_loss < 100:
                move_quality = MoveQuality.INACCURACY.value
            elif cp_loss < 300:
                move_quality = MoveQuality.MISTAKE.value
            else:
                move_quality = MoveQuality.BLUNDER.value

        # Analyze thinking trace if present
        thinking_chars = None
        thinking_words = None
        thinking_has_structured = None
        thinking_mentions_tactics = None
        thinking_mentions_strategy = None
        thinking_mentions_time_pressure = None
        thinking_mentions_material = None
        thinking_mentions_positional = None
        thinking_mentions_king_safety = None

        if thinking_trace and thinking_trace.strip():
            from chessbench.models.thinking import analyze_thinking
            trace = analyze_thinking(thinking_trace)
            thinking_chars = trace.char_count
            thinking_words = trace.word_count
            thinking_has_structured = trace.has_structured_reasoning
            thinking_mentions_tactics = trace.mentions_tactics
            thinking_mentions_strategy = trace.mentions_strategy
            thinking_mentions_time_pressure = trace.mentions_time_pressure
            thinking_mentions_material = trace.mentions_material
            thinking_mentions_positional = trace.mentions_positional
            thinking_mentions_king_safety = trace.mentions_king_safety

        entry = MoveLogEntry(
            game_id=self.current_game_id,
            move_number=move_number,
            player=player,
            color=color,
            fen_before=fen_before,
            move_uci=move_uci,
            move_san=move_san,
            llm_latency_ms=llm_latency_ms,
            llm_tokens_in=llm_tokens_in,
            llm_tokens_out=llm_tokens_out,
            llm_raw_response=llm_raw_response,
            thinking_trace=thinking_trace,
            prompt_hash=prompt_hash,
            validation_retries=validation_retries,
            timestamp_utc=datetime.now(UTC).isoformat() + 'Z',
            eval_cp_score=eval_cp_score,
            eval_mate_in=eval_mate_in,
            eval_best_move_uci=eval_best_move_uci,
            eval_best_move_cp=eval_best_move_cp,
            eval_top3_moves=eval_top3_moves,
            eval_depth=eval_depth,
            eval_time_ms=eval_time_ms,
            cp_loss=cp_loss,
            move_quality=move_quality,
            is_best_move=is_best_move,
            thinking_chars=thinking_chars,
            thinking_words=thinking_words,
            thinking_has_structured=thinking_has_structured,
            thinking_mentions_tactics=thinking_mentions_tactics,
            thinking_mentions_strategy=thinking_mentions_strategy,
            thinking_mentions_time_pressure=thinking_mentions_time_pressure,
            thinking_mentions_material=thinking_mentions_material,
            thinking_mentions_positional=thinking_mentions_positional,
            thinking_mentions_king_safety=thinking_mentions_king_safety,
        )
        self.current_game_moves.append(entry)

        # Write immediately to moves.jsonl
        with open(self.move_log_path, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')

    def end_game(self, result: str, result_numeric: float,
                 total_moves: int, game_duration_sec: float,
                 termination_reason: str = "unknown") -> None:
        """End current game and write complete game log."""
        game_entry = GameLogEntry(
            game_id=self.current_game_id,
            white_player=self.current_game_info['white_player'],
            black_player=self.current_game_info['black_player'],
            white_provider=self.current_game_info['white_provider'],
            black_provider=self.current_game_info['black_provider'],
            opening_eco=self.current_game_info['opening_eco'],
            opening_name=self.current_game_info['opening_name'],
            opening_fen=self.current_game_info['opening_fen'],
            result=result,
            result_numeric=result_numeric,
            moves=self.current_game_moves,
            total_moves=total_moves,
            game_duration_sec=game_duration_sec,
            timestamp_utc=datetime.now(UTC).isoformat() + 'Z',
            config=self.run_config,
            termination_reason=termination_reason
        )

        self.games_completed.append(game_entry)

        # Write to games.jsonl
        with open(self.game_log_path, 'a') as f:
            f.write(json.dumps(asdict(game_entry)) + '\n')

        # Write PGN
        self._write_pgn(game_entry)

        self.current_game_id = ""
        self.current_game_moves = []
        self.current_game_info = {}

    def _write_pgn(self, game: GameLogEntry) -> None:
        """Write game to PGN file."""
        pgn_lines = [
            '[Event "Chess LLM Benchmark"]',
            '[Site "Local"]',
            f'[Date "{datetime.now(UTC).strftime("%Y.%m.%d")}"]',
            f'[Round "{game.game_id}"]',
            f'[White "{game.white_player}"]',
            f'[Black "{game.black_player}"]',
            f'[Result "{game.result}"]',
            f'[WhiteProvider "{game.white_provider}"]',
            f'[BlackProvider "{game.black_provider}"]',
            f'[OpeningECO "{game.opening_eco or "?"}"]',
            f'[OpeningName "{game.opening_name or "?"}"]',
            f'[GameDuration "{game.game_duration_sec:.1f}"]',
            '',
        ]

        # Convert stored UCI moves to SAN by replaying them from the opening
        # FEN. SAN requires board context (disambiguation, capture/check
        # markers), so it can't be captured once at log time and must be derived
        # from the move-by-move board state here.
        board = chess.Board(game.opening_fen)
        move_text = []
        for ply, move_log in enumerate(game.moves):
            move = chess.Move.from_uci(move_log.move_uci)
            if move in board.legal_moves:
                san = board.san(move)
                board.push(move)
            else:
                san = move_log.move_uci

            if ply % 2 == 0:
                move_text.append(f'{ply//2 + 1}. {san}')
            else:
                move_text.append(san)

        pgn_lines.append(' '.join(move_text))
        pgn_lines.append(game.result)
        pgn_lines.append('')

        with open(self.pgn_path, 'a') as f:
            f.write('\n'.join(pgn_lines) + '\n\n')

    def write_summary(self) -> None:
        """Write run summary."""
        summary = {
            'run_id': self.run_dir.name,
            'timestamp_utc': datetime.now(UTC).isoformat() + 'Z',
            'config': self.run_config,
            'total_games': len(self.games_completed),
            'results': {
                'white_wins': sum(1 for g in self.games_completed if g.result == '1-0'),
                'black_wins': sum(1 for g in self.games_completed if g.result == '0-1'),
                'draws': sum(1 for g in self.games_completed if g.result == '1/2-1/2'),
            },
            'players': list(set(
                [g.white_player for g in self.games_completed] +
                [g.black_player for g in self.games_completed]
            )),
            'total_moves': sum(g.total_moves for g in self.games_completed),
            'total_duration_sec': sum(g.game_duration_sec for g in self.games_completed),
        }

        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def get_pgn_content(self) -> str:
        """Get all PGN content."""
        if self.pgn_path.exists():
            return self.pgn_path.read_text()
        return ""

    def log_error(self, game_index: int, white: str, black: str, error: str) -> None:
        """Record a per-game error to errors.jsonl.

        Called when a single game fails (auth/rate-limit errors are fatal and
        abort the run, not recorded here).
        """
        entry = {
            "game_index": game_index,
            "white": white,
            "black": black,
            "error": error,
            "timestamp_utc": datetime.now(UTC).isoformat() + 'Z',
        }
        with open(self.error_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    # Test
    logger = BenchmarkLogger("/tmp/test_run")
    logger.start_run({"test": True})

    logger.start_game("GPT-4o", "Claude-3.5-Sonnet", "openai", "anthropic", "A00", "Polish Opening")

    board = chess.Board()
    logger.log_move(1, "GPT-4o", "white", board.fen(), "b2b4", "b4", 500, 100, 5, "b4", "<thinking>...</thinking>", "hash1", 0)
    board.push(chess.Move.from_uci("b2b4"))

    logger.log_move(1, "Claude-3.5-Sonnet", "black", board.fen(), "e7e5", "e5", 400, 100, 5, "e5", "<thinking>...</thinking>", "hash2", 0)
    board.push(chess.Move.from_uci("e7e5"))

    logger.end_game("1-0", 1.0, 2, 2.0)
    logger.write_summary()

    print("Test complete!")
    print(logger.get_pgn_content())
