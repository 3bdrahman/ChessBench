"""Structured logging for benchmark games."""

import json
import uuid
from dataclasses import asdict, dataclass, field
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
    # Rich Telemetry & Board Phase Metrics
    game_phase: str | None = None
    material_white: int | None = None
    material_black: int | None = None
    material_imbalance: int | None = None
    position_complexity: int | None = None
    illegal_attempts_count: int = 0
    attempted_illegal_moves: list[str] = field(default_factory=list)
    clock_remaining_sec: float | None = None
    move_duration_sec: float | None = None
    reasoning_token_ratio: float | None = None


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
    white_acpl: float | None = None
    black_acpl: float | None = None
    white_blunder_count: int | None = None
    black_blunder_count: int | None = None
    white_accuracy_pct: float | None = None
    black_accuracy_pct: float | None = None


class BenchmarkLogger:
    """Structured JSONL logger for benchmark runs."""

    def __init__(self, run_dir: str, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume

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
        if not self.resume:
            # Clear previous logs only for fresh runs
            for path in [self.game_log_path, self.move_log_path, self.pgn_path, self.error_log_path]:
                if path.exists():
                    path.unlink()
        else:
            # Resume: load completed game indices
            self._load_completed_games()

    def _load_completed_games(self) -> None:
        """Load completed game IDs from existing games.jsonl to skip on resume."""
        if self.game_log_path.exists():
            try:
                with open(self.game_log_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            import json
                            game = json.loads(line)
                            if 'game_id' in game:
                                self.games_completed.append(GameLogEntry(**game))
            except Exception:
                pass  # If loading fails, start fresh

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
                       thinking_trace: str | None,
                       validation_retries: int = 0,
                       eval_cp_score: int | None = None,
                       eval_mate_in: int | None = None,
                       eval_best_move_uci: str | None = None,
                       eval_best_move_cp: int | None = None,
                       eval_top3_moves: list[dict[str, Any]] | None = None,
                       eval_depth: int | None = None,
                       eval_time_ms: int | None = None,
                       game_phase: str | None = None,
                       material_white: int | None = None,
                       material_black: int | None = None,
                       material_imbalance: int | None = None,
                       position_complexity: int | None = None,
                       illegal_attempts_count: int = 0,
                       attempted_illegal_moves: list[str] | None = None,
                       clock_remaining_sec: float | None = None,
                       move_duration_sec: float | None = None,
                       reasoning_token_ratio: float | None = None,
                       prompt_hash: str | None = None) -> None:
            """Log a single move."""
            # Handle backward compatibility: prompt_hash was added later, so default to empty string
            effective_prompt_hash = prompt_hash if prompt_hash is not None else ""
            # Calculate move quality metrics from Stockfish evaluation
            cp_loss = None
            move_quality = None
            is_best_move = False

            if eval_cp_score is not None and eval_best_move_cp is not None:
                # cp_loss = best_move_cp - move_cp (positive means loss)
                cp_loss = max(0, eval_best_move_cp - eval_cp_score)
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
                prompt_hash=effective_prompt_hash,
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
                game_phase=game_phase,
                material_white=material_white,
                material_black=material_black,
                material_imbalance=material_imbalance,
                position_complexity=position_complexity,
                illegal_attempts_count=illegal_attempts_count,
                attempted_illegal_moves=attempted_illegal_moves or [],
                clock_remaining_sec=clock_remaining_sec,
                move_duration_sec=move_duration_sec,
    reasoning_token_ratio=reasoning_token_ratio,
                )
            self.current_game_moves.append(entry)

    # Write immediately to moves.jsonl
            with open(self.move_log_path, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')

    def end_game(self, result: str, result_numeric: float,
                   total_moves: int, game_duration_sec: float,
                   termination_reason: str = "unknown") -> None:
        """End current game and write complete game log."""
        import math

        white_losses = [m.cp_loss for m in self.current_game_moves if m.color.lower() in ("white", "w") and m.cp_loss is not None]
        black_losses = [m.cp_loss for m in self.current_game_moves if m.color.lower() in ("black", "b") and m.cp_loss is not None]

        white_acpl = float(sum(white_losses) / len(white_losses)) if white_losses else None
        black_acpl = float(sum(black_losses) / len(black_losses)) if black_losses else None

        white_blunders = sum(1 for m in self.current_game_moves if m.color.lower() in ("white", "w") and m.move_quality == "blunder")
        black_blunders = sum(1 for m in self.current_game_moves if m.color.lower() in ("black", "b") and m.move_quality == "blunder")

        def _calc_acc(losses: list[int]) -> float | None:
            if not losses:
                return None
            accs = [100.0 * math.exp(-0.005 * loss) for loss in losses]
            return float(sum(accs) / len(accs))

        white_acc = _calc_acc(white_losses)
        black_acc = _calc_acc(black_losses)

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
            termination_reason=termination_reason,
            white_acpl=white_acpl,
            black_acpl=black_acpl,
            white_blunder_count=white_blunders,
            black_blunder_count=black_blunders,
            white_accuracy_pct=white_acc,
            black_accuracy_pct=black_acc,
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

    def _compute_player_provenance(self) -> dict[str, Any]:
        """Compute per-player prompt provenance from config and completed games.

        Returns:
            Dict mapping player specs to their provenance info by color
        """
        import hashlib

        provenance: dict[str, dict[str, dict[str, Any]]] = {}

        # Get the set of player specs from the config
        player_specs = set(self.run_config.get('players', []))

        # For each player, determine what colors they played as
        player_colors: dict[str, set[str]] = {spec: set() for spec in player_specs}
        for game in self.games_completed:
            white_spec = game.white_player
            black_spec = game.black_player
            if white_spec in player_colors:
                player_colors[white_spec].add('white')
            if black_spec in player_colors:
                player_colors[black_spec].add('black')

        # Compute provenance for each player
        for spec in player_specs:
            provenance[spec] = {}

            # Check what colors this player actually played as
            for color in player_colors[spec]:
                # Determine prompts with precedence: color-specific > spec-keyed > None
                system_prompt = None
                turn_prompt = None

                # Try color-specific override first
                color_key = color.lower()
                if color_key in self.run_config.get('system_prompts_by_color', {}):
                    system_prompt = self.run_config['system_prompts_by_color'][color_key]
                if color_key in self.run_config.get('turn_prompts_by_color', {}):
                    turn_prompt = self.run_config['turn_prompts_by_color'][color_key]

                # Fall back to spec-keyed prompts
                if system_prompt is None:
                    system_prompt = self.run_config.get('system_prompts', {}).get(spec)
                if turn_prompt is None:
                    turn_prompt = self.run_config.get('turn_prompts', {}).get(spec)

                # Determine if fallback occurred
                # First choice availability and whether we got it
                system_first_choice_available = False
                system_got_first_choice = False
                if color:
                    # First choice: color-specific
                    system_first_choice_available = color_key in self.run_config.get('system_prompts_by_color', {})
                    if system_first_choice_available:
                        system_got_first_choice = True
                else:
                    # First choice: spec-keyed
                    system_first_choice_available = spec in self.run_config.get('system_prompts', {})
                    if system_first_choice_available:
                        system_got_first_choice = True

                turn_first_choice_available = False
                turn_got_first_choice = False
                if color:
                    # First choice: color-specific
                    turn_first_choice_available = color_key in self.run_config.get('turn_prompts_by_color', {})
                    if turn_first_choice_available:
                        turn_got_first_choice = True
                else:
                    # First choice: spec-keyed
                    turn_first_choice_available = spec in self.run_config.get('turn_prompts', {})
                    if turn_first_choice_available:
                        turn_got_first_choice = True

                used_fallback = not (system_got_first_choice and turn_got_first_choice)

                # Compute strategy hash
                strategy_hash = ""
                if system_prompt is not None or turn_prompt is not None:
                    # Use null byte as separator as specified in requirements
                    system_part = system_prompt if system_prompt is not None else ""
                    turn_part = turn_prompt if turn_prompt is not None else ""
                    strategy_string = f"{system_part}\x00{turn_part}"
                    strategy_hash = hashlib.sha256(strategy_string.encode()).hexdigest()[:16]

                # Determine source (simplified - we don't have access to preset versions)
                # For now, we'll say "custom" if we got a non-None prompt from config, else "preset"
                source = "custom"
                if system_prompt is None and turn_prompt is None:
                    source = "preset"

                provenance[spec][color] = {
                    "system_prompt": system_prompt,
                    "turn_prompt": turn_prompt,
                    "strategy_hash": strategy_hash,
                    "source": source,
                    "used_fallback": used_fallback,
                }

        return provenance

    def write_summary(self) -> None:
        """Write run summary."""
        provenance = self._compute_player_provenance()
        summary = {
            'run_id': self.run_dir.name,
            'timestamp_utc': datetime.now(UTC).isoformat() + 'Z',
            'config': self.run_config,
            'provenance': provenance,
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

        # Atomic write: a crash mid-dump must not leave a truncated summary.
        tmp_path = self.summary_path.with_suffix(".json.tmp")
        with open(tmp_path, 'w') as f:
            json.dump(summary, f, indent=2)
        tmp_path.replace(self.summary_path)

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
    logger.log_move(1, "GPT-4o", "white", board.fen(), "b2b4", "b4", 500, 100, 5, "b4", "<thinking>...", 0, prompt_hash="hash1")
    board.push(chess.Move.from_uci("b2b4"))

    logger.log_move(1, "Claude-3.5-Sonnet", "black", board.fen(), "e7e5", "e5", 400, 100, 5, "e5", "<thinking>...", 0, prompt_hash="hash2")
    board.push(chess.Move.from_uci("e7e5"))

    logger.end_game("1-0", 1.0, 2, 2.0)
    logger.write_summary()

    print("Test complete!")
    print(logger.get_pgn_content())
