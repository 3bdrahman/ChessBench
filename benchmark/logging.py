"""Structured logging for benchmark games."""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path
import chess


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
    llm_tokens_in: Optional[int]
    llm_tokens_out: Optional[int]
    llm_raw_response: str
    thinking_trace: Optional[str]
    prompt_hash: str
    validation_retries: int
    timestamp_utc: str


@dataclass
class GameLogEntry:
    """Complete game log entry."""
    game_id: str
    white_player: str
    black_player: str
    white_provider: str
    black_provider: str
    opening_eco: Optional[str]
    opening_name: Optional[str]
    opening_fen: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    result_numeric: float  # 1.0, 0.0, 0.5
    moves: List[MoveLogEntry]
    total_moves: int
    game_duration_sec: float
    timestamp_utc: str
    config: Dict[str, Any]


class BenchmarkLogger:
    """Structured JSONL logger for benchmark runs."""
    
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.game_log_path = self.run_dir / "games.jsonl"
        self.move_log_path = self.run_dir / "moves.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.pgn_path = self.run_dir / "games.pgn"
        
        self.current_game_id: Optional[str] = None
        self.current_game_moves: List[MoveLogEntry] = []
        self.current_game_info: Dict[str, Any] = {}
        self.run_config: Dict[str, Any] = {}
        self.games_completed: List[GameLogEntry] = []
    
    def start_run(self, config: Dict[str, Any]):
        """Start a new benchmark run."""
        self.run_config = config
        # Clear previous logs
        for path in [self.game_log_path, self.move_log_path, self.pgn_path]:
            if path.exists():
                path.unlink()
    
    def start_game(self, white_player: str, black_player: str, 
                   white_provider: str, black_provider: str,
                   opening_eco: Optional[str] = None,
                   opening_name: Optional[str] = None,
                   opening_fen: str = None):
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
                 llm_latency_ms: int, llm_tokens_in: Optional[int],
                 llm_tokens_out: Optional[int], llm_raw_response: str,
                 thinking_trace: Optional[str], prompt_hash: str,
                 validation_retries: int):
        """Log a single move."""
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
            timestamp_utc=datetime.now(timezone.utc).isoformat() + 'Z'
        )
        self.current_game_moves.append(entry)
        
        # Write immediately to moves.jsonl
        with open(self.move_log_path, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
    
    def end_game(self, result: str, result_numeric: float, 
                 total_moves: int, game_duration_sec: float):
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
            timestamp_utc=datetime.now(timezone.utc).isoformat() + 'Z',
            config=self.run_config
        )
        
        self.games_completed.append(game_entry)
        
        # Write to games.jsonl
        with open(self.game_log_path, 'a') as f:
            f.write(json.dumps(asdict(game_entry)) + '\n')
        
        # Write PGN
        self._write_pgn(game_entry)
        
        self.current_game_id = None
        self.current_game_moves = []
        self.current_game_info = {}
    
    def _write_pgn(self, game: GameLogEntry):
        """Write game to PGN file."""
        pgn_lines = [
            f'[Event "Chess LLM Benchmark"]',
            f'[Site "Local"]',
            f'[Date "{datetime.now(timezone.utc).strftime("%Y.%m.%d")}"]',
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
    
    def write_summary(self):
        """Write run summary."""
        summary = {
            'run_id': self.run_dir.name,
            'timestamp_utc': datetime.now(timezone.utc).isoformat() + 'Z',
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