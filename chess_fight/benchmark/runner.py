"""Benchmark configuration and headless runner."""

import asyncio
import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import yaml

from chess_fight.benchmark.elo import BayesianElo
from chess_fight.benchmark.logging import BenchmarkLogger, GameLogEntry
from chess_fight.benchmark.openings import OpeningBook
from chess_fight.game.async_game import AsyncChessGame, GameState
from chess_fight.providers import get_provider, list_providers
from chess_fight.providers.chess_ai import ProviderChessAI

_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)


def _extract_thinking(raw: str | None) -> str | None:
    if not raw:
        return None
    match = _THINKING_RE.search(raw)
    return match.group(1).strip() if match else None


def _uci_to_san(fen: str, uci: str) -> str:
    try:
        board = chess.Board(fen)
        return board.san(chess.Move.from_uci(uci))
    except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
        return uci


def _prompt_hash(fen: str) -> str:
    return hashlib.sha256(fen.encode()).hexdigest()[:16]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    # Game rules
    time_control_seconds_per_move: int = 30
    opening_book: str = "eco_balanced"  # "eco_balanced", "eco_all", "startpos"
    games_per_pairing: int = 10
    colors: str = "alternating"  # "alternating", "fixed"

    # Model params (benchmark mode)
    temperature: float = 0.0
    max_tokens: int = 100
    seed: int | None = 42

    # Concurrency
    max_parallel_games: int = 4

    # Players (provider:model_id)
    players: list[str] = field(default_factory=list)  # e.g., ["openai:gpt-4o", "anthropic:claude-3-5-sonnet"]

    # Output
    output_dir: str = "runs"
    run_name: str | None = None

    # API keys (loaded from env or config)
    api_keys: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> 'BenchmarkConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class BenchmarkRunner:
    """Headless benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.run_id = config.run_name or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.output_dir) / self.run_id
        self.logger = BenchmarkLogger(str(self.run_dir))

        # Initialize players
        self.players = self._initialize_players()

        # Initialize openings
        self.opening_book = OpeningBook()
        self.openings = self._select_openings()

        # ELO calculator
        self.elo = BayesianElo()

        # Results
        self.results: list[Any] = []

    def _initialize_players(self) -> dict[str, ProviderChessAI]:
        """Initialize player AIs from config."""
        players = {}
        for player_spec in self.config.players:
            provider_name, model_id = player_spec.split(':', 1)
            api_key = self.config.api_keys.get(provider_name, '')

            if not api_key:
                raise ValueError(f"No API key for provider {provider_name}")

            provider = get_provider(provider_name)
            if not provider:
                raise ValueError(f"Unknown provider: {provider_name}")

            if not provider.validate_key(api_key):
                raise ValueError(f"Invalid API key for {provider_name}")

            ai = ProviderChessAI(
                provider_name=provider_name,
                model_id=model_id,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            players[player_spec] = ai

        return players

    def _select_openings(self) -> list[dict]:
        """Select openings based on config."""
        if self.config.opening_book == "eco_all":
            return self.opening_book.get_all_openings()
        elif self.config.opening_book == "eco_balanced":
            # Get balanced set: 1 opening per pairing per color
            n_pairings = len(self.config.players) * (len(self.config.players) - 1) // 2
            n_openings = n_pairings * self.config.games_per_pairing * 2  # both colors
            return self.opening_book.get_balanced_set(n_openings)
        else:
            # Single starting position
            return [{
                'eco': 'START',
                'name': 'Starting Position',
                'moves': [],
                'fen': chess.STARTING_FEN,
                'ply': 0
            }]

    async def run_pairing(self, white_spec: str, black_spec: str,
                          opening: dict, game_idx: int) -> GameLogEntry:
        """Run a single game between two players."""
        white_ai = self.players[white_spec]
        black_ai = self.players[black_spec]

        # Set up opening position
        board = chess.Board(opening['fen'])

        self.logger.start_game(
            white_player=white_spec,
            black_player=black_spec,
            white_provider=white_spec.split(':')[0],
            black_provider=black_spec.split(':')[0],
            opening_eco=opening['eco'],
            opening_name=opening['name'],
            opening_fen=opening['fen']
        )

        # Track moves for logging
        move_logs: list[Any] = []

        async def ui_callback(state: GameState):
            if state.moves and len(state.moves) > len(move_logs):
                last_move = state.moves[-1]
                fen_before = state.fen_before or state.board.fen()
                cr = state.last_completion_result
                self.logger.log_move(
                    move_number=len(state.moves),
                    player=last_move.player,
                    color="white" if last_move.player == white_spec else "black",
                    fen_before=fen_before,
                    move_uci=last_move.move,
                    move_san=_uci_to_san(fen_before, last_move.move),
                    llm_latency_ms=cr.latency_ms if cr and cr.latency_ms else 0,
                    llm_tokens_in=cr.tokens_in if cr else None,
                    llm_tokens_out=cr.tokens_out if cr else None,
                    llm_raw_response=cr.text if cr else "",
                    thinking_trace=_extract_thinking(cr.text if cr else None),
                    prompt_hash=_prompt_hash(fen_before),
                    validation_retries=0,
                )
                move_logs.append(last_move)

        # Create game with custom starting position
        game = AsyncChessGame(white_ai, black_ai)
        game.board = board.copy()

        # Play the game
        stats = await game.play_game(ui_callback, delay=0.01)

        # Determine result
        if stats.winner == white_spec:
            result = "1-0"
            result_numeric = 1.0
        elif stats.winner == black_spec:
            result = "0-1"
            result_numeric = 0.0
        else:
            result = "1/2-1/2"
            result_numeric = 0.5

        # End game logging
        self.logger.end_game(
            result=result,
            result_numeric=result_numeric,
            total_moves=stats.total_moves,
            game_duration_sec=stats.game_duration
        )

        # Update ELO
        self.elo.add_game(white_spec, black_spec, result_numeric, opening['eco'])

        return self.logger.games_completed[-1]

    async def run_benchmark(self):
        """Run the full benchmark."""
        print(f"Starting benchmark: {self.run_id}")
        print(f"Players: {list(self.players.keys())}")
        print(f"Games per pairing: {self.config.games_per_pairing}")
        print(f"Openings: {len(self.openings)}")
        print(f"Max parallel games: {self.config.max_parallel_games}")

        self.logger.start_run(self.config.to_dict())

        # Generate pairings
        pairings = []
        players = list(self.players.keys())
        for i, white in enumerate(players):
            for j, black in enumerate(players):
                if i != j:
                    pairings.append((white, black))

        # Create all game tasks
        game_tasks = []
        game_count = 0
        for white, black in pairings:
            for _game_num in range(self.config.games_per_pairing):
                if game_count < len(self.openings):
                    opening = self.openings[game_count]
                else:
                    opening = self.openings[game_count % len(self.openings)]

                game_tasks.append((white, black, opening, game_count))
                game_count += 1

        # Run games with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_parallel_games)

        async def run_game_task(white, black, opening, game_idx):
            async with semaphore:
                print(f"  Game {game_idx + 1}: {white} vs {black} ({opening['eco']}: {opening['name']})")
                return await self.run_pairing(white, black, opening, game_idx)

        # Execute all games concurrently with limit
        results = await asyncio.gather(*[run_game_task(*task) for task in game_tasks], return_exceptions=True)

        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = game_tasks[i]
                print(f"  Game {i + 1} ({task[0]} vs {task[1]}) failed: {result}")

        self.logger.write_summary()

        # Print leaderboard
        print("\n=== FINAL LEADERBOARD ===")
        for row in self.elo.leaderboard():
            print(f"  {row['name']}: {row['rating']} ± {row['deviation']} (95% CI: {row['ci_low']}-{row['ci_high']})")

        print(f"\nResults saved to: {self.run_dir}")
        return self.run_dir


async def main():
    """Main entry point for benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Chess LLM Benchmark Runner")
    parser.add_argument("--config", default="benchmark.yaml", help="Config file path")
    parser.add_argument("--output", default="runs", help="Output directory")
    parser.add_argument("--players", nargs="+", help="Player specs (provider:model)")
    parser.add_argument("--games", type=int, help="Games per pairing")
    parser.add_argument("--parallel", type=int, help="Max parallel games")

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config = BenchmarkConfig.from_yaml(args.config)
    else:
        config = BenchmarkConfig()

    # Override from command line
    if args.players:
        config.players = args.players
    if args.games:
        config.games_per_pairing = args.games
    if args.parallel:
        config.max_parallel_games = args.parallel
    config.output_dir = args.output

    # Load API keys from environment
    import os
    for provider in list_providers():
        key = os.getenv(f"{provider.upper()}_API_KEY")
        if key:
            config.api_keys[provider] = key

    if not config.players:
        print("Error: No players specified. Use --players or config file.")
        return

    runner = BenchmarkRunner(config)
    await runner.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
