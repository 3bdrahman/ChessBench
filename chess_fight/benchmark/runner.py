"""Benchmark configuration and headless runner."""

import asyncio
import hashlib
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import yaml

from chess_fight import constants
from chess_fight.benchmark.elo import BayesianElo
from chess_fight.benchmark.evaluator import StockfishEvaluator
from chess_fight.benchmark.logging import BenchmarkLogger, GameLogEntry
from chess_fight.benchmark.openings import OpeningBook
from chess_fight.common.exceptions import (
    FatalBenchmarkError,
    GameTimeoutError,
    InvalidApiKeyError,
    LimiterExhaustedError,
    MoveExhaustedError,
    NoProvidersConfiguredError,
    ProviderError,
    RateLimitError,
    SetupError,
)
from chess_fight.game.async_game import AsyncChessGame, GameState
from chess_fight.game.clock import GameClock
from chess_fight.models import GameStats
from chess_fight.providers import get_provider, list_providers
from chess_fight.providers.chess_ai import ProviderChessAI
from chess_fight.providers.ratelimit import ProviderRateLimiter

_log = logging.getLogger(__name__)

_THINKING_RE = re.compile(r"<(?:think|thinking)>(.*?)</(?:think|thinking)>", re.DOTALL | re.IGNORECASE)

# Termination reasons that represent a *clean* chess conclusion — the board
# reached a real terminal position (or its automatic draw variants). Games that
# end with one of these reasons count as finished: they are logged via
# ``end_game`` and contribute to ELO. Anything else (cancelled, error,
# timeout, max_moves) is a *problem*: the game did NOT reach a natural chess
# outcome and must NOT be counted as concluded — it is logged via ``log_error``
# and the benchmark pauses for the user instead of starting the next game.
_CLEAN_TERMINATIONS: frozenset[str] = frozenset({
    "checkmate",
    "stalemate",
    "insufficient_material",
    "fifty_moves",
    "threefold_repetition",
    "seventyfive_moves",
    "fivefold_repetition",
    "variant_win",
    "variant_loss",
    "variant_draw",
    "draw",
})


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
    time_control_seconds_per_move: int = constants.DEFAULT_TIME_CONTROL_SECONDS_PER_MOVE
    opening_book: str = constants.DEFAULT_OPENING_BOOK  # "eco_balanced", "eco_all", "startpos"
    games_per_pairing: int = constants.DEFAULT_GAMES_PER_PAIRING
    colors: str = constants.DEFAULT_COLORS_MODE  # "alternating", "fixed"

    # Model params (benchmark mode)
    temperature: float = constants.DEFAULT_BENCHMARK_TEMPERATURE
    max_tokens: int | None = constants.DEFAULT_MAX_TOKENS_BENCHMARK
    reasoning_level: str = constants.DEFAULT_REASONING_LEVEL
    seed: int | None = constants.DEFAULT_SEED

    # Concurrency
    max_parallel_games: int = constants.DEFAULT_MAX_PARALLEL_GAMES

    # Timeouts
    move_timeout_seconds: int = constants.DEFAULT_MOVE_TIMEOUT_SECONDS
    game_timeout_seconds: int = constants.DEFAULT_GAME_TIMEOUT_SECONDS

    # Players (provider:model_id)
    players: list[str] = field(default_factory=list)  # e.g., ["openai:gpt-4o", "anthropic:claude-3-5-sonnet"]

    # Output
    output_dir: str = constants.DEFAULT_OUTPUT_DIR
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

    def save_yaml(self, path: str) -> None:
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
        self.start_time = time.time()

        # Initialize rate limiter
        self.rate_limiter = ProviderRateLimiter()

        # Initialize players
        self.players = self._initialize_players()

        # Initialize openings
        self.opening_book = OpeningBook()
        self.openings = self._select_openings()

        # ELO calculator
        self.elo = BayesianElo()

        # Results
        self.results: list[GameLogEntry] = []

        # Benchmark-level pause. On a *problem* game (cancelled/error/timeout/
        # max_moves), the runner notifies the UI via ui_callback with a paused
        # GameState, then blocks on this threading.Event until the UI signals
        # continue/abort. threading.Event is intentional: the UI sets it from
        # the Streamlit thread; the runner awaits it via asyncio.to_thread.
        self._continue_after_problem_event: threading.Event = threading.Event()
        self._abort_requested: bool = False
        # Failed game's state snapshot — populated by the runner before
        # signalling the UI so the UI can render the paused board/error.
        self.problem_state: GameState | None = None

    def _initialize_players(self) -> dict[str, ProviderChessAI]:
        """Initialize player AIs from config.

        Raises:
            NoProvidersConfiguredError: When no providers are registered.
            SetupError: When configuration is invalid (empty players list,
                unknown provider, missing API key).
            InvalidApiKeyError: When an API key fails format validation.
        """
        available_providers = list_providers()
        if not available_providers:
            raise NoProvidersConfiguredError()

        if not self.config.players:
            raise SetupError(
                "No players configured. Pass --players or set the players "
                "list in benchmark.yaml."
            )

        if len(self.config.players) < 2:
            raise SetupError(
                "At least 2 players are required to run a benchmark match. "
                "Specify two or more players (or pass the same player twice for self-play)."
            )

        players: dict[str, ProviderChessAI] = {}
        for player_spec in self.config.players:
            try:
                provider_name, model_id = player_spec.split(':', 1)
            except ValueError as exc:
                raise SetupError(
                    f"Invalid player spec '{player_spec}': expected 'provider:model' format"
                ) from exc

            api_key = self.config.api_keys.get(provider_name, '')

            provider = get_provider(provider_name)
            if not provider:
                raise SetupError(
                    f"Unknown provider '{provider_name}'. "
                    f"Available: {', '.join(sorted(available_providers))}"
                )

            if provider.requires_api_key:
                if not api_key:
                    raise SetupError(
                        f"No API key configured for provider '{provider_name}'. "
                        f"Set the {provider_name.upper()}_API_KEY environment variable "
                        f"or pass --api-keys."
                    )
                if not provider.validate_key(api_key):
                    hint_method = getattr(provider, "_key_prefix_hint", None)
                    expected = hint_method() if hint_method else "<valid format>"
                    raise InvalidApiKeyError(
                        provider=provider_name,
                        got_prefix=api_key[:8] + "…" if api_key else "",
                        expected_prefix=expected,
                        http_status=401,
                    )
            else:
                api_key = api_key or ''

            ai = ProviderChessAI(
                provider_name=provider_name,
                model_id=model_id,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                reasoning_level=self.config.reasoning_level,
            )
            players[player_spec] = ai

        return players

    def _select_openings(self) -> list[dict[str, Any]]:
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

    def _create_player(self, player_spec: str) -> ProviderChessAI:
        """Create a fresh AI instance for a game."""
        provider_name, model_id = player_spec.split(':', 1)
        api_key = self.config.api_keys.get(provider_name, '')
        return ProviderChessAI(
            provider_name=provider_name,
            model_id=model_id,
            api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            reasoning_level=self.config.reasoning_level,
        )

    def request_continue_after_problem(self) -> None:
        """UI thread: dismiss the problem pause and proceed to the next game."""
        self._abort_requested = False
        self._continue_after_problem_event.set()

    def request_abort_after_problem(self) -> None:
        """UI thread: dismiss the problem pause and abort the benchmark."""
        self._abort_requested = True
        self._continue_after_problem_event.set()

    def resume_game(self, retry: bool = True, force_move: bool = False) -> None:
        """UI thread: resume the currently paused game."""
        if hasattr(self, 'current_game') and self.current_game:
            self.current_game.resume(retry_current_turn=retry, force_move=force_move)

    async def run_pairing(
        self,
        white_spec: str,
        black_spec: str,
        opening: dict[str, Any],
        game_idx: int,
        user_callback: Callable[[GameState], Awaitable[None]] | None = None,
        alternating: bool = False,
        on_game_start: Callable[[AsyncChessGame], None] | None = None,
    ) -> GameLogEntry | None:
        """Run a single game between two players.

        Returns a ``GameLogEntry`` when the game reached a clean chess terminal
        (checkmate, stalemate, automatic draws); returns ``None`` when the game
        had a problem (cancelled, move timeout, max-moves failsafe, or runtime
        exception). Problem games are recorded via ``log_error`` and do not
        contribute to ELO, so the leaderboard is not polluted by games that
        never reached a real chess outcome.

        ``user_callback`` (if provided) is invoked after the per-move logger
        has written the JSONL record — so a UI consumer sees the same state
        the on-disk logs reflect.
        """
        # Create fresh AI instances for every game to isolate state (move history, etc.)
        white_ai = self._create_player(white_spec)
        black_ai = self._create_player(black_spec)

        # Set up opening position
        board = chess.Board(opening['fen'])

        # Start Stockfish evaluator for this game
        async with StockfishEvaluator() as evaluator:

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
            move_logs: list[GameLogEntry] = []

            async def ui_callback(state: GameState) -> None:
                if state.moves and len(state.moves) > len(move_logs):
                    last_move = state.moves[-1]
                    fen_before = state.fen_before or state.board.fen()
                    cr = state.last_completion_result

                    # Evaluate position before the move with Stockfish
                    board_before = chess.Board(fen_before)
                    eval_result = await evaluator.evaluate(board_before)

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
                        eval_cp_score=eval_result.cp_score if eval_result else None,
                        eval_mate_in=eval_result.mate_in if eval_result else None,
                        eval_best_move_uci=eval_result.best_move_uci if eval_result else None,
                        eval_best_move_cp=eval_result.best_move_cp if eval_result else None,
                        eval_top3_moves=eval_result.top3_moves if eval_result else None,
                        eval_depth=eval_result.depth if eval_result else None,
                        eval_time_ms=eval_result.time_ms if eval_result else None,
                    )
                    move_logs.append(last_move)
                if user_callback is not None:
                    await user_callback(state)

            # Create game with custom starting position
            # Create clock with time control
            clock = GameClock.from_seconds(
                self.config.time_control_seconds_per_move,
                0  # No increment for now, could be added to config later
            )
            game = AsyncChessGame(white_ai, black_ai, clock=clock, evaluator=evaluator)
            game.board = board.copy()
            self.current_game = game

            # Notify UI that a new game has started
            if on_game_start:
                on_game_start(game)

            # Play the game with move timeout
            stats = None
            try:
                stats = await game.play_game(
                    ui_callback,
                    delay=0.01,
                    move_timeout_seconds=self.config.move_timeout_seconds
                )
            except Exception as exc:
                run_exc: Exception | None = exc
                _log.warning(
                    "Game %d (%s vs %s) play_game raised: %s",
                    game_idx + 1, white_spec, black_spec, exc,
                )
            else:
                run_exc = None
            finally:
                if stats is None:
                    stats = game.stats
                if not stats.game_duration:
                    stats.game_duration = time.time() - game.start_time

                termination = getattr(stats, 'termination_reason', 'unknown')
                is_clean = termination in _CLEAN_TERMINATIONS

                if is_clean and run_exc is None:
                    result_numeric: float | None
                    if stats.winner == white_spec:
                        result = "1-0"
                        result_numeric = 1.0
                    elif stats.winner == black_spec:
                        result = "0-1"
                        result_numeric = 0.0
                    elif stats.winner in ("1/2-1/2", "Draw", None):
                        result = "1/2-1/2"
                        result_numeric = 0.5
                    else:
                        result = "Error"
                        result_numeric = None

                    if result_numeric is None:
                        is_clean = False
                    else:
                        self.logger.end_game(
                            result=result,
                            result_numeric=result_numeric,
                            total_moves=stats.total_moves,
                            game_duration_sec=stats.game_duration,
                            termination_reason=termination,
                        )
                        self.elo.add_game(
                            white_spec, black_spec, result_numeric, opening['eco']
                        )

                if not is_clean:
                    problem = (
                        f"{run_exc}" if run_exc is not None
                        else f"game did not reach a clean chess terminal "
                             f"(termination={termination}, winner={stats.winner})"
                    )
                    self.logger.log_error(
                        game_index=game_idx + 1,
                        white=white_spec,
                        black=black_spec,
                        error=f"premature_conclusion: {problem}",
                    )
                    _log.warning(
                        "Game %d (%s vs %s) NOT concluded: %s",
                        game_idx + 1, white_spec, black_spec, problem,
                    )

            # Stop evaluator after game
            await evaluator.stop()

            if (
                self.logger.games_completed
                and self.logger.games_completed[-1].termination_reason in _CLEAN_TERMINATIONS
                and self.logger.games_completed[-1].white_player == white_spec
                and self.logger.games_completed[-1].black_player == black_spec
            ):
                return self.logger.games_completed[-1]
            return None

    async def run_benchmark(self) -> Path:
        """Run the full benchmark."""
        return await self.run_benchmark_with_callback(None)

    async def _pause_on_problems(
        self,
        ui_callback: Callable[[GameState], Awaitable[None]],
        problems: list[tuple[int, str, str]],
    ) -> None:
        """Block the runner until the UI dismisses a problem-game pause.

        For each problem game, synthesize a paused GameState snapshot and push
        it through ``ui_callback`` so the UI renders the paused board + error
        alongside the same buttons used for mid-game move pauses
        (``pause_reason="game_failed"``). Then await the cross-thread
        ``threading.Event`` (the UI sets it on Continue/Abort) without blocking
        the asyncio loop: ``asyncio.to_thread`` runs the blocking wait in a
        worker thread while the loop stays responsive.
        """
        game_idx, white_spec, black_spec = problems[-1]
        board = chess.Board()
        moves_summary = ", ".join(
            f"game {g + 1} {w} vs {b}" for g, w, b in problems
        )
        self.problem_state = GameState(
            board=board,
            moves=[],
            stats=GameStats(),
            current_player="",
            is_game_over=False,
            winner=None,
            fen_before=board.fen(),
            is_paused=True,
            pause_reason="game_failed",
            pause_error=(
                f"{len(problems)} game(s) ended with a problem and were not "
                f"concluded: {moves_summary}. Decide whether to continue or "
                f"abort the benchmark."
            ),
            paused_player=f"{white_spec} vs {black_spec}",
            paused_turn=game_idx,
        )
        await ui_callback(self.problem_state)
        # asyncio.to_thread is required (not `await event.wait()`): the UI sets
        # the event via the loop-bound callback, so a blocking loop would
        # deadlock waiting for a signal it can never receive.
        await asyncio.to_thread(self._continue_after_problem_event.wait)
        self._continue_after_problem_event.clear()

    async def run_benchmark_with_callback(
        self,
        ui_callback: Callable[[GameState], Awaitable[None]] | None,
    ) -> Path:
        """Run the full benchmark, optionally streaming GameState to ``ui_callback``.

        When ``max_parallel_games == 1`` (the in-Streamlit case), the callback
        is safe to call from the single game loop. With parallel games, the
        callback receives interleaved events from multiple game states — the
        UI layer should treat each callback as a live tick, not a snapshot of
        a single game.

        Raises:
            GameExecutionError: When a fatal error (auth, rate-limit, missing
                providers) halts the run. The cause is attached for inspection.
        """
        _log.info("Starting benchmark: %s", self.run_id)
        _log.info("Players: %s", list(self.players.keys()))
        _log.info("Games per pairing: %s", self.config.games_per_pairing)
        _log.info("Openings: %s", len(self.openings))
        _log.info("Max parallel games: %s", self.config.max_parallel_games)

        self.logger.start_run(self.config.to_dict())

        # Generate pairings
        pairings: list[tuple[str, str]] = []
        pairing_alternating_colors: list[bool] = []  # Track if pairing alternates colors per game
        players = list(self.config.players)
        if self.config.colors == "fixed" and len(players) >= 2:
            pairings.append((players[0], players[1]))
            pairing_alternating_colors.append(False)
        elif self.config.colors == "alternating" and len(players) == 2:
            # For 2 players with alternating colors: single pairing, alternate colors per game
            pairings.append((players[0], players[1]))
            pairing_alternating_colors.append(True)
        else:
            # Multiple players or other modes: all pairings
            for i, white in enumerate(players):
                for j, black in enumerate(players):
                    if i != j:
                        pairings.append((white, black))
                        pairing_alternating_colors.append(False)

        fatal_exception: Exception | None = None

        # Run games pairing by pairing (for proper Glicko-2 rating periods)
        for pairing_idx, (white, black) in enumerate(pairings):
            alternating = pairing_alternating_colors[pairing_idx]
            _log.info("=== Pairing %d/%d: %s vs %s (alternating=%s) ===", pairing_idx + 1, len(pairings), white, black, alternating)

            # Create all game tasks for this pairing
            pairing_game_tasks: list[tuple[str, str, dict[str, Any], int, bool]] = []
            for game_num in range(self.config.games_per_pairing):
                game_count = pairing_idx * self.config.games_per_pairing + game_num
                if game_count < len(self.openings):
                    opening = self.openings[game_count]
                else:
                    opening = self.openings[game_count % len(self.openings)]

                # For alternating pairing, swap colors for odd-numbered games
                if alternating and game_num % 2 == 1:
                    game_white, game_black = black, white
                else:
                    game_white, game_black = white, black

                pairing_game_tasks.append((game_white, game_black, opening, game_count, alternating))

            # Run games for this pairing with concurrency limit
            semaphore = asyncio.Semaphore(self.config.max_parallel_games)
            pairing_game_count = 0

            async def run_pairing_game_task(
                white_spec: str,
                black_spec: str,
                opening: dict[str, Any],
                game_idx: int,
                alternating: bool = False,
                sem: asyncio.Semaphore = semaphore,
            ) -> GameLogEntry | None:
                async with sem:
                    nonlocal pairing_game_count
                    pairing_game_count += 1
                    print(f"  Game {pairing_game_count}: {white_spec} vs {black_spec} ({opening['eco']}: {opening['name']})")

                    # Acquire rate limit permit
                    white_provider = white_spec.split(':')[0]
                    black_provider = black_spec.split(':')[0]

                    # Acquire permits for both providers
                    wait_time = await self.rate_limiter.acquire(white_provider, tokens=1)
                    if wait_time is None:
                        raise LimiterExhaustedError(
                            provider=white_provider,
                            game_index=game_idx,
                            white=white_spec,
                            black=black_spec,
                        )

                    wait_time = await self.rate_limiter.acquire(black_provider, tokens=1)
                    if wait_time is None:
                        self.rate_limiter.release(white_provider)
                        raise LimiterExhaustedError(
                            provider=black_provider,
                            game_index=game_idx,
                            white=white_spec,
                            black=black_spec,
                        )

                    try:
                        return await asyncio.wait_for(
                            self.run_pairing(
                                white_spec, black_spec, opening, game_idx, ui_callback, alternating,
                                on_game_start=lambda g: setattr(
                                    __import__('streamlit').session_state, 'benchmark_current_game', g
                                )
                            ),
                            timeout=self.config.game_timeout_seconds
                        )
                    except TimeoutError:
                        _log.error(
                            "Game %d (%s vs %s) timed out after %d seconds",
                            game_idx + 1, white_spec, black_spec, self.config.game_timeout_seconds
                        )
                        raise GameTimeoutError(
                            timeout_seconds=self.config.game_timeout_seconds,
                            game_index=game_idx,
                            white=white_spec,
                            black=black_spec,
                        ) from None
                    finally:
                        self.rate_limiter.release(white_provider)
                        self.rate_limiter.release(black_provider)

            # Execute all games for this pairing concurrently with limit
            pairing_results = await asyncio.gather(
                *[run_pairing_game_task(*task) for task in pairing_game_tasks],
                return_exceptions=True
            )

            # Classify exceptions for this pairing
            pairing_game_failures: list[tuple[int, str, str, Exception]] = []
            paused_problem_games: list[tuple[int, str, str]] = []

            for i, result in enumerate(pairing_results):
                task = pairing_game_tasks[i]
                white_spec, black_spec = task[0], task[1]
                game_idx = task[3]

                if result is None:
                    paused_problem_games.append((game_idx, white_spec, black_spec))
                    continue

                if isinstance(result, Exception):
                    # Fatal: limiter permanently saturated → continuing just queues
                    # every subsequent game behind the same stall.
                    if isinstance(result, LimiterExhaustedError):
                        _log.error(
                            "Fatal error at game %d (%s vs %s): %s",
                            game_idx + 1, white_spec, black_spec, result,
                        )
                        if fatal_exception is None:
                            fatal_exception = result
                        break
                    # Fatal: auth failure or upstream rate-limit → the whole run is broken.
                    if isinstance(result, (InvalidApiKeyError, RateLimitError)):
                        _log.error(
                            "Fatal error at game %d (%s vs %s): %s",
                            game_idx + 1, white_spec, black_spec, result,
                        )
                        if fatal_exception is None:
                            fatal_exception = result
                        break
                    # Fatal: other ProviderErrors (network, quota, API) — no point continuing.
                    if isinstance(result, ProviderError) and not isinstance(result, RateLimitError):
                        _log.error(
                            "Fatal provider error at game %d (%s vs %s): %s",
                            game_idx + 1, white_spec, black_spec, result,
                        )
                        if fatal_exception is None:
                            fatal_exception = result
                        break
                    # Per-game timeout: chess self-terminates normally; a wall-clock
                    # failsafe trip means a pathological game that should be skipped, not
                    # a signal to abort the whole tournament.
                    if isinstance(result, GameTimeoutError):
                        _log.warning(
                            "Game %d (%s vs %s) timed out after %g seconds: %s",
                            game_idx + 1, white_spec, black_spec,
                            result.timeout_seconds, result,
                        )
                        self.logger.log_error(
                            game_index=game_idx + 1,
                            white=white_spec,
                            black=black_spec,
                            error=f"game_timeout: {result}",
                        )
                        pairing_game_failures.append((game_idx, white_spec, black_spec, result))
                        _log.warning(
                            "Game %d (%s vs %s) failed: %s",
                            game_idx + 1, white_spec, black_spec, result,
                        )
                        print(f"  Game {game_idx + 1} ({white_spec} vs {black_spec}) failed: {result}")
                        paused_problem_games.append((game_idx, white_spec, black_spec))
                        continue
                    # Per-game move-exhaustion — log and continue.
                    if isinstance(result, MoveExhaustedError):
                        _log.warning(
                            "Game %d (%s vs %s) exhausted move retries: %s",
                            game_idx + 1, white_spec, black_spec, result,
                        )
                        self.logger.log_error(
                            game_index=game_idx + 1,
                            white=white_spec,
                            black=black_spec,
                            error=f"move_exhausted: {result}",
                        )
                        pairing_game_failures.append((game_idx, white_spec, black_spec, result))
                        _log.warning(
                            "Game %d (%s vs %s) failed: %s",
                            game_idx + 1, white_spec, black_spec, result,
                        )
                        print(f"  Game {game_idx + 1} ({white_spec} vs {black_spec}) failed: {result}")
                        paused_problem_games.append((game_idx, white_spec, black_spec))
                        continue
                    # Residual per-game failure (evaluator crash, other GameExecutionError,
                    # unexpected Exception). Log and continue so one bad game doesn't abort
                    # the whole tournament.
                    pairing_game_failures.append((game_idx, white_spec, black_spec, result))
                    self.logger.log_error(
                        game_index=game_idx + 1,
                        white=white_spec,
                        black=black_spec,
                        error=str(result),
                    )
                    _log.warning(
                        "Game %d (%s vs %s) failed: %s",
                        game_idx + 1, white_spec, black_spec, result,
                    )
                    print(f"  Game {game_idx + 1} ({white_spec} vs {black_spec}) failed: {result}")
                    paused_problem_games.append((game_idx, white_spec, black_spec))

            if fatal_exception is not None:
                break

            # If any game in this pairing had a problem (returned None or was a
            # per-game exception), PAUSE before moving to the next pairing —
            # but only when a UI callback is attached (the CLI headless runner
            # has no UI to pause on, so it skips and continues as before).
            if paused_problem_games and ui_callback is not None:
                await self._pause_on_problems(
                    ui_callback, paused_problem_games
                )
                if self._abort_requested:
                    raise FatalBenchmarkError(
                        "Benchmark aborted by user after a problem game.",
                        cause=None,
                    )

            # Finalize rating period after this pairing's games complete
            self.elo.finalize_period()

            if pairing_game_failures:
                print(f"\n{len(pairing_game_failures)} game(s) failed in pairing {white} vs {black}.")

        if fatal_exception is not None:
            raise FatalBenchmarkError(
                f"Benchmark aborted due to fatal error: {fatal_exception}",
                cause=fatal_exception,
            ) from fatal_exception

        self.logger.write_summary()

        # Print leaderboard
        print("\n=== FINAL LEADERBOARD ===")
        for row in self.elo.leaderboard():
            print(f"  {row['name']}: {row['rating']} ± {row['deviation']} (95% CI: {row['ci_low']}-{row['ci_high']})")

        print(f"\nResults saved to: {self.run_dir}")

        return self.run_dir


async def main() -> None:
    """Main entry point for benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Chess LLM Benchmark Runner")
    parser.add_argument("--config", default="benchmark.yaml", help="Config file path")
    parser.add_argument("--output", default="runs", help="Output directory")
    parser.add_argument("--players", nargs="+", help="Player specs (provider:model)")
    parser.add_argument("--games", type=int, help="Games per pairing")
    parser.add_argument("--parallel", type=int, help="Max parallel games")
    parser.add_argument(
        "--move-timeout",
        type=int,
        help="Per-move timeout in seconds (overrides config)",
    )
    parser.add_argument(
        "--game-timeout",
        type=int,
        help="Per-game wall-clock failsafe in seconds (overrides config)",
    )

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
    if args.move_timeout is not None:
        config.move_timeout_seconds = args.move_timeout
    if args.game_timeout is not None:
        config.game_timeout_seconds = args.game_timeout
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
