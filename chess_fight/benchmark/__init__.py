"""Benchmark module exports."""

from chess_fight.benchmark.adversarial import (
    AdversarialConfig,
    AdversarialEvaluator,
    AdversarialReport,
    DepthResult,
)
from chess_fight.benchmark.elo import BayesianElo, GameResult, Glicko2
from chess_fight.benchmark.evaluator import EvaluationResult, StockfishEvaluator
from chess_fight.benchmark.export import (
    compute_config_hash,
    create_reproducibility_metadata,
    export_all_formats,
    export_csv,
    export_parquet,
    export_pgn_with_eval,
)
from chess_fight.benchmark.logging import BenchmarkLogger, GameLogEntry, MoveLogEntry, MoveQuality
from chess_fight.benchmark.openings import OpeningBook
from chess_fight.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chess_fight.benchmark.statistics import (
    PairingStats,
    binom_confidence_interval,
    binomial_test,
    bootstrap_ci,
    compute_pairing_stats,
    effective_sample_size,
    glicko2_bootstrap_ci,
    rating_convergence,
    rating_stability_metric,
)
from chess_fight.benchmark.verify import (
    ReproductionReport,
    run_reproducibility_cli,
    verify_run_reproducibility,
)

__all__ = [
    # Adversarial
    "AdversarialConfig",
    "AdversarialEvaluator",
    "AdversarialReport",
    # ELO
    "BayesianElo",
    # Runner
    "BenchmarkConfig",
    # Logging
    "BenchmarkLogger",
    "BenchmarkRunner",
    "DepthResult",
    "EvaluationResult",
    "GameLogEntry",
    "GameResult",
    "Glicko2",
    "MoveLogEntry",
    "MoveQuality",
    # Openings
    "OpeningBook",
    # Statistics
    "PairingStats",
    # Verify
    "ReproductionReport",
    # Evaluator
    "StockfishEvaluator",
    "binom_confidence_interval",
    "binomial_test",
    "bootstrap_ci",
    "compute_config_hash",
    "compute_pairing_stats",
    "create_reproducibility_metadata",
    "effective_sample_size",
    "export_all_formats",
    "export_csv",
    # Export
    "export_parquet",
    "export_pgn_with_eval",
    "glicko2_bootstrap_ci",
    "rating_convergence",
    "rating_stability_metric",
    "run_reproducibility_cli",
    "verify_run_reproducibility",
]
