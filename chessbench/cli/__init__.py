"""ChessBench CLI — LLM Critical Thinking Benchmark via Chess."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

logging.getLogger("streamlit").setLevel(logging.ERROR)

from chessbench.benchmark.adversarial import AdversarialConfig, AdversarialEvaluator  # noqa: E402
from chessbench.benchmark.results_view import list_runs  # noqa: E402
from chessbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner  # noqa: E402
from chessbench.benchmark.verify import verify_run_reproducibility  # noqa: E402
from chessbench.providers import get_provider, list_providers  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_keys() -> dict[str, str]:
    """Load API keys from environment variables."""
    keys = {}
    for provider in list_providers():
        key = os.getenv(f"{provider.upper()}_API_KEY")
        if key:
            keys[provider] = key
    return keys


def _parse_players(ctx: click.Context, param: click.Parameter, value: tuple[str, ...] | str | None) -> list[str]:
    """Parse player specs from command line, supporting comma, space, or repeated flags."""
    if not value:
        return []
    players: list[str] = []
    items = value if isinstance(value, (tuple, list)) else [value]
    for item in items:
        for part in item.replace(",", " ").split():
            part_str = part.strip()
            if part_str:
                players.append(part_str)
    return players


# ---------------------------------------------------------------------------
# Main CLI Group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="chessbench")
def cli() -> None:
    """ChessBench — LLM Critical Thinking Benchmark via Chess.

    Evaluate LLM reasoning capabilities through structured chess tournaments.
    """


# ---------------------------------------------------------------------------
# Run Command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to benchmark config YAML file",
)
@click.option(
    "--players",
    "-p",
    multiple=True,
    callback=_parse_players,
    help="Player specs as provider:model (e.g., openai:gpt-4o).",
)
@click.argument("extra_players", nargs=-1)
@click.option("--games", "-g", type=int, help="Games per pairing")
@click.option("--parallel", "-j", type=int, help="Max parallel games")
@click.option("--opening-book", type=click.Choice(["eco_balanced", "eco_all", "startpos"]), help="Opening book to use")
@click.option("--time-control", type=int, help="Seconds per move")
@click.option("--temperature", type=float, help="Model temperature (0.0 for deterministic)")
@click.option("--max-tokens", type=int, help="Max tokens per completion")
@click.option("--reasoning-level", type=click.Choice(["low", "mid", "high"]), help="Reasoning detail level")
@click.option("--move-timeout", type=int, help="Per-move timeout in seconds")
@click.option("--game-timeout", type=int, help="Per-game wall-clock failsafe in seconds")
@click.option("--output", "-o", type=click.Path(path_type=Path), default="runs", help="Output directory")
def _sanitize_run_name(ctx, param, value):
    """Sanitize run name to prevent path traversal."""
    if value is None:
        return value
    import os
    # Reject path separators and parent directory references
    if os.path.sep in value or (os.path.altsep and os.path.altsep in value):
        raise click.BadParameter(f"Run name cannot contain path separators: {value}")
    if os.path.basename(value) != value:
        raise click.BadParameter(f"Run name cannot contain directory components: {value}")
    return value


@click.option("--name", "-n", help="Run name (default: timestamp)", callback=_sanitize_run_name)
@click.option("--seed", type=int, help="Random seed for reproducibility")
@click.option("--colors", type=click.Choice(["alternating", "fixed"]), help="Color assignment mode")
def run(
    config: Path | None,
    players: list[str],
    extra_players: tuple[str, ...],
    games: int | None,
    parallel: int | None,
    opening_book: str | None,
    time_control: int | None,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_level: str | None,
    move_timeout: int | None,
    game_timeout: int | None,
    output: Path,
    name: str | None,
    seed: int | None,
    colors: str | None,
) -> None:
    """Run a benchmark tournament between LLM models."""

    # Load config from file or create default
    if config:
        click.echo(f"Loading config from {config}")
        benchmark_config = BenchmarkConfig.from_yaml(str(config))
    else:
        benchmark_config = BenchmarkConfig()

    # Override from CLI
    all_players = list(players) + list(extra_players)
    if all_players:
        benchmark_config.players = all_players
    if games is not None:
        benchmark_config.games_per_pairing = games
    if parallel is not None:
        benchmark_config.max_parallel_games = parallel
    if opening_book:
        benchmark_config.opening_book = opening_book
    if time_control is not None:
        benchmark_config.time_control_seconds_per_move = time_control
    if temperature is not None:
        benchmark_config.temperature = temperature
    if max_tokens is not None:
        benchmark_config.max_tokens = max_tokens
    if reasoning_level:
        benchmark_config.reasoning_level = reasoning_level
    if move_timeout is not None:
        benchmark_config.move_timeout_seconds = move_timeout
    if game_timeout is not None:
        benchmark_config.game_timeout_seconds = game_timeout
    if seed is not None:
        benchmark_config.seed = seed
    if colors:
        benchmark_config.colors = colors
    benchmark_config.output_dir = str(output)
    if name:
        benchmark_config.run_name = name

    # Load API keys from environment
    benchmark_config.api_keys = _load_api_keys()

    # Validate
    if not benchmark_config.players:
        click.echo("Error: No players specified. Use --players or config file.", err=True)
        sys.exit(1)

    if len(benchmark_config.players) < 2:
        click.echo("Error: At least 2 players required.", err=True)
        sys.exit(1)

    click.echo(f"Starting benchmark: {benchmark_config.run_name or 'auto'}")
    click.echo(f"Players: {benchmark_config.players}")
    click.echo(f"Games per pairing: {benchmark_config.games_per_pairing}")
    click.echo(f"Parallel games: {benchmark_config.max_parallel_games}")
    click.echo(f"Output: {output / (benchmark_config.run_name or 'auto')}")

    try:
        run_dir = asyncio.run(_run_benchmark(benchmark_config))
        click.echo(f"\n✅ Benchmark complete! Results in: {run_dir}")
    except Exception as e:
        click.echo(f"\n❌ Benchmark failed: {e}", err=True)
        sys.exit(1)


async def _run_benchmark(config: BenchmarkConfig) -> Path:
    """Run benchmark asynchronously."""
    runner = BenchmarkRunner(config)
    return await runner.run_benchmark()


# ---------------------------------------------------------------------------
# Evaluate Command (Adversarial)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "-m", required=True, help="Model to evaluate (provider:model)")
@click.option("--depths", default="8,12,16,20", help="Comma-separated Stockfish depths")
@click.option("--games", "-g", default=4, help="Games per depth")
@click.option("--colors", type=click.Choice(["alternating", "white", "black"]), default="alternating")
@click.option("--time-control", default=30, help="Seconds per move")
@click.option("--parallel", "-j", default=2, help="Max parallel games")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file")
def evaluate(
    model: str,
    depths: str,
    games: int,
    colors: str,
    time_control: int,
    parallel: int,
    output: Path | None,
) -> None:
    """Evaluate a model against Stockfish at calibrated depths (adversarial mode)."""

    depth_list = [int(d.strip()) for d in depths.split(",")]

    config = AdversarialConfig(
        stockfish_depths=depth_list,
        games_per_depth=games,
        colors=colors,
        time_control_seconds=time_control,
        max_parallel_games=parallel,
    )

    api_keys = _load_api_keys()

    click.echo(f"Evaluating {model} vs Stockfish depths: {depth_list}")
    click.echo(f"Games per depth: {games}")

    try:
        report = asyncio.run(_run_adversarial(model, config, api_keys))

        # Print summary
        click.echo("\n=== ADVERSARIAL REPORT ===")
        click.echo(f"Model: {report.model_name}")
        click.echo(f"Equivalent Stockfish Depth: {report.equivalent_depth:.1f}" if report.equivalent_depth else "Equivalent Depth: N/A")
        click.echo()
        for dr in report.depth_results:
            click.echo(f"  Depth {dr.depth}: {dr.games} games, "
                       f"Score: {dr.llm_score:.1%}, "
                       f"Avg CP Loss: {dr.llm_cp_loss:.1f}, "
                       f"W/D/L: {dr.win_rate:.1%}/{dr.draw_rate:.1%}/{dr.loss_rate:.1%}")

        if output:
            output.write_text(json.dumps(report.to_dict(), indent=2))
            click.echo(f"\n📄 Report saved to {output}")

    except Exception as e:
        click.echo(f"\n❌ Evaluation failed: {e}", err=True)
        sys.exit(1)


async def _run_adversarial(
    model_spec: str,
    config: AdversarialConfig,
    api_keys: dict[str, str],
) -> Any:
    """Run adversarial evaluation."""
    evaluator = AdversarialEvaluator(config)
    return await evaluator.evaluate_model(model_spec)


# ---------------------------------------------------------------------------
# Report Command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "-f", type=click.Choice(["html", "pdf", "parquet", "csv", "json"]), default="html")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file/directory")
@click.option("--open/--no-open", default=False, help="Open HTML report in browser")
def report(run_dir: Path, format: str, output: Path | None, open: bool) -> None:
    """Generate a report from a benchmark run."""

    click.echo(f"Generating {format.upper()} report from {run_dir}...")

    try:
        if format in ("parquet", "csv"):
            from chessbench.benchmark.export import export_csv, export_parquet
            if format == "parquet":
                out = export_parquet(run_dir, output)
            else:
                out = export_csv(run_dir, output)
            click.echo(f"✅ Exported to {out}")

        elif format == "html":
            from chessbench.benchmark.export import export_all_formats
            results = export_all_formats(run_dir, output)
            click.echo(f"✅ Exported: {results}")
            if open and "pgn_eval" in results:
                import webbrowser
                webbrowser.open(f"file://{results['pgn_eval'].absolute()}")

        elif format == "json":
            from chessbench.benchmark.results_view import load_run
            run = load_run(run_dir)
            if run is None:
                raise ValueError("No valid run data found")
            out_path = output or (run_dir / "report.json")
            out_path.write_text(json.dumps({
                "run_id": run.run_id,
                "config": run.config,
                "leaderboard": [
                    {"player": k, **asdict(v)} for k, v in run.player_stats.items()
                ],
            }, indent=2, default=str))
            click.echo(f"✅ JSON report saved to {out_path}")

        elif format == "pdf":
            click.echo("⚠️  PDF export requires weasyprint. Use HTML and print to PDF.")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Report generation failed: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Verify Command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--move-tolerance", type=int, default=100, help="Move timing tolerance (ms)")
@click.option("--token-tolerance", type=int, default=5, help="Token count tolerance")
@click.option("--full-behavioral", is_flag=True, help="Run full behavioral check (requires API keys)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def verify(run_dir: Path, move_tolerance: int, token_tolerance: int, full_behavioral: bool, json_output: bool) -> None:
    """Verify reproducibility of a benchmark run."""

    click.echo(f"Verifying reproducibility of {run_dir}...")

    try:
        report = asyncio.run(verify_run_reproducibility(
            run_dir,
            move_timing_tolerance_ms=move_tolerance,
            token_tolerance=token_tolerance,
            full_behavioral_check=full_behavioral,
        ))

        if json_output:
            click.echo(json.dumps(report.to_dict(), indent=2))
        else:
            click.echo(f"Reproducibility Check: {report.status}")
            if report.config_hash_match is not None:
                click.echo(f"  Config Hash Match: {'YES' if report.config_hash_match else 'NO'}")
                click.echo(f"  Original Hash: {report.original_hash}")
                click.echo(f"  New Hash:      {report.new_hash}")
            if report.diffs:
                click.echo(f"  Differences ({len(report.diffs)}):")
                for diff in report.diffs[:20]:
                    click.echo(f"    - {diff}")
                if len(report.diffs) > 20:
                    click.echo(f"    ... and {len(report.diffs) - 20} more")
            if report.error:
                click.echo(f"  Error: {report.error}")

        sys.exit(0 if report.status == "PASS" else 1)

    except Exception as e:
        click.echo(f"❌ Verification failed: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# History Command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--runs-dir", type=click.Path(exists=True, path_type=Path), default="runs")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def history(runs_dir: Path, json_output: bool) -> None:
    """List all benchmark runs."""

    runs = list_runs(str(runs_dir))

    if not runs:
        click.echo("No benchmark runs found.")
        return

    if json_output:
        click.echo(json.dumps([
            {
                "run_id": r.run_id,
                "total_games": r.total_games,
                "providers": list(r.providers_seen),
                "timestamp": r.timestamp_utc,
            }
            for r in runs
        ], indent=2))
        return

    click.echo(f"Found {len(runs)} benchmark run(s) in {runs_dir}:\n")
    for r in runs:
        click.echo(f"  {r.run_id}")
        click.echo(f"    Games: {r.total_games} | Providers: {', '.join(r.providers_seen)} | Time: {r.timestamp_utc}")
        if r.player_stats:
            top = max(r.player_stats.items(), key=lambda x: x[1].score_pct or 0)
            click.echo(f"    Leader: {top[0]} ({top[1].score_pct:.1f}%)")
        click.echo()


# ---------------------------------------------------------------------------
# Models Command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--provider", "-p", help="Filter by provider")
@click.option("--filter", "-f", help="Filter models by name (supports 'free' for free tier)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def models(provider: str | None, filter: str | None, json_output: bool) -> None:
    """List available models from providers."""

    api_keys = _load_api_keys()
    providers_to_check = [provider] if provider else list_providers()

    all_models = {}

    for prov_name in providers_to_check:
        prov = get_provider(prov_name)
        if not prov:
            continue
        if prov_name not in api_keys and prov.requires_api_key:
            if provider:
                click.echo(f"Warning: {prov_name.upper()}_API_KEY environment variable is not set.", err=True)
            continue

        try:
            models_list = asyncio.run(prov.list_models(api_keys.get(prov_name, "")))
            for model in models_list:
                if filter and filter.lower() not in model.name.lower() and filter.lower() not in model.id.lower():
                    continue
                if filter == "free" and not (":free" in model.id.lower() or "free" in model.name.lower()):
                    continue
                key = f"{prov_name}:{model.id}"
                all_models[key] = {
                    "provider": prov_name,
                    "id": model.id,
                    "name": model.name,
                    "context_window": model.context_window,
                }
        except Exception as e:
            click.echo(f"Warning: Failed to fetch models from {prov_name}: {e}", err=True)

    if json_output:
        click.echo(json.dumps(all_models, indent=2))
        return

    if not all_models:
        click.echo("No models found. Check API keys and provider availability.")
        return

    click.echo(f"Available models ({len(all_models)}):\n")
    for key, info in sorted(all_models.items()):
        click.echo(f"  {key}")
        click.echo(f"    Name: {info['name']}")
        click.echo(f"    Context: {info['context_window']:,} tokens")
        click.echo()


# ---------------------------------------------------------------------------
# Config Command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--output", "-o", type=click.Path(path_type=Path), default="benchmark.yaml")
@click.option("--players", "-p", multiple=True, help="Player specs to include in config")
def config(output: Path, players: tuple[str, ...]) -> None:
    """Generate a benchmark configuration file."""

    default_players = list(players) if players else [
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-20241022",
        "google:gemini-1.5-pro",
        "openrouter:anthropic/claude-3.5-sonnet",
    ]

    config = BenchmarkConfig(
        players=default_players,
    )

    config.save_yaml(str(output))
    click.echo(f"✅ Config saved to {output}")
    click.echo("Edit the file to customize your benchmark, then run:")
    click.echo(f"  chessbench run --config {output}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    cli()


if __name__ == "__main__":
    main()
