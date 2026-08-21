import asyncio
from chess_fight.benchmark.runner import BenchmarkConfig, BenchmarkRunner

async def main():
    config = BenchmarkConfig(
        players=["stockfish:15", "stockfish:15"],
        games_per_pairing=1,
        max_parallel_games=1,
        move_timeout_seconds=5,
    )
    runner = BenchmarkRunner(config)
    
    async def ui_callback(state):
        if state.is_paused:
            print("PAUSED:", state.pause_reason, state.pause_error)
            runner.request_continue_after_problem()
    
    await runner.run_benchmark_with_callback(ui_callback)

if __name__ == "__main__":
    asyncio.run(main())
