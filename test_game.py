import asyncio
import logging
from chess_fight.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chess_fight.providers.registry import register_provider
from tests.test_integration import MockProvider

logging.basicConfig(level=logging.INFO)

async def main():
    register_provider("mock_registered", MockProvider)
    config = BenchmarkConfig(
        players=['mock_registered:mock', 'mock_registered:mock'],
        games_per_pairing=1,
        colors='fixed'
    )
    runner = BenchmarkRunner(config)
    await runner.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main())
