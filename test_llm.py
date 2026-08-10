import asyncio
import os
import sys

from chess_fight.providers.chess_ai import ProviderChessAI
from chess_fight.common.exceptions import MoveExhaustedError

async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("No OPENROUTER_API_KEY set.")
        return

    ai = ProviderChessAI(
        provider_name="openrouter",
        model_id="inclusionai/ling-3.0-tiny:free",
        api_key=api_key,
        temperature=0.0,
        max_tokens=1500
    )
    
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    try:
        move = await ai.get_move(fen)
        print("Successfully got move:", move)
    except MoveExhaustedError as e:
        print("Failed!")
        print("RAW TEXT:")
        print(e.raw_text)

if __name__ == "__main__":
    asyncio.run(main())
