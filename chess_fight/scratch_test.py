import asyncio
from chess_fight.models.chess_ai import ChessAI
from chess_fight.game.async_game import AsyncChessGame, GameState
import logging

class DummyAI(ChessAI):
    def __init__(self, name="Dummy", moves=[]):
        super().__init__(name)
        self.moves = moves
        self.idx = 0
    
    async def _get_move_from_model(self, fen: str, validation_attempt: int = 0) -> str:
        if self.idx < len(self.moves):
            move = self.moves[self.idx]
            self.idx += 1
            return move
        return "<move>e2e4</move>"

async def main():
    logging.basicConfig(level=logging.DEBUG)
    p1 = DummyAI("W", ["<move>e2e4</move>", "<move>d2d4</move>"])
    p2 = DummyAI("B", ["<move>e7e5</move>", "<move>d7d5</move>"])
    
    async def ui_cb(state: GameState):
        if state.is_game_over:
            print(f"GAME OVER: {state.winner}")
        else:
            print(f"Turn: {state.current_player}")

    game = AsyncChessGame(p1, p2)
    stats = await game.play_game(ui_cb, delay=0)
    print("Stats:", stats)

asyncio.run(main())
