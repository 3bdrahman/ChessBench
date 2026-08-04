"""Lichess integration for game import/analysis."""

import asyncio
import json
import webbrowser
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import chess
import chess.pgn
from io import StringIO


@dataclass
class LichessGame:
    """Lichess game data."""
    id: str
    url: str
    white: str
    black: str
    result: str
    opening: Optional[str]
    opening_eco: Optional[str]
    moves: str
    pgn: str
    created_at: int
    speed: str  # bullet, blitz, rapid, classical
    perf: Dict[str, Any]


class LichessClient:
    """Client for interacting with Lichess API."""
    
    BASE_URL = "https://lichess.org/api"
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.session = aiohttp.ClientSession(headers=headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_game(self, game_id: str) -> LichessGame:
        """Get a single game by ID."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        async with self.session.get(f"{self.BASE_URL}/game/{game_id}") as resp:
            if resp.status == 404:
                raise ValueError(f"Game not found: {game_id}")
            data = await resp.json()
        
        return self._parse_game(data)
    
    async def get_user_games(
        self, 
        username: str, 
        max_games: int = 100,
        rated: bool = True,
        perf_type: Optional[str] = None
    ) -> List[LichessGame]:
        """Get games for a user."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        params = {"max": max_games}
        if rated:
            params["rated"] = "true"
        if perf_type:
            params["perfType"] = perf_type
        
        async with self.session.get(f"{self.BASE_URL}/games/user/{username}", params=params) as resp:
            if resp.status == 404:
                raise ValueError(f"User not found: {username}")
            # NDJSON format
            text = await resp.text()
        
        games = []
        for line in text.strip().split('\n'):
            if line:
                data = json.loads(line)
                games.append(self._parse_game(data))
        
        return games
    
    async def import_pgn(self, pgn: str) -> Dict[str, Any]:
        """Import a PGN to Lichess (requires token)."""
        if not self.token:
            raise ValueError("Token required for importing games")
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        data = aiohttp.FormData()
        data.add_field('pgn', pgn)
        
        async with self.session.post(f"{self.BASE_URL}/import", data=data) as resp:
            return await resp.json()
    
    async def get_cloud_eval(self, fen: str, multi_pv: int = 3) -> Dict[str, Any]:
        """Get cloud evaluation for a position."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        params = {"fen": fen, "multiPv": multi_pv}
        async with self.session.get(f"{self.BASE_URL}/cloud-eval", params=params) as resp:
            return await resp.json()
    
    def _parse_game(self, data: Dict) -> LichessGame:
        """Parse Lichess API game response."""
        players = data.get("players", {})
        white = players.get("white", {}).get("user", {}).get("name", "Anonymous")
        black = players.get("black", {}).get("user", {}).get("name", "Anonymous")
        
        # Extract moves
        moves = data.get("moves", "")
        
        # Get opening
        opening = data.get("opening", {})
        opening_name = opening.get("name") if opening else None
        opening_eco = opening.get("eco") if opening else None
        
        return LichessGame(
            id=data.get("id", ""),
            url=f"https://lichess.org/{data.get('id', '')}",
            white=white,
            black=black,
            result=data.get("winner", "draw") if data.get("winner") else "1/2-1/2",
            opening=opening_name,
            opening_eco=opening_eco,
            moves=moves,
            pgn=self._moves_to_pgn(data),
            created_at=data.get("createdAt", 0),
            speed=data.get("speed", "unknown"),
            perf=data.get("perf", {})
        )
    
    def _moves_to_pgn(self, data: Dict) -> str:
        """Convert game data to PGN."""
        pgn_lines = [
            f'[Event "{data.get("event", "Lichess Game")}"]',
            f'[Site "https://lichess.org/{data.get("id", "")}"]',
            f'[Date "{self._timestamp_to_date(data.get("createdAt", 0))}"]',
            f'[White "{data.get("players", {}).get("white", {}).get("user", {}).get("name", "Anonymous")}"]',
            f'[Black "{data.get("players", {}).get("black", {}).get("user", {}).get("name", "Anonymous")}"]',
            f'[Result "{self._result_to_pgn(data.get("winner"))}"]',
            f'[WhiteElo "{data.get("players", {}).get("white", {}).get("rating", "?")}"]',
            f'[BlackElo "{data.get("players", {}).get("black", {}).get("rating", "?")}"]',
            f'[TimeControl "{data.get("clock", {}).get("initial", 0)}+{data.get("clock", {}).get("increment", 0)}"]',
            f'[Opening "{data.get("opening", {}).get("name", "?")}"]',
            f'[ECO "{data.get("opening", {}).get("eco", "?")}"]',
            '',
            data.get("moves", "") + " " + self._result_to_pgn(data.get("winner")),
            ''
        ]
        return '\n'.join(pgn_lines)
    
    def _result_to_pgn(self, winner: Optional[str]) -> str:
        if winner == "white":
            return "1-0"
        elif winner == "black":
            return "0-1"
        return "1/2-1/2"
    
    def _timestamp_to_date(self, timestamp: int) -> str:
        from datetime import datetime
        return datetime.utcfromtimestamp(timestamp / 1000).strftime("%Y.%m.%d")


class LichessAnalyzer:
    """Analyze games using Lichess cloud evaluation."""
    
    def __init__(self, client: LichessClient):
        self.client = client
    
    async def analyze_game(
        self, 
        game: LichessGame, 
        sample_moves: int = 10
    ) -> Dict[str, Any]:
        """Analyze key positions in a game."""
        board = chess.Board()
        moves = game.moves.split()
        
        analyses = []
        for i, move_uci in enumerate(moves[:sample_moves]):
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    
                    # Analyze every few moves or critical positions
                    if i % 3 == 0 or board.is_check() or board.is_capture(move):
                        eval_data = await self.client.get_cloud_eval(board.fen())
                        analyses.append({
                            "move_number": i + 1,
                            "move": move_uci,
                            "fen": board.fen(),
                            "evaluation": eval_data
                        })
            except Exception:
                break
        
        return {
            "game_id": game.id,
            "analyses": analyses,
            "total_moves_analyzed": len(analyses)
        }
    
    async def find_blunders(self, game: LichessGame, threshold: float = 2.0) -> List[Dict]:
        """Find blunders in a game (eval drop > threshold)."""
        board = chess.Board()
        moves = game.moves.split()
        prev_eval = None
        blunders = []
        
        for i, move_uci in enumerate(moves):
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    break
                board.push(move)
                
                eval_data = await self.client.get_cloud_eval(board.fen())
                
                # Extract evaluation
                current_eval = self._extract_eval(eval_data)
                
                if prev_eval is not None and current_eval is not None:
                    # Check for blunder from perspective of player who just moved
                    eval_diff = prev_eval - current_eval
                    if eval_diff > threshold:
                        blunders.append({
                            "move_number": i + 1,
                            "player": "white" if i % 2 == 0 else "black",
                            "move": move_uci,
                            "eval_before": prev_eval,
                            "eval_after": current_eval,
                            "eval_drop": eval_diff,
                            "fen": board.fen()
                        })
                
                prev_eval = current_eval
                
            except Exception:
                break
        
        return blunders
    
    def _extract_eval(self, eval_data: Dict) -> Optional[float]:
        """Extract numeric evaluation from Lichess cloud eval."""
        try:
            if "pvs" in eval_data and eval_data["pvs"]:
                pv = eval_data["pvs"][0]
                if "cp" in pv:
                    return pv["cp"] / 100.0  # Convert centipawns to pawns
                elif "mate" in pv:
                    return 10.0 if pv["mate"] > 0 else -10.0
        except Exception:
            pass
        return None


async def export_game_to_lichess(pgn: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Export a game PGN to Lichess."""
    async with LichessClient(token) as client:
        return await client.import_pgn(pgn)


async def analyze_lichess_game(game_id: str) -> Dict[str, Any]:
    """Analyze a Lichess game by ID."""
    async with LichessClient() as client:
        game = await client.get_game(game_id)
        analyzer = LichessAnalyzer(client)
        
        analysis = await analyzer.analyze_game(game)
        blunders = await analyzer.find_blunders(game)
        
        return {
            "game": {
                "id": game.id,
                "url": game.url,
                "white": game.white,
                "black": game.black,
                "result": game.result,
                "opening": game.opening,
                "opening_eco": game.opening_eco,
            },
            "analysis": analysis,
            "blunders": blunders
        }


def open_lichess_game(game_id: str):
    """Open a Lichess game in browser."""
    webbrowser.open(f"https://lichess.org/{game_id}")


def open_lichess_analysis(fen: str):
    """Open Lichess analysis board for a position."""
    webbrowser.open(f"https://lichess.org/analysis/{fen}")


if __name__ == "__main__":
    # Demo: analyze a famous game
    async def demo():
        # Use a known game ID (this is a placeholder)
        result = await analyze_lichess_game("X7Xz7Q8")  # Example ID
        print(json.dumps(result, indent=2))
    
    # asyncio.run(demo())
    print("Lichess integration module ready")