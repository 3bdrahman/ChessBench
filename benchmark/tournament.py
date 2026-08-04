"""Tournament mode with Swiss/Round-Robin pairing and bracket UI."""

import asyncio
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import itertools

from providers.chess_ai import ProviderChessAI
from game.async_game import AsyncChessGame
from benchmark.elo import BayesianElo
from benchmark.openings import OpeningBook


class TournamentType(Enum):
    ROUND_ROBIN = "round_robin"
    SWISS = "swiss"
    ELIMINATION = "elimination"


class TournamentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class TournamentPlayer:
    """Player in a tournament."""
    id: str
    name: str
    provider: str
    model: str
    api_key: str
    score: float = 0.0
    buchholz: float = 0.0  # Tiebreak
    games_played: int = 0
    color_history: List[str] = field(default_factory=list)  # 'white' or 'black'
    opponents: List[str] = field(default_factory=list)


@dataclass
class TournamentGame:
    """Single game in tournament."""
    id: str
    white_id: str
    black_id: str
    round_num: int
    board_num: int
    result: Optional[str] = None  # "1-0", "0-1", "1/2-1/2"
    result_numeric: Optional[float] = None  # 1.0, 0.0, 0.5
    opening_eco: Optional[str] = None
    opening_name: Optional[str] = None
    moves: int = 0
    duration: float = 0.0
    status: str = "pending"  # "pending", "in_progress", "completed"


@dataclass
class TournamentRound:
    """A round of games."""
    round_num: int
    games: List[TournamentGame]
    status: str = "pending"  # "pending", "in_progress", "completed"


class Tournament:
    """Tournament manager supporting multiple formats."""
    
    def __init__(
        self,
        name: str,
        tournament_type: TournamentType,
        players: List[TournamentPlayer],
        games_per_pairing: int = 1,
        opening_book: str = "eco_balanced",
        time_control: int = 30,
        temperature: float = 0.0,
        max_tokens: int = 100
    ):
        self.name = name
        self.tournament_type = tournament_type
        self.players = {p.id: p for p in players}
        self.games_per_pairing = games_per_pairing
        self.opening_book = OpeningBook()
        self.openings = self.opening_book.get_balanced_set(len(players) * (len(players) - 1) // 2 * games_per_pairing * 2)
        self.time_control = time_control
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.rounds: List[TournamentRound] = []
        self.current_round = 0
        self.status = TournamentStatus.PENDING
        self.elo = BayesianElo()
        
        # Initialize AI instances
        self.ai_instances = {}
        for player in players:
            self.ai_instances[player.id] = ProviderChessAI(
                provider_name=player.provider,
                model_id=player.model,
                api_key=player.api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        # Generate pairings
        self._generate_pairings()
    
    def _generate_pairings(self):
        """Generate all pairings based on tournament type."""
        if self.tournament_type == TournamentType.ROUND_ROBIN:
            self._generate_round_robin()
        elif self.tournament_type == TournamentType.SWISS:
            self._generate_swiss()
        elif self.tournament_type == TournamentType.ELIMINATION:
            self._generate_elimination()
    
    def _generate_round_robin(self):
        """Generate round-robin pairings (each player plays each other)."""
        player_ids = list(self.players.keys())
        n = len(player_ids)
        
        # Each pair plays twice (once as white, once as black) per games_per_pairing
        for _ in range(self.games_per_pairing):
            for i in range(n):
                for j in range(i + 1, n):
                    # Create two games for each pairing (alternating colors)
                    for game_num in range(2):
                        if game_num % 2 == 0:
                            white, black = player_ids[i], player_ids[j]
                        else:
                            white, black = player_ids[j], player_ids[i]
                        
                        game = TournamentGame(
                            id=f"r{len(self.rounds)+1}_b{len(self.rounds[0].games) if self.rounds else 0}_{white}_{black}",
                            white_id=white,
                            black_id=black,
                            round_num=len(self.rounds) + 1,
                            board_num=len(self.rounds[-1].games) + 1 if self.rounds else 1,
                        )
                        
                        # Add to current or new round
                        if not self.rounds or len(self.rounds[-1].games) >= n // 2:
                            self.rounds.append(TournamentRound(round_num=len(self.rounds) + 1, games=[]))
                        self.rounds[-1].games.append(game)
                        
                        # Track color history
                        self.players[white].color_history.append('white')
                        self.players[black].color_history.append('black')
                        self.players[white].opponents.append(black)
                        self.players[black].opponents.append(white)
    
    def _generate_swiss(self):
        """Generate Swiss system pairings (simplified)."""
        # For Swiss, we pair round by round based on scores
        # This is a simplified version - real Swiss is more complex
        player_ids = list(self.players.keys())
        num_rounds = min(len(player_ids) - 1, 7)  # Typical Swiss rounds
        
        for round_num in range(1, num_rounds + 1):
            round_obj = TournamentRound(round_num=round_num, games=[])
            
            # Sort by score (descending), then by rating/seed
            sorted_players = sorted(
                player_ids,
                key=lambda pid: (
                    -self.players[pid].score,
                    self.players[pid].name  # tiebreaker
                )
            )
            
            # Pair adjacent players (simple Swiss pairing)
            paired = set()
            for i, pid1 in enumerate(sorted_players):
                if pid1 in paired:
                    continue
                for pid2 in sorted_players[i+1:]:
                    if pid2 in paired:
                        continue
                    if pid2 in self.players[pid1].opponents:
                        continue  # Already played
                    
                    # Determine colors (balance as much as possible)
                    white_count = self.players[pid1].color_history.count('white')
                    black_count = self.players[pid1].color_history.count('black')
                    
                    if white_count <= black_count:
                        white, black = pid1, pid2
                    else:
                        white, black = pid2, pid1
                    
                    game = TournamentGame(
                        id=f"r{round_num}_b{len(round_obj.games)+1}_{white}_{black}",
                        white_id=white,
                        black_id=black,
                        round_num=round_num,
                        board_num=len(round_obj.games) + 1,
                    )
                    round_obj.games.append(game)
                    paired.add(pid1)
                    paired.add(pid2)
                    
                    # Update color history
                    self.players[white].color_history.append('white')
                    self.players[black].color_history.append('black')
                    self.players[white].opponents.append(black)
                    self.players[black].opponents.append(white)
                    break
            
            self.rounds.append(round_obj)
    
    def _generate_elimination(self):
        """Generate single-elimination bracket."""
        player_ids = list(self.players.keys())
        random.shuffle(player_ids)
        
        # Power of 2
        n = len(player_ids)
        target = 1
        while target < n:
            target *= 2
        
        # Add byes
        byes = target - n
        if byes > 0:
            # Random byes
            bye_players = random.sample(player_ids, byes)
            for pid in bye_players:
                # Auto-advance
                pass
            player_ids = [p for p in player_ids if p not in bye_players]
        
        round_num = 1
        while len(player_ids) > 1:
            round_obj = TournamentRound(round_num=round_num, games=[])
            next_round_players = []
            
            for i in range(0, len(player_ids), 2):
                if i + 1 < len(player_ids):
                    white, black = player_ids[i], player_ids[i+1]
                    game = TournamentGame(
                        id=f"r{round_num}_b{len(round_obj.games)+1}_{white}_{black}",
                        white_id=white,
                        black_id=black,
                        round_num=round_num,
                        board_num=len(round_obj.games) + 1,
                    )
                    round_obj.games.append(game)
                    
                    # Winner placeholder - will be filled after game
                    next_round_players.append(f"winner_{game.id}")
            
            self.rounds.append(round_obj)
            player_ids = next_round_players
            round_num += 1
    
    def get_standings(self) -> List[Dict]:
        """Get current tournament standings."""
        standings = []
        for player in self.players.values():
            # Calculate Buchholz (sum of opponents' scores)
            buchholz = sum(self.players[opp].score for opp in player.opponents) if player.opponents else 0
            
            standings.append({
                'id': player.id,
                'name': player.name,
                'provider': player.provider,
                'model': player.model,
                'score': player.score,
                'games_played': player.games_played,
                'wins': int(player.score - player.games_played * 0.5 + sum(1 for g in self._get_player_games(player.id) if g.result_numeric == 1.0)),
                'draws': int(sum(1 for g in self._get_player_games(player.id) if g.result_numeric == 0.5)),
                'losses': int(sum(1 for g in self._get_player_games(player.id) if g.result_numeric == 0.0)),
                'buchholz': buchholz,
                'color_balance': player.color_history.count('white') - player.color_history.count('black'),
            })
        
        # Sort by score, then Buchholz
        standings.sort(key=lambda x: (-x['score'], -x['buchholz'], x['name']))
        
        # Add rank
        for i, s in enumerate(standings):
            s['rank'] = i + 1
        
        return standings
    
    def _get_player_games(self, player_id: str) -> List[TournamentGame]:
        """Get all games for a player."""
        games = []
        for round_obj in self.rounds:
            for game in round_obj.games:
                if game.white_id == player_id or game.black_id == player_id:
                    games.append(game)
        return games
    
    async def play_round(self, round_num: int, ui_callback=None) -> List[TournamentGame]:
        """Play all games in a round concurrently."""
        if round_num > len(self.rounds):
            return []
        
        round_obj = self.rounds[round_num - 1]
        round_obj.status = "in_progress"
        
        games_to_play = [g for g in round_obj.games if g.status == "pending"]
        
        # Play games with limited concurrency
        semaphore = asyncio.Semaphore(4)  # Max 4 concurrent games
        
        async def play_single_game(game: TournamentGame):
            async with semaphore:
                return await self._play_game(game, ui_callback)
        
        tasks = [play_single_game(game) for game in games_to_play]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            game = games_to_play[i]
            if isinstance(result, Exception):
                game.status = "error"
                game.result = "Error"
            else:
                game.status = "completed"
                game.result, game.result_numeric, game.moves, game.duration = result
                
                # Update player scores
                white = self.players[game.white_id]
                black = self.players[game.black_id]
                
                if game.result_numeric == 1.0:
                    white.score += 1
                elif game.result_numeric == 0.5:
                    white.score += 0.5
                    black.score += 0.5
                else:
                    black.score += 1
                
                white.games_played += 1
                black.games_played += 1
                
                # Update ELO
                self.elo.add_game(white.name, black.name, game.result_numeric or 0.5, game.opening_eco)
        
        round_obj.status = "completed"
        return games_to_play
    
    async def _play_game(self, game: TournamentGame, ui_callback=None):
        """Play a single tournament game."""
        white_ai = self.ai_instances[game.white_id]
        black_ai = self.ai_instances[game.black_id]
        
        # Select opening
        if game.opening_eco:
            opening = self.opening_book.get_opening_by_eco(game.opening_eco)
        else:
            opening = self.openings.pop() if self.openings else {'eco': 'START', 'name': 'Start', 'fen': 'startpos', 'moves': []}
        
        game.opening_eco = opening['eco']
        game.opening_name = opening['name']
        
        # Play game
        async_game = AsyncChessGame(white_ai, black_ai)
        
        if opening.get('fen') and opening['fen'] != 'startpos':
            async_game.board = chess.Board(opening['fen'])
        
        import time
        start_time = time.time()
        
        if ui_callback:
            await ui_callback({"type": "game_start", "game": game})
        
        stats = await async_game.play_game(
            lambda state: ui_callback({"type": "game_state", "game": game, "state": state}) if ui_callback else None,
            delay=0.05
        )
        
        duration = time.time() - start_time
        
        # Determine result
        if stats.winner == white_ai.name:
            result = "1-0"
            result_numeric = 1.0
        elif stats.winner == black_ai.name:
            result = "0-1"
            result_numeric = 0.0
        else:
            result = "1/2-1/2"
            result_numeric = 0.5
        
        if ui_callback:
            await ui_callback({"type": "game_end", "game": game, "result": result})
        
        return result, result_numeric, stats.total_moves, duration
    
    async def play_tournament(self, ui_callback=None):
        """Play the entire tournament."""
        self.status = TournamentStatus.IN_PROGRESS
        
        for round_obj in self.rounds:
            await self.play_round(round_obj.round_num, ui_callback)
        
        self.status = TournamentStatus.COMPLETED
        
        return self.get_standings()


class TournamentBracket:
    """Generate bracket visualization for elimination tournaments."""
    
    @staticmethod
    def generate_bracket_html(tournament: Tournament) -> str:
        """Generate HTML for tournament bracket."""
        if tournament.tournament_type != TournamentType.ELIMINATION:
            return "<p>Bracket only available for elimination tournaments</p>"
        
        html = ['<div class="tournament-bracket">']
        
        for round_obj in tournament.rounds:
            html.append(f'<div class="bracket-round" data-round="{round_obj.round_num}">')
            html.append(f'<h4>Round {round_obj.round_num}</h4>')
            html.append('<div class="bracket-games">')
            
            for game in round_obj.games:
                white_name = tournament.players[game.white_id].name
                black_name = tournament.players[game.black_id].name
                
                result_class = ""
                if game.result == "1-0":
                    result_class = "winner-white"
                elif game.result == "0-1":
                    result_class = "winner-black"
                elif game.result == "1/2-1/2":
                    result_class = "draw"
                
                html.append(f'''
                <div class="bracket-game {result_class}">
                    <div class="player white">{white_name}</div>
                    <div class="player black">{black_name}</div>
                    <div class="result">{game.result or "vs"}</div>
                </div>
                ''')
            
            html.append('</div></div>')
        
        html.append('</div>')
        
        # Add CSS
        css = '''
        <style>
        .tournament-bracket { display: flex; flex-direction: column; gap: 20px; padding: 20px; }
        .bracket-round { background: #f8f9fa; padding: 15px; border-radius: 8px; }
        .bracket-round h4 { margin-top: 0; }
        .bracket-games { display: flex; flex-direction: column; gap: 10px; }
        .bracket-game { display: flex; align-items: center; gap: 10px; padding: 10px; background: white; border-radius: 5px; border: 1px solid #ddd; }
        .bracket-game .player { flex: 1; }
        .bracket-game .player.white { font-weight: bold; }
        .bracket-game .result { background: #e9ecef; padding: 5px 15px; border-radius: 3px; font-family: monospace; }
        .bracket-game.winner-white .player.white { color: #27ae60; }
        .bracket-game.winner-black .player.black { color: #27ae60; }
        .bracket-game.draw .result { background: #fff3cd; }
        </style>
        '''
        
        return css + '\n'.join(html)


if __name__ == "__main__":
    import chess
    # Demo
    players = [
        TournamentPlayer("1", "GPT-4o", "openai", "gpt-4o", "sk-test"),
        TournamentPlayer("2", "Claude", "anthropic", "claude-3-5-sonnet", "sk-test"),
        TournamentPlayer("3", "Gemini", "google", "gemini-1.5-pro", "sk-test"),
        TournamentPlayer("4", "Llama", "ollama", "llama3.2", ""),
    ]
    
    tourney = Tournament("Test Tournament", TournamentType.ROUND_ROBIN, players, games_per_pairing=1)
    print(f"Generated {len(tourney.rounds)} rounds with {sum(len(r.games) for r in tourney.rounds)} games")
    
    for round_obj in tourney.rounds:
        print(f"  Round {round_obj.round_num}: {len(round_obj.games)} games")
        for game in round_obj.games:
            print(f"    {tourney.players[game.white_id].name} vs {tourney.players[game.black_id].name}")