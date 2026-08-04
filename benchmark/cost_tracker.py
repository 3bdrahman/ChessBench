"""Cost tracking dashboard for benchmark runs."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class ModelPricing:
    """Pricing information for a model."""
    provider: str
    model_id: str
    input_price_per_1k: float  # $ per 1K input tokens
    output_price_per_1k: float  # $ per 1K output tokens
    currency: str = "USD"

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token usage."""
        return (input_tokens / 1000 * self.input_price_per_1k +
                output_tokens / 1000 * self.output_price_per_1k)


# Current pricing (as of 2024 - update as needed)
DEFAULT_PRICING = {
    "openai": {
        "gpt-4o": ModelPricing("openai", "gpt-4o", 0.005, 0.015),
        "gpt-4o-mini": ModelPricing("openai", "gpt-4o-mini", 0.00015, 0.0006),
        "o1-preview": ModelPricing("openai", "o1-preview", 0.015, 0.06),
        "o1-mini": ModelPricing("openai", "o1-mini", 0.003, 0.012),
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": ModelPricing("anthropic", "claude-3-5-sonnet-20241022", 0.003, 0.015),
        "claude-3-5-haiku-20241022": ModelPricing("anthropic", "claude-3-5-haiku-20241022", 0.001, 0.005),
        "claude-3-opus-20240229": ModelPricing("anthropic", "claude-3-opus-20240229", 0.015, 0.075),
    },
    "google": {
        "gemini-1.5-pro": ModelPricing("google", "gemini-1.5-pro", 0.0035, 0.0105),
        "gemini-1.5-flash": ModelPricing("google", "gemini-1.5-flash", 0.000075, 0.0003),
    },
    "openrouter": {
        # OpenRouter pricing varies by model
    },
    "nim": {
        # NVIDIA NIM pricing varies
    },
    "ollama": {
        # Local - free
        "llama3.2": ModelPricing("ollama", "llama3.2", 0.0, 0.0),
    },
}


@dataclass
class GameCost:
    """Cost breakdown for a single game."""
    game_id: str
    white_player: str
    black_player: str
    white_cost: float
    black_cost: float
    total_cost: float
    white_tokens_in: int
    white_tokens_out: int
    black_tokens_in: int
    black_tokens_out: int
    duration_sec: float
    cost_per_move: float


class CostTracker:
    """Track and analyze costs for benchmark runs."""

    def __init__(self, pricing: dict | None = None):
        self.pricing = pricing or DEFAULT_PRICING
        self.game_costs: list[GameCost] = []
        self.player_totals: dict[str, dict] = {}

    def get_pricing(self, provider: str, model: str) -> ModelPricing | None:
        """Get pricing for a model."""
        provider_pricing = self.pricing.get(provider, {})
        return provider_pricing.get(model)

    def add_game(
        self,
        game_id: str,
        white_player: str,
        black_player: str,
        white_provider: str,
        black_provider: str,
        white_model: str,
        black_model: str,
        white_tokens_in: int,
        white_tokens_out: int,
        black_tokens_in: int,
        black_tokens_out: int,
        duration_sec: float,
        total_moves: int
    ):
        """Add a game's cost data."""
        white_pricing = self.get_pricing(white_provider, white_model)
        black_pricing = self.get_pricing(black_provider, black_model)

        white_cost = white_pricing.calculate_cost(white_tokens_in, white_tokens_out) if white_pricing else 0.0
        black_cost = black_pricing.calculate_cost(black_tokens_in, black_tokens_out) if black_pricing else 0.0

        total_cost = white_cost + black_cost
        cost_per_move = total_cost / total_moves if total_moves > 0 else 0.0

        game_cost = GameCost(
            game_id=game_id,
            white_player=white_player,
            black_player=black_player,
            white_cost=white_cost,
            black_cost=black_cost,
            total_cost=total_cost,
            white_tokens_in=white_tokens_in,
            white_tokens_out=white_tokens_out,
            black_tokens_in=black_tokens_in,
            black_tokens_out=black_tokens_out,
            duration_sec=duration_sec,
            cost_per_move=cost_per_move
        )

        self.game_costs.append(game_cost)

        # Update player totals
        for player, cost, tokens_in, tokens_out in [
            (white_player, white_cost, white_tokens_in, white_tokens_out),
            (black_player, black_cost, black_tokens_in, black_tokens_out)
        ]:
            if player not in self.player_totals:
                self.player_totals[player] = {
                    "total_cost": 0.0,
                    "total_games": 0,
                    "total_tokens_in": 0,
                    "total_tokens_out": 0,
                    "total_moves": 0,
                    "total_duration": 0.0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                }

            self.player_totals[player]["total_cost"] += cost
            self.player_totals[player]["total_games"] += 1
            self.player_totals[player]["total_tokens_in"] += tokens_in
            self.player_totals[player]["total_tokens_out"] += tokens_out
            self.player_totals[player]["total_moves"] += total_moves // 2  # Approximate
            self.player_totals[player]["total_duration"] += duration_sec

    def update_result(self, player: str, result: str):
        """Update win/loss/draw for a player."""
        if player in self.player_totals:
            if result == "1-0":
                self.player_totals[player]["wins"] += 1
            elif result == "0-1":
                self.player_totals[player]["losses"] += 1
            else:
                self.player_totals[player]["draws"] += 1

    def get_summary(self) -> dict:
        """Get cost summary."""
        total_cost = sum(g.total_cost for g in self.game_costs)
        total_games = len(self.game_costs)
        total_moves = sum(g.white_tokens_in + g.white_tokens_out + g.black_tokens_in + g.black_tokens_out for g in self.game_costs)

        return {
            "total_cost": total_cost,
            "total_games": total_games,
            "total_moves": total_moves,
            "avg_cost_per_game": total_cost / total_games if total_games > 0 else 0,
            "avg_cost_per_move": total_cost / sum(g.white_tokens_in + g.white_tokens_out + g.black_tokens_in + g.black_tokens_out for g in self.game_costs) * 1000 if total_moves > 0 else 0,
            "cost_by_player": self.get_cost_by_player(),
            "cost_by_provider": self.get_cost_by_provider(),
        }

    def get_cost_by_player(self) -> dict[str, dict]:
        """Get cost breakdown by player."""
        result = {}
        for player, data in self.player_totals.items():
            cost_per_elo = None
            # Would need ELO data to calculate
            result[player] = {
                "total_cost": data["total_cost"],
                "total_games": data["total_games"],
                "total_tokens_in": data["total_tokens_in"],
                "total_tokens_out": data["total_tokens_out"],
                "avg_cost_per_game": data["total_cost"] / data["total_games"] if data["total_games"] > 0 else 0,
                "wins": data["wins"],
                "losses": data["losses"],
                "draws": data["draws"],
            }
        return result

    def get_cost_by_provider(self) -> dict[str, dict]:
        """Get cost breakdown by provider."""
        provider_costs = {}
        for game in self.game_costs:
            for player, provider, cost, tokens_in, tokens_out in [
                (game.white_player, game.white_player.split(':')[0] if ':' in game.white_player else "unknown", game.white_cost, game.white_tokens_in, game.white_tokens_out),
                (game.black_player, game.black_player.split(':')[0] if ':' in game.black_player else "unknown", game.black_cost, game.black_tokens_in, game.black_tokens_out)
            ]:
                if provider not in provider_costs:
                    provider_costs[provider] = {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "games": 0}
                provider_costs[provider]["cost"] += cost
                provider_costs[provider]["tokens_in"] += tokens_in
                provider_costs[provider]["tokens_out"] += tokens_out
                provider_costs[provider]["games"] += 1

        return provider_costs

    def export_csv(self, path: str):
        """Export cost data to CSV."""
        df = pd.DataFrame([{
            "game_id": g.game_id,
            "white_player": g.white_player,
            "black_player": g.black_player,
            "white_cost": g.white_cost,
            "black_cost": g.black_cost,
            "total_cost": g.total_cost,
            "white_tokens_in": g.white_tokens_in,
            "white_tokens_out": g.white_tokens_out,
            "black_tokens_in": g.black_tokens_in,
            "black_tokens_out": g.black_tokens_out,
            "duration_sec": g.duration_sec,
            "cost_per_move": g.cost_per_move,
        } for g in self.game_costs])

        df.to_csv(path, index=False)

    def generate_dashboard_html(self, output_path: str | None = None) -> str:
        """Generate interactive HTML dashboard."""
        if not self.game_costs:
            return "<p>No cost data available</p>"

        # Create plots
        summary = self.get_summary()
        player_costs = self.get_cost_by_player()
        provider_costs = self.get_cost_by_provider()

        # Cost per player
        players = list(player_costs.keys())
        costs = [player_costs[p]["total_cost"] for p in players]
        games = [player_costs[p]["total_games"] for p in players]

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=players,
            y=costs,
            name="Total Cost ($)",
            marker_color='#1f77b4',
            yaxis='y',
            text=[f"${c:.4f}" for c in costs],
            textposition='auto'
        ))
        fig1.add_trace(go.Bar(
            x=players,
            y=games,
            name="Games Played",
            marker_color='#ff7f0e',
            yaxis='y2',
            text=[str(g) for g in games],
            textposition='auto'
        ))
        fig1.update_layout(
            title="Cost & Games by Player",
            xaxis_title="Player",
            yaxis=dict(title="Cost ($)", side="left"),
            yaxis2=dict(title="Games", side="right", overlaying="y"),
            barmode='group',
            height=400,
            plot_bgcolor='white'
        )

        # Cost over time (games)
        fig2 = go.Figure()
        cumulative_cost = 0
        cum_costs = []
        game_labels = []
        for i, g in enumerate(self.game_costs):
            cumulative_cost += g.total_cost
            cum_costs.append(cumulative_cost)
            game_labels.append(f"Game {i+1}")

        fig2.add_trace(go.Scatter(
            x=game_labels,
            y=cum_costs,
            mode='lines+markers',
            name='Cumulative Cost',
            line=dict(color='#27ae60', width=2),
            marker=dict(size=6)
        ))
        fig2.update_layout(
            title="Cumulative Cost Over Games",
            xaxis_title="Game",
            yaxis_title="Cumulative Cost ($)",
            height=400,
            plot_bgcolor='white'
        )

        # Provider cost breakdown
        providers = list(provider_costs.keys())
        prov_costs = [provider_costs[p]["cost"] for p in providers]

        fig3 = px.pie(
            values=prov_costs,
            names=providers,
            title="Cost Distribution by Provider"
        )
        fig3.update_layout(height=400)

        # Cost per move
        moves = [g.white_tokens_in + g.white_tokens_out + g.black_tokens_in + g.black_tokens_out for g in self.game_costs]
        cost_per_move = [g.cost_per_move * 1000 for g in self.game_costs]  # per 1K tokens

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=moves,
            y=cost_per_move,
            mode='markers',
            marker=dict(size=10, color='#9b59b6'),
            name='Cost per 1K tokens',
            hovertemplate='Tokens: %{x}<br>Cost/1K: $%{y:.4f}<extra></extra>'
        ))
        fig4.update_layout(
            title="Cost Efficiency (Cost per 1K tokens vs Total Tokens)",
            xaxis_title="Total Tokens in Game",
            yaxis_title="Cost per 1K tokens ($)",
            height=400,
            plot_bgcolor='white'
        )

        # HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cost Tracking Dashboard - {datetime.utcnow().strftime('%Y-%m-%d')}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f6fa; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .header h1 {{ margin: 0; color: #2c3e50; }}
        .header .meta {{ color: #7f8c8d; margin-top: 10px; }}
        .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .card-value {{ font-size: 36px; font-weight: 700; color: #1f77b4; }}
        .card-label {{ color: #7f8c8d; margin-top: 5px; font-size: 14px; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .plot-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-card h3 {{ margin-top: 0; color: #2c3e50; }}
        .table-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .cost-positive {{ color: #e74c3c; }}
        .cost-negative {{ color: #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💰 Cost Tracking Dashboard</h1>
            <div class="meta">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
        </div>
        
        <div class="summary-cards">
            <div class="card">
                <div class="card-value">${summary['total_cost']:.4f}</div>
                <div class="card-label">Total Cost</div>
            </div>
            <div class="card">
                <div class="card-value">{summary['total_games']}</div>
                <div class="card-label">Total Games</div>
            </div>
            <div class="card">
                <div class="card-value">${summary['avg_cost_per_game']:.4f}</div>
                <div class="card-label">Avg Cost/Game</div>
            </div>
            <div class="card">
                <div class="card-value">${summary['avg_cost_per_move']:.6f}</div>
                <div class="card-label">Cost per 1K tokens</div>
            </div>
        </div>
        
        <div class="plot-grid">
            <div class="plot-card">
                <h3>Cost & Games by Player</h3>
                <div id="plot1">{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            </div>
            <div class="plot-card">
                <h3>Cumulative Cost</h3>
                <div id="plot2">{fig2.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            </div>
            <div class="plot-card">
                <h3>Cost by Provider</h3>
                <div id="plot3">{fig3.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            </div>
            <div class="plot-card">
                <h3>Cost Efficiency</h3>
                <div id="plot4">{fig4.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            </div>
        </div>
        
        <div class="table-container">
            <h3>Player Cost Details</h3>
            <table>
                <thead>
                    <tr>
                        <th>Player</th>
                        <th>Total Cost</th>
                        <th>Games</th>
                        <th>Tokens In</th>
                        <th>Tokens Out</th>
                        <th>Avg Cost/Game</th>
                        <th>Wins</th>
                        <th>Draws</th>
                        <th>Losses</th>
                    </tr>
                </thead>
                <tbody>
"""

        for player, data in player_costs.items():
            html += f"""
                <tr>
                    <td><strong>{player}</strong></td>
                    <td>${data['total_cost']:.4f}</td>
                    <td>{data['total_games']}</td>
                    <td>{data['total_tokens_in']:,}</td>
                    <td>{data['total_tokens_out']:,}</td>
                    <td>${data['avg_cost_per_game']:.4f}</td>
                    <td>{data['wins']}</td>
                    <td>{data['draws']}</td>
                    <td>{data['losses']}</td>
                </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
        
        <div class="table-container" style="margin-top: 20px;">
            <h3>Game-by-Game Costs</h3>
            <table>
                <thead>
                    <tr>
                        <th>Game ID</th>
                        <th>White</th>
                        <th>Black</th>
                        <th>White Cost</th>
                        <th>Black Cost</th>
                        <th>Total Cost</th>
                        <th>Cost/Move</th>
                        <th>Duration (s)</th>
                    </tr>
                </thead>
                <tbody>
"""

        for g in self.game_costs:
            html += f"""
                <tr>
                    <td>{g.game_id}</td>
                    <td>{g.white_player}</td>
                    <td>{g.black_player}</td>
                    <td>${g.white_cost:.6f}</td>
                    <td>${g.black_cost:.6f}</td>
                    <td>${g.total_cost:.6f}</td>
                    <td>${g.cost_per_move:.6f}</td>
                    <td>{g.duration_sec:.1f}</td>
                </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

        if output_path:
            Path(output_path).write_text(html)

        return html


def load_costs_from_run(run_dir: str) -> CostTracker:
    """Load cost data from a benchmark run directory."""
    tracker = CostTracker()

    games_path = Path(run_dir) / "games.jsonl"
    if not games_path.exists():
        return tracker

    with open(games_path) as f:
        for line in f:
            game = json.loads(line)

            # Extract token usage from moves
            # This would need move-level data from moves.jsonl
            moves_path = Path(run_dir) / "moves.jsonl"
            if moves_path.exists():
                moves_data = []
                with open(moves_path) as mf:
                    for mline in mf:
                        move = json.loads(mline)
                        if move['game_id'] == game['game_id']:
                            moves_data.append(move)

                white_tokens_in = sum(m.get('llm_tokens_in', 0) for m in moves_data if m['color'] == 'white')
                white_tokens_out = sum(m.get('llm_tokens_out', 0) for m in moves_data if m['color'] == 'white')
                black_tokens_in = sum(m.get('llm_tokens_in', 0) for m in moves_data if m['color'] == 'black')
                black_tokens_out = sum(m.get('llm_tokens_out', 0) for m in moves_data if m['color'] == 'black')
            else:
                white_tokens_in = white_tokens_out = black_tokens_in = black_tokens_out = 0

            tracker.add_game(
                game_id=game['game_id'],
                white_player=game['white_player'],
                black_player=game['black_player'],
                white_provider=game['white_provider'],
                black_provider=game['black_provider'],
                white_model=game['white_player'].split(':')[-1] if ':' in game['white_player'] else 'unknown',
                black_model=game['black_player'].split(':')[-1] if ':' in game['black_player'] else 'unknown',
                white_tokens_in=white_tokens_in,
                white_tokens_out=white_tokens_out,
                black_tokens_in=black_tokens_in,
                black_tokens_out=black_tokens_out,
                duration_sec=game['game_duration_sec'],
                total_moves=game['total_moves']
            )

            tracker.update_result(game['white_player'], game['result'])
            # For black player result
            black_result = "1-0" if game['result'] == "0-1" else "0-1" if game['result'] == "1-0" else "1/2-1/2"
            tracker.update_result(game['black_player'], black_result)

    return tracker


if __name__ == "__main__":
    # Demo
    tracker = CostTracker()

    # Add some sample games
    tracker.add_game("g1", "openai:gpt-4o", "anthropic:claude-3.5", "openai", "anthropic", "gpt-4o", "claude-3.5", 1000, 500, 800, 400, 30.0, 40)
    tracker.add_game("g2", "anthropic:claude-3.5", "openai:gpt-4o", "anthropic", "openai", "claude-3.5", "gpt-4o", 900, 450, 1100, 550, 35.0, 45)
    tracker.add_game("g3", "google:gemini-1.5", "openai:gpt-4o", "google", "openai", "gemini-1.5", "gpt-4o", 1200, 600, 1000, 500, 40.0, 50)

    tracker.update_result("openai:gpt-4o", "1-0")
    tracker.update_result("anthropic:claude-3.5", "0-1")
    tracker.update_result("anthropic:claude-3.5", "1-0")
    tracker.update_result("openai:gpt-4o", "0-1")
    tracker.update_result("google:gemini-1.5", "1/2-1/2")
    tracker.update_result("openai:gpt-4o", "1/2-1/2")

    html = tracker.generate_dashboard_html("cost_dashboard.html")
    print("Dashboard generated: cost_dashboard.html")
