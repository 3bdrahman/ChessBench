"""Report generation for benchmark runs."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from benchmark.elo import BayesianElo
from benchmark.logging import BenchmarkLogger


class ReportGenerator:
    """Generate HTML reports from benchmark runs."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.logger = BenchmarkLogger(str(self.run_dir))
        self._load_data()

    def _load_data(self):
        """Load all data from run directory."""
        # Load games
        self.games = []
        games_path = self.run_dir / "games.jsonl"
        if games_path.exists():
            with open(games_path) as f:
                for line in f:
                    self.games.append(json.loads(line))

        # Load moves
        self.moves = []
        moves_path = self.run_dir / "moves.jsonl"
        if moves_path.exists():
            with open(moves_path) as f:
                for line in f:
                    self.moves.append(json.loads(line))

        # Load summary
        summary_path = self.run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                self.summary = json.load(f)
        else:
            self.summary = {}

        # Load config
        self.config = self.summary.get('config', {})

    def generate_html(self, output_path: str | None = None) -> str:
        """Generate complete HTML report."""
        if output_path is None:
            output_path = self.run_dir / "report.html"

        html = self._generate_html_report()

        with open(output_path, 'w') as f:
            f.write(html)

        return str(output_path)

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        # Calculate statistics
        players = self._get_players()
        elo_ratings = self._calculate_elo_ratings()

        # Generate plots
        elo_plot = self._create_elo_plot(elo_ratings)
        results_heatmap = self._create_results_heatmap()
        move_time_plot = self._create_move_time_plot()
        token_usage_plot = self._create_token_usage_plot()
        opening_performance = self._create_opening_performance()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chess LLM Benchmark Report - {self.run_dir.name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #fafafa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 40px; }}
        h3 {{ color: #34495e; }}
        .meta {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 30px; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .meta-item {{ background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #1f77b4; }}
        .meta-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .meta-value {{ font-size: 18px; font-weight: 600; color: #1a1a2e; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #1f77b4; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .rating {{ font-family: monospace; font-weight: 600; }}
        .rating-high {{ color: #27ae60; }}
        .rating-mid {{ color: #f39c12; }}
        .rating-low {{ color: #e74c3c; }}
        .plot-container {{ margin: 30px 0; }}
        .summary-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: 700; color: #1f77b4; }}
        .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>♟️ Chess LLM Benchmark Report</h1>
        
        <div class="meta">
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">Run ID</div>
                    <div class="meta-value">{self.run_dir.name}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Date</div>
                    <div class="meta-value">{self.summary.get('timestamp_utc', 'Unknown')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Total Games</div>
                    <div class="meta-value">{self.summary.get('total_games', 0)}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Players</div>
                    <div class="meta-value">{len(players)}</div>
                </div>
            </div>
        </div>
        
        <h2>📊 Summary Statistics</h2>
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{self.summary.get('results', {}).get('white_wins', 0)}</div>
                <div class="stat-label">White Wins</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.summary.get('results', {}).get('black_wins', 0)}</div>
                <div class="stat-label">Black Wins</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.summary.get('results', {}).get('draws', 0)}</div>
                <div class="stat-label">Draws</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.summary.get('total_moves', 0)}</div>
                <div class="stat-label">Total Moves</div>
            </div>
        </div>
        
        <h2>🏆 ELO Ratings</h2>
        <div class="plot-container" id="elo-plot">{elo_plot}</div>
        
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Player</th>
                    <th>Rating</th>
                    <th>Deviation</th>
                    <th>95% CI</th>
                    <th>Volatility</th>
                </tr>
            </thead>
            <tbody>
"""

        for i, rating in enumerate(elo_ratings, 1):
            ci_low, ci_high = rating['ci_low'], rating['ci_high']
            rating_class = 'rating-high' if i <= 2 else 'rating-mid' if i <= len(elo_ratings)//2 else 'rating-low'
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{rating['name']}</strong></td>
                    <td class="rating {rating_class}">{rating['rating']}</td>
                    <td>±{rating['deviation']}</td>
                    <td>{ci_low} - {ci_high}</td>
                    <td>{rating['volatility']}</td>
                </tr>
"""

        html += f"""
            </tbody>
        </table>
        
        <h2>📈 Cross-Table (Results Heatmap)</h2>
        <div class="plot-container" id="results-heatmap">{results_heatmap}</div>
        
        <h2>⏱️ Move Time Distribution</h2>
        <div class="plot-container" id="move-time">{move_time_plot}</div>
        
        <h2>🔤 Token Usage</h2>
        <div class="plot-container" id="token-usage">{token_usage_plot}</div>
        
        <h2>📚 Opening Performance</h2>
        <div class="plot-container" id="opening-perf">{opening_performance}</div>
        
        <h2>🎮 Game Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Game ID</th>
                    <th>White</th>
                    <th>Black</th>
                    <th>Opening</th>
                    <th>Result</th>
                    <th>Moves</th>
                    <th>Duration (s)</th>
                </tr>
            </thead>
            <tbody>
"""

        for game in self.games:
            html += f"""
                <tr>
                    <td>{game['game_id']}</td>
                    <td>{game['white_player']}</td>
                    <td>{game['black_player']}</td>
                    <td>{game['opening_eco'] or '?'}: {game['opening_name'] or 'Unknown'}</td>
                    <td>{game['result']}</td>
                    <td>{game['total_moves']}</td>
                    <td>{game['game_duration_sec']:.1f}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
        
        <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 5px; text-align: center; color: #666;">
            Generated by Chess LLM Benchmark on """ + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + """ UTC
        </div>
    </div>
</body>
</html>
"""
        return html

    def _get_players(self) -> list[str]:
        """Get unique players from games."""
        players = set()
        for game in self.games:
            players.add(game['white_player'])
            players.add(game['black_player'])
        return sorted(players)

    def _calculate_elo_ratings(self) -> list[dict]:
        """Calculate ELO ratings from games."""
        elo = BayesianElo()
        for game in self.games:
            elo.add_game(game['white_player'], game['black_player'], game['result_numeric'], game['opening_eco'])
        return elo.leaderboard()

    def _create_elo_plot(self, ratings: list[dict]) -> str:
        """Create ELO rating plot."""
        if not ratings:
            return "<p>No rating data available</p>"

        names = [r['name'] for r in ratings]
        rating_vals = [r['rating'] for r in ratings]
        deviations = [r['deviation'] for r in ratings]
        ci_lows = [r['ci_low'] for r in ratings]
        ci_highs = [r['ci_high'] for r in ratings]

        fig = go.Figure()

        # Add error bars for 95% CI
        fig.add_trace(go.Scatter(
            x=names,
            y=rating_vals,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[h - r for r, h in zip(rating_vals, ci_highs)],
                arrayminus=[r - l for r, l in zip(rating_vals, ci_lows)],
                visible=True,
                color='rgba(31, 119, 180, 0.3)',
                thickness=2,
                width=10
            ),
            mode='markers',
            marker=dict(size=12, color='#1f77b4'),
            name='Rating ± 95% CI',
            hovertemplate='<b>%{x}</b><br>Rating: %{y:.0f}<br>95% CI: [%{customdata[0]:.0f}, %{customdata[1]:.0f}]<extra></extra>',
            customdata=list(zip(ci_lows, ci_highs))
        ))

        fig.update_layout(
            title='ELO Ratings with 95% Confidence Intervals',
            xaxis_title='Player',
            yaxis_title='Rating',
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(tickangle=-45)
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def _create_results_heatmap(self) -> str:
        """Create results cross-table heatmap."""
        if not self.games:
            return "<p>No game data available</p>"

        players = self._get_players()
        n = len(players)
        player_idx = {p: i for i, p in enumerate(players)}

        # Results matrix: rows=white, cols=black
        results = np.zeros((n, n, 3))  # [wins, draws, losses] from white perspective

        for game in self.games:
            w_idx = player_idx[game['white_player']]
            b_idx = player_idx[game['black_player']]
            if game['result'] == '1-0':
                results[w_idx, b_idx, 0] += 1
            elif game['result'] == '0-1':
                results[w_idx, b_idx, 2] += 1
            else:
                results[w_idx, b_idx, 1] += 1

        # Win rate from white perspective
        total = results.sum(axis=2, keepdims=True)
        win_rate = np.divide(results[:, :, 0], total[:, :, 0], where=total[:, :, 0] > 0)

        fig = go.Figure(data=go.Heatmap(
            z=win_rate,
            x=players,
            y=players,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            text=[[f"{win_rate[i,j]:.1%}" if total[i,j,0] > 0 else "—" for j in range(n)] for i in range(n)],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='White: %{y}<br>Black: %{x}<br>White Win Rate: %{z:.1%}<extra></extra>'
        ))

        fig.update_layout(
            title='White Win Rate by Pairing',
            height=400,
            xaxis_title='Black Player',
            yaxis_title='White Player',
            plot_bgcolor='white'
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def _create_move_time_plot(self) -> str:
        """Create move time distribution plot."""
        if not self.moves:
            return "<p>No move timing data available</p>"

        df = pd.DataFrame(self.moves)

        fig = px.box(
            df, x='player', y='llm_latency_ms',
            title='LLM Latency by Player',
            labels={'llm_latency_ms': 'Latency (ms)', 'player': 'Player'}
        )
        fig.update_layout(height=400, plot_bgcolor='white')

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def _create_token_usage_plot(self) -> str:
        """Create token usage plot."""
        if not self.moves:
            return "<p>No token usage data available</p>"

        df = pd.DataFrame(self.moves)
        df['tokens_total'] = df['llm_tokens_in'].fillna(0) + df['llm_tokens_out'].fillna(0)

        fig = px.bar(
            df.groupby('player')['tokens_total'].sum().reset_index(),
            x='player', y='tokens_total',
            title='Total Tokens Used by Player',
            labels={'tokens_total': 'Total Tokens', 'player': 'Player'}
        )
        fig.update_layout(height=400, plot_bgcolor='white')

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def _create_opening_performance(self) -> str:
        """Create opening performance plot."""
        if not self.games:
            return "<p>No opening data available</p>"

        # Win rate by opening
        opening_stats = {}
        for game in self.games:
            eco = game['opening_eco'] or '?'
            if eco not in opening_stats:
                opening_stats[eco] = {'white_wins': 0, 'black_wins': 0, 'draws': 0, 'total': 0}
            opening_stats[eco]['total'] += 1
            if game['result'] == '1-0':
                opening_stats[eco]['white_wins'] += 1
            elif game['result'] == '0-1':
                opening_stats[eco]['black_wins'] += 1
            else:
                opening_stats[eco]['draws'] += 1

        ecos = list(opening_stats.keys())
        white_rates = [opening_stats[e]['white_wins']/opening_stats[e]['total'] for e in ecos]
        black_rates = [opening_stats[e]['black_wins']/opening_stats[e]['total'] for e in ecos]
        draw_rates = [opening_stats[e]['draws']/opening_stats[e]['total'] for e in ecos]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='White Win', x=ecos, y=white_rates, marker_color='#27ae60'))
        fig.add_trace(go.Bar(name='Draw', x=ecos, y=draw_rates, marker_color='#f39c12'))
        fig.add_trace(go.Bar(name='Black Win', x=ecos, y=black_rates, marker_color='#e74c3c'))

        fig.update_layout(
            title='Result Distribution by Opening (ECO)',
            barmode='stack',
            height=400,
            xaxis_title='ECO Code',
            yaxis_title='Proportion',
            plot_bgcolor='white'
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Find latest run
        runs_dir = Path("runs")
        if runs_dir.exists():
            run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
            if run_dirs:
                run_dir = run_dirs[-1]
            else:
                print("No runs found")
                sys.exit(1)
        else:
            print("No runs directory")
            sys.exit(1)

    generator = ReportGenerator(run_dir)
    output = generator.generate_html()
    print(f"Report generated: {output}")
