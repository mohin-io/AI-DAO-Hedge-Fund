"""
Automated Report Generation System
Professional PDF and HTML reports for trading performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import json
import logging
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Automated trading performance report generator
    Creates professional HTML and PDF reports
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def generate_daily_report(
        self,
        date: str,
        portfolio_data: Dict,
        trades: List[Dict],
        performance_metrics: Dict,
        agent_performance: Dict
    ) -> str:
        """
        Generate daily trading report

        Args:
            date: Report date
            portfolio_data: Portfolio snapshot
            trades: List of trades executed
            performance_metrics: Performance metrics
            agent_performance: AI agent performance data

        Returns:
            Path to generated HTML report
        """
        logger.info(f"Generating daily report for {date}")

        # Create visualizations
        equity_chart = self._create_equity_chart(portfolio_data['equity_history'])
        pnl_chart = self._create_pnl_distribution(trades)
        agent_chart = self._create_agent_performance_chart(agent_performance)

        # Prepare template data
        template_data = {
            'date': date,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio': portfolio_data,
            'trades': trades,
            'metrics': performance_metrics,
            'agents': agent_performance,
            'equity_chart': equity_chart,
            'pnl_chart': pnl_chart,
            'agent_chart': agent_chart,
        }

        # Render HTML report
        html_content = self._render_daily_template(template_data)

        # Save report
        report_filename = f"daily_report_{date}.html"
        report_path = self.output_dir / report_filename

        with open(report_path, 'w') as f:
            f.write(html_content)

        logger.info(f"✓ Daily report saved: {report_path}")
        return str(report_path)

    def generate_monthly_report(
        self,
        month: str,
        portfolio_summary: Dict,
        all_trades: List[Dict],
        monthly_metrics: Dict
    ) -> str:
        """
        Generate monthly performance report

        Args:
            month: Report month (YYYY-MM)
            portfolio_summary: Monthly portfolio summary
            all_trades: All trades for the month
            monthly_metrics: Monthly performance metrics

        Returns:
            Path to generated HTML report
        """
        logger.info(f"Generating monthly report for {month}")

        # Create comprehensive visualizations
        monthly_equity = self._create_monthly_equity_chart(portfolio_summary)
        returns_dist = self._create_returns_distribution(monthly_metrics['daily_returns'])
        drawdown_chart = self._create_drawdown_chart(portfolio_summary)
        strategy_comparison = self._create_strategy_comparison(all_trades)

        template_data = {
            'month': month,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': portfolio_summary,
            'trades': all_trades,
            'metrics': monthly_metrics,
            'monthly_equity': monthly_equity,
            'returns_dist': returns_dist,
            'drawdown_chart': drawdown_chart,
            'strategy_comparison': strategy_comparison,
        }

        html_content = self._render_monthly_template(template_data)

        report_filename = f"monthly_report_{month}.html"
        report_path = self.output_dir / report_filename

        with open(report_path, 'w') as f:
            f.write(html_content)

        logger.info(f"✓ Monthly report saved: {report_path}")
        return str(report_path)

    def _create_equity_chart(self, equity_history: List[float]) -> str:
        """Create equity curve chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(equity_history, linewidth=2, color='#1976D2')
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        return self._fig_to_base64(fig)

    def _create_pnl_distribution(self, trades: List[Dict]) -> str:
        """Create P&L distribution histogram"""
        if not trades:
            return ""

        pnls = [t.get('pnl', 0) for t in trades]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(pnls, bins=30, color='#4CAF50', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def _create_agent_performance_chart(self, agent_data: Dict) -> str:
        """Create agent performance comparison chart"""
        if not agent_data:
            return ""

        agents = list(agent_data.keys())
        performance = [agent_data[a].get('performance_score', 0) for a in agents]

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ['#1976D2', '#FF9800', '#9C27B0']
        ax.barh(agents, performance, color=colors[:len(agents)])
        ax.set_title('AI Agent Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Performance Score')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')

        return self._fig_to_base64(fig)

    def _create_monthly_equity_chart(self, portfolio_summary: Dict) -> str:
        """Create monthly equity chart with daily data points"""
        equity_data = portfolio_summary.get('daily_equity', [])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(equity_data, linewidth=2, color='#1976D2', marker='o', markersize=3)
        ax.fill_between(range(len(equity_data)), equity_data, alpha=0.3, color='#64B5F6')
        ax.set_title('Monthly Equity Progression', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        return self._fig_to_base64(fig)

    def _create_returns_distribution(self, daily_returns: List[float]) -> str:
        """Create returns distribution with normal curve"""
        if not daily_returns:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(daily_returns, bins=50, density=True, alpha=0.7, color='#4CAF50', edgecolor='black')

        # Overlay normal distribution
        mu = np.mean(daily_returns)
        sigma = np.std(daily_returns)
        x = np.linspace(min(daily_returns), max(daily_returns), 100)
        ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)),
                linewidth=2, color='red', label='Normal Distribution')

        ax.axvline(x=mu, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mu:.4f}')
        ax.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def _create_drawdown_chart(self, portfolio_summary: Dict) -> str:
        """Create drawdown chart"""
        equity = portfolio_summary.get('daily_equity', [])
        if not equity:
            return ""

        equity_series = pd.Series(equity)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.fill_between(range(len(drawdown)), drawdown * 100, 0, alpha=0.7, color='#F44336')
        ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def _create_strategy_comparison(self, trades: List[Dict]) -> str:
        """Create strategy performance comparison"""
        if not trades:
            return ""

        # Group by strategy
        strategy_pnl = {}
        for trade in trades:
            strategy = trade.get('strategy', 'Unknown')
            pnl = trade.get('pnl', 0)
            strategy_pnl[strategy] = strategy_pnl.get(strategy, 0) + pnl

        strategies = list(strategy_pnl.keys())
        pnls = list(strategy_pnl.values())

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ['#4CAF50' if p > 0 else '#F44336' for p in pnls]
        ax.bar(strategies, pnls, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total P&L ($)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

    def _render_daily_template(self, data: Dict) -> str:
        """Render daily report HTML template"""
        template = """
<!DOCTYPE html>
<html>
<head>
    <title>Daily Trading Report - {{ date }}</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #1976D2; border-bottom: 3px solid #1976D2; padding-bottom: 10px; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
        .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-label { font-size: 12px; color: #666; margin-bottom: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #1976D2; }
        .section { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .positive { color: #4CAF50; }
        .negative { color: #F44336; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { background: #f5f5f5; font-weight: 600; }
        img { max-width: 100%; height: auto; }
        .footer { text-align: center; color: #666; margin-top: 40px; padding: 20px; border-top: 1px solid #e0e0e0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Daily Trading Report</h1>
        <p><strong>Date:</strong> {{ date }}</p>
        <p><strong>Generated:</strong> {{ generated_at }}</p>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-label">Portfolio Value</div>
            <div class="metric-value">${{ "%.2f"|format(portfolio.equity) }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Daily P&L</div>
            <div class="metric-value {% if portfolio.daily_pnl >= 0 %}positive{% else %}negative{% endif %}">
                {{ "+" if portfolio.daily_pnl >= 0 else "" }}${{ "%.2f"|format(portfolio.daily_pnl) }}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{{ trades|length }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{{ "%.1f"|format(metrics.win_rate * 100) }}%</div>
        </div>
    </div>

    <div class="section">
        <h2>Equity Curve</h2>
        <img src="{{ equity_chart }}" alt="Equity Curve">
    </div>

    <div class="section">
        <h2>Trades Executed</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Quantity</th>
                    <th>Price</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody>
                {% for trade in trades %}
                <tr>
                    <td>{{ trade.timestamp }}</td>
                    <td>{{ trade.symbol }}</td>
                    <td>{{ trade.side }}</td>
                    <td>{{ trade.quantity }}</td>
                    <td>${{ "%.2f"|format(trade.price) }}</td>
                    <td class="{% if trade.pnl >= 0 %}positive{% else %}negative{% endif %}">
                        {{ "+" if trade.pnl >= 0 else "" }}${{ "%.2f"|format(trade.pnl) }}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>AI Agent Performance</h2>
        <img src="{{ agent_chart }}" alt="Agent Performance">
    </div>

    <div class="footer">
        <p>🤖 Generated by AI DAO Hedge Fund Automated Reporting System</p>
        <p>For internal use only. Confidential.</p>
    </div>
</body>
</html>
        """

        return Template(template).render(**data)

    def _render_monthly_template(self, data: Dict) -> str:
        """Render monthly report HTML template"""
        # Similar structure to daily template but with more comprehensive sections
        template = """
<!DOCTYPE html>
<html>
<head>
    <title>Monthly Report - {{ month }}</title>
    <style>
        /* Same styles as daily template */
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #1976D2; }
        .section { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>📈 Monthly Trading Report - {{ month }}</h1>
    <div class="section">
        <h2>Monthly Performance</h2>
        <img src="{{ monthly_equity }}" alt="Monthly Equity">
    </div>
    <div class="section">
        <h2>Returns Distribution</h2>
        <img src="{{ returns_dist }}" alt="Returns Distribution">
    </div>
    <div class="section">
        <h2>Drawdown Analysis</h2>
        <img src="{{ drawdown_chart }}" alt="Drawdown">
    </div>
</body>
</html>
        """

        return Template(template).render(**data)


if __name__ == "__main__":
    # Example usage
    generator = ReportGenerator()

    # Mock data
    portfolio_data = {
        'equity': 105000,
        'daily_pnl': 2500,
        'equity_history': [100000 + i * 250 for i in range(20)]
    }

    trades = [
        {'timestamp': '2025-10-04 09:30', 'symbol': 'AAPL', 'side': 'BUY', 'quantity': 10, 'price': 180.50, 'pnl': 150},
        {'timestamp': '2025-10-04 10:15', 'symbol': 'GOOGL', 'side': 'SELL', 'quantity': 5, 'price': 142.30, 'pnl': -50},
    ]

    metrics = {'win_rate': 0.65}
    agents = {'Momentum': {'performance_score': 0.85}, 'Arbitrage': {'performance_score': 0.72}}

    report_path = generator.generate_daily_report('2025-10-04', portfolio_data, trades, metrics, agents)
    print(f"\n✓ Report generated: {report_path}")
