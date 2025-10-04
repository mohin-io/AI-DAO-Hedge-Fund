"""
Export Utilities for AI DAO Hedge Fund
Handles PDF report generation, CSV exports, and file downloads
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO, StringIO
import base64


def generate_portfolio_report_html(portfolio_data, metrics_data, trades_data):
    """
    Generate comprehensive HTML portfolio report

    Args:
        portfolio_data: Dict with portfolio information
        metrics_data: Dict with performance metrics
        trades_data: DataFrame with trade history

    Returns:
        str: HTML content for the report
    """

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI DAO Hedge Fund - Portfolio Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

            * {{
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            }}

            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }}

            h1 {{
                font-size: 2.5rem;
                font-weight: 900;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }}

            .subtitle {{
                font-size: 1.1rem;
                color: #666;
                margin-bottom: 5px;
            }}

            .timestamp {{
                font-size: 0.9rem;
                color: #999;
            }}

            .section {{
                margin: 30px 0;
            }}

            .section-title {{
                font-size: 1.5rem;
                font-weight: 700;
                color: #333;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
            }}

            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}

            .metric-card {{
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 12px;
                padding: 20px;
                border-left: 4px solid #667eea;
            }}

            .metric-label {{
                font-size: 0.9rem;
                color: #666;
                margin-bottom: 5px;
            }}

            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
                color: #333;
            }}

            .metric-delta {{
                font-size: 0.9rem;
                margin-top: 5px;
            }}

            .positive {{
                color: #00cc00;
            }}

            .negative {{
                color: #ff4444;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}

            th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}

            td {{
                padding: 12px;
                border-bottom: 1px solid #eee;
            }}

            tr:hover {{
                background: rgba(102, 126, 234, 0.05);
            }}

            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #eee;
                text-align: center;
                color: #999;
                font-size: 0.9rem;
            }}

            .disclaimer {{
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
                font-size: 0.85rem;
                color: #856404;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖⛓️📈 AI DAO Hedge Fund</h1>
                <div class="subtitle">Decentralized Autonomous Hedge Fund - Portfolio Report</div>
                <div class="timestamp">Generated: {current_time}</div>
            </div>

            <div class="section">
                <div class="section-title">📊 Portfolio Overview</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Portfolio Value</div>
                        <div class="metric-value">${portfolio_data['value']:,.2f}</div>
                        <div class="metric-delta positive">+${portfolio_data['daily_pnl']:,.2f} (+{portfolio_data['daily_pnl_pct']:.2f}%)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value">{portfolio_data['total_return']:.2f}%</div>
                        <div class="metric-delta">Since Inception</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{metrics_data['sharpe']:.2f}</div>
                        <div class="metric-delta">Institutional Grade</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value">{metrics_data['max_drawdown']:.2f}%</div>
                        <div class="metric-delta">Low Risk Profile</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">🎯 Performance Metrics</div>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td>{metrics_data['win_rate']:.1f}%</td>
                        <td><span class="positive">✓ Excellent</span></td>
                    </tr>
                    <tr>
                        <td>Daily Volatility</td>
                        <td>{metrics_data['daily_volatility']:.2f}%</td>
                        <td><span class="positive">✓ Within Limits</span></td>
                    </tr>
                    <tr>
                        <td>Annual Volatility</td>
                        <td>{metrics_data['annual_volatility']:.2f}%</td>
                        <td><span class="positive">✓ Moderate</span></td>
                    </tr>
                    <tr>
                        <td>Value at Risk (95%)</td>
                        <td>{metrics_data['var_95']:.2f}%</td>
                        <td><span class="positive">✓ Acceptable</span></td>
                    </tr>
                    <tr>
                        <td>Market Beta</td>
                        <td>{metrics_data['beta']:.2f}</td>
                        <td><span class="positive">✓ Well Diversified</span></td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <div class="section-title">🤖 AI Agent Performance</div>
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Strategy</th>
                        <th>P&L</th>
                        <th>Win Rate</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td><strong>Momentum (PPO)</strong></td>
                        <td>Trend Following</td>
                        <td class="positive">+$42,567</td>
                        <td>71.2%</td>
                        <td><span class="positive">✓ Active</span></td>
                    </tr>
                    <tr>
                        <td><strong>Arbitrage (DQN)</strong></td>
                        <td>Mean Reversion</td>
                        <td class="positive">+$28,934</td>
                        <td>65.8%</td>
                        <td><span class="positive">✓ Active</span></td>
                    </tr>
                    <tr>
                        <td><strong>Hedging (SAC)</strong></td>
                        <td>Risk Protection</td>
                        <td class="positive">+$15,890</td>
                        <td>58.3%</td>
                        <td><span class="positive">✓ Active</span></td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <div class="section-title">📋 Recent Trades (Last 10)</div>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Agent</th>
                        <th>Action</th>
                        <th>Asset</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>P&L</th>
                        <th>Confidence</th>
                    </tr>
                    {generate_trades_table_rows(trades_data)}
                </table>
            </div>

            <div class="disclaimer">
                <strong>⚠️ Disclaimer:</strong> This is a live demonstration of the AI DAO Hedge Fund system.
                Performance metrics shown are based on historical backtesting with simulated data.
                Past performance does not guarantee future results.
                All investment decisions should be made in accordance with your risk tolerance and investment objectives.
            </div>

            <div class="footer">
                <div><strong>AI DAO Hedge Fund</strong> - Decentralized Autonomous Hedge Fund</div>
                <div>Powered by Multi-Agent Reinforcement Learning & Blockchain DAO</div>
                <div style="margin-top: 10px;">
                    <a href="https://github.com/mohin-io/AI-DAO-Hedge-Fund" style="color: #667eea; text-decoration: none;">
                        GitHub Repository
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


def generate_trades_table_rows(trades_data):
    """Generate HTML table rows for trades"""
    rows = ""
    for _, trade in trades_data.head(10).iterrows():
        pnl_class = "positive" if '+' in str(trade['P&L']) else "negative"
        rows += f"""
        <tr>
            <td>{trade['Time']}</td>
            <td>{trade['Agent']}</td>
            <td><strong>{trade['Action']}</strong></td>
            <td>{trade['Asset']}</td>
            <td>{trade['Quantity']}</td>
            <td>{trade['Price']}</td>
            <td class="{pnl_class}">{trade['P&L']}</td>
            <td>{trade['Confidence']}</td>
        </tr>
        """
    return rows


def generate_metrics_csv(portfolio_data, metrics_data, agent_performance):
    """
    Generate CSV export of all metrics

    Args:
        portfolio_data: Dict with portfolio information
        metrics_data: Dict with performance metrics
        agent_performance: Dict with agent-level data

    Returns:
        str: CSV content
    """

    # Create comprehensive metrics DataFrame
    metrics_df = pd.DataFrame([
        ["Portfolio Metrics", "", ""],
        ["Portfolio Value", f"${portfolio_data['value']:,.2f}", ""],
        ["Daily P&L", f"${portfolio_data['daily_pnl']:,.2f}", f"{portfolio_data['daily_pnl_pct']:.2f}%"],
        ["Total Return", f"{portfolio_data['total_return']:.2f}%", "Since Inception"],
        ["", "", ""],
        ["Performance Metrics", "", ""],
        ["Sharpe Ratio", f"{metrics_data['sharpe']:.2f}", "Institutional Grade"],
        ["Max Drawdown", f"{metrics_data['max_drawdown']:.2f}%", "Low Risk"],
        ["Win Rate", f"{metrics_data['win_rate']:.1f}%", "Excellent"],
        ["Daily Volatility", f"{metrics_data['daily_volatility']:.2f}%", ""],
        ["Annual Volatility", f"{metrics_data['annual_volatility']:.2f}%", ""],
        ["Value at Risk (95%)", f"{metrics_data['var_95']:.2f}%", ""],
        ["Market Beta", f"{metrics_data['beta']:.2f}", ""],
        ["", "", ""],
        ["Agent Performance", "P&L", "Win Rate"],
        ["Momentum (PPO)", "$42,567", "71.2%"],
        ["Arbitrage (DQN)", "$28,934", "65.8%"],
        ["Hedging (SAC)", "$15,890", "58.3%"],
        ["", "", ""],
        ["Export Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""],
    ], columns=["Metric", "Value", "Note"])

    # Convert to CSV
    csv_buffer = StringIO()
    metrics_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def generate_trades_csv(trades_data):
    """
    Generate CSV export of trade log

    Args:
        trades_data: DataFrame with trade history

    Returns:
        str: CSV content
    """

    # Add metadata header
    csv_buffer = StringIO()
    csv_buffer.write(f"# AI DAO Hedge Fund - Trade Log Export\n")
    csv_buffer.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    csv_buffer.write(f"# Total Trades: {len(trades_data)}\n")
    csv_buffer.write("\n")

    # Export trades
    trades_data.to_csv(csv_buffer, index=False)

    return csv_buffer.getvalue()


def create_download_link(content, filename, file_type="text/html"):
    """
    Create a download link for content

    Args:
        content: String or bytes content to download
        filename: Name of the file
        file_type: MIME type

    Returns:
        str: HTML download link
    """

    if isinstance(content, str):
        content = content.encode()

    b64 = base64.b64encode(content).decode()

    return f'<a href="data:{file_type};base64,{b64}" download="{filename}" style="text-decoration: none; color: white; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px 24px; border-radius: 8px; font-weight: 600; display: inline-block; margin: 10px 0;">📥 Download {filename}</a>'


def get_sample_data():
    """
    Generate sample data for demonstration
    Returns portfolio data, metrics, and trades
    """

    # Portfolio data
    portfolio_data = {
        'value': 1247893.45,
        'daily_pnl': 8234.56,
        'daily_pnl_pct': 0.66,
        'total_return': 34.2
    }

    # Performance metrics
    metrics_data = {
        'sharpe': 2.14,
        'max_drawdown': -12.3,
        'win_rate': 67.8,
        'daily_volatility': 1.8,
        'annual_volatility': 18.3,
        'var_95': -2.1,
        'beta': 0.87
    }

    # Agent performance
    agent_performance = {
        'momentum': {'pnl': 42567, 'win_rate': 71.2},
        'arbitrage': {'pnl': 28934, 'win_rate': 65.8},
        'hedging': {'pnl': 15890, 'win_rate': 58.3}
    }

    # Recent trades
    trades_data = pd.DataFrame({
        'Time': [
            (datetime.now() - timedelta(minutes=5)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(minutes=15)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(minutes=32)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(hours=1, minutes=12)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(hours=2, minutes=5)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(hours=3, minutes=22)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(hours=4, minutes=45)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(hours=5, minutes=10)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(hours=6, minutes=33)).strftime('%H:%M:%S'),
            (datetime.now() - timedelta(hours=7, minutes=8)).strftime('%H:%M:%S'),
        ],
        'Agent': ['Momentum', 'Arbitrage', 'Hedging', 'Momentum', 'Arbitrage', 'Hedging', 'Momentum', 'Arbitrage', 'Momentum', 'Hedging'],
        'Action': ['BUY', 'LONG/SHORT', 'BUY', 'SELL', 'CLOSE', 'BUY', 'SELL', 'CLOSE', 'BUY', 'SELL'],
        'Asset': ['AAPL', 'MSFT/GOOGL', 'SPY PUT', 'TSLA', 'BTC-USD', 'QQQ PUT', 'NVDA', 'ETH-USD', 'META', 'VIX CALL'],
        'Quantity': ['100', '50/50', '10 contracts', '75', '0.5 BTC', '15 contracts', '50', '2 ETH', '80', '20 contracts'],
        'Price': ['$182.45', 'Spread: 1.2%', '$420.50', '$245.80', '$43,256', '$385.20', '$512.30', '$2,845', '$385.60', '$18.50'],
        'P&L': ['+$1,234', '+$890', '-$156', '+$2,145', '+$567', '-$234', '+$1,890', '+$345', '+$1,123', '+$456'],
        'Confidence': ['87%', '72%', '91%', '83%', '68%', '88%', '75%', '70%', '82%', '79%']
    })

    return portfolio_data, metrics_data, agent_performance, trades_data
