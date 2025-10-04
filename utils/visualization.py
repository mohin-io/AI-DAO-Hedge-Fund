"""
Visualization Utilities for AI DAO Hedge Fund
Creates plots and charts for performance analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class PerformanceVisualizer:
    """Creates comprehensive performance visualizations"""

    def __init__(self, output_dir: str = "simulations/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cumulative_returns(
        self,
        portfolio_values: pd.DataFrame,
        benchmark_values: Optional[pd.Series] = None,
        title: str = "Cumulative Returns",
        save_name: str = "cumulative_returns.png"
    ):
        """
        Plot cumulative returns for portfolio and benchmark

        Args:
            portfolio_values: DataFrame with portfolio values (can have multiple agents)
            benchmark_values: Benchmark returns (e.g., S&P 500)
            title: Plot title
            save_name: Filename to save
        """
        fig = go.Figure()

        # Plot portfolio(s)
        if isinstance(portfolio_values, pd.DataFrame):
            for col in portfolio_values.columns:
                returns = (portfolio_values[col] / portfolio_values[col].iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(
                    x=portfolio_values.index,
                    y=returns,
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                ))
        else:
            returns = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=portfolio_values.index,
                y=returns,
                mode='lines',
                name='Portfolio',
                line=dict(width=2, color='blue')
            ))

        # Plot benchmark
        if benchmark_values is not None:
            bench_returns = (benchmark_values / benchmark_values.iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=benchmark_values.index,
                y=bench_returns,
                mode='lines',
                name='Benchmark (S&P 500)',
                line=dict(width=2, color='gray', dash='dash')
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        # Save
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path.with_suffix('.html')))
        fig.write_image(str(save_path), width=1200, height=500)

        logger.info(f"Cumulative returns plot saved to {save_path}")

    def plot_drawdown(
        self,
        portfolio_values: pd.Series,
        title: str = "Drawdown Analysis",
        save_name: str = "drawdown.png"
    ):
        """Plot drawdown over time"""
        # Calculate drawdown
        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=drawdown,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        # Save
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path.with_suffix('.html')))
        fig.write_image(str(save_path), width=1200, height=400)

        logger.info(f"Drawdown plot saved to {save_path}")

    def plot_agent_comparison(
        self,
        agent_metrics: Dict[str, Dict],
        save_name: str = "agent_comparison.png"
    ):
        """
        Compare performance across agents

        Args:
            agent_metrics: Dict of agent_name -> metrics dict
        """
        agents = list(agent_metrics.keys())
        metrics_to_plot = ['sharpe_ratio', 'total_return', 'win_rate', 'max_drawdown']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sharpe Ratio', 'Total Return (%)',
                          'Win Rate (%)', 'Max Drawdown (%)')
        )

        # Sharpe Ratio
        sharpe_values = [agent_metrics[a].get('sharpe_ratio', 0) for a in agents]
        fig.add_trace(
            go.Bar(x=agents, y=sharpe_values, name='Sharpe', marker_color='blue'),
            row=1, col=1
        )

        # Total Return
        return_values = [agent_metrics[a].get('total_return', 0) * 100 for a in agents]
        fig.add_trace(
            go.Bar(x=agents, y=return_values, name='Return', marker_color='green'),
            row=1, col=2
        )

        # Win Rate
        winrate_values = [agent_metrics[a].get('win_rate', 0) * 100 for a in agents]
        fig.add_trace(
            go.Bar(x=agents, y=winrate_values, name='Win Rate', marker_color='orange'),
            row=2, col=1
        )

        # Max Drawdown
        dd_values = [agent_metrics[a].get('max_drawdown', 0) * 100 for a in agents]
        fig.add_trace(
            go.Bar(x=agents, y=dd_values, name='Drawdown', marker_color='red'),
            row=2, col=2
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Agent Performance Comparison",
            template='plotly_white'
        )

        # Save
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path.with_suffix('.html')))
        fig.write_image(str(save_path), width=1200, height=600)

        logger.info(f"Agent comparison plot saved to {save_path}")

    def plot_agent_allocation_over_time(
        self,
        allocation_history: pd.DataFrame,
        save_name: str = "agent_allocation.png"
    ):
        """
        Plot how agent allocations change over time

        Args:
            allocation_history: DataFrame with columns for each agent's weight over time
        """
        fig = go.Figure()

        for col in allocation_history.columns:
            fig.add_trace(go.Scatter(
                x=allocation_history.index,
                y=allocation_history[col] * 100,
                mode='lines',
                name=col,
                stackgroup='one'  # For stacked area chart
            ))

        fig.update_layout(
            title="Agent Allocation Over Time",
            xaxis_title="Date/Step",
            yaxis_title="Allocation (%)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        # Save
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path.with_suffix('.html')))
        fig.write_image(str(save_path), width=1200, height=500)

        logger.info(f"Agent allocation plot saved to {save_path}")

    def plot_risk_metrics(
        self,
        returns: pd.Series,
        save_name: str = "risk_metrics.png"
    ):
        """Plot risk-related metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Returns distribution
        axes[0, 0].hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Returns Distribution')
        axes[0, 0].set_xlabel('Daily Return')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)

        # Rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
        axes[0, 1].plot(rolling_vol.index, rolling_vol, color='orange')
        axes[0, 1].set_title('Rolling 20-Day Volatility (Annualized)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Volatility (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot for normality check
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)

        # Value at Risk
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)

        axes[1, 1].hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].axvline(var_95, color='orange', linestyle='--',
                          linewidth=2, label=f'VaR 95%: {var_95:.2%}')
        axes[1, 1].axvline(var_99, color='red', linestyle='--',
                          linewidth=2, label=f'VaR 99%: {var_99:.2%}')
        axes[1, 1].set_title('Value at Risk (VaR)')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()

        plt.tight_layout()

        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Risk metrics plot saved to {save_path}")

    def create_dashboard_summary(
        self,
        metrics: Dict,
        portfolio_values: pd.Series,
        save_name: str = "dashboard_summary.png"
    ):
        """Create a comprehensive dashboard summary"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Portfolio Value',
                'Key Metrics',
                'Monthly Returns',
                'Drawdown',
                'Return Distribution',
                'Rolling Sharpe'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "indicator"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "histogram"}, {"type": "scatter"}]
            ]
        )

        # 1. Portfolio Value
        fig.add_trace(
            go.Scatter(x=portfolio_values.index, y=portfolio_values,
                      mode='lines', name='Portfolio Value', line=dict(color='blue')),
            row=1, col=1
        )

        # 2. Key Metrics (using indicator)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.get('total_return', 0) * 100,
                title={"text": "Total Return (%)"},
                delta={'reference': 0},
            ),
            row=1, col=2
        )

        # 3. Monthly Returns (example - simplified)
        returns = portfolio_values.pct_change().dropna()
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        fig.add_trace(
            go.Bar(x=monthly_returns.index, y=monthly_returns,
                  name='Monthly Return', marker_color='green'),
            row=1, col=3
        )

        # 4. Drawdown
        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax * 100
        fig.add_trace(
            go.Scatter(x=portfolio_values.index, y=drawdown, fill='tozeroy',
                      name='Drawdown', line=dict(color='red')),
            row=2, col=1
        )

        # 5. Return Distribution
        fig.add_trace(
            go.Histogram(x=returns * 100, name='Returns', marker_color='blue'),
            row=2, col=2
        )

        # 6. Rolling Sharpe
        rolling_sharpe = returns.rolling(window=60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        fig.add_trace(
            go.Scatter(x=portfolio_values.index, y=rolling_sharpe,
                      mode='lines', name='Rolling Sharpe', line=dict(color='purple')),
            row=2, col=3
        )

        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="AI DAO Hedge Fund - Performance Dashboard",
            template='plotly_white'
        )

        # Save
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path.with_suffix('.html')))
        fig.write_image(str(save_path), width=1600, height=800)

        logger.info(f"Dashboard summary saved to {save_path}")


def create_architecture_diagram_description():
    """
    Description for creating architecture diagram
    (Actual diagram would be created using draw.io or similar tool)
    """
    description = """
AI DAO Hedge Fund - System Architecture

┌─────────────────────────────────────────────────────────────┐
│                     DAO Governance Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ DAOGovernance│  │TreasuryManager│  │ AgentRegistry  │  │
│  │   Contract   │  │   Contract    │  │   Contract     │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Multi-Agent Coordinator                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  - Weighted Voting                                    │  │
│  │  - Market Regime Detection                            │  │
│  │  - Dynamic Allocation                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Momentum   │   │  Arbitrage   │   │   Hedging    │
│    Agent     │   │    Agent     │   │    Agent     │
│    (PPO)     │   │    (DQN)     │   │    (SAC)     │
└──────────────┘   └──────────────┘   └──────────────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Explainability Layer                            │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │SHAP Analyzer│  │  Attention   │  │  Risk Explainer   │ │
│  │             │  │  Visualizer  │  │                    │ │
│  └────────────┘  └──────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Dashboard & API                             │
│  ┌──────────────┐                   ┌──────────────┐       │
│  │   FastAPI    │                   │   React UI   │       │
│  │   Backend    │◄─────────────────►│   Dashboard  │       │
│  └──────────────┘                   └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
"""
    return description


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Visualization utilities loaded successfully")
    print("\nArchitecture:")
    print(create_architecture_diagram_description())
