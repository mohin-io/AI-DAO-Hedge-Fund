"""
Base Agent Class for RL Trading Agents
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents

    Each agent implements:
    - Training logic
    - Prediction/action selection
    - Performance tracking
    - Save/load functionality
    """

    def __init__(
        self,
        name: str,
        strategy: str,
        agent_id: Optional[int] = None
    ):
        self.name = name
        self.strategy = strategy
        self.agent_id = agent_id

        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }

        # Trade history
        self.trades = []

        # Model placeholder
        self.model = None

    @abstractmethod
    def train(self, env, total_timesteps: int, **kwargs):
        """
        Train the agent

        Args:
            env: Trading environment
            total_timesteps: Number of timesteps to train
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action given observation

        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy

        Returns:
            action: Action to take
        """
        pass

    def record_trade(
        self,
        timestamp: pd.Timestamp,
        asset: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float
    ):
        """Record a trade for performance tracking"""
        trade = {
            'timestamp': timestamp,
            'asset': asset,
            'action': action,
            'quantity': quantity,
            'price': price,
            'pnl': pnl
        }

        self.trades.append(trade)
        self.metrics['total_trades'] += 1

        if pnl > 0:
            self.metrics['winning_trades'] += 1
        elif pnl < 0:
            self.metrics['losing_trades'] += 1

        self.metrics['total_pnl'] += pnl

        # Update derived metrics
        self._update_metrics()

    def _update_metrics(self):
        """Update derived performance metrics"""
        if self.metrics['total_trades'] == 0:
            return

        # Win rate
        self.metrics['win_rate'] = (
            self.metrics['winning_trades'] / self.metrics['total_trades']
        )

        # Average win/loss
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]

        self.metrics['avg_win'] = np.mean(wins) if wins else 0.0
        self.metrics['avg_loss'] = np.mean(losses) if losses else 0.0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        if total_losses > 0:
            self.metrics['profit_factor'] = total_wins / total_losses
        else:
            self.metrics['profit_factor'] = float('inf') if total_wins > 0 else 0.0

        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            pnls = [t['pnl'] for t in self.trades]
            if np.std(pnls) > 0:
                self.metrics['sharpe_ratio'] = np.mean(pnls) / np.std(pnls) * np.sqrt(252)

    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.metrics.copy()

    def get_trade_history(self) -> List[Dict]:
        """Get full trade history"""
        return self.trades.copy()

    def save(self, path: str):
        """
        Save agent to disk

        Args:
            path: Directory path to save agent
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model is not None:
            model_path = save_path / f"{self.name}_model"
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")

        # Save metrics and trades
        metadata = {
            'name': self.name,
            'strategy': self.strategy,
            'agent_id': self.agent_id,
            'metrics': self.metrics,
            'trades': self.trades
        }

        metadata_path = save_path / f"{self.name}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Agent metadata saved to {metadata_path}")

    def load(self, path: str):
        """
        Load agent from disk

        Args:
            path: Directory path to load agent from
        """
        load_path = Path(path)

        # Load metadata
        metadata_path = load_path / f"{self.name}_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.metrics = metadata['metrics']
            self.trades = metadata['trades']

            logger.info(f"Agent metadata loaded from {metadata_path}")

        # Model loading is handled by subclasses

    def reset_metrics(self):
        """Reset performance metrics and trade history"""
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
        self.trades = []

    def get_explanation(self, observation: np.ndarray, action: np.ndarray) -> str:
        """
        Generate human-readable explanation for an action

        Args:
            observation: Current state
            action: Action taken

        Returns:
            Explanation string
        """
        # Base implementation - override in subclasses for strategy-specific explanations
        return f"{self.name} ({self.strategy}): Executed action {action}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', strategy='{self.strategy}')"


class AgentEvaluator:
    """Utility class to evaluate agent performance"""

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        return float(sharpe)

    @staticmethod
    def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) == 0:
            return 0.0

        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (cummax - portfolio_values) / cummax
        max_dd = np.max(drawdowns)

        return float(max_dd)

    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)

        return float(sortino)

    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, portfolio_values: np.ndarray) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        if len(returns) == 0 or len(portfolio_values) == 0:
            return 0.0

        annual_return = np.mean(returns) * 252
        max_dd = AgentEvaluator.calculate_max_drawdown(portfolio_values)

        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0

        calmar = annual_return / max_dd

        return float(calmar)

    @staticmethod
    def generate_performance_report(agent: BaseAgent) -> str:
        """Generate a comprehensive performance report"""
        metrics = agent.get_metrics()

        report = f"""
{'='*60}
Performance Report: {agent.name}
Strategy: {agent.strategy}
{'='*60}

Trade Statistics:
  Total Trades:      {metrics['total_trades']:>10}
  Winning Trades:    {metrics['winning_trades']:>10}
  Losing Trades:     {metrics['losing_trades']:>10}
  Win Rate:          {metrics['win_rate']:>10.2%}

Profitability:
  Total P&L:         ${metrics['total_pnl']:>10,.2f}
  Average Win:       ${metrics['avg_win']:>10,.2f}
  Average Loss:      ${metrics['avg_loss']:>10,.2f}
  Profit Factor:     {metrics['profit_factor']:>10.2f}

Risk Metrics:
  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}
  Max Drawdown:      {metrics['max_drawdown']:>10.2%}

{'='*60}
"""

        return report
