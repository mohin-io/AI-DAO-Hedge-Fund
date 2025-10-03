"""
Trading Environment for Multi-Agent RL
OpenAI Gymnasium compatible environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agents

    State Space:
        - Portfolio: cash, positions, total value
        - Market: prices, volumes, technical indicators
        - Time: time-based features

    Action Space:
        - Discrete: [0=hold, 1=buy, 2=sell] for each asset
        - OR Continuous: [-1, 1] position size for each asset

    Reward:
        - Sharpe ratio maximization
        - Max drawdown penalty
        - Transaction cost consideration
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.3,
        lookback_window: int = 20,
        action_type: str = 'continuous'
    ):
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.action_type = action_type

        # Get number of assets
        self.n_assets = len([col for col in data.columns if col.endswith('_close')])

        # Define action space
        if action_type == 'discrete':
            # 3 actions per asset: hold, buy, sell
            self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        else:
            # Continuous action: position size [-1, 1] for each asset
            self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(self.n_assets,),
                dtype=np.float32
            )

        # Define observation space
        # Portfolio (3) + Market data per asset + Technical indicators
        n_features_per_asset = 10  # price, volume, + 8 indicators
        obs_dim = 3 + self.n_assets * n_features_per_asset

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.cash = self.initial_balance
        self.positions = np.zeros(self.n_assets)
        self.portfolio_values = [self.initial_balance]
        self.trades_history = []

        # Performance tracking
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to position changes
        position_changes = self._process_action(action)

        # Execute trades
        self._execute_trades(position_changes)

        # Move to next time step
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = self._get_portfolio_value() <= 0  # Bankruptcy

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Convert raw action to position changes"""
        if self.action_type == 'discrete':
            # 0=hold, 1=buy, 2=sell
            position_changes = np.zeros(self.n_assets)
            for i, act in enumerate(action):
                if act == 1:  # Buy
                    position_changes[i] = 0.1  # Buy 10% of portfolio
                elif act == 2:  # Sell
                    position_changes[i] = -0.1  # Sell 10% of position
        else:
            # Continuous: direct position sizing
            position_changes = action * self.max_position_size

        return position_changes

    def _execute_trades(self, position_changes: np.ndarray):
        """Execute trades with transaction costs and slippage"""
        current_prices = self._get_current_prices()

        for i, change in enumerate(position_changes):
            if abs(change) < 1e-6:  # Skip tiny trades
                continue

            current_price = current_prices[i]
            current_value = self._get_portfolio_value()

            # Calculate shares to trade
            target_value = current_value * change
            shares = target_value / current_price

            # Apply slippage
            if shares > 0:  # Buying
                execution_price = current_price * (1 + self.slippage)
            else:  # Selling
                execution_price = current_price * (1 - self.slippage)

            # Calculate cost
            trade_cost = abs(shares * execution_price)
            transaction_fee = trade_cost * self.transaction_cost

            # Check if we have enough cash
            if shares > 0:  # Buying
                total_cost = trade_cost + transaction_fee
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.positions[i] += shares
                    self._record_trade('buy', i, shares, execution_price, transaction_fee)
            else:  # Selling
                if self.positions[i] >= abs(shares):
                    proceeds = trade_cost - transaction_fee
                    self.cash += proceeds
                    self.positions[i] += shares  # shares is negative
                    self._record_trade('sell', i, abs(shares), execution_price, transaction_fee)

    def _record_trade(self, action: str, asset: int, shares: float,
                     price: float, fee: float):
        """Record trade for analysis"""
        self.trades_history.append({
            'step': self.current_step,
            'action': action,
            'asset': asset,
            'shares': shares,
            'price': price,
            'fee': fee
        })

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on:
        1. Portfolio return
        2. Sharpe ratio
        3. Drawdown penalty
        """
        current_value = self._get_portfolio_value()
        self.portfolio_values.append(current_value)

        # Update max value and drawdown
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value

        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Calculate returns
        if len(self.portfolio_values) < 2:
            return 0.0

        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        recent_returns = returns[-20:]  # Last 20 steps

        # Sharpe ratio approximation
        if len(recent_returns) > 1 and np.std(recent_returns) > 0:
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
        else:
            sharpe = 0.0

        # Drawdown penalty
        dd_penalty = -10 * drawdown if drawdown > 0.2 else 0

        # Combine rewards
        reward = sharpe + dd_penalty

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation

        Includes:
        - Portfolio state: cash %, positions %, total value change
        - Market data: normalized prices, volumes
        - Technical indicators: RSI, MACD, etc.
        """
        portfolio_value = self._get_portfolio_value()

        # Portfolio features
        cash_pct = self.cash / portfolio_value
        positions_value = np.sum(self.positions * self._get_current_prices())
        positions_pct = positions_value / portfolio_value
        value_change = (portfolio_value / self.initial_balance) - 1

        portfolio_features = [cash_pct, positions_pct, value_change]

        # Market features for each asset
        market_features = []
        for i in range(self.n_assets):
            asset_data = self._get_asset_window(i)
            market_features.extend(asset_data)

        obs = np.array(portfolio_features + market_features, dtype=np.float32)

        return obs

    def _get_asset_window(self, asset_idx: int) -> list:
        """Get normalized features for an asset"""
        # Get column names for this asset
        price_col = f'asset_{asset_idx}_close'
        volume_col = f'asset_{asset_idx}_volume'

        if price_col not in self.data.columns:
            # Fallback for different naming convention
            price_col = self.data.columns[asset_idx * 2]
            volume_col = self.data.columns[asset_idx * 2 + 1]

        # Get recent window
        window = self.data.iloc[
            max(0, self.current_step - self.lookback_window):self.current_step + 1
        ]

        if price_col in window.columns:
            prices = window[price_col].values
            volumes = window[volume_col].values if volume_col in window.columns else np.zeros_like(prices)

            # Normalize
            price_norm = (prices[-1] / prices[0]) - 1 if len(prices) > 0 and prices[0] != 0 else 0
            volume_norm = volumes[-1] / (np.mean(volumes) + 1e-6) if len(volumes) > 0 else 0

            # Simple technical indicators
            rsi = self._calculate_rsi(prices)
            macd = self._calculate_macd(prices)

            features = [
                price_norm,
                volume_norm,
                rsi / 100.0,  # Normalize RSI
                macd,
                np.mean(prices[-5:]) / prices[-1] - 1 if len(prices) >= 5 else 0,  # MA5
                np.mean(prices[-10:]) / prices[-1] - 1 if len(prices) >= 10 else 0,  # MA10
                (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-6),  # Stochastic
                np.std(prices) / (np.mean(prices) + 1e-6),  # Volatility
            ]
        else:
            features = [0.0] * 8

        # Pad to 10 features
        features = features[:10] + [0.0] * max(0, 10 - len(features))

        return features

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(self, prices: np.ndarray) -> float:
        """Calculate MACD (simplified)"""
        if len(prices) < 26:
            return 0.0

        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        macd = (ema12 - ema26) / (ema26 + 1e-6)

        return float(macd)

    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all assets"""
        prices = []
        for i in range(self.n_assets):
            price_col = f'asset_{i}_close'
            if price_col in self.data.columns:
                prices.append(self.data.iloc[self.current_step][price_col])
            else:
                # Fallback
                prices.append(self.data.iloc[self.current_step, i * 2])

        return np.array(prices)

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = np.sum(self.positions * self._get_current_prices())
        return self.cash + positions_value

    def _get_info(self) -> dict:
        """Get additional information"""
        return {
            'step': self.current_step,
            'portfolio_value': self._get_portfolio_value(),
            'cash': self.cash,
            'positions': self.positions.copy(),
            'max_drawdown': self.max_drawdown,
            'num_trades': len(self.trades_history),
            'sharpe_ratio': self._calculate_sharpe()
        }

    def _calculate_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.portfolio_values) < 2:
            return 0.0

        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        return float(sharpe)

    def render(self):
        """Render environment state"""
        if hasattr(self, 'current_step'):
            portfolio_value = self._get_portfolio_value()
            returns = (portfolio_value / self.initial_balance - 1) * 100

            print(f"\n=== Step {self.current_step} ===")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Returns: {returns:.2f}%")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Positions: {self.positions}")
            print(f"Max Drawdown: {self.max_drawdown * 100:.2f}%")
            print(f"Sharpe Ratio: {self._calculate_sharpe():.2f}")
