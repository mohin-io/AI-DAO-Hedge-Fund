"""
Options Trading Agent with Greeks Calculation
Implements Black-Scholes model and advanced options strategies
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlackScholesModel:
    """
    Black-Scholes option pricing model with Greeks calculation
    """

    @staticmethod
    def calculate_d1_d2(
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        r: float,  # Risk-free rate
        sigma: float  # Volatility
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price"""
        if T <= 0:
            return max(S - K, 0)

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price"""
        if T <= 0:
            return max(K - S, 0)

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price


class GreeksCalculator:
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
    """

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Delta: rate of change of option price with respect to stock price"""
        if T <= 0:
            return 0

        d1, _ = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)

        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Gamma: rate of change of Delta with respect to stock price"""
        if T <= 0:
            return 0

        d1, _ = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Theta: rate of change of option price with respect to time"""
        if T <= 0:
            return 0

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)

        common_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type.lower() == 'call':
            theta = common_term - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = common_term + r * K * np.exp(-r * T) * norm.cdf(-d2)

        return theta / 365  # Daily theta

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Vega: rate of change of option price with respect to volatility"""
        if T <= 0:
            return 0

        d1, _ = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega / 100  # Per 1% change in volatility

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Rho: rate of change of option price with respect to interest rate"""
        if T <= 0:
            return 0

        _, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)

        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return rho / 100  # Per 1% change in interest rate

    @staticmethod
    def calculate_all_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str
    ) -> Dict[str, float]:
        """Calculate all Greeks at once"""
        return {
            'delta': GreeksCalculator.delta(S, K, T, r, sigma, option_type),
            'gamma': GreeksCalculator.gamma(S, K, T, r, sigma),
            'theta': GreeksCalculator.theta(S, K, T, r, sigma, option_type),
            'vega': GreeksCalculator.vega(S, K, T, r, sigma),
            'rho': GreeksCalculator.rho(S, K, T, r, sigma, option_type)
        }


class OptionsStrategy:
    """
    Advanced options strategies
    """

    @staticmethod
    def covered_call(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        shares_owned: int = 100
    ) -> Dict[str, float]:
        """
        Covered Call: Own stock + Sell call option
        Strategy for generating income on stocks you already own
        """
        stock_value = S * shares_owned
        call_premium = BlackScholesModel.call_price(S, K, T, r, sigma) * shares_owned

        return {
            'stock_value': stock_value,
            'call_premium_received': call_premium,
            'total_value': stock_value + call_premium,
            'max_profit': (K - S) * shares_owned + call_premium,
            'breakeven': S - (call_premium / shares_owned)
        }

    @staticmethod
    def protective_put(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        shares_owned: int = 100
    ) -> Dict[str, float]:
        """
        Protective Put: Own stock + Buy put option
        Strategy for downside protection
        """
        stock_value = S * shares_owned
        put_cost = BlackScholesModel.put_price(S, K, T, r, sigma) * shares_owned

        return {
            'stock_value': stock_value,
            'put_cost_paid': put_cost,
            'total_cost': stock_value + put_cost,
            'max_loss': (S - K) * shares_owned + put_cost,
            'breakeven': S + (put_cost / shares_owned)
        }

    @staticmethod
    def bull_call_spread(
        S: float,
        K_long: float,
        K_short: float,
        T: float,
        r: float,
        sigma: float
    ) -> Dict[str, float]:
        """
        Bull Call Spread: Buy call at lower strike + Sell call at higher strike
        Strategy for moderate bullish outlook
        """
        long_call = BlackScholesModel.call_price(S, K_long, T, r, sigma)
        short_call = BlackScholesModel.call_price(S, K_short, T, r, sigma)

        net_cost = long_call - short_call

        return {
            'long_call_cost': long_call,
            'short_call_premium': short_call,
            'net_cost': net_cost,
            'max_profit': (K_short - K_long) - net_cost,
            'max_loss': net_cost,
            'breakeven': K_long + net_cost
        }

    @staticmethod
    def iron_condor(
        S: float,
        K_put_long: float,
        K_put_short: float,
        K_call_short: float,
        K_call_long: float,
        T: float,
        r: float,
        sigma: float
    ) -> Dict[str, float]:
        """
        Iron Condor: Sell OTM put spread + Sell OTM call spread
        Strategy for low volatility markets
        """
        put_long = BlackScholesModel.put_price(S, K_put_long, T, r, sigma)
        put_short = BlackScholesModel.put_price(S, K_put_short, T, r, sigma)
        call_short = BlackScholesModel.call_price(S, K_call_short, T, r, sigma)
        call_long = BlackScholesModel.call_price(S, K_call_long, T, r, sigma)

        net_credit = (put_short - put_long) + (call_short - call_long)

        return {
            'net_credit': net_credit,
            'max_profit': net_credit,
            'max_loss': (K_put_short - K_put_long) - net_credit,
            'breakeven_lower': K_put_short - net_credit,
            'breakeven_upper': K_call_short + net_credit
        }


class OptionsEnvironment(gym.Env):
    """
    Gymnasium environment for options trading with RL
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 1.0,
        risk_free_rate: float = 0.05
    ):
        super().__init__()

        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_free_rate = risk_free_rate

        # Action space: [strategy_type, strike_offset, quantity]
        # strategy_type: 0=covered_call, 1=protective_put, 2=bull_spread, 3=iron_condor
        self.action_space = spaces.Box(
            low=np.array([0, -0.2, 0]),
            high=np.array([3, 0.2, 10]),
            dtype=np.float32
        )

        # Observation space: [price, volatility, time_to_expiry, portfolio_value, greeks...]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.positions = []

        # Market simulation
        self.current_price = 100.0
        self.volatility = 0.25
        self.time_step = 0
        self.max_steps = 252  # 1 year of trading days

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        days_to_expiry = 30  # Example: 30-day options

        # Calculate Greeks for ATM option
        greeks = GreeksCalculator.calculate_all_greeks(
            S=self.current_price,
            K=self.current_price,
            T=days_to_expiry / 365,
            r=self.risk_free_rate,
            sigma=self.volatility,
            option_type='call'
        )

        obs = np.array([
            self.current_price,
            self.volatility,
            days_to_expiry,
            self.portfolio_value,
            self.capital,
            len(self.positions),
            greeks['delta'],
            greeks['gamma'],
            greeks['theta'],
            greeks['vega'],
            greeks['rho'],
            self.time_step / self.max_steps,
            np.random.randn(),  # Market sentiment
            np.random.randn(),  # Technical indicator
            np.random.randn()   # Volume indicator
        ], dtype=np.float32)

        return obs

    def step(self, action):
        # Simulate price movement (Geometric Brownian Motion)
        dt = 1/252
        drift = self.risk_free_rate * dt
        shock = self.volatility * np.sqrt(dt) * np.random.randn()
        self.current_price *= np.exp(drift + shock)

        # Execute strategy based on action
        strategy_type = int(action[0])
        strike_offset = action[1]
        quantity = int(action[2])

        # Calculate reward based on portfolio P&L
        previous_value = self.portfolio_value
        self.portfolio_value = self._calculate_portfolio_value()
        reward = (self.portfolio_value - previous_value) / previous_value

        self.time_step += 1
        done = self.time_step >= self.max_steps
        truncated = False

        return self._get_observation(), reward, done, truncated, {}

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value including all positions"""
        # Simplified: just return capital for now
        return self.capital


class OptionsAgent:
    """
    RL-based Options Trading Agent using PPO
    """

    def __init__(self, initial_capital: float = 100000):
        self.env = OptionsEnvironment(initial_capital=initial_capital)
        self.model = None

    def train(self, total_timesteps: int = 100000):
        """Train the options agent using PPO"""
        logger.info("Training Options Agent...")

        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1
        )

        self.model.learn(total_timesteps=total_timesteps)
        logger.info("✓ Options Agent training complete")

    def save(self, path: str):
        """Save trained model"""
        self.model.save(path)

    def load(self, path: str):
        """Load trained model"""
        self.model = PPO.load(path, env=self.env)


if __name__ == "__main__":
    # Example: Calculate option price and Greeks
    print("=" * 80)
    print("BLACK-SCHOLES OPTION PRICING & GREEKS")
    print("=" * 80)

    S = 100  # Current stock price
    K = 105  # Strike price
    T = 30/365  # 30 days to expiration
    r = 0.05  # 5% risk-free rate
    sigma = 0.25  # 25% volatility

    call_price = BlackScholesModel.call_price(S, K, T, r, sigma)
    put_price = BlackScholesModel.put_price(S, K, T, r, sigma)

    print(f"\nOption Prices:")
    print(f"  Call: ${call_price:.2f}")
    print(f"  Put: ${put_price:.2f}")

    print(f"\nCall Greeks:")
    greeks = GreeksCalculator.calculate_all_greeks(S, K, T, r, sigma, 'call')
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.4f}")

    # Example: Bull Call Spread
    print(f"\n" + "=" * 80)
    print("BULL CALL SPREAD STRATEGY")
    print("=" * 80)

    spread = OptionsStrategy.bull_call_spread(
        S=100,
        K_long=100,
        K_short=110,
        T=30/365,
        r=0.05,
        sigma=0.25
    )

    print(f"\nStrategy Details:")
    for key, value in spread.items():
        print(f"  {key}: ${value:.2f}")

    print("\n✓ Options calculations complete!")
