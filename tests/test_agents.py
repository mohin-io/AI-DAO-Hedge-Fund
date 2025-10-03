"""
Unit tests for trading agents
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.momentum_agent import MomentumAgent
from agents.arbitrage_agent import ArbitrageAgent
from agents.hedging_agent import HedgingAgent
from agents.multi_agent_coordinator import MultiAgentCoordinator
from environment.trading_env import TradingEnvironment
from environment.data_loader import generate_synthetic_data


class TestAgents:
    """Test suite for trading agents"""

    @pytest.fixture
    def sample_environment(self):
        """Create a sample trading environment"""
        data = generate_synthetic_data(n_assets=3, n_days=100, seed=42)
        env = TradingEnvironment(data=data, initial_balance=10000)
        return env

    @pytest.fixture
    def sample_observation(self):
        """Create a sample observation"""
        return np.random.randn(33)

    def test_momentum_agent_initialization(self):
        """Test that MomentumAgent initializes correctly"""
        agent = MomentumAgent(agent_id=0)

        assert agent.name == "Momentum Trader"
        assert agent.agent_id == 0
        assert agent.strategy == "Momentum Trading - RSI/MACD/MA-based"
        assert agent.model is None  # Not trained yet

    def test_arbitrage_agent_initialization(self):
        """Test that ArbitrageAgent initializes correctly"""
        agent = ArbitrageAgent(agent_id=1)

        assert agent.name == "Arbitrage Hunter"
        assert agent.agent_id == 1
        assert agent.strategy == "Statistical Arbitrage - Mean Reversion"

    def test_hedging_agent_initialization(self):
        """Test that HedgingAgent initializes correctly"""
        agent = HedgingAgent(agent_id=2)

        assert agent.name == "Risk Hedger"
        assert agent.agent_id == 2
        assert "Risk Management" in agent.strategy

    def test_agent_metrics_tracking(self):
        """Test that agents track performance metrics"""
        agent = MomentumAgent()

        # Record some trades
        agent.record_trade(
            timestamp=pd.Timestamp('2024-01-01'),
            asset='AAPL',
            action='buy',
            quantity=10,
            price=150.0,
            pnl=100.0
        )

        agent.record_trade(
            timestamp=pd.Timestamp('2024-01-02'),
            asset='AAPL',
            action='sell',
            quantity=10,
            price=155.0,
            pnl=-50.0
        )

        metrics = agent.get_metrics()

        assert metrics['total_trades'] == 2
        assert metrics['winning_trades'] == 1
        assert metrics['losing_trades'] == 1
        assert metrics['total_pnl'] == 50.0
        assert metrics['win_rate'] == 0.5

    def test_multi_agent_coordinator(self):
        """Test multi-agent coordinator initialization and prediction"""
        agents = [
            MomentumAgent(agent_id=0),
            ArbitrageAgent(agent_id=1),
            HedgingAgent(agent_id=2)
        ]

        coordinator = MultiAgentCoordinator(agents=agents)

        assert len(coordinator.agents) == 3
        assert len(coordinator.weights) == 3
        assert np.allclose(np.sum(coordinator.weights), 1.0)

    def test_regime_detection(self, sample_observation):
        """Test market regime detection"""
        coordinator = MultiAgentCoordinator()

        # Test different market conditions
        bull_obs = sample_observation.copy()
        bull_obs[2] = 0.02  # Strong positive value change

        bear_obs = sample_observation.copy()
        bear_obs[2] = -0.02  # Strong negative value change

        volatile_obs = sample_observation.copy()
        volatile_obs[2] = 0.05  # High volatility

        bull_regime = coordinator.detect_market_regime(bull_obs)
        bear_regime = coordinator.detect_market_regime(bear_obs)
        volatile_regime = coordinator.detect_market_regime(volatile_obs)

        assert bull_regime == 'BULL'
        assert bear_regime == 'BEAR'
        assert volatile_regime == 'VOLATILE'

    def test_agent_save_load(self, tmp_path):
        """Test agent save and load functionality"""
        agent = MomentumAgent(agent_id=0)

        # Record some metrics
        agent.metrics['total_pnl'] = 1000.0
        agent.metrics['total_trades'] = 50

        # Save
        save_path = tmp_path / "test_agent"
        agent.save(str(save_path))

        # Load
        new_agent = MomentumAgent(agent_id=0)
        new_agent.load(str(save_path))

        assert new_agent.metrics['total_pnl'] == 1000.0
        assert new_agent.metrics['total_trades'] == 50


class TestEnvironment:
    """Test suite for trading environment"""

    def test_environment_initialization(self):
        """Test that environment initializes correctly"""
        data = generate_synthetic_data(n_assets=3, n_days=100)
        env = TradingEnvironment(data=data, initial_balance=10000)

        assert env.initial_balance == 10000
        assert env.n_assets == 3
        assert env.observation_space.shape[0] == 33  # 3 + 3*10

    def test_environment_reset(self):
        """Test environment reset"""
        data = generate_synthetic_data(n_assets=3, n_days=100)
        env = TradingEnvironment(data=data, initial_balance=10000)

        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert info['portfolio_value'] == 10000
        assert info['step'] == env.lookback_window

    def test_environment_step(self):
        """Test environment step function"""
        data = generate_synthetic_data(n_assets=3, n_days=100)
        env = TradingEnvironment(data=data, initial_balance=10000)

        obs, info = env.reset()
        action = np.array([0.1, -0.05, 0.0])  # Buy, sell, hold

        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'portfolio_value' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
