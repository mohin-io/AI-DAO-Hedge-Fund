"""
Multi-Agent Coordinator
Combines decisions from multiple agents using ensemble methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from agents.base_agent import BaseAgent
from agents.momentum_agent import MomentumAgent
from agents.arbitrage_agent import ArbitrageAgent
from agents.hedging_agent import HedgingAgent

logger = logging.getLogger(__name__)


class MultiAgentCoordinator:
    """
    Coordinates multiple trading agents

    Strategies:
    - Weighted voting based on recent performance
    - Dynamic allocation based on market regime
    - Conflict resolution via DAO governance
    - Meta-learning for agent selection
    """

    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        ensemble_method: str = 'weighted_voting',
        rebalance_frequency: int = 10
    ):
        """
        Initialize coordinator

        Args:
            agents: List of trading agents
            ensemble_method: Method for combining predictions
            rebalance_frequency: How often to rebalance weights (steps)
        """
        if agents is None:
            # Initialize default agents
            self.agents = [
                MomentumAgent(agent_id=0),
                ArbitrageAgent(agent_id=1),
                HedgingAgent(agent_id=2)
            ]
        else:
            self.agents = agents

        self.ensemble_method = ensemble_method
        self.rebalance_frequency = rebalance_frequency

        # Initialize weights (equal weighting)
        self.weights = np.ones(len(self.agents)) / len(self.agents)

        # Performance tracking
        self.agent_performance = {
            agent.agent_id: {
                'recent_returns': [],
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'weight': 1.0 / len(self.agents)
            }
            for agent in self.agents
        }

        # Market regime detection
        self.current_regime = 'NEUTRAL'
        self.regime_history = []

        # Steps counter
        self.steps = 0

    def predict(
        self,
        observation: np.ndarray,
        market_regime: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get combined prediction from all agents

        Args:
            observation: Current state
            market_regime: Optional market regime override

        Returns:
            combined_action, agent_actions
        """
        # Detect market regime if not provided
        if market_regime is None:
            market_regime = self.detect_market_regime(observation)

        self.current_regime = market_regime

        # Get predictions from each agent
        agent_actions = {}
        for agent in self.agents:
            try:
                action = agent.predict(observation, deterministic=True)
                agent_actions[agent.agent_id] = action
            except Exception as e:
                logger.warning(f"Agent {agent.name} failed to predict: {e}")
                # Default to hold action
                agent_actions[agent.agent_id] = np.zeros(3)  # Assuming 3 assets

        # Combine actions based on ensemble method
        if self.ensemble_method == 'weighted_voting':
            combined_action = self._weighted_voting(agent_actions)
        elif self.ensemble_method == 'regime_based':
            combined_action = self._regime_based_selection(agent_actions, market_regime)
        elif self.ensemble_method == 'performance_weighted':
            combined_action = self._performance_weighted(agent_actions)
        else:
            # Default: equal weighting
            combined_action = self._equal_weighting(agent_actions)

        # Rebalance weights periodically
        self.steps += 1
        if self.steps % self.rebalance_frequency == 0:
            self._rebalance_weights()

        return combined_action, agent_actions

    def _weighted_voting(self, agent_actions: Dict) -> np.ndarray:
        """Combine actions using weighted voting"""
        combined = np.zeros_like(list(agent_actions.values())[0])

        for i, agent in enumerate(self.agents):
            if agent.agent_id in agent_actions:
                combined += self.weights[i] * agent_actions[agent.agent_id]

        return combined

    def _equal_weighting(self, agent_actions: Dict) -> np.ndarray:
        """Simple average of all actions"""
        actions = list(agent_actions.values())
        return np.mean(actions, axis=0)

    def _performance_weighted(self, agent_actions: Dict) -> np.ndarray:
        """Weight by recent performance (Sharpe ratio)"""
        combined = np.zeros_like(list(agent_actions.values())[0])

        # Get performance-based weights
        perf_weights = []
        for agent in self.agents:
            sharpe = self.agent_performance[agent.agent_id]['sharpe_ratio']
            # Convert Sharpe to weight (exponential for emphasis on good performers)
            weight = np.exp(sharpe) if sharpe > 0 else 0.1
            perf_weights.append(weight)

        # Normalize
        perf_weights = np.array(perf_weights)
        perf_weights = perf_weights / (perf_weights.sum() + 1e-6)

        # Combine
        for i, agent in enumerate(self.agents):
            if agent.agent_id in agent_actions:
                combined += perf_weights[i] * agent_actions[agent.agent_id]

        return combined

    def _regime_based_selection(self, agent_actions: Dict, regime: str) -> np.ndarray:
        """
        Select agent based on market regime

        Regimes:
        - BULL: Favor momentum agent
        - BEAR: Favor hedging agent
        - SIDEWAYS: Favor arbitrage agent
        - VOLATILE: Favor hedging agent
        """
        regime_preferences = {
            'BULL': [0.5, 0.2, 0.3],      # Momentum, Arbitrage, Hedging
            'BEAR': [0.1, 0.2, 0.7],      # Heavy on hedging
            'SIDEWAYS': [0.2, 0.6, 0.2],  # Favor arbitrage
            'VOLATILE': [0.2, 0.1, 0.7],  # Heavy on hedging
            'NEUTRAL': [0.33, 0.33, 0.34] # Equal weight
        }

        weights = regime_preferences.get(regime, [0.33, 0.33, 0.34])

        combined = np.zeros_like(list(agent_actions.values())[0])
        for i, agent in enumerate(self.agents):
            if agent.agent_id in agent_actions:
                combined += weights[i] * agent_actions[agent.agent_id]

        return combined

    def detect_market_regime(self, observation: np.ndarray) -> str:
        """
        Detect current market regime from observation

        Args:
            observation: Current state

        Returns:
            Market regime: BULL, BEAR, SIDEWAYS, VOLATILE, NEUTRAL
        """
        # Extract features from observation
        # observation[2] is value_change in our environment

        if len(observation) < 3:
            return 'NEUTRAL'

        value_change = observation[2]
        volatility = abs(value_change)

        # Simple regime detection
        if volatility > 0.03:  # High volatility
            regime = 'VOLATILE'
        elif value_change > 0.01:  # Uptrend
            regime = 'BULL'
        elif value_change < -0.01:  # Downtrend
            regime = 'BEAR'
        elif abs(value_change) < 0.005:  # Low movement
            regime = 'SIDEWAYS'
        else:
            regime = 'NEUTRAL'

        self.regime_history.append({
            'step': self.steps,
            'regime': regime,
            'value_change': value_change
        })

        return regime

    def _rebalance_weights(self):
        """Rebalance agent weights based on recent performance"""
        logger.info(f"Rebalancing weights at step {self.steps}")

        new_weights = []

        for agent in self.agents:
            perf = self.agent_performance[agent.agent_id]

            # Calculate performance score
            # Combine Sharpe ratio and win rate
            sharpe_score = max(0, perf['sharpe_ratio'])
            win_rate_score = perf['win_rate']

            # Weighted combination
            score = 0.7 * sharpe_score + 0.3 * win_rate_score

            new_weights.append(score)

        # Normalize weights
        new_weights = np.array(new_weights)
        total = new_weights.sum()

        if total > 0:
            self.weights = new_weights / total
        else:
            # Fallback to equal weighting
            self.weights = np.ones(len(self.agents)) / len(self.agents)

        # Update performance tracking
        for i, agent in enumerate(self.agents):
            self.agent_performance[agent.agent_id]['weight'] = self.weights[i]

        logger.info(f"New weights: {dict(zip([a.name for a in self.agents], self.weights))}")

    def update_agent_performance(
        self,
        agent_id: int,
        returns: float,
        sharpe_ratio: float,
        win_rate: float
    ):
        """Update performance metrics for an agent"""
        if agent_id in self.agent_performance:
            perf = self.agent_performance[agent_id]

            perf['recent_returns'].append(returns)
            # Keep only last 50 returns
            if len(perf['recent_returns']) > 50:
                perf['recent_returns'] = perf['recent_returns'][-50:]

            perf['sharpe_ratio'] = sharpe_ratio
            perf['win_rate'] = win_rate

    def get_agent_allocations(self) -> Dict:
        """Get current agent allocations"""
        allocations = {}
        for i, agent in enumerate(self.agents):
            allocations[agent.name] = {
                'weight': self.weights[i],
                'sharpe_ratio': self.agent_performance[agent.agent_id]['sharpe_ratio'],
                'win_rate': self.agent_performance[agent.agent_id]['win_rate']
            }

        return allocations

    def get_explanation(self, observation: np.ndarray, combined_action: np.ndarray,
                       agent_actions: Dict) -> str:
        """
        Generate comprehensive explanation for ensemble decision

        Args:
            observation: Current state
            combined_action: Final combined action
            agent_actions: Individual agent actions

        Returns:
            Detailed explanation string
        """
        explanation = f"""
{'='*70}
Multi-Agent Coordinator Decision
{'='*70}

Market Regime: {self.current_regime}
Ensemble Method: {self.ensemble_method}

Individual Agent Predictions:
"""

        for agent in self.agents:
            if agent.agent_id in agent_actions:
                action = agent_actions[agent.agent_id]
                weight = self.weights[agent.agent_id]

                explanation += f"\n{agent.name} (Weight: {weight:.2%}):\n"
                explanation += f"  Action: {action}\n"
                explanation += f"  Sharpe: {self.agent_performance[agent.agent_id]['sharpe_ratio']:.2f}\n"
                explanation += f"  Win Rate: {self.agent_performance[agent.agent_id]['win_rate']:.2%}\n"

        explanation += f"\nCombined Action: {combined_action}\n"
        explanation += f"\nRationale: "

        if self.ensemble_method == 'regime_based':
            explanation += f"In {self.current_regime} market, prioritizing "
            if self.current_regime == 'BULL':
                explanation += "momentum strategies for trend following."
            elif self.current_regime == 'BEAR':
                explanation += "hedging strategies for risk protection."
            elif self.current_regime == 'SIDEWAYS':
                explanation += "arbitrage strategies for range-bound trading."
            else:
                explanation += "balanced approach across all strategies."
        else:
            explanation += f"Combining agent predictions using {self.ensemble_method}."

        explanation += f"\n{'='*70}"

        return explanation

    def save(self, path: str):
        """Save coordinator state and all agents"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save each agent
        for agent in self.agents:
            agent_path = save_path / agent.name.replace(' ', '_')
            agent.save(str(agent_path))

        # Save coordinator state
        import pickle
        state = {
            'weights': self.weights,
            'agent_performance': self.agent_performance,
            'regime_history': self.regime_history,
            'ensemble_method': self.ensemble_method
        }

        with open(save_path / 'coordinator_state.pkl', 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Coordinator saved to {save_path}")

    def load(self, path: str):
        """Load coordinator state and all agents"""
        load_path = Path(path)

        # Load each agent
        for agent in self.agents:
            agent_path = load_path / agent.name.replace(' ', '_')
            if agent_path.exists():
                agent.load(str(agent_path))

        # Load coordinator state
        import pickle
        state_file = load_path / 'coordinator_state.pkl'
        if state_file.exists():
            with open(state_file, 'rb') as f:
                state = pickle.load(f)

            self.weights = state['weights']
            self.agent_performance = state['agent_performance']
            self.regime_history = state.get('regime_history', [])

            logger.info(f"Coordinator loaded from {load_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test coordinator
    coordinator = MultiAgentCoordinator()

    # Simulate observation
    observation = np.random.randn(33)  # Example observation

    # Get prediction
    action, agent_actions = coordinator.predict(observation)

    print(f"Combined action: {action}")
    print(f"\nAgent actions: {agent_actions}")

    # Get explanation
    explanation = coordinator.get_explanation(observation, action, agent_actions)
    print(explanation)
