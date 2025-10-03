"""
Risk Hedging Agent
Uses SAC algorithm for portfolio protection
"""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class HedgingAgent(BaseAgent):
    """
    Risk Hedging Strategy Agent

    Strategy:
        - Portfolio protection and volatility management
        - Tail risk hedging (black swan protection)
        - Dynamic position sizing based on market volatility
        - Correlation-based risk reduction

    Algorithm: SAC (Soft Actor-Critic)
        - Maximum entropy framework (explores risky scenarios)
        - Excellent for continuous control
        - More stable than vanilla policy gradients
        - Good for risk-sensitive objectives
    """

    def __init__(self, name: str = "Risk Hedger", agent_id: Optional[int] = None):
        super().__init__(
            name=name,
            strategy="Risk Management - Volatility & Tail Risk Hedging",
            agent_id=agent_id
        )

        # SAC specific parameters
        self.learning_rate = 3e-4
        self.buffer_size = 100000
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.ent_coef = 'auto'  # Automatic entropy tuning

        # Risk management parameters
        self.target_volatility = 0.15  # 15% target vol
        self.max_portfolio_var = 0.25  # 25% max variance

    def train(
        self,
        env,
        total_timesteps: int = 100000,
        eval_env=None,
        save_path: str = "models/hedging",
        **kwargs
    ):
        """
        Train the hedging agent using SAC

        Args:
            env: Trading environment
            total_timesteps: Total training steps
            eval_env: Evaluation environment
            save_path: Path to save checkpoints
        """
        logger.info(f"Training {self.name} with SAC for {total_timesteps} timesteps...")

        # Wrap environment
        vec_env = DummyVecEnv([lambda: env])

        # Initialize SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            ent_coef=self.ent_coef,
            verbose=1,
            tensorboard_log=f"./logs/{self.name.replace(' ', '_')}",
            **kwargs
        )

        # Setup callbacks
        callbacks = []

        # Checkpoint callback
        save_path_obj = Path(save_path)
        save_path_obj.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(save_path_obj),
            name_prefix="hedging_sac"
        )
        callbacks.append(checkpoint_callback)

        # Eval callback
        if eval_env is not None:
            eval_vec_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=str(save_path_obj / "best_model"),
                log_path=str(save_path_obj / "eval_logs"),
                eval_freq=5000,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        logger.info(f"{self.name} training completed!")

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action using trained SAC model

        Args:
            observation: Current state
            deterministic: Use deterministic policy

        Returns:
            action: Trading action
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        action, _ = self.model.predict(observation, deterministic=deterministic)

        return action

    def assess_risk(self, observation: np.ndarray) -> dict:
        """
        Assess current portfolio risk

        Args:
            observation: Current state

        Returns:
            Risk assessment dictionary
        """
        # Extract risk-related features from observation
        # This is simplified - real implementation would be more comprehensive

        risk_assessment = {
            'volatility': 0.0,
            'var_95': 0.0,  # Value at Risk (95%)
            'cvar_95': 0.0,  # Conditional VaR
            'portfolio_beta': 0.0,
            'correlation_risk': 0.0,
            'tail_risk': 'LOW',
            'recommendation': ''
        }

        # Placeholder calculations
        # In production, these would be calculated from actual observation features

        # Estimate current volatility (simplified)
        # observation[2] is value_change in our environment
        if len(observation) > 2:
            recent_change = observation[2]
            estimated_vol = abs(recent_change) * np.sqrt(252)
            risk_assessment['volatility'] = estimated_vol

            # Determine risk level
            if estimated_vol > self.max_portfolio_var:
                risk_assessment['tail_risk'] = 'HIGH'
                risk_assessment['recommendation'] = 'HEDGE: Increase protective positions'
            elif estimated_vol > self.target_volatility:
                risk_assessment['tail_risk'] = 'MEDIUM'
                risk_assessment['recommendation'] = 'MONITOR: Consider reducing exposure'
            else:
                risk_assessment['tail_risk'] = 'LOW'
                risk_assessment['recommendation'] = 'NORMAL: Maintain current hedges'

        return risk_assessment

    def get_explanation(self, observation: np.ndarray, action: np.ndarray) -> str:
        """
        Generate explanation for the action

        Args:
            observation: Current state
            action: Action taken

        Returns:
            Human-readable explanation
        """
        risk_info = self.assess_risk(observation)

        explanation = f"""{self.name} Decision:

Current Risk Assessment:
  Estimated Volatility: {risk_info['volatility']:.2%}
  Target Volatility: {self.target_volatility:.2%}
  Tail Risk Level: {risk_info['tail_risk']}

Hedging Action:
"""

        # Interpret actions
        n_assets = len(action)
        for i, act in enumerate(action):
            if abs(act) < 0.01:
                action_str = "No change"
            elif act < 0:  # Reducing exposure
                action_str = f"Reduce exposure by {abs(act)*100:.1f}%"
            else:  # Increasing hedges
                action_str = f"Add hedge position: {act*100:.1f}%"

            explanation += f"\n  Asset {i}: {action_str}"

        explanation += f"\n\nRationale: {risk_info['recommendation']}"
        explanation += f"\nStrategy: Protecting portfolio from {risk_info['tail_risk'].lower()} volatility scenario."

        return explanation

    def calculate_hedge_ratio(self, portfolio_beta: float, target_beta: float = 0.5) -> float:
        """
        Calculate optimal hedge ratio

        Args:
            portfolio_beta: Current portfolio beta
            target_beta: Desired beta

        Returns:
            Hedge ratio
        """
        if portfolio_beta == 0:
            return 0.0

        hedge_ratio = (portfolio_beta - target_beta) / portfolio_beta

        return max(0.0, min(1.0, hedge_ratio))  # Clip to [0, 1]

    def save(self, path: str):
        """Save model and metadata"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            model_path = save_path / "hedging_sac_model"
            self.model.save(model_path)
            logger.info(f"SAC model saved to {model_path}")

        super().save(path)

    def load(self, path: str):
        """Load model and metadata"""
        load_path = Path(path)

        # Load SAC model
        model_path = load_path / "hedging_sac_model.zip"
        if model_path.exists():
            self.model = SAC.load(model_path)
            logger.info(f"SAC model loaded from {model_path}")

        super().load(path)
