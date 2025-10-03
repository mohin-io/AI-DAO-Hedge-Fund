"""
Momentum Trading Agent
Uses PPO algorithm to follow market trends
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from pathlib import Path

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MomentumAgent(BaseAgent):
    """
    Momentum Trading Strategy Agent

    Strategy:
        - Follows trends using technical indicators (RSI, MACD, Moving Averages)
        - Buys assets showing upward momentum
        - Sells assets showing downward momentum
        - Holds positions during strong trends

    Algorithm: PPO (Proximal Policy Optimization)
        - Stable policy gradient method
        - Good for continuous action spaces
        - Handles non-stationary financial data well
    """

    def __init__(self, name: str = "Momentum Trader", agent_id: Optional[int] = None):
        super().__init__(
            name=name,
            strategy="Momentum Trading - RSI/MACD/MA-based",
            agent_id=agent_id
        )

        # PPO specific parameters
        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def train(
        self,
        env,
        total_timesteps: int = 100000,
        eval_env=None,
        save_path: str = "models/momentum",
        **kwargs
    ):
        """
        Train the momentum agent using PPO

        Args:
            env: Trading environment
            total_timesteps: Total training steps
            eval_env: Evaluation environment
            save_path: Path to save checkpoints
        """
        logger.info(f"Training {self.name} with PPO for {total_timesteps} timesteps...")

        # Wrap environment
        vec_env = DummyVecEnv([lambda: env])

        # Initialize PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
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
            name_prefix="momentum_ppo"
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
        Predict action using trained PPO model

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

    def get_explanation(self, observation: np.ndarray, action: np.ndarray) -> str:
        """
        Generate explanation for the action

        Args:
            observation: Current state
            action: Action taken

        Returns:
            Human-readable explanation
        """
        # Extract relevant features from observation
        # observation structure: [cash%, positions%, value_change, ...market features]

        explanations = []

        # Interpret action
        n_assets = len(action)
        for i, act in enumerate(action):
            if abs(act) < 0.01:  # Holding
                action_str = "HOLD"
                reason = "No strong momentum signal"
            elif act > 0:  # Buying
                action_str = f"BUY {abs(act)*100:.1f}%"

                # Check indicators from observation (simplified)
                # In real implementation, access actual indicator values
                reason = "Positive momentum: RSI oversold, MACD bullish crossover, price above MA"
            else:  # Selling
                action_str = f"SELL {abs(act)*100:.1f}%"
                reason = "Negative momentum: RSI overbought, MACD bearish crossover, price below MA"

            explanations.append(f"Asset {i}: {action_str} - {reason}")

        full_explanation = f"{self.name} Decision:\n" + "\n".join(explanations)

        return full_explanation

    def save(self, path: str):
        """Save model and metadata"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            model_path = save_path / "momentum_ppo_model"
            self.model.save(model_path)
            logger.info(f"PPO model saved to {model_path}")

        super().save(path)

    def load(self, path: str):
        """Load model and metadata"""
        load_path = Path(path)

        # Load PPO model
        model_path = load_path / "momentum_ppo_model.zip"
        if model_path.exists():
            self.model = PPO.load(model_path)
            logger.info(f"PPO model loaded from {model_path}")

        super().load(path)


# Make sure this import exists at the top
from typing import Optional
