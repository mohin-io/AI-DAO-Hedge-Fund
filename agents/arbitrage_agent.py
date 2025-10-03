"""
Arbitrage Trading Agent
Uses DQN algorithm to exploit price differences
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ArbitrageAgent(BaseAgent):
    """
    Arbitrage Trading Strategy Agent

    Strategy:
        - Identifies price discrepancies across assets
        - Exploits mean-reversion opportunities
        - Quick entry/exit (high frequency trading style)
        - Monitors correlation breakdowns

    Algorithm: DQN (Deep Q-Network)
        - Good for discrete action spaces
        - Efficient for quick decision-making
        - Experience replay helps with rare arbitrage events
    """

    def __init__(self, name: str = "Arbitrage Hunter", agent_id: Optional[int] = None):
        super().__init__(
            name=name,
            strategy="Statistical Arbitrage - Mean Reversion",
            agent_id=agent_id
        )

        # DQN specific parameters
        self.learning_rate = 1e-4
        self.buffer_size = 100000
        self.batch_size = 32
        self.gamma = 0.95
        self.exploration_fraction = 0.3
        self.exploration_final_eps = 0.05

    def train(
        self,
        env,
        total_timesteps: int = 100000,
        eval_env=None,
        save_path: str = "models/arbitrage",
        **kwargs
    ):
        """
        Train the arbitrage agent using DQN

        Args:
            env: Trading environment (should use discrete actions)
            total_timesteps: Total training steps
            eval_env: Evaluation environment
            save_path: Path to save checkpoints
        """
        logger.info(f"Training {self.name} with DQN for {total_timesteps} timesteps...")

        # Wrap environment
        vec_env = DummyVecEnv([lambda: env])

        # Initialize DQN model
        self.model = DQN(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            exploration_fraction=self.exploration_fraction,
            exploration_final_eps=self.exploration_final_eps,
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
            name_prefix="arbitrage_dqn"
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
        Predict action using trained DQN model

        Args:
            observation: Current state
            deterministic: Use deterministic policy (exploit vs explore)

        Returns:
            action: Trading action
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        action, _ = self.model.predict(observation, deterministic=deterministic)

        return action

    def detect_arbitrage_opportunity(self, observation: np.ndarray) -> dict:
        """
        Analyze observation for arbitrage opportunities

        Returns:
            dict with opportunity details
        """
        # Extract price correlation and spreads from observation
        # This is a simplified version - real implementation would be more complex

        opportunity = {
            'detected': False,
            'type': None,
            'assets': [],
            'expected_profit': 0.0,
            'confidence': 0.0
        }

        # Example: Check if prices have diverged from historical correlation
        # (In production, this would analyze the actual observation features)

        # Placeholder logic
        threshold = 0.02  # 2% spread threshold

        # Simplified arbitrage detection
        # In reality, would analyze observation to find:
        # 1. Cross-exchange arbitrage
        # 2. Statistical arbitrage (pairs trading)
        # 3. Triangular arbitrage

        return opportunity

    def get_explanation(self, observation: np.ndarray, action: np.ndarray) -> str:
        """
        Generate explanation for the action

        Args:
            observation: Current state
            action: Action taken

        Returns:
            Human-readable explanation
        """
        opportunity = self.detect_arbitrage_opportunity(observation)

        if opportunity['detected']:
            explanation = f"""{self.name} Decision:
Type: {opportunity['type']}
Assets: {', '.join(map(str, opportunity['assets']))}
Expected Profit: {opportunity['expected_profit']:.2%}
Confidence: {opportunity['confidence']:.2%}
Action: {action}
Rationale: Detected {opportunity['type']} opportunity with favorable risk/reward ratio.
"""
        else:
            explanation = f"""{self.name} Decision:
No arbitrage opportunity detected.
Action: HOLD (waiting for market inefficiency)
Rationale: Current spreads are too narrow or risks are too high.
"""

        return explanation

    def save(self, path: str):
        """Save model and metadata"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            model_path = save_path / "arbitrage_dqn_model"
            self.model.save(model_path)
            logger.info(f"DQN model saved to {model_path}")

        super().save(path)

    def load(self, path: str):
        """Load model and metadata"""
        load_path = Path(path)

        # Load DQN model
        model_path = load_path / "arbitrage_dqn_model.zip"
        if model_path.exists():
            self.model = DQN.load(model_path)
            logger.info(f"DQN model loaded from {model_path}")

        super().load(path)
