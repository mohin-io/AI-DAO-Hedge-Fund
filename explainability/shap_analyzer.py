"""
SHAP-based Explainability for Trading Agents
Provides feature importance and decision explanations
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analyzer for agent decisions

    Provides:
    - Feature importance for each decision
    - Waterfall plots showing contribution
    - Summary plots across multiple decisions
    """

    def __init__(self, agent, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP analyzer

        Args:
            agent: Trading agent with a trained model
            feature_names: Names of features in observation
        """
        self.agent = agent
        self.feature_names = feature_names

        # Initialize explainer (lazy loading)
        self.explainer = None

        # Cache of explanations
        self.explanations = []

    def _initialize_explainer(self, background_data: np.ndarray):
        """
        Initialize SHAP explainer with background data

        Args:
            background_data: Sample of training data for baseline
        """
        try:
            # Use TreeExplainer for tree-based models, DeepExplainer for neural nets
            # For now, use KernelExplainer as it's model-agnostic
            logger.info("Initializing SHAP KernelExplainer...")

            def model_predict(x):
                """Wrapper function for model prediction"""
                predictions = []
                for observation in x:
                    action = self.agent.predict(observation, deterministic=True)
                    # Convert action to scalar for SHAP (use first action if multi-asset)
                    if isinstance(action, np.ndarray):
                        predictions.append(action[0] if len(action) > 0 else 0)
                    else:
                        predictions.append(action)
                return np.array(predictions)

            # Sample background data to reduce computation
            if len(background_data) > 100:
                indices = np.random.choice(len(background_data), 100, replace=False)
                background_sample = background_data[indices]
            else:
                background_sample = background_data

            self.explainer = shap.KernelExplainer(
                model_predict,
                background_sample
            )

            logger.info("SHAP explainer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None

    def explain_decision(
        self,
        observation: np.ndarray,
        background_data: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Explain a single decision

        Args:
            observation: Single observation to explain
            background_data: Background data for explainer initialization

        Returns:
            Dictionary with SHAP values and explanation
        """
        if self.explainer is None:
            if background_data is None:
                raise ValueError("Background data needed for first explanation")
            self._initialize_explainer(background_data)

        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(observation.reshape(1, -1))

            # Extract values (handle different SHAP value formats)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            if shap_values.ndim > 1:
                shap_values = shap_values[0]

            # Get feature importance ranking
            feature_importance = np.abs(shap_values)
            top_indices = np.argsort(feature_importance)[::-1]

            # Create explanation
            explanation = {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'top_features': top_indices[:5],  # Top 5 features
                'observation': observation
            }

            # Add to cache
            self.explanations.append(explanation)

            return explanation

        except Exception as e:
            logger.error(f"Failed to explain decision: {e}")
            return {
                'shap_values': None,
                'feature_importance': None,
                'error': str(e)
            }

    def plot_waterfall(
        self,
        explanation: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create waterfall plot for a single explanation

        Args:
            explanation: Explanation dict from explain_decision
            save_path: Path to save plot
            show: Whether to display plot
        """
        if explanation['shap_values'] is None:
            logger.warning("No SHAP values to plot")
            return

        try:
            # Create SHAP explanation object
            shap_exp = shap.Explanation(
                values=explanation['shap_values'],
                data=explanation['observation'],
                feature_names=self.feature_names
            )

            # Create waterfall plot
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap_exp, show=False)

            plt.title(f"SHAP Waterfall Plot - {self.agent.name}")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Waterfall plot saved to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {e}")

    def plot_summary(
        self,
        observations: np.ndarray,
        max_display: int = 10,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create summary plot across multiple observations

        Args:
            observations: Array of observations
            max_display: Number of features to display
            save_path: Path to save plot
            show: Whether to display plot
        """
        if self.explainer is None:
            logger.warning("Explainer not initialized")
            return

        try:
            # Calculate SHAP values for all observations
            logger.info(f"Calculating SHAP values for {len(observations)} observations...")
            shap_values = self.explainer.shap_values(observations)

            # Handle different formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                observations,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )

            plt.title(f"SHAP Summary Plot - {self.agent.name}")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Summary plot saved to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            logger.error(f"Failed to create summary plot: {e}")

    def get_text_explanation(self, explanation: Dict) -> str:
        """
        Generate text explanation from SHAP values

        Args:
            explanation: Explanation dict

        Returns:
            Human-readable text explanation
        """
        if explanation['shap_values'] is None:
            return "Unable to generate explanation"

        shap_values = explanation['shap_values']
        observation = explanation['observation']

        # Get top contributing features
        top_indices = explanation['top_features']

        text = f"\n{self.agent.name} - Decision Explanation\n"
        text += "=" * 50 + "\n\n"
        text += "Top Contributing Features:\n\n"

        for i, idx in enumerate(top_indices, 1):
            feature_name = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
            shap_val = shap_values[idx]
            obs_val = observation[idx]

            impact = "increases" if shap_val > 0 else "decreases"

            text += f"{i}. {feature_name}\n"
            text += f"   Value: {obs_val:.4f}\n"
            text += f"   SHAP: {shap_val:.4f} ({impact} action)\n\n"

        # Overall decision direction
        total_shap = np.sum(shap_values)
        text += f"Overall Impact: {total_shap:.4f}\n"
        text += f"Decision: {'BUY' if total_shap > 0 else 'SELL' if total_shap < 0 else 'HOLD'}\n"

        return text

    def generate_feature_importance_df(
        self,
        observations: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate feature importance dataframe

        Args:
            observations: Array of observations

        Returns:
            DataFrame with feature importance
        """
        if self.explainer is None:
            logger.warning("Explainer not initialized")
            return pd.DataFrame()

        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(observations)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            # Create dataframe
            feature_names = self.feature_names or [f"Feature_{i}" for i in range(len(mean_abs_shap))]

            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_abs_shap
            })

            df = df.sort_values('Importance', ascending=False)

            return df

        except Exception as e:
            logger.error(f"Failed to generate importance dataframe: {e}")
            return pd.DataFrame()


def generate_default_feature_names(n_assets: int = 3) -> List[str]:
    """
    Generate default feature names for trading environment

    Args:
        n_assets: Number of assets

    Returns:
        List of feature names
    """
    names = ['Cash %', 'Positions %', 'Portfolio Change']

    # Add asset-specific features
    for i in range(n_assets):
        asset_prefix = f'Asset_{i}'
        names.extend([
            f'{asset_prefix}_Price',
            f'{asset_prefix}_Volume',
            f'{asset_prefix}_RSI',
            f'{asset_prefix}_MACD',
            f'{asset_prefix}_MA5',
            f'{asset_prefix}_MA10',
            f'{asset_prefix}_Stochastic',
            f'{asset_prefix}_Volatility',
            f'{asset_prefix}_Feature8',  # Padding
            f'{asset_prefix}_Feature9'   # Padding
        ])

    return names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage (requires a trained agent)
    print("SHAP Analyzer module loaded successfully")
    print("Example feature names:", generate_default_feature_names(3)[:10])
