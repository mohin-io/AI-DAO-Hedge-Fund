"""
Multi-Agent Training Script
Trains all agents and evaluates ensemble performance
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import logging
from datetime import datetime

from environment.trading_env import TradingEnvironment
from environment.data_loader import generate_synthetic_data, MarketDataLoader
from agents.momentum_agent import MomentumAgent
from agents.arbitrage_agent import ArbitrageAgent
from agents.hedging_agent import HedgingAgent
from agents.multi_agent_coordinator import MultiAgentCoordinator
from utils.visualization import PerformanceVisualizer
from explainability.shap_analyzer import SHAPAnalyzer, generate_default_feature_names

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""

    logger.info("="*70)
    logger.info("AI DAO HEDGE FUND - Multi-Agent Training Pipeline")
    logger.info("="*70)

    # 1. Generate/Load Data
    logger.info("\n[1/6] Loading market data...")

    # Use synthetic data for demo (replace with real data loader for production)
    use_synthetic = True

    if use_synthetic:
        logger.info("Generating synthetic market data...")
        data = generate_synthetic_data(n_assets=3, n_days=1000, seed=42)
    else:
        logger.info("Loading real market data...")
        loader = MarketDataLoader()
        data = loader.load_data()
        data = loader.prepare_for_environment(data)

    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    logger.info(f"Training data: {len(train_data)} days")
    logger.info(f"Testing data: {len(test_data)} days")

    # 2. Create Environments
    logger.info("\n[2/6] Creating trading environments...")

    train_env = TradingEnvironment(
        data=train_data,
        initial_balance=100000,
        action_type='continuous'
    )

    test_env = TradingEnvironment(
        data=test_data,
        initial_balance=100000,
        action_type='continuous'
    )

    logger.info(f"Observation space: {train_env.observation_space.shape}")
    logger.info(f"Action space: {train_env.action_space.shape}")

    # 3. Train Individual Agents
    logger.info("\n[3/6] Training individual agents...")

    # Initialize agents
    momentum_agent = MomentumAgent(agent_id=0)
    arbitrage_agent = ArbitrageAgent(agent_id=1)
    hedging_agent = HedgingAgent(agent_id=2)

    agents = [momentum_agent, arbitrage_agent, hedging_agent]

    # Training parameters
    timesteps_per_agent = 50000  # Reduced for demo, use 500k+ for production

    for agent in agents:
        logger.info(f"\nTraining {agent.name}...")
        try:
            agent.train(
                env=train_env,
                total_timesteps=timesteps_per_agent,
                eval_env=test_env,
                save_path=f"models/{agent.name.replace(' ', '_').lower()}"
            )
            logger.info(f"{agent.name} training completed successfully!")
        except Exception as e:
            logger.error(f"Failed to train {agent.name}: {e}")

    # 4. Evaluate Individual Agents
    logger.info("\n[4/6] Evaluating individual agents...")

    agent_results = {}

    for agent in agents:
        logger.info(f"\nEvaluating {agent.name}...")

        obs, info = test_env.reset()
        done = False
        total_reward = 0
        steps = 0

        portfolio_values = [test_env.initial_balance]

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            total_reward += reward
            portfolio_values.append(info['portfolio_value'])
            steps += 1

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        max_dd = np.max((np.maximum.accumulate(portfolio_values) - portfolio_values) /
                       np.maximum.accumulate(portfolio_values))
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        agent_results[agent.name] = {
            'total_reward': total_reward,
            'final_value': portfolio_values[-1],
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'steps': steps,
            'portfolio_values': portfolio_values
        }

        logger.info(f"{agent.name} Results:")
        logger.info(f"  Total Return: {total_return*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_dd*100:.2f}%")

    # 5. Train and Evaluate Ensemble
    logger.info("\n[5/6] Evaluating multi-agent ensemble...")

    coordinator = MultiAgentCoordinator(
        agents=agents,
        ensemble_method='weighted_voting'
    )

    obs, info = test_env.reset()
    done = False
    total_reward = 0
    steps = 0

    ensemble_portfolio_values = [test_env.initial_balance]
    allocation_history = []

    while not done:
        # Get ensemble action
        action, agent_actions = coordinator.predict(obs)

        # Execute action
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        total_reward += reward
        ensemble_portfolio_values.append(info['portfolio_value'])

        # Track allocations
        allocations = coordinator.get_agent_allocations()
        allocation_history.append({
            'step': steps,
            **{name: alloc['weight'] for name, alloc in allocations.items()}
        })

        steps += 1

    # Calculate ensemble metrics
    ensemble_portfolio_values = np.array(ensemble_portfolio_values)
    ensemble_returns = np.diff(ensemble_portfolio_values) / ensemble_portfolio_values[:-1]

    ensemble_sharpe = np.mean(ensemble_returns) / (np.std(ensemble_returns) + 1e-6) * np.sqrt(252)
    ensemble_max_dd = np.max((np.maximum.accumulate(ensemble_portfolio_values) - ensemble_portfolio_values) /
                             np.maximum.accumulate(ensemble_portfolio_values))
    ensemble_total_return = (ensemble_portfolio_values[-1] / ensemble_portfolio_values[0]) - 1

    logger.info(f"\nEnsemble Results:")
    logger.info(f"  Total Return: {ensemble_total_return*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {ensemble_sharpe:.2f}")
    logger.info(f"  Max Drawdown: {ensemble_max_dd*100:.2f}%")

    agent_results['Ensemble'] = {
        'total_return': ensemble_total_return,
        'sharpe_ratio': ensemble_sharpe,
        'max_drawdown': ensemble_max_dd,
        'portfolio_values': ensemble_portfolio_values
    }

    # 6. Visualize Results
    logger.info("\n[6/6] Creating visualizations...")

    visualizer = PerformanceVisualizer(output_dir="simulations/plots")

    # Cumulative returns comparison
    portfolio_df = pd.DataFrame({
        name: results['portfolio_values']
        for name, results in agent_results.items()
    })

    visualizer.plot_cumulative_returns(
        portfolio_df,
        title="Multi-Agent Performance Comparison",
        save_name="cumulative_returns.png"
    )

    # Agent comparison
    comparison_metrics = {
        name: {
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'total_return': results.get('total_return', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'win_rate': 0.55  # Placeholder
        }
        for name, results in agent_results.items()
    }

    visualizer.plot_agent_comparison(
        comparison_metrics,
        save_name="agent_comparison.png"
    )

    # Agent allocation over time
    allocation_df = pd.DataFrame(allocation_history).set_index('step')
    visualizer.plot_agent_allocation_over_time(
        allocation_df,
        save_name="agent_allocation.png"
    )

    # Dashboard summary
    ensemble_values = pd.Series(ensemble_portfolio_values)
    visualizer.create_dashboard_summary(
        metrics=comparison_metrics['Ensemble'],
        portfolio_values=ensemble_values,
        save_name="dashboard_summary.png"
    )

    logger.info("\nAll visualizations saved to simulations/plots/")

    # Save results summary
    results_df = pd.DataFrame({
        name: {
            'Total Return (%)': results.get('total_return', 0) * 100,
            'Sharpe Ratio': results.get('sharpe_ratio', 0),
            'Max Drawdown (%)': results.get('max_drawdown', 0) * 100,
        }
        for name, results in agent_results.items()
    }).T

    results_path = Path("simulations/results/agent_comparison.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path)

    logger.info(f"\nResults summary saved to {results_path}")
    logger.info("\n" + "="*70)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*70)

    print("\n" + results_df.to_string())


if __name__ == "__main__":
    main()
