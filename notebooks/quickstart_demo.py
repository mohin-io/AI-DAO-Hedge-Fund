"""
Quick Start Demo - AI DAO Hedge Fund
This script demonstrates the basic usage of the system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Import our modules
from environment.trading_env import TradingEnvironment
from environment.data_loader import generate_synthetic_data
from agents.momentum_agent import MomentumAgent
from agents.arbitrage_agent import ArbitrageAgent
from agents.hedging_agent import HedgingAgent
from agents.multi_agent_coordinator import MultiAgentCoordinator
from utils.visualization import PerformanceVisualizer

print("="*70)
print("AI DAO Hedge Fund - Quick Start Demo")
print("="*70)

# Step 1: Generate sample data
print("\n[Step 1] Generating synthetic market data...")
data = generate_synthetic_data(n_assets=3, n_days=500, seed=42)
print(f"Generated {len(data)} days of data for {3} assets")
print(f"\nData preview:\n{data.head()}")

# Step 2: Create trading environment
print("\n[Step 2] Creating trading environment...")
env = TradingEnvironment(
    data=data,
    initial_balance=100000,
    action_type='continuous'
)
print(f"Environment created with:")
print(f"  - Initial balance: ${env.initial_balance:,}")
print(f"  - Observation space: {env.observation_space.shape}")
print(f"  - Action space: {env.action_space.shape}")

# Step 3: Initialize agents (without training for demo)
print("\n[Step 3] Initializing AI agents...")
agents = [
    MomentumAgent(agent_id=0),
    ArbitrageAgent(agent_id=1),
    HedgingAgent(agent_id=2)
]

for agent in agents:
    print(f"  ✓ {agent.name} ({agent.strategy})")

# Step 4: Create multi-agent coordinator
print("\n[Step 4] Creating multi-agent coordinator...")
coordinator = MultiAgentCoordinator(
    agents=agents,
    ensemble_method='weighted_voting'
)
print(f"Coordinator initialized with {len(coordinator.agents)} agents")
print(f"Initial weights: {coordinator.weights}")

# Step 5: Demonstrate prediction (without trained models, this is just structure demo)
print("\n[Step 5] Demonstrating system structure...")
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial portfolio value: ${info['portfolio_value']:,.2f}")

# Detect market regime
regime = coordinator.detect_market_regime(obs)
print(f"\nDetected market regime: {regime}")

# Step 6: Show explainability structure
print("\n[Step 6] Explainability demonstration...")
print("\nExample explanation (structure):")
print("""
Momentum Agent Decision:
  Feature: AAPL_RSI = 28.5 (Oversold)
  SHAP Value: +0.42 (Strong BUY signal)

  Feature: AAPL_MACD = 0.015 (Bullish crossover)
  SHAP Value: +0.31

  Overall Impact: +0.73 → BUY
  Confidence: 85%
""")

# Step 7: Show performance tracking
print("\n[Step 7] Performance tracking capabilities...")
print("\nAgent metrics tracked:")
metrics = agents[0].get_metrics()
for key, value in metrics.items():
    print(f"  - {key}: {value}")

# Step 8: Visualization capabilities
print("\n[Step 8] Visualization capabilities...")
visualizer = PerformanceVisualizer(output_dir="notebooks/demo_plots")
print("Available visualizations:")
print("  - Cumulative returns comparison")
print("  - Drawdown analysis")
print("  - Agent performance comparison")
print("  - Agent allocation over time")
print("  - Risk metrics dashboard")

# Step 9: Blockchain integration (mock)
print("\n[Step 9] Blockchain integration...")
from utils.blockchain_interface import MockBlockchainInterface

blockchain = MockBlockchainInterface()

# Create a proposal
tx_hash = blockchain.create_proposal(
    "mock_key",
    "Enable new momentum agent with improved signals",
    0,  # ENABLE_AGENT
    b""
)
print(f"DAO Proposal created: {tx_hash}")

# Record a trade
tx_hash = blockchain.record_trade("mock_key", 0, 250)  # 2.5% profit
print(f"Trade recorded on-chain: {tx_hash}")

# Get performance
perf = blockchain.get_agent_performance(0)
print(f"\nAgent performance from blockchain:")
print(f"  - Name: {perf['name']}")
print(f"  - Total trades: {perf['total_trades']}")
print(f"  - Total PnL: {perf['total_pnl']} bps")

# Summary
print("\n" + "="*70)
print("Quick Start Demo Complete!")
print("="*70)
print("\nNext steps:")
print("  1. Train agents: python simulations/backtest/run_multi_agent_training.py")
print("  2. View results: check simulations/plots/")
print("  3. Deploy contracts: cd contracts && hardhat deploy")
print("  4. Explore docs: see docs/PLAN.md for detailed roadmap")
print("\nFor full documentation: https://github.com/mohin-io/AI-DAO-Hedge-Fund")
print("="*70)
