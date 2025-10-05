"""
Home page for Decentralized Autonomous Hedge Fund AI DAO Streamlit app
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

def render():
    """Render the home page"""

    # Main header
    st.markdown('<h1 class="main-header">🤖⛓️📈 Decentralized Autonomous Hedge Fund AI DAO</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Decentralized Autonomous Hedge Fund powered by Multi-Agent RL and Blockchain DAO</p>', unsafe_allow_html=True)

    # Quick stats with consistent sizing
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card" style="padding: 1.8rem; min-height: 150px; display: flex; flex-direction: column; justify-content: space-between;">
            <h3 style="margin: 0; font-size: 0.9rem; opacity: 0.7;">Portfolio Value</h3>
            <h2 style="margin: 0.8rem 0; font-size: 2rem; font-weight: 800;">$1,247,893</h2>
            <p style="margin: 0; opacity: 0.8; font-size: 0.85rem;">+34.2% Total Return</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="padding: 1.8rem; min-height: 150px; display: flex; flex-direction: column; justify-content: space-between;">
            <h3 style="margin: 0; font-size: 0.9rem; opacity: 0.7;">Sharpe Ratio</h3>
            <h2 style="margin: 0.8rem 0; font-size: 2rem; font-weight: 800;">2.14</h2>
            <p style="margin: 0; opacity: 0.8; font-size: 0.85rem;">Institutional Grade</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="padding: 1.8rem; min-height: 150px; display: flex; flex-direction: column; justify-content: space-between;">
            <h3 style="margin: 0; font-size: 0.9rem; opacity: 0.7;">Max Drawdown</h3>
            <h2 style="margin: 0.8rem 0; font-size: 2rem; font-weight: 800;">-12.3%</h2>
            <p style="margin: 0; opacity: 0.8; font-size: 0.85rem;">Low Risk Profile</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card" style="padding: 1.8rem; min-height: 150px; display: flex; flex-direction: column; justify-content: space-between;">
            <h3 style="margin: 0; font-size: 0.9rem; opacity: 0.7;">Active Agents</h3>
            <h2 style="margin: 0.8rem 0; font-size: 2rem; font-weight: 800;">3/3</h2>
            <p style="margin: 0; opacity: 0.8; font-size: 0.85rem;">All Operational</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # System Architecture Overview
    st.header("🏗️ System Architecture")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Multi-Layer Architecture

        The AI DAO Hedge Fund operates on a sophisticated multi-layer architecture:

        **1. 🤖 AI Agent Layer**
        - **Momentum Agent (PPO)**: Trend-following strategies using RSI, MACD, and Bollinger Bands
        - **Arbitrage Agent (DQN)**: Mean-reversion and statistical arbitrage opportunities
        - **Hedging Agent (SAC)**: Risk management and portfolio protection

        **2. 🧠 Coordination Layer**
        - Market regime detection (Bull/Bear/Sideways/Volatile)
        - Dynamic agent weight allocation
        - Ensemble decision-making with weighted voting

        **3. ⛓️ Blockchain Layer**
        - DAOGovernance.sol: Proposal voting and execution
        - TreasuryManager.sol: Fund management and fee distribution
        - AgentRegistry.sol: Performance tracking and reputation scoring

        **4. 🔍 Explainability Layer**
        - SHAP analysis for every trade decision
        - Feature importance visualization
        - Risk attribution and transparency
        """)

    with col2:
        st.markdown("""
        ### Quick Actions
        """)

        if st.button("📊 View Live Portfolio", use_container_width=True):
            st.info("Navigate to 'Portfolio Dashboard' from the sidebar")

        if st.button("🤖 Control AI Agents", use_container_width=True):
            st.info("Navigate to 'AI Agents Control' from the sidebar")

        if st.button("⛓️ DAO Proposals", use_container_width=True):
            st.info("Navigate to 'DAO Governance' from the sidebar")

        if st.button("🎮 Run Simulation", use_container_width=True):
            st.info("Navigate to 'Trading Simulator' from the sidebar")

        st.markdown("---")

        st.markdown("""
        ### System Status

        - ✅ All agents operational
        - ✅ Blockchain connected
        - ✅ Real-time data feed active
        - ✅ Risk limits within bounds
        """)

    # Key Features
    st.header("✨ Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🧠 Multi-Agent AI

        - **PPO** for momentum trading
        - **DQN** for arbitrage
        - **SAC** for risk hedging
        - Dynamic regime detection
        - Ensemble decision-making
        """)

    with col2:
        st.markdown("""
        ### ⛓️ Blockchain DAO

        - Transparent governance
        - On-chain voting
        - Immutable audit trail
        - Smart contract execution
        - Community-driven strategy
        """)

    with col3:
        st.markdown("""
        ### 🔍 Explainability

        - SHAP analysis per trade
        - Feature importance ranking
        - Risk decomposition
        - Waterfall plots
        - Full transparency
        """)

    # Performance Comparison
    st.header("📈 Performance vs Benchmark")

    # Create comparison chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['AI DAO Fund', 'S&P 500'],
        y=[34.2, 18.6],
        name='Total Return (%)',
        marker_color=['#667eea', '#f5576c'],
        text=['+34.2%', '+18.6%'],
        textposition='outside'
    ))

    fig.update_layout(
        title="Total Return Comparison (2020-2025)",
        yaxis_title="Return (%)",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recent Activity
    st.header("📋 Recent Activity")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 AI Agent Activity")
        st.markdown("""
        - **Momentum Agent**: BUY AAPL (+2.3% position)
        - **Arbitrage Agent**: LONG MSFT / SHORT GOOGL (spread: 1.2%)
        - **Hedging Agent**: ADD SPY puts (VaR protection)
        - **Ensemble**: Overall BULLISH (65% confidence)
        """)

    with col2:
        st.subheader("⛓️ DAO Governance Activity")
        st.markdown("""
        - **Proposal #12**: Increase hedging allocation to 40%
          - Status: ✅ PASSED (78% approval)
        - **Proposal #13**: Add new DeFi yield farming strategy
          - Status: 🗳️ VOTING (2 days remaining)
        - **Active Voters**: 142 DAO members
        """)

    # Tech Stack
    st.header("🛠️ Technology Stack")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        **AI/ML**
        - PyTorch 2.0+
        - Stable-Baselines3
        - SHAP
        - Gymnasium
        """)

    with col2:
        st.markdown("""
        **Blockchain**
        - Solidity 0.8.20
        - Hardhat
        - Web3.py
        - OpenZeppelin
        """)

    with col3:
        st.markdown("""
        **Data & Finance**
        - yfinance
        - pandas-ta
        - NumPy/Pandas
        - Plotly
        """)

    with col4:
        st.markdown("""
        **Frontend**
        - Streamlit
        - Plotly
        - Altair
        - Matplotlib
        """)

    # Call to Action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h2>Ready to Explore?</h2>
        <p>Use the sidebar to navigate through different modules of the AI DAO Hedge Fund system.</p>
        <p><strong>Start with the Portfolio Dashboard to see real-time performance!</strong></p>
    </div>
    """, unsafe_allow_html=True)
