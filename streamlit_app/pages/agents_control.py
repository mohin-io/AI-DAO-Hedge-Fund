"""
AI Agents Control - Monitor and control individual AI trading agents
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def render():
    """Render the AI agents control page"""

    st.title("🤖 AI Agents Control Center")
    st.markdown("Monitor, control, and configure individual AI trading agents")

    # Agent Status Overview
    st.markdown("### 📊 Agent Status Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;">
            <h3 style="margin: 0;">📈 Momentum Agent</h3>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Algorithm: PPO</p>
            <h4 style="margin: 0.5rem 0;">Status: 🟢 ACTIVE</h4>
            <p style="margin: 0; font-size: 0.9rem;">P&L: +$42,567</p>
            <p style="margin: 0; font-size: 0.9rem;">Win Rate: 62.3%</p>
            <p style="margin: 0; font-size: 0.9rem;">Active Positions: 8</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white;">
            <h3 style="margin: 0;">💹 Arbitrage Agent</h3>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Algorithm: DQN</p>
            <h4 style="margin: 0.5rem 0;">Status: 🟢 ACTIVE</h4>
            <p style="margin: 0; font-size: 0.9rem;">P&L: +$28,934</p>
            <p style="margin: 0; font-size: 0.9rem;">Win Rate: 68.7%</p>
            <p style="margin: 0; font-size: 0.9rem;">Active Positions: 5</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 10px; color: white;">
            <h3 style="margin: 0;">🛡️ Hedging Agent</h3>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Algorithm: SAC</p>
            <h4 style="margin: 0.5rem 0;">Status: 🟢 ACTIVE</h4>
            <p style="margin: 0; font-size: 0.9rem;">P&L: +$15,890</p>
            <p style="margin: 0; font-size: 0.9rem;">Win Rate: 54.2%</p>
            <p style="margin: 0; font-size: 0.9rem;">Active Positions: 12</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Agent Selection for Detailed View
    st.markdown("### 🔍 Agent Details")

    selected_agent = st.selectbox(
        "Select Agent for Detailed Analysis",
        ["📈 Momentum Agent (PPO)", "💹 Arbitrage Agent (DQN)", "🛡️ Hedging Agent (SAC)"]
    )

    # Agent-specific configurations
    agent_configs = {
        "📈 Momentum Agent (PPO)": {
            "algorithm": "PPO (Proximal Policy Optimization)",
            "strategy": "Trend Following",
            "indicators": ["RSI", "MACD", "SMA 20/50", "Bollinger Bands"],
            "model_path": "models/momentum_agent.zip",
            "training_steps": "500,000",
            "last_updated": "2025-10-03 14:32:00",
            "performance": {
                "total_return": "+42.5%",
                "sharpe": "1.87",
                "max_dd": "-15.7%",
                "avg_trade": "+$534"
            }
        },
        "💹 Arbitrage Agent (DQN)": {
            "algorithm": "DQN (Deep Q-Network)",
            "strategy": "Statistical Arbitrage & Mean Reversion",
            "indicators": ["Spread Analysis", "Correlation", "Cointegration"],
            "model_path": "models/arbitrage_agent.zip",
            "training_steps": "500,000",
            "last_updated": "2025-10-03 14:30:00",
            "performance": {
                "total_return": "+28.9%",
                "sharpe": "1.52",
                "max_dd": "-8.4%",
                "avg_trade": "+$289"
            }
        },
        "🛡️ Hedging Agent (SAC)": {
            "algorithm": "SAC (Soft Actor-Critic)",
            "strategy": "Risk Management & Portfolio Protection",
            "indicators": ["VaR 95%", "CVaR", "Beta", "Volatility"],
            "model_path": "models/hedging_agent.zip",
            "training_steps": "500,000",
            "last_updated": "2025-10-03 14:28:00",
            "performance": {
                "total_return": "+15.9%",
                "sharpe": "1.38",
                "max_dd": "-9.2%",
                "avg_trade": "+$198"
            }
        }
    }

    config = agent_configs[selected_agent]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"#### Agent Configuration")

        st.markdown(f"""
        **Algorithm:** {config['algorithm']}

        **Strategy:** {config['strategy']}

        **Technical Indicators:**
        {', '.join(['`' + ind + '`' for ind in config['indicators']])}

        **Model Path:** `{config['model_path']}`

        **Training Steps:** {config['training_steps']}

        **Last Updated:** {config['last_updated']}
        """)

        # Performance metrics
        st.markdown("#### Performance Metrics")

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

        with perf_col1:
            st.metric("Total Return", config['performance']['total_return'])

        with perf_col2:
            st.metric("Sharpe Ratio", config['performance']['sharpe'])

        with perf_col3:
            st.metric("Max Drawdown", config['performance']['max_dd'])

        with perf_col4:
            st.metric("Avg Trade", config['performance']['avg_trade'])

    with col2:
        st.markdown("#### Agent Controls")

        agent_status = st.radio(
            "Agent Status",
            ["🟢 Active", "🟡 Paused", "🔴 Stopped"],
            index=0
        )

        st.markdown("---")

        st.markdown("**Risk Parameters**")

        max_position = st.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )

        stop_loss = st.slider(
            "Stop Loss (%)",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

        st.markdown("---")

        if st.button("💾 Save Configuration", use_container_width=True):
            st.success("Configuration saved!")

        if st.button("🔄 Retrain Agent", use_container_width=True):
            st.info("Retraining initiated...")

        if st.button("📥 Download Model", use_container_width=True):
            st.success("Model downloaded!")

    # Agent Performance Chart
    st.markdown("---")
    st.markdown("### 📊 Agent Performance Over Time")

    # Generate performance data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    np.random.seed(hash(selected_agent) % 2**32)
    returns = np.random.normal(0.002, 0.018, len(dates))
    cumulative_pnl = np.cumsum(returns) * 100000

    fig_pnl = go.Figure()

    fig_pnl.add_trace(go.Scatter(
        x=dates,
        y=cumulative_pnl,
        name='Cumulative P&L',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))

    fig_pnl.update_layout(
        height=350,
        yaxis_title='Cumulative P&L ($)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    st.plotly_chart(fig_pnl, use_container_width=True)

    # Action Distribution
    st.markdown("### 🎯 Action Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Action counts
        actions_df = pd.DataFrame({
            'Action': ['BUY', 'SELL', 'HOLD', 'CLOSE'],
            'Count': [234, 189, 456, 198]
        })

        fig_actions = go.Figure(data=[
            go.Bar(
                x=actions_df['Action'],
                y=actions_df['Count'],
                marker_color=['#00ff00', '#ff0000', '#ffff00', '#00ffff'],
                text=actions_df['Count'],
                textposition='outside'
            )
        ])

        fig_actions.update_layout(
            height=350,
            yaxis_title='Number of Actions',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )

        st.plotly_chart(fig_actions, use_container_width=True)

    with col2:
        # Trade duration distribution
        durations = np.random.exponential(scale=4, size=1000)
        durations = durations[durations < 20]  # Cap at 20 days

        fig_duration = go.Figure(data=[
            go.Histogram(
                x=durations,
                nbinsx=20,
                marker_color='#764ba2',
                opacity=0.7
            )
        ])

        fig_duration.update_layout(
            height=350,
            title='Trade Duration Distribution',
            xaxis_title='Duration (days)',
            yaxis_title='Frequency',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )

        st.plotly_chart(fig_duration, use_container_width=True)

    # Recent Actions Table
    st.markdown("### 📋 Recent Agent Actions")

    recent_actions = pd.DataFrame({
        'Timestamp': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=18),
            datetime.now() - timedelta(minutes=42),
            datetime.now() - timedelta(hours=1, minutes=15),
            datetime.now() - timedelta(hours=2, minutes=30),
        ],
        'Action': ['BUY', 'SELL', 'HOLD', 'BUY', 'CLOSE'],
        'Asset': ['AAPL', 'MSFT', '-', 'GOOGL', 'TSLA'],
        'Quantity': [100, 75, 0, 50, 120],
        'Price': ['$182.45', '$415.30', '-', '$138.20', '$245.80'],
        'Confidence': [0.87, 0.72, 0.95, 0.83, 0.78],
        'Reason': [
            'RSI oversold + MACD crossover',
            'Take profit at resistance',
            'Waiting for confirmation',
            'Bullish trend continuation',
            'Stop loss triggered'
        ]
    })

    recent_actions['Timestamp'] = recent_actions['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(
        recent_actions,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Confidence': st.column_config.ProgressColumn(
                'Confidence',
                format='%.0f%%',
                min_value=0,
                max_value=1,
            )
        }
    )

    # Training Metrics
    st.markdown("---")
    st.markdown("### 📚 Training Metrics")

    col1, col2 = st.columns(2)

    with col1:
        # Reward curve
        episodes = np.arange(1, 501)
        rewards = 100 + 50 * np.log(episodes) + np.random.normal(0, 10, len(episodes))

        fig_rewards = go.Figure()

        fig_rewards.add_trace(go.Scatter(
            x=episodes,
            y=rewards,
            mode='lines',
            name='Episode Reward',
            line=dict(color='rgba(102, 126, 234, 0.3)', width=1)
        ))

        # Add moving average
        window = 20
        ma_rewards = pd.Series(rewards).rolling(window=window).mean()

        fig_rewards.add_trace(go.Scatter(
            x=episodes,
            y=ma_rewards,
            mode='lines',
            name=f'{window}-Episode MA',
            line=dict(color='#667eea', width=3)
        ))

        fig_rewards.update_layout(
            height=350,
            title='Training Reward Curve',
            xaxis_title='Episode',
            yaxis_title='Reward',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )

        st.plotly_chart(fig_rewards, use_container_width=True)

    with col2:
        # Loss curve
        loss_values = 1000 * np.exp(-episodes/100) + np.random.normal(0, 20, len(episodes))
        loss_values = np.maximum(loss_values, 0)  # Ensure non-negative

        fig_loss = go.Figure()

        fig_loss.add_trace(go.Scatter(
            x=episodes,
            y=loss_values,
            mode='lines',
            name='Loss',
            line=dict(color='rgba(245, 87, 108, 0.3)', width=1)
        ))

        # Add moving average
        ma_loss = pd.Series(loss_values).rolling(window=window).mean()

        fig_loss.add_trace(go.Scatter(
            x=episodes,
            y=ma_loss,
            mode='lines',
            name=f'{window}-Episode MA',
            line=dict(color='#f5576c', width=3)
        ))

        fig_loss.update_layout(
            height=350,
            title='Training Loss Curve',
            xaxis_title='Episode',
            yaxis_title='Loss',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )

        st.plotly_chart(fig_loss, use_container_width=True)

    # Hyperparameters
    st.markdown("### ⚙️ Hyperparameters")

    with st.expander("View/Edit Hyperparameters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            learning_rate = st.number_input("Learning Rate", value=0.0003, format="%.6f", step=0.0001)
            gamma = st.slider("Gamma (Discount Factor)", 0.0, 1.0, 0.99, 0.01)
            batch_size = st.selectbox("Batch Size", [32, 64, 128, 256, 512], index=2)

        with col2:
            n_steps = st.number_input("N Steps", value=2048, step=128)
            ent_coef = st.number_input("Entropy Coefficient", value=0.01, format="%.4f", step=0.001)
            vf_coef = st.number_input("Value Function Coefficient", value=0.5, format="%.2f", step=0.1)

        with col3:
            max_grad_norm = st.number_input("Max Gradient Norm", value=0.5, format="%.2f", step=0.1)
            gae_lambda = st.slider("GAE Lambda", 0.0, 1.0, 0.95, 0.01)
            clip_range = st.number_input("Clip Range", value=0.2, format="%.2f", step=0.05)

        if st.button("💾 Save Hyperparameters"):
            st.success("Hyperparameters saved! Retrain the agent to apply changes.")
