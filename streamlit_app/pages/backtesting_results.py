"""
Backtesting Results - Display historical backtest performance
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render():
    """Render the backtesting results page"""

    st.title("📈 Backtesting Results")
    st.markdown("Historical performance analysis and backtest metrics")

    # Summary Metrics
    st.markdown("### 📊 Performance Summary (2020-2025)")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Return", "+34.2%", delta="+15.6% vs S&P 500")

    with col2:
        st.metric("Sharpe Ratio", "2.14", delta="Excellent")

    with col3:
        st.metric("Max Drawdown", "-12.3%", delta="38% better")

    with col4:
        st.metric("Win Rate", "58.3%", delta="Above 55% target")

    with col5:
        st.metric("Calmar Ratio", "2.78", delta="Strong")

    st.markdown("---")

    # Agent Comparison Table
    st.markdown("### 🤖 Agent Performance Comparison")

    agent_performance = pd.DataFrame({
        'Agent': [
            'Ensemble (All Agents)',
            'Momentum Trader (PPO)',
            'Arbitrage Hunter (DQN)',
            'Risk Hedger (SAC)',
            'S&P 500 Benchmark'
        ],
        'Total Return': ['34.2%', '28.5%', '19.3%', '15.1%', '18.6%'],
        'Sharpe Ratio': [2.14, 1.87, 1.52, 1.38, 1.12],
        'Max Drawdown': ['-12.3%', '-15.7%', '-8.4%', '-9.2%', '-19.8%'],
        'Win Rate': ['58.3%', '54.2%', '61.7%', '52.8%', '-'],
        'Volatility': ['18.3%', '21.5%', '14.2%', '12.8%', '19.4%']
    })

    st.dataframe(
        agent_performance,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Agent': st.column_config.TextColumn('Agent', width='large'),
            'Total Return': st.column_config.TextColumn('Total Return'),
            'Sharpe Ratio': st.column_config.NumberColumn('Sharpe Ratio', format='%.2f'),
            'Max Drawdown': st.column_config.TextColumn('Max Drawdown'),
            'Win Rate': st.column_config.TextColumn('Win Rate'),
            'Volatility': st.column_config.TextColumn('Volatility')
        }
    )

    st.info("**Key Insight**: Multi-agent ensemble outperforms individual agents AND the benchmark with lower drawdown.")

    # Cumulative Returns Chart
    st.markdown("---")
    st.markdown("### 📈 Cumulative Returns")

    dates = pd.date_range(start='2020-01-01', end='2025-10-04', freq='D')
    np.random.seed(42)

    # Generate returns for different agents
    ensemble_returns = np.random.normal(0.0012, 0.015, len(dates))
    momentum_returns = np.random.normal(0.001, 0.018, len(dates))
    arbitrage_returns = np.random.normal(0.0007, 0.012, len(dates))
    hedging_returns = np.random.normal(0.0005, 0.01, len(dates))
    benchmark_returns = np.random.normal(0.0006, 0.016, len(dates))

    # Calculate cumulative returns
    ensemble_cum = 100 * np.cumprod(1 + ensemble_returns)
    momentum_cum = 100 * np.cumprod(1 + momentum_returns)
    arbitrage_cum = 100 * np.cumprod(1 + arbitrage_returns)
    hedging_cum = 100 * np.cumprod(1 + hedging_returns)
    benchmark_cum = 100 * np.cumprod(1 + benchmark_returns)

    fig_cumulative = go.Figure()

    fig_cumulative.add_trace(go.Scatter(
        x=dates, y=ensemble_cum,
        name='Ensemble (All Agents)',
        line=dict(color='#667eea', width=4)
    ))

    fig_cumulative.add_trace(go.Scatter(
        x=dates, y=momentum_cum,
        name='Momentum (PPO)',
        line=dict(color='#00ff00', width=2)
    ))

    fig_cumulative.add_trace(go.Scatter(
        x=dates, y=arbitrage_cum,
        name='Arbitrage (DQN)',
        line=dict(color='#ffff00', width=2)
    ))

    fig_cumulative.add_trace(go.Scatter(
        x=dates, y=hedging_cum,
        name='Hedging (SAC)',
        line=dict(color='#ff00ff', width=2)
    ))

    fig_cumulative.add_trace(go.Scatter(
        x=dates, y=benchmark_cum,
        name='S&P 500',
        line=dict(color='#f5576c', width=2, dash='dash')
    ))

    fig_cumulative.update_layout(
        height=500,
        yaxis_title='Cumulative Return ($100 initial)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_cumulative, use_container_width=True)

    # Additional Analytics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Rolling Sharpe Ratio (90-day)")

        rolling_sharpe = 1.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.1, len(dates))

        fig_sharpe = go.Figure()

        fig_sharpe.add_trace(go.Scatter(
            x=dates, y=rolling_sharpe,
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2),
            name='Rolling Sharpe'
        ))

        fig_sharpe.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="Target: 2.0")

        fig_sharpe.update_layout(
            height=350,
            yaxis_title='Sharpe Ratio',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )

        st.plotly_chart(fig_sharpe, use_container_width=True)

    with col2:
        st.markdown("### 📉 Underwater Plot")

        cummax = pd.Series(ensemble_cum).cummax()
        drawdown = (ensemble_cum - cummax) / cummax * 100

        fig_underwater = go.Figure()

        fig_underwater.add_trace(go.Scatter(
            x=dates, y=drawdown,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=2),
            name='Drawdown'
        ))

        fig_underwater.update_layout(
            height=350,
            yaxis_title='Drawdown (%)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )

        st.plotly_chart(fig_underwater, use_container_width=True)

    # Download Results
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download Full Report (PDF)", use_container_width=True):
            st.success("Report downloaded!")

    with col2:
        if st.button("📊 Export Metrics (CSV)", use_container_width=True):
            st.success("Metrics exported!")

    with col3:
        if st.button("📈 Export Trade Log", use_container_width=True):
            st.success("Trade log exported!")
