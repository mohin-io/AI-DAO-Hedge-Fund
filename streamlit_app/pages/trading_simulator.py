"""
Trading Simulator - Interactive backtesting and live simulation
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def render():
    """Render the trading simulator page"""

    st.title("🎮 Trading Simulator")
    st.markdown("Backtest strategies and simulate live trading scenarios")

    # Simulation Mode Selection
    st.markdown("### 🎯 Simulation Mode")

    sim_mode = st.radio(
        "Select Mode",
        ["📈 Historical Backtest", "🔴 Live Simulation", "🎲 Monte Carlo Simulation"],
        horizontal=True
    )

    st.markdown("---")

    if sim_mode == "📈 Historical Backtest":
        render_historical_backtest()
    elif sim_mode == "🔴 Live Simulation":
        render_live_simulation()
    else:
        render_monte_carlo()


def render_historical_backtest():
    """Render historical backtesting interface"""

    st.markdown("### 📈 Historical Backtest Configuration")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now()
        )

        assets = st.multiselect(
            "Select Assets",
            ["AAPL", "MSFT", "GOOGL", "TSLA", "BTC-USD", "ETH-USD", "SPY", "QQQ"],
            default=["AAPL", "MSFT", "BTC-USD"]
        )

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=1000000,
            step=10000
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now()
        )

        agents_enabled = st.multiselect(
            "Enable Agents",
            ["Momentum (PPO)", "Arbitrage (DQN)", "Hedging (SAC)"],
            default=["Momentum (PPO)", "Arbitrage (DQN)", "Hedging (SAC)"]
        )

        transaction_cost = st.slider(
            "Transaction Cost (bps)",
            min_value=0,
            max_value=100,
            value=10,
            step=5,
            help="Transaction cost in basis points (1 bps = 0.01%)"
        )

    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2, col3 = st.columns(3)

        with col1:
            rebalance_freq = st.selectbox(
                "Rebalance Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=0
            )

            max_leverage = st.slider("Max Leverage", 1.0, 3.0, 1.0, 0.1)

        with col2:
            risk_limit = st.slider("Max Drawdown Limit (%)", 5, 50, 20, 5)
            position_limit = st.slider("Max Position Size (%)", 5, 50, 20, 5)

        with col3:
            use_stop_loss = st.checkbox("Use Stop Loss", value=True)
            if use_stop_loss:
                stop_loss_pct = st.slider("Stop Loss (%)", 1, 20, 5, 1)

    # Run Backtest Button
    if st.button("🚀 Run Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running backtest..."):
            # Simulate backtest
            import time
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            st.success("✅ Backtest completed successfully!")

            # Display Results
            display_backtest_results(initial_capital, start_date, end_date)


def display_backtest_results(initial_capital, start_date, end_date):
    """Display backtest results"""

    st.markdown("---")
    st.markdown("### 📊 Backtest Results")

    # Performance Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Return", "+34.2%", delta="+15.6% vs S&P 500")

    with col2:
        st.metric("Sharpe Ratio", "2.14", delta="+0.32")

    with col3:
        st.metric("Max Drawdown", "-12.3%", delta="7.5% better")

    with col4:
        st.metric("Win Rate", "58.3%", delta="+3.1%")

    with col5:
        st.metric("Total Trades", "1,247", delta="Avg 4/day")

    # Equity Curve
    st.markdown("#### 📈 Equity Curve")

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, len(dates))
    portfolio_values = initial_capital * np.cumprod(1 + returns)

    # Benchmark
    benchmark_returns = np.random.normal(0.0005, 0.012, len(dates))
    benchmark_values = initial_capital * np.cumprod(1 + benchmark_returns)

    fig_equity = go.Figure()

    fig_equity.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        name='AI DAO Strategy',
        line=dict(color='#667eea', width=3)
    ))

    fig_equity.add_trace(go.Scatter(
        x=dates,
        y=benchmark_values,
        name='S&P 500 Benchmark',
        line=dict(color='#f5576c', width=2, dash='dash')
    ))

    fig_equity.update_layout(
        height=400,
        yaxis_title='Portfolio Value ($)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_equity, use_container_width=True)

    # Drawdown Chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📉 Drawdown Analysis")

        # Calculate drawdown
        cummax = pd.Series(portfolio_values).cummax()
        drawdown = (portfolio_values - cummax) / cummax * 100

        fig_dd = go.Figure()

        fig_dd.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red', width=2),
            name='Drawdown'
        ))

        fig_dd.update_layout(
            height=350,
            yaxis_title='Drawdown (%)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        st.plotly_chart(fig_dd, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Monthly Returns Heatmap")

        # Generate monthly returns
        monthly_returns = np.random.uniform(-5, 10, (5, 12))

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = [2021, 2022, 2023, 2024, 2025]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=monthly_returns,
            x=months,
            y=years,
            colorscale='RdYlGn',
            text=monthly_returns,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Return (%)")
        ))

        fig_heatmap.update_layout(
            height=350,
            xaxis_title='Month',
            yaxis_title='Year',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Trade Analysis
    st.markdown("#### 📋 Trade Analysis")

    trade_stats = pd.DataFrame({
        'Metric': [
            'Total Trades',
            'Winning Trades',
            'Losing Trades',
            'Win Rate',
            'Average Win',
            'Average Loss',
            'Profit Factor',
            'Max Consecutive Wins',
            'Max Consecutive Losses'
        ],
        'Value': [
            '1,247',
            '727',
            '520',
            '58.3%',
            '+$542',
            '-$312',
            '1.74',
            '8',
            '5'
        ]
    })

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(trade_stats, use_container_width=True, hide_index=True)

    with col2:
        # Win/Loss distribution
        returns_dist = np.random.normal(100, 300, 1000)

        fig_dist = go.Figure()

        fig_dist.add_trace(go.Histogram(
            x=returns_dist,
            nbinsx=50,
            marker_color='#667eea',
            opacity=0.7,
            name='Trade Returns'
        ))

        fig_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Breakeven")

        fig_dist.update_layout(
            height=300,
            title='Trade Returns Distribution',
            xaxis_title='Return ($)',
            yaxis_title='Frequency',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.plotly_chart(fig_dist, use_container_width=True)


def render_live_simulation():
    """Render live simulation interface"""

    st.markdown("### 🔴 Live Trading Simulation")

    st.info("Live simulation mode allows you to test strategies in real-time with simulated market data")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Simulation Controls
        st.markdown("#### Simulation Controls")

        sim_speed = st.select_slider(
            "Simulation Speed",
            options=["1x", "2x", "5x", "10x", "50x"],
            value="10x"
        )

        market_regime = st.selectbox(
            "Market Regime",
            ["Random (Realistic)", "Bull Market", "Bear Market", "High Volatility", "Low Volatility"],
            index=0
        )

    with col2:
        st.markdown("#### Current Status")

        if 'sim_running' not in st.session_state:
            st.session_state.sim_running = False

        if st.button("▶️ Start Simulation" if not st.session_state.sim_running else "⏸️ Pause Simulation",
                     use_container_width=True):
            st.session_state.sim_running = not st.session_state.sim_running

        if st.button("🔄 Reset Simulation", use_container_width=True):
            st.session_state.sim_running = False
            st.success("Simulation reset!")

    # Live Metrics
    if st.session_state.sim_running:
        st.markdown("---")
        st.markdown("### 📊 Live Metrics")

        # Animated metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Portfolio Value", "$1,023,456", delta="+2.3%")

        with col2:
            st.metric("Current P&L", "+$23,456", delta="+5 min")

        with col3:
            st.metric("Open Positions", "8", delta="+2")

        with col4:
            st.metric("Agent Activity", "HIGH", delta="Momentum active")

        # Live chart placeholder
        st.line_chart(np.random.randn(50).cumsum())


def render_monte_carlo():
    """Render Monte Carlo simulation"""

    st.markdown("### 🎲 Monte Carlo Simulation")

    st.info("Run thousands of simulations to understand potential outcomes and risk scenarios")

    col1, col2 = st.columns(2)

    with col1:
        n_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)
        time_horizon = st.slider("Time Horizon (days)", 30, 365, 90, 30)

    with col2:
        confidence_level = st.selectbox("Confidence Level", ["90%", "95%", "99%"], index=1)
        initial_capital_mc = st.number_input("Initial Capital", value=1000000, step=100000)

    if st.button("🎲 Run Monte Carlo Simulation", use_container_width=True, type="primary"):
        with st.spinner(f"Running {n_simulations} simulations..."):
            import time
            time.sleep(2)

        st.markdown("---")
        st.markdown("### 📊 Monte Carlo Results")

        # Generate simulations
        np.random.seed(42)
        simulations = []

        for _ in range(min(n_simulations, 100)):  # Limit for visualization
            returns = np.random.normal(0.001, 0.02, time_horizon)
            path = initial_capital_mc * np.cumprod(1 + returns)
            simulations.append(path)

        simulations = np.array(simulations)

        # Plot simulations
        fig_mc = go.Figure()

        for i in range(min(50, len(simulations))):
            fig_mc.add_trace(go.Scatter(
                y=simulations[i],
                mode='lines',
                line=dict(color='rgba(102, 126, 234, 0.2)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add percentiles
        percentiles = np.percentile(simulations, [5, 50, 95], axis=0)

        fig_mc.add_trace(go.Scatter(
            y=percentiles[2],
            name='95th Percentile',
            line=dict(color='green', width=3)
        ))

        fig_mc.add_trace(go.Scatter(
            y=percentiles[1],
            name='Median',
            line=dict(color='#667eea', width=3)
        ))

        fig_mc.add_trace(go.Scatter(
            y=percentiles[0],
            name='5th Percentile',
            line=dict(color='red', width=3)
        ))

        fig_mc.update_layout(
            height=500,
            title=f'{n_simulations} Monte Carlo Simulations ({time_horizon} days)',
            yaxis_title='Portfolio Value ($)',
            xaxis_title='Days',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        st.plotly_chart(fig_mc, use_container_width=True)

        # Summary statistics
        col1, col2, col3 = st.columns(3)

        final_values = simulations[:, -1]

        with col1:
            st.metric("Median Final Value", f"${np.median(final_values):,.0f}")
            st.metric("Mean Final Value", f"${np.mean(final_values):,.0f}")

        with col2:
            st.metric("95th Percentile", f"${np.percentile(final_values, 95):,.0f}")
            st.metric("5th Percentile", f"${np.percentile(final_values, 5):,.0f}")

        with col3:
            prob_profit = (final_values > initial_capital_mc).sum() / len(final_values) * 100
            st.metric("Probability of Profit", f"{prob_profit:.1f}%")

            expected_return = (np.median(final_values) - initial_capital_mc) / initial_capital_mc * 100
            st.metric("Expected Return", f"{expected_return:.1f}%")
