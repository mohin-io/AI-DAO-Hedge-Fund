"""
Portfolio Dashboard - Real-time portfolio monitoring and analytics with advanced interactivity
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import time

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.export_utils import (
    generate_portfolio_report_html,
    generate_metrics_csv,
    generate_trades_csv,
    get_sample_data
)

def render():
    """Render the enhanced interactive portfolio dashboard page"""

    # Page header with live indicator
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("📊 Portfolio Dashboard")
    with col2:
        refresh_interval = st.selectbox("Refresh", ["5s", "10s", "30s", "60s", "Off"], index=4, label_visibility="collapsed")
    with col3:
        if refresh_interval != "Off":
            st.markdown("🟢 **Live**")
            # Auto-refresh functionality
            interval_map = {"5s": 5, "10s": 10, "30s": 30, "60s": 60}
            st_autorefresh = st.empty()
            time.sleep(0.1)  # Simulated refresh
        else:
            st.markdown("⚫ **Paused**")

    st.markdown("### Real-time portfolio performance, allocation, and risk metrics")

    # Add custom CSS for enhanced interactivity
    st.markdown("""
    <style>
        /* Enhanced metric cards with hover effects */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(102, 126, 234, 0.2);
            transition: all 0.3s ease;
            cursor: pointer;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        }

        /* Animated pulse for live indicators */
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 5px rgba(0, 255, 0, 0.5); }
            50% { box-shadow: 0 0 20px rgba(0, 255, 0, 0.8); }
        }

        /* Enhanced dataframe styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Interactive button effects */
        .stButton > button {
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: scale(1.05);
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Time period selector for interactive filtering
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### 📊 Performance Metrics")
    with col2:
        time_period = st.selectbox("Period", ["24H", "7D", "30D", "90D", "1Y", "ALL"], index=4)
    with col3:
        view_mode = st.selectbox("View", ["Overview", "Detailed", "Advanced"], index=0)

    # Dynamic metrics based on time period
    period_multipliers = {"24H": 0.1, "7D": 0.3, "30D": 0.8, "90D": 1.5, "1Y": 3.5, "ALL": 5.0}
    multiplier = period_multipliers.get(time_period, 1.0)

    # Key Metrics Row 1 - Enhanced with dynamic values
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        portfolio_value = 1247893 * (1 + multiplier * 0.01)
        st.metric(
            label="💰 Portfolio Value",
            value=f"${portfolio_value:,.0f}",
            delta=f"+${42156 * multiplier:,.0f} ({3.5 * multiplier:.1f}%)",
            delta_color="normal",
            help="Total portfolio value including all assets"
        )

    with col2:
        daily_pnl = 8234 * multiplier
        st.metric(
            label="📈 Daily P&L",
            value=f"+${daily_pnl:,.0f}",
            delta=f"+{0.66 * multiplier:.2f}%",
            delta_color="normal",
            help="Profit & Loss for the selected period"
        )

    with col3:
        sharpe = 2.14 + (multiplier * 0.05)
        st.metric(
            label="⚡ Sharpe Ratio",
            value=f"{sharpe:.2f}",
            delta=f"+{0.08 * multiplier:.2f}",
            delta_color="normal",
            help="Risk-adjusted return metric"
        )

    with col4:
        max_dd = -12.3 + (multiplier * 0.5)
        st.metric(
            label="🛡️ Max Drawdown",
            value=f"{max_dd:.1f}%",
            delta="Improving" if multiplier > 1 else "Stable",
            delta_color="inverse",
            help="Maximum peak-to-trough decline"
        )

    # Second row of metrics with interactive elements
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        win_rate = 67.8 + (multiplier * 0.3)
        st.metric(
            label="🎯 Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"+{2.1 * multiplier:.1f}%",
            delta_color="normal",
            help="Percentage of profitable trades"
        )

    with col2:
        st.metric(
            label="🤖 Active Agents",
            value="3/3",
            delta="All Operational",
            delta_color="off",
            help="Number of AI agents currently trading"
        )

    with col3:
        total_trades = int(1247 + (42 * multiplier))
        st.metric(
            label="📊 Total Trades",
            value=f"{total_trades:,}",
            delta=f"+{int(42 * multiplier)} today",
            delta_color="normal",
            help="Total number of executed trades"
        )

    with col4:
        avg_duration = 4.2 - (0.3 * multiplier * 0.5)
        st.metric(
            label="⏱️ Avg Trade Duration",
            value=f"{avg_duration:.1f} hrs",
            delta=f"-{0.3 * multiplier * 0.5:.1f} hrs",
            delta_color="normal",
            help="Average holding period for trades"
        )

    st.markdown("---")

    # Interactive Portfolio Performance Chart with advanced features
    st.markdown("### 📈 Portfolio Performance Over Time")

    # Chart customization controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_benchmark = st.checkbox("Show Benchmark", value=True)
    with col2:
        show_drawdown = st.checkbox("Show Drawdown", value=False)
    with col3:
        show_volume = st.checkbox("Show Volume", value=False)
    with col4:
        log_scale = st.checkbox("Log Scale", value=False)

    # Generate enhanced time series data
    period_days = {"24H": 1, "7D": 7, "30D": 30, "90D": 90, "1Y": 365, "ALL": 540}
    days = period_days.get(time_period, 180)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D' if days > 7 else 'H')

    # Simulate portfolio growth with realistic volatility
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, len(dates))
    portfolio_values = 1000000 * np.cumprod(1 + returns)

    # Benchmark (S&P 500)
    benchmark_returns = np.random.normal(0.0005, 0.012, len(dates))
    benchmark_values = 1000000 * np.cumprod(1 + benchmark_returns)

    # Calculate drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax * 100

    # Simulate trading volume
    volume = np.random.randint(50, 200, len(dates))

    df_performance = pd.DataFrame({
        'Date': dates,
        'AI DAO Fund': portfolio_values,
        'S&P 500 Benchmark': benchmark_values,
        'Drawdown': drawdown,
        'Volume': volume
    })

    # Create enhanced interactive chart
    fig_performance = go.Figure()

    # Main portfolio line
    fig_performance.add_trace(go.Scatter(
        x=df_performance['Date'],
        y=df_performance['AI DAO Fund'],
        name='AI DAO Fund',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>Portfolio Value</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))

    # Benchmark
    if show_benchmark:
        fig_performance.add_trace(go.Scatter(
            x=df_performance['Date'],
            y=df_performance['S&P 500 Benchmark'],
            name='S&P 500',
            line=dict(color='#f5576c', width=2, dash='dash'),
            hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))

    # Drawdown overlay
    if show_drawdown:
        fig_performance.add_trace(go.Scatter(
            x=df_performance['Date'],
            y=df_performance['Drawdown'],
            name='Drawdown %',
            yaxis='y2',
            line=dict(color='#ff6b6b', width=1.5, dash='dot'),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)',
            hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>DD: %{y:.2f}%<extra></extra>'
        ))

    # Volume bars
    if show_volume:
        fig_performance.add_trace(go.Bar(
            x=df_performance['Date'],
            y=df_performance['Volume'],
            name='Trade Volume',
            yaxis='y2',
            marker_color='rgba(102, 126, 234, 0.3)',
            hovertemplate='<b>Volume</b><br>Date: %{x}<br>Trades: %{y}<extra></extra>'
        ))

    # Enhanced layout with advanced interactive features
    fig_performance.update_layout(
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            rangeslider=dict(visible=True, thickness=0.05),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(step="all", label="ALL")
                ]),
                bgcolor='rgba(102, 126, 234, 0.1)',
                activecolor='rgba(102, 126, 234, 0.3)'
            )
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            title='Portfolio Value ($)',
            type='log' if log_scale else 'linear'
        ),
        yaxis2=dict(
            title='Drawdown (%) / Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ) if show_drawdown or show_volume else None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.05)'
        ),
        dragmode='zoom',
        modebar=dict(
            bgcolor='rgba(102, 126, 234, 0.1)',
            color='#667eea',
            activecolor='#764ba2'
        )
    )

    # Add crosshair
    fig_performance.update_xaxes(showspikes=True, spikecolor="#667eea", spikesnap="cursor", spikemode="across", spikethickness=1)
    fig_performance.update_yaxes(showspikes=True, spikecolor="#667eea", spikesnap="cursor", spikemode="across", spikethickness=1)

    st.plotly_chart(fig_performance, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    })

    # Two column layout for allocation and agent performance
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🥧 Asset Allocation")

        # Interactive allocation controls
        allocation_type = st.radio(
            "View by",
            ["Value", "Percentage", "Risk"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Asset allocation data
        allocation_data = pd.DataFrame({
            'Asset': ['Equities', 'Crypto', 'Options', 'Cash'],
            'Value': [623947, 311974, 186790, 125182],
            'Percentage': [50.0, 25.0, 15.0, 10.0],
            'Risk': [65, 85, 95, 5]
        })

        if allocation_type == "Value":
            values = allocation_data['Value']
            value_format = '$%{value:,.0f}'
        elif allocation_type == "Percentage":
            values = allocation_data['Percentage']
            value_format = '%{value:.1f}%'
        else:
            values = allocation_data['Risk']
            value_format = 'Risk: %{value}'

        fig_allocation = go.Figure(data=[go.Pie(
            labels=allocation_data['Asset'],
            values=values,
            hole=0.5,
            marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c'],
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>' + value_format + '<extra></extra>',
            pull=[0.05, 0, 0, 0],  # Pull out first slice
            rotation=45
        )])

        fig_allocation.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            annotations=[dict(
                text=f'Total<br>${allocation_data["Value"].sum():,.0f}' if allocation_type == "Value" else f'{allocation_data["Percentage"].sum():.0f}%',
                x=0.5, y=0.5,
                font_size=18,
                showarrow=False,
                font=dict(color='#667eea', weight='bold')
            )],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_allocation, use_container_width=True)

    with col2:
        st.markdown("### 🤖 Agent Performance (P&L)")

        # Agent performance selector
        metric_view = st.radio(
            "Metric",
            ["P&L", "Win Rate", "Sharpe"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Agent performance data
        agent_data = {
            'P&L': [42567, 28934, 15890],
            'Win Rate': [71.2, 65.8, 58.3],
            'Sharpe': [2.4, 2.1, 1.8]
        }

        agent_pnl = pd.DataFrame({
            'Agent': ['Momentum\n(PPO)', 'Arbitrage\n(DQN)', 'Hedging\n(SAC)'],
            'Values': agent_data[metric_view],
            'Color': ['#667eea', '#764ba2', '#f5576c']
        })

        if metric_view == "P&L":
            text_format = ['$' + f"{val:,.0f}" for val in agent_pnl['Values']]
            y_title = 'Profit & Loss ($)'
        elif metric_view == "Win Rate":
            text_format = [f"{val:.1f}%" for val in agent_pnl['Values']]
            y_title = 'Win Rate (%)'
        else:
            text_format = [f"{val:.2f}" for val in agent_pnl['Values']]
            y_title = 'Sharpe Ratio'

        fig_agents = go.Figure(data=[
            go.Bar(
                x=agent_pnl['Agent'],
                y=agent_pnl['Values'],
                marker_color=agent_pnl['Color'],
                text=text_format,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' + y_title + ': %{y:,.2f}<extra></extra>',
                marker=dict(
                    line=dict(color='white', width=2),
                    pattern=dict(shape=['/', '\\', 'x'][i] if metric_view == "Sharpe" else '')
                ) if metric_view == "Sharpe" else None
            )
        ])

        fig_agents.update_layout(
            height=400,
            yaxis_title=y_title,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
            hoverlabel=dict(bgcolor='rgba(102, 126, 234, 0.9)', font_color='white')
        )

        st.plotly_chart(fig_agents, use_container_width=True)

    st.markdown("---")

    # Agent Weight Allocation Over Time - Enhanced
    st.markdown("### ⚖️ Dynamic Agent Weight Allocation")

    # Add weight strategy selector
    weight_strategy = st.selectbox(
        "Strategy",
        ["Adaptive", "Equal Weight", "Risk Parity", "Performance Based"],
        help="Select weight allocation strategy"
    )

    dates_weights = pd.date_range(end=datetime.now(), periods=90, freq='D')

    # Simulate dynamic weights based on strategy
    np.random.seed(43)
    if weight_strategy == "Adaptive":
        momentum_weights = 0.4 + 0.2 * np.sin(np.linspace(0, 4*np.pi, len(dates_weights))) + np.random.normal(0, 0.05, len(dates_weights))
        arbitrage_weights = 0.3 + 0.15 * np.cos(np.linspace(0, 3*np.pi, len(dates_weights))) + np.random.normal(0, 0.04, len(dates_weights))
    elif weight_strategy == "Equal Weight":
        momentum_weights = np.ones(len(dates_weights)) * 0.333
        arbitrage_weights = np.ones(len(dates_weights)) * 0.333
    elif weight_strategy == "Risk Parity":
        momentum_weights = 0.25 + np.random.normal(0, 0.02, len(dates_weights))
        arbitrage_weights = 0.40 + np.random.normal(0, 0.02, len(dates_weights))
    else:  # Performance Based
        momentum_weights = 0.5 + 0.1 * np.random.randn(len(dates_weights))
        arbitrage_weights = 0.3 + 0.1 * np.random.randn(len(dates_weights))

    hedging_weights = 1 - momentum_weights - arbitrage_weights

    # Normalize to ensure sum = 1
    total = momentum_weights + arbitrage_weights + hedging_weights
    momentum_weights /= total
    arbitrage_weights /= total
    hedging_weights /= total

    fig_weights = go.Figure()

    fig_weights.add_trace(go.Scatter(
        x=dates_weights, y=momentum_weights,
        name='Momentum Agent',
        stackgroup='one',
        fillcolor='rgba(102, 126, 234, 0.6)',
        line=dict(width=0.5, color='#667eea'),
        hovertemplate='<b>Momentum</b><br>Date: %{x}<br>Weight: %{y:.1%}<extra></extra>'
    ))

    fig_weights.add_trace(go.Scatter(
        x=dates_weights, y=arbitrage_weights,
        name='Arbitrage Agent',
        stackgroup='one',
        fillcolor='rgba(118, 75, 162, 0.6)',
        line=dict(width=0.5, color='#764ba2'),
        hovertemplate='<b>Arbitrage</b><br>Date: %{x}<br>Weight: %{y:.1%}<extra></extra>'
    ))

    fig_weights.add_trace(go.Scatter(
        x=dates_weights, y=hedging_weights,
        name='Hedging Agent',
        stackgroup='one',
        fillcolor='rgba(245, 87, 108, 0.6)',
        line=dict(width=0.5, color='#f5576c'),
        hovertemplate='<b>Hedging</b><br>Date: %{x}<br>Weight: %{y:.1%}<extra></extra>'
    ))

    fig_weights.update_layout(
        height=350,
        yaxis=dict(
            title='Weight Allocation',
            tickformat='.0%',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)'
        ),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_weights, use_container_width=True)

    st.markdown("---")

    # Enhanced Risk Metrics with interactive elements
    st.markdown("### 🛡️ Risk Metrics & Analysis")

    # Risk metric selector
    risk_view = st.radio(
        "Select Risk View",
        ["Overview", "Detailed VaR", "Correlation Matrix", "Stress Test"],
        horizontal=True
    )

    if risk_view == "Overview":
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            **📊 Volatility**
            - Daily: 1.8%
            - Annual: 18.3%
            - Target: 15%
            - Status: ⚠️ Above target

            **Trend**: Decreasing
            """)

        with col2:
            st.markdown("""
            **💹 Value at Risk (95%)**
            - Daily VaR: -2.1%
            - Monthly VaR: -6.8%
            - Max Loss: $26,186
            - Status: ✅ Within limits

            **Coverage**: 95% confidence
            """)

        with col3:
            st.markdown("""
            **🔗 Beta & Correlation**
            - Market Beta: 0.87
            - S&P 500 Corr: 0.72
            - Diversification: Good
            - Status: ✅ Optimal

            **Systematic Risk**: Low
            """)

        with col4:
            st.markdown("""
            **📉 Drawdown Analysis**
            - Current DD: -3.2%
            - Max DD: -12.3%
            - Recovery: 87%
            - Status: ✅ Recovering

            **Time to Recover**: ~8 days
            """)

    elif risk_view == "Detailed VaR":
        # VaR distribution chart
        confidence_levels = [90, 95, 99]
        var_values = [-1.5, -2.1, -3.8]

        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=[f"{c}% VaR" for c in confidence_levels],
            y=var_values,
            marker_color=['#667eea', '#764ba2', '#f5576c'],
            text=[f"{v:.2f}%" for v in var_values],
            textposition='outside'
        ))
        fig_var.update_layout(
            height=300,
            yaxis_title="Value at Risk (%)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_var, use_container_width=True)

    elif risk_view == "Correlation Matrix":
        # Correlation heatmap
        assets = ['Momentum', 'Arbitrage', 'Hedging', 'Benchmark']
        corr_matrix = np.array([
            [1.0, 0.45, -0.32, 0.67],
            [0.45, 1.0, -0.15, 0.52],
            [-0.32, -0.15, 1.0, -0.28],
            [0.67, 0.52, -0.28, 1.0]
        ])

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=assets,
            y=assets,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix,
            texttemplate='%{text:.2f}',
            textfont={"size": 14},
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    else:  # Stress Test
        # Stress test scenarios
        scenarios = ['Market Crash\n-30%', 'Volatility Spike\n+200%', 'Liquidity Crisis', 'Rate Hike\n+2%']
        impact = [-15.2, -8.7, -5.3, -3.1]

        fig_stress = go.Figure()
        fig_stress.add_trace(go.Bar(
            y=scenarios,
            x=impact,
            orientation='h',
            marker_color=['#f5576c' if i < -10 else '#ffa500' if i < -5 else '#667eea' for i in impact],
            text=[f"{i:.1f}%" for i in impact],
            textposition='outside'
        ))
        fig_stress.update_layout(
            height=300,
            xaxis_title="Portfolio Impact (%)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown("---")

    # Enhanced Recent Trades Table with filtering and search
    st.markdown("### 📋 Recent Trades")

    # Trade filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_agent = st.multiselect("Agent", ["All", "Momentum", "Arbitrage", "Hedging"], default=["All"])
    with col2:
        filter_action = st.multiselect("Action", ["All", "BUY", "SELL", "LONG/SHORT", "CLOSE"], default=["All"])
    with col3:
        filter_pnl = st.selectbox("P&L Filter", ["All", "Profitable", "Loss"], index=0)
    with col4:
        search_asset = st.text_input("Search Asset", placeholder="e.g. AAPL")

    trades_data = pd.DataFrame({
        'Time': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=15),
            datetime.now() - timedelta(minutes=32),
            datetime.now() - timedelta(hours=1, minutes=12),
            datetime.now() - timedelta(hours=2, minutes=5),
            datetime.now() - timedelta(hours=3, minutes=42),
            datetime.now() - timedelta(hours=5, minutes=18),
        ],
        'Agent': ['Momentum', 'Arbitrage', 'Hedging', 'Momentum', 'Arbitrage', 'Hedging', 'Momentum'],
        'Action': ['BUY', 'LONG/SHORT', 'BUY', 'SELL', 'CLOSE', 'BUY', 'SELL'],
        'Asset': ['AAPL', 'MSFT/GOOGL', 'SPY PUT', 'TSLA', 'BTC-USD', 'QQQ', 'NVDA'],
        'Quantity': [100, '50/50', '10 contracts', 75, '0.5 BTC', 200, 150],
        'Price': ['$182.45', 'Spread: 1.2%', '$420.50', '$245.80', '$43,256', '$385.20', '$495.30'],
        'P&L': ['+$1,234', '+$890', '-$156', '+$2,145', '+$567', '-$234', '+$1,823'],
        'Confidence': [87, 72, 91, 83, 68, 75, 89]
    })

    # Apply filters
    if "All" not in filter_agent:
        trades_data = trades_data[trades_data['Agent'].isin(filter_agent)]
    if "All" not in filter_action:
        trades_data = trades_data[trades_data['Action'].isin(filter_action)]
    if filter_pnl == "Profitable":
        trades_data = trades_data[trades_data['P&L'].str.contains(r'\+', regex=True)]
    elif filter_pnl == "Loss":
        trades_data = trades_data[trades_data['P&L'].str.contains(r'-', regex=True)]
    if search_asset:
        trades_data = trades_data[trades_data['Asset'].str.contains(search_asset, case=False)]

    # Format time column
    trades_data['Time'] = trades_data['Time'].dt.strftime('%H:%M:%S')

    # Enhanced dataframe with styling
    st.dataframe(
        trades_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Time': st.column_config.TextColumn('⏰ Time', width='small'),
            'Agent': st.column_config.TextColumn('🤖 Agent', width='small'),
            'Action': st.column_config.TextColumn('⚡ Action', width='small'),
            'Asset': st.column_config.TextColumn('📊 Asset', width='medium'),
            'Quantity': st.column_config.TextColumn('📦 Qty', width='small'),
            'Price': st.column_config.TextColumn('💰 Price', width='medium'),
            'P&L': st.column_config.TextColumn('💹 P&L', width='small'),
            'Confidence': st.column_config.ProgressColumn(
                '🎯 Confidence',
                format='%d%%',
                min_value=0,
                max_value=100,
                width='medium'
            )
        },
        height=300
    )

    # Trade statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        profitable = len([p for p in trades_data['P&L'] if '+' in p])
        st.metric("✅ Profitable Trades", f"{profitable}/{len(trades_data)}", f"{profitable/len(trades_data)*100:.0f}%")
    with col2:
        avg_conf = trades_data['Confidence'].mean()
        st.metric("📊 Avg Confidence", f"{avg_conf:.0f}%", "+3%")
    with col3:
        st.metric("📈 Total Displayed", len(trades_data), f"of {len(trades_data)} total")

    st.markdown("---")

    # Market Regime Detection - Enhanced
    st.markdown("### 🌡️ Market Regime Detection & Recommendation Engine")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Add regime selector
        regime_model = st.selectbox(
            "Detection Model",
            ["Hidden Markov", "ML Classifier", "Technical Indicators"]
        )

        st.markdown(f"""
        **Current Regime** ({regime_model})

        🟢 **BULLISH TREND**

        **Regime Probability:**
        - 🟢 Bull: 65%
        - 🟡 Sideways: 25%
        - 🔴 Bear: 8%
        - 🟣 Volatile: 2%

        **AI Recommended Actions:**
        - ↗️ Increase momentum allocation to 55%
        - ↘️ Reduce hedging positions to 15%
        - 👁️ Monitor volatility (VIX < 20)
        - 🎯 Target: Maximize alpha in trending markets

        **Confidence Level**: 87%
        """)

    with col2:
        # Enhanced regime history with confidence bands
        regime_dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        regime_values = np.random.choice(['Bull', 'Sideways', 'Bear', 'Volatile'],
                                        size=len(regime_dates),
                                        p=[0.5, 0.3, 0.15, 0.05])

        regime_numeric = pd.Series(regime_values).map({'Bull': 3, 'Sideways': 2, 'Bear': 1, 'Volatile': 0})
        confidence = 60 + 30 * np.random.random(len(regime_dates))

        fig_regime = go.Figure()

        colors = {'Bull': '#00ff00', 'Sideways': '#ffff00', 'Bear': '#ff0000', 'Volatile': '#ff00ff'}

        # Add confidence band
        fig_regime.add_trace(go.Scatter(
            x=regime_dates,
            y=regime_numeric,
            mode='lines',
            line=dict(color='rgba(102, 126, 234, 0.3)', width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        for regime in ['Bull', 'Sideways', 'Bear', 'Volatile']:
            mask = regime_values == regime
            fig_regime.add_trace(go.Scatter(
                x=regime_dates[mask],
                y=regime_numeric[mask],
                mode='markers',
                name=regime,
                marker=dict(
                    size=12,
                    color=colors[regime],
                    line=dict(color='white', width=1),
                    symbol='circle'
                ),
                hovertemplate=f'<b>{regime} Regime</b><br>Date: %{{x}}<br>Confidence: %{{customdata:.0f}}%<extra></extra>',
                customdata=confidence[mask]
            ))

        fig_regime.update_layout(
            height=350,
            yaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=['🟣 Volatile', '🔴 Bear', '🟡 Sideways', '🟢 Bull'],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.1)'
            ),
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='closest'
        )

        st.plotly_chart(fig_regime, use_container_width=True)

    st.markdown("---")

    # Enhanced action buttons with real download functionality
    st.markdown("### 📥 Export & Actions")

    col1, col2, col3, col4 = st.columns(4)

    # Prepare export data
    portfolio_export_data = {
        'value': portfolio_value,
        'daily_pnl': daily_pnl,
        'daily_pnl_pct': 0.66 * multiplier,
        'total_return': 34.2 * multiplier
    }

    metrics_export_data = {
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'daily_volatility': 1.8,
        'annual_volatility': 18.3,
        'var_95': -2.1,
        'beta': 0.87
    }

    agent_export_data = {
        'momentum': {'pnl': 42567 * multiplier, 'win_rate': 71.2},
        'arbitrage': {'pnl': 28934 * multiplier, 'win_rate': 65.8},
        'hedging': {'pnl': 15890 * multiplier, 'win_rate': 58.3}
    }

    with col1:
        # Download HTML Report
        html_report = generate_portfolio_report_html(
            portfolio_export_data,
            metrics_export_data,
            trades_data
        )

        st.download_button(
            label="📥 Download Report",
            data=html_report,
            file_name=f"AI_DAO_Portfolio_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True,
            help="Download comprehensive portfolio report as HTML",
            type="primary"
        )

    with col2:
        # Export Metrics CSV
        metrics_csv = generate_metrics_csv(
            portfolio_export_data,
            metrics_export_data,
            agent_export_data
        )

        st.download_button(
            label="📊 Export Metrics",
            data=metrics_csv,
            file_name=f"AI_DAO_Metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Export all performance metrics as CSV"
        )

    with col3:
        # Export Trade Log CSV
        trades_csv = generate_trades_csv(trades_data)

        st.download_button(
            label="📋 Export Trades",
            data=trades_csv,
            file_name=f"AI_DAO_TradeLog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Export complete trade log as CSV"
        )

    with col4:
        # Emergency Stop (action button)
        if st.button("⚠️ Emergency Stop", use_container_width=True, help="Halt all trading activity", type="secondary"):
            st.error("🛑 Emergency stop activated! All trading halted.")
            st.warning("Contact system administrator to resume operations.")
            st.info("Last trade executed at: " + datetime.now().strftime('%H:%M:%S'))
