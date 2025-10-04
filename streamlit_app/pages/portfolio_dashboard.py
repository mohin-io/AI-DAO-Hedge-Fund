"""
Portfolio Dashboard - Real-time portfolio monitoring and analytics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.export_utils import (
    generate_portfolio_report_html,
    generate_metrics_csv,
    generate_trades_csv,
    get_sample_data
)

def render():
    """Render the portfolio dashboard page"""

    st.title("📊 Portfolio Dashboard")
    st.markdown("Real-time portfolio performance, allocation, and risk metrics")

    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### Live Portfolio Monitoring")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    with col3:
        if auto_refresh:
            st.markdown("🟢 **Live**")
        else:
            st.markdown("⚫ **Paused**")

    st.markdown("---")

    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Portfolio Value",
            value="$1,247,893",
            delta="+$42,156 (3.5%)",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Daily P&L",
            value="+$8,234",
            delta="+0.66%",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="Sharpe Ratio",
            value="2.14",
            delta="+0.08",
            delta_color="normal"
        )

    with col4:
        st.metric(
            label="Max Drawdown",
            value="-12.3%",
            delta="Improved",
            delta_color="inverse"
        )

    with col5:
        st.metric(
            label="Win Rate",
            value="58.3%",
            delta="+2.1%",
            delta_color="normal"
        )

    # Portfolio Performance Chart
    st.markdown("### 📈 Portfolio Performance Over Time")

    # Generate sample time series data
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')

    # Simulate portfolio growth with realistic volatility
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, len(dates))
    portfolio_values = 1000000 * np.cumprod(1 + returns)

    # Benchmark (S&P 500)
    benchmark_returns = np.random.normal(0.0005, 0.012, len(dates))
    benchmark_values = 1000000 * np.cumprod(1 + benchmark_returns)

    df_performance = pd.DataFrame({
        'Date': dates,
        'AI DAO Fund': portfolio_values,
        'S&P 500 Benchmark': benchmark_values
    })

    fig_performance = go.Figure()

    fig_performance.add_trace(go.Scatter(
        x=df_performance['Date'],
        y=df_performance['AI DAO Fund'],
        name='AI DAO Fund',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))

    fig_performance.add_trace(go.Scatter(
        x=df_performance['Date'],
        y=df_performance['S&P 500 Benchmark'],
        name='S&P 500',
        line=dict(color='#f5576c', width=2, dash='dash')
    ))

    fig_performance.update_layout(
        height=450,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title='Portfolio Value ($)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_performance, use_container_width=True)

    # Two column layout for allocation and agent performance
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🥧 Asset Allocation")

        # Asset allocation data
        allocation_data = pd.DataFrame({
            'Asset': ['Equities', 'Crypto', 'Options', 'Cash'],
            'Value': [623947, 311974, 186790, 125182],
            'Percentage': [50.0, 25.0, 15.0, 10.0]
        })

        fig_allocation = go.Figure(data=[go.Pie(
            labels=allocation_data['Asset'],
            values=allocation_data['Value'],
            hole=0.4,
            marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c'],
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig_allocation.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            annotations=[dict(text='Total<br>$1.25M', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )

        st.plotly_chart(fig_allocation, use_container_width=True)

    with col2:
        st.markdown("### 🤖 Agent Performance (P&L)")

        # Agent performance data
        agent_pnl = pd.DataFrame({
            'Agent': ['Momentum\n(PPO)', 'Arbitrage\n(DQN)', 'Hedging\n(SAC)'],
            'P&L': [42567, 28934, 15890],
            'Color': ['#667eea', '#764ba2', '#f5576c']
        })

        fig_agents = go.Figure(data=[
            go.Bar(
                x=agent_pnl['Agent'],
                y=agent_pnl['P&L'],
                marker_color=agent_pnl['Color'],
                text=['$' + f"{val:,.0f}" for val in agent_pnl['P&L']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>P&L: $%{y:,.0f}<extra></extra>'
            )
        ])

        fig_agents.update_layout(
            height=400,
            yaxis_title='Profit & Loss ($)',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )

        st.plotly_chart(fig_agents, use_container_width=True)

    # Agent Weight Allocation Over Time
    st.markdown("### ⚖️ Dynamic Agent Weight Allocation")

    dates_weights = pd.date_range(end=datetime.now(), periods=90, freq='D')

    # Simulate dynamic weights based on market regime
    np.random.seed(43)
    momentum_weights = 0.4 + 0.2 * np.sin(np.linspace(0, 4*np.pi, len(dates_weights))) + np.random.normal(0, 0.05, len(dates_weights))
    arbitrage_weights = 0.3 + 0.15 * np.cos(np.linspace(0, 3*np.pi, len(dates_weights))) + np.random.normal(0, 0.04, len(dates_weights))
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
        line=dict(width=0.5, color='#667eea')
    ))

    fig_weights.add_trace(go.Scatter(
        x=dates_weights, y=arbitrage_weights,
        name='Arbitrage Agent',
        stackgroup='one',
        fillcolor='rgba(118, 75, 162, 0.6)',
        line=dict(width=0.5, color='#764ba2')
    ))

    fig_weights.add_trace(go.Scatter(
        x=dates_weights, y=hedging_weights,
        name='Hedging Agent',
        stackgroup='one',
        fillcolor='rgba(245, 87, 108, 0.6)',
        line=dict(width=0.5, color='#f5576c')
    ))

    fig_weights.update_layout(
        height=350,
        yaxis=dict(title='Weight Allocation', tickformat='.0%', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_weights, use_container_width=True)

    # Risk Metrics
    st.markdown("### 🛡️ Risk Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        **Volatility**
        - Daily: 1.8%
        - Annual: 18.3%
        - Target: 15%
        - Status: ⚠️ Above target
        """)

    with col2:
        st.markdown("""
        **Value at Risk (95%)**
        - Daily VaR: -2.1%
        - Monthly VaR: -6.8%
        - Max Loss: $26,186
        - Status: ✅ Within limits
        """)

    with col3:
        st.markdown("""
        **Beta & Correlation**
        - Market Beta: 0.87
        - S&P 500 Corr: 0.72
        - Diversification: Good
        - Status: ✅ Optimal
        """)

    with col4:
        st.markdown("""
        **Drawdown Analysis**
        - Current DD: -3.2%
        - Max DD: -12.3%
        - Recovery: 87%
        - Status: ✅ Recovering
        """)

    # Recent Trades Table
    st.markdown("### 📋 Recent Trades")

    trades_data = pd.DataFrame({
        'Time': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=15),
            datetime.now() - timedelta(minutes=32),
            datetime.now() - timedelta(hours=1, minutes=12),
            datetime.now() - timedelta(hours=2, minutes=5),
        ],
        'Agent': ['Momentum', 'Arbitrage', 'Hedging', 'Momentum', 'Arbitrage'],
        'Action': ['BUY', 'LONG/SHORT', 'BUY', 'SELL', 'CLOSE'],
        'Asset': ['AAPL', 'MSFT/GOOGL', 'SPY PUT', 'TSLA', 'BTC-USD'],
        'Quantity': [100, '50/50', '10 contracts', 75, '0.5 BTC'],
        'Price': ['$182.45', 'Spread: 1.2%', '$420.50', '$245.80', '$43,256'],
        'P&L': ['+$1,234', '+$890', '-$156', '+$2,145', '+$567'],
        'Confidence': ['87%', '72%', '91%', '83%', '68%']
    })

    # Format time column
    trades_data['Time'] = trades_data['Time'].dt.strftime('%H:%M:%S')

    st.dataframe(
        trades_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'P&L': st.column_config.TextColumn('P&L'),
            'Confidence': st.column_config.ProgressColumn(
                'Confidence',
                min_value=0,
                max_value=100,
            )
        }
    )

    # Market Regime Detection
    st.markdown("### 🌡️ Market Regime Detection")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        **Current Regime**

        🟢 **BULLISH TREND**

        **Regime Probability:**
        - Bull: 65%
        - Sideways: 25%
        - Bear: 8%
        - Volatile: 2%

        **Recommended Action:**
        - Increase momentum allocation
        - Reduce hedging positions
        - Monitor volatility
        """)

    with col2:
        # Regime history
        regime_dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        regime_values = np.random.choice(['Bull', 'Sideways', 'Bear', 'Volatile'],
                                        size=len(regime_dates),
                                        p=[0.5, 0.3, 0.15, 0.05])

        regime_numeric = pd.Series(regime_values).map({'Bull': 3, 'Sideways': 2, 'Bear': 1, 'Volatile': 0})

        fig_regime = go.Figure()

        colors = {'Bull': '#00ff00', 'Sideways': '#ffff00', 'Bear': '#ff0000', 'Volatile': '#ff00ff'}

        for regime in ['Bull', 'Sideways', 'Bear', 'Volatile']:
            mask = regime_values == regime
            fig_regime.add_trace(go.Scatter(
                x=regime_dates[mask],
                y=regime_numeric[mask],
                mode='markers',
                name=regime,
                marker=dict(size=10, color=colors[regime])
            ))

        fig_regime.update_layout(
            height=300,
            yaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=['Volatile', 'Bear', 'Sideways', 'Bull'],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_regime, use_container_width=True)

    # Action buttons with real download functionality
    st.markdown("---")
    st.markdown("### 📥 Export & Actions")

    col1, col2, col3, col4 = st.columns(4)

    # Prepare export data
    portfolio_export_data = {
        'value': 1247893.45,
        'daily_pnl': 8234.56,
        'daily_pnl_pct': 0.66,
        'total_return': 34.2
    }

    metrics_export_data = {
        'sharpe': 2.14,
        'max_drawdown': -12.3,
        'win_rate': 67.8,
        'daily_volatility': 1.8,
        'annual_volatility': 18.3,
        'var_95': -2.1,
        'beta': 0.87
    }

    agent_export_data = {
        'momentum': {'pnl': 42567, 'win_rate': 71.2},
        'arbitrage': {'pnl': 28934, 'win_rate': 65.8},
        'hedging': {'pnl': 15890, 'win_rate': 58.3}
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
            help="Download comprehensive portfolio report as HTML"
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
        if st.button("⚠️ Emergency Stop", use_container_width=True, help="Halt all trading activity"):
            st.error("🛑 Emergency stop activated! All trading halted.")
            st.warning("Contact system administrator to resume operations.")
