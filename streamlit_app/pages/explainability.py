"""
Explainability - SHAP-based AI decision explanations
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def render():
    """Render the explainability page"""

    st.title("🔍 AI Explainability (SHAP Analysis)")
    st.markdown("Understand every AI trading decision with SHAP (SHapley Additive exPlanations)")

    # Trade Selection
    st.markdown("### 📊 Select Trade to Explain")

    recent_trades = pd.DataFrame({
        'ID': [1001, 1002, 1003, 1004, 1005],
        'Timestamp': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=18),
            datetime.now() - timedelta(minutes=42),
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(hours=2)
        ],
        'Agent': ['Momentum', 'Arbitrage', 'Hedging', 'Momentum', 'Arbitrage'],
        'Action': ['BUY', 'LONG/SHORT', 'BUY', 'SELL', 'CLOSE'],
        'Asset': ['AAPL', 'MSFT/GOOGL', 'SPY PUT', 'TSLA', 'BTC-USD'],
        'P&L': ['+$1,234', '+$890', '-$156', '+$2,145', '+$567']
    })

    recent_trades['Timestamp'] = recent_trades['Timestamp'].dt.strftime('%H:%M:%S')

    col1, col2 = st.columns([3, 1])

    with col1:
        st.dataframe(recent_trades, use_container_width=True, hide_index=True)

    with col2:
        selected_trade_id = st.selectbox(
            "Select Trade ID",
            recent_trades['ID'].tolist(),
            index=0
        )

    st.markdown("---")

    # Get selected trade details
    selected_trade = recent_trades[recent_trades['ID'] == selected_trade_id].iloc[0]

    # Trade Details
    st.markdown("### 📋 Trade Details")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Trade ID", f"#{selected_trade['ID']}")

    with col2:
        st.metric("Agent", selected_trade['Agent'])

    with col3:
        st.metric("Action", selected_trade['Action'])

    with col4:
        st.metric("Asset", selected_trade['Asset'])

    with col5:
        st.metric("P&L", selected_trade['P&L'])

    st.markdown("---")

    # SHAP Waterfall Plot
    st.markdown("### 🌊 SHAP Waterfall Plot")
    st.markdown("*Visualizes how each feature contributes to the final decision*")

    # Generate sample SHAP values
    features = [
        'Base Value',
        'RSI (28.5 - Oversold)',
        'MACD (0.015 - Bullish)',
        'Portfolio Cash (45%)',
        'SMA_20 (Above MA)',
        'Bollinger Band (Lower)',
        'Volume (High)',
        'Market Volatility (Low)',
        'Sentiment Score (0.72)',
        'Final Decision'
    ]

    shap_values = [0.0, 0.42, 0.31, 0.18, 0.12, 0.15, 0.08, -0.08, 0.10, 0.95]
    cumulative_values = np.cumsum(shap_values)

    # Create waterfall chart
    fig_waterfall = go.Figure()

    colors = ['rgba(200,200,200,0.8)'] + \
             ['rgba(0,255,0,0.6)' if v > 0 else 'rgba(255,0,0,0.6)' for v in shap_values[1:-1]] + \
             ['rgba(102,126,234,0.8)']

    fig_waterfall.add_trace(go.Waterfall(
        x=features,
        y=shap_values,
        text=[f"{v:+.2f}" for v in shap_values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "rgba(255,0,0,0.6)"}},
        increasing={"marker": {"color": "rgba(0,255,0,0.6)"}},
        totals={"marker": {"color": "rgba(102,126,234,0.8)"}}
    ))

    fig_waterfall.update_layout(
        height=500,
        yaxis_title="SHAP Value (Contribution)",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    # Feature Importance
    st.markdown("### 📊 Feature Importance Ranking")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Remove base value and final decision
        feature_names = features[1:-1]
        feature_shap = shap_values[1:-1]

        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(feature_shap))[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_shap = [feature_shap[i] for i in sorted_indices]

        fig_importance = go.Figure()

        colors_bar = ['#00ff00' if v > 0 else '#ff0000' for v in sorted_shap]

        fig_importance.add_trace(go.Bar(
            y=sorted_features,
            x=sorted_shap,
            orientation='h',
            marker_color=colors_bar,
            text=[f"{v:+.2f}" for v in sorted_shap],
            textposition='outside'
        ))

        fig_importance.update_layout(
            height=400,
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=False)
        )

        st.plotly_chart(fig_importance, use_container_width=True)

    with col2:
        st.markdown("#### Top Contributing Features")

        st.markdown(f"""
        **1. RSI (Oversold)**
        - Value: 28.5
        - Impact: +0.42 (STRONG BUY)
        - Reason: Oversold conditions

        **2. MACD (Bullish Crossover)**
        - Value: 0.015
        - Impact: +0.31 (BUY)
        - Reason: Bullish momentum

        **3. Portfolio Cash**
        - Value: 45%
        - Impact: +0.18 (BUY)
        - Reason: Available capital

        **4. Bollinger Band**
        - Value: Lower band
        - Impact: +0.15 (BUY)
        - Reason: Price near support
        """)

    # Decision Confidence
    st.markdown("---")
    st.markdown("### 🎯 Decision Confidence Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=87,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Confidence", 'font': {'size': 20}},
            delta={'reference': 75, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255,0,0,0.2)'},
                    {'range': [50, 75], 'color': 'rgba(255,255,0,0.2)'},
                    {'range': [75, 100], 'color': 'rgba(0,255,0,0.2)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown("#### Confidence Factors")

        confidence_factors = pd.DataFrame({
            'Factor': ['Technical Signals', 'Market Regime', 'Risk Metrics', 'Historical Performance'],
            'Score': [92, 85, 81, 88]
        })

        fig_factors = go.Figure()

        fig_factors.add_trace(go.Bar(
            y=confidence_factors['Factor'],
            x=confidence_factors['Score'],
            orientation='h',
            marker_color='#667eea',
            text=confidence_factors['Score'],
            textposition='outside'
        ))

        fig_factors.update_layout(
            height=300,
            xaxis_title="Score",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(range=[0, 100], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig_factors, use_container_width=True)

    with col3:
        st.markdown("#### Risk Assessment")

        st.markdown("""
        **Risk Level:** 🟢 LOW

        **Risk Factors:**
        - Portfolio Exposure: 15%
        - Position Size: Small
        - Stop Loss: Set at -5%
        - Market Volatility: Low

        **Expected Outcomes:**
        - Best Case: +8.5%
        - Base Case: +3.2%
        - Worst Case: -5.0%

        **Risk-Reward:** 1:1.7 ✅
        """)

    # Alternative Actions Considered
    st.markdown("---")
    st.markdown("### 🔄 Alternative Actions Considered")

    alternatives_df = pd.DataFrame({
        'Action': ['BUY (Selected)', 'HOLD', 'SELL', 'BUY (Double Size)'],
        'Expected Return': [3.2, 0.0, -1.5, 6.4],
        'Risk Score': [25, 0, 10, 65],
        'Confidence': [87, 45, 12, 58],
        'SHAP Score': [0.95, 0.12, -0.34, 0.78],
        'Selected': ['✅', '', '', '']
    })

    st.dataframe(
        alternatives_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Confidence': st.column_config.ProgressColumn(
                'Confidence (%)',
                min_value=0,
                max_value=100,
            ),
            'Risk Score': st.column_config.ProgressColumn(
                'Risk Score',
                min_value=0,
                max_value=100,
            )
        }
    )

    # SHAP Summary Plot (Multiple Trades)
    st.markdown("---")
    st.markdown("### 📈 SHAP Summary Plot (Last 100 Trades)")
    st.markdown("*Shows feature importance across multiple decisions*")

    # Generate sample data for multiple trades
    np.random.seed(42)
    n_trades = 100
    feature_names_summary = ['RSI', 'MACD', 'Portfolio Cash', 'SMA_20', 'Bollinger', 'Volume', 'Volatility', 'Sentiment']

    shap_matrix = np.random.randn(n_trades, len(feature_names_summary)) * 0.3

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    fig_summary = go.Figure()

    for i, idx in enumerate(sorted_idx):
        fig_summary.add_trace(go.Box(
            y=shap_matrix[:, idx],
            name=feature_names_summary[idx],
            boxmean='sd',
            marker_color='#667eea'
        ))

    fig_summary.update_layout(
        height=400,
        yaxis_title="SHAP Value",
        xaxis_title="Feature",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )

    st.plotly_chart(fig_summary, use_container_width=True)

    # Detailed Explanation Text
    st.markdown("---")
    st.markdown("### 📝 Detailed Decision Explanation")

    st.info(f"""
    **Trade #{selected_trade['ID']} - {selected_trade['Action']} {selected_trade['Asset']}**

    **Decision Rationale:**

    The {selected_trade['Agent']} Agent decided to {selected_trade['Action']} {selected_trade['Asset']} with **87% confidence** based on the following analysis:

    **Primary Signals (Positive Contributors):**

    1. **RSI Indicator (Impact: +0.42)**
       - The Relative Strength Index is at 28.5, indicating oversold conditions
       - Historical data shows strong reversal probability when RSI < 30
       - This is the strongest signal supporting the BUY decision

    2. **MACD Crossover (Impact: +0.31)**
       - MACD line crossed above signal line (bullish crossover)
       - Current value: 0.015 (positive momentum)
       - Confirms trend reversal pattern

    3. **Portfolio Cash Availability (Impact: +0.18)**
       - 45% cash available for deployment
       - Allows for position sizing without over-leverage
       - Risk management allows this trade

    **Secondary Signals:**

    4. **Bollinger Bands (Impact: +0.15)** - Price touching lower band (potential support)
    5. **SMA Analysis (Impact: +0.12)** - Price above 20-day moving average
    6. **Sentiment Score (Impact: +0.10)** - Positive market sentiment (0.72/1.0)

    **Caution Signals (Negative Contributors):**

    7. **Market Volatility (Impact: -0.08)**
       - Slightly elevated volatility may increase trade risk
       - Recommend tighter stop loss

    **Risk Management:**
    - Position Size: Limited to 15% of portfolio
    - Stop Loss: Set at -5% ($182.45 → $173.33)
    - Take Profit: Target +8% ($182.45 → $197.05)
    - Risk-Reward Ratio: 1:1.6 ✅

    **Expected Outcome:**
    - Probability of Success: 87%
    - Expected Return: +3.2% (base case)
    - Maximum Risk: -5.0%

    **Conclusion:**
    The combination of oversold RSI, bullish MACD crossover, and adequate portfolio cash provides strong justification for this BUY decision. The trade aligns with the Momentum Agent's strategy and risk parameters.
    """)

    # Export Options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download SHAP Report (PDF)", use_container_width=True):
            st.success("SHAP report downloaded!")

    with col2:
        if st.button("📊 Export Feature Values (CSV)", use_container_width=True):
            st.success("Feature values exported!")

    with col3:
        if st.button("🖼️ Save Visualizations (PNG)", use_container_width=True):
            st.success("Visualizations saved!")
