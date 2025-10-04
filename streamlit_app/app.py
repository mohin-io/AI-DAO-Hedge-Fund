"""
AI DAO Hedge Fund - Streamlit Agentic Application
Main entry point for the Streamlit-based interactive dashboard with live demo
"""

import streamlit as st
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="AI DAO Hedge Fund - Live Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/mohin-io/AI-DAO-Hedge-Fund',
        'Report a bug': "https://github.com/mohin-io/AI-DAO-Hedge-Fund/issues",
        'About': "# AI DAO Hedge Fund\nDecentralized Autonomous Hedge Fund powered by Multi-Agent RL and Blockchain DAO"
    }
)

# Enhanced Custom CSS with animations and modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main header with gradient animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradientShift 4s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        opacity: 0.85;
        margin-bottom: 2rem;
        color: #cbd5e0;
    }

    /* Enhanced metric cards with glassmorphism */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #667eea;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-left-color: #764ba2;
    }

    /* Live indicator with pulse animation */
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #00ff00;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
        margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7);
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7);
        }
        50% {
            opacity: 0.7;
            transform: scale(1.1);
            box-shadow: 0 0 0 10px rgba(0, 255, 0, 0);
        }
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        cursor: pointer;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem;
        animation: fadeIn 0.5s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .status-online {
        background: rgba(0, 255, 0, 0.15);
        border: 2px solid #00ff00;
        color: #00ff00;
    }

    .status-warning {
        background: rgba(255, 165, 0, 0.15);
        border: 2px solid #ffa500;
        color: #ffa500;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #667eea;
        backdrop-filter: blur(10px);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Glassmorphism container */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Sidebar metrics */
    .sidebar-metric {
        background: rgba(102, 126, 234, 0.1);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }

    /* Loading animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-spinner {
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation with enhanced UI
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0;">
    <h1 style="margin: 0; font-size: 2.5rem;">🤖⛓️📈</h1>
    <h2 style="margin: 0.5rem 0; font-size: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;">AI DAO Fund</h2>
    <p style="margin: 0; font-size: 0.85rem; opacity: 0.7; color: #cbd5e0;">Live Interactive Demo</p>
</div>
""", unsafe_allow_html=True)

# Live status indicator
st.sidebar.markdown("""
<div style="text-align: center; margin: 1rem 0; padding: 1rem; background: rgba(0, 255, 0, 0.1); border-radius: 12px; border: 2px solid rgba(0, 255, 0, 0.3); box-shadow: 0 4px 15px rgba(0, 255, 0, 0.2);">
    <span class="live-indicator"></span>
    <strong style="color: #00ff00; font-size: 1rem;">SYSTEM LIVE</strong>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📍 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Portfolio Dashboard",
        "🤖 AI Agents Control",
        "⛓️ DAO Governance",
        "🔍 Explainability (SHAP)",
        "🎮 Trading Simulator",
        "🔗 Blockchain Integration",
        "📈 Backtesting Results"
    ],
    index=0,
    label_visibility="collapsed"
)

# Quick stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Portfolio", "$1.25M", "+3.5%", delta_color="normal")
with col2:
    st.metric("Daily P&L", "$8.2K", "+0.66%", delta_color="normal")

st.sidebar.metric("Active Agents", "3/3", "100%", delta_color="normal")

# System status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; line-height: 1.8;">
    <div style="margin: 0.3rem 0;">✅ <strong>AI Agents:</strong> <span style="color: #00ff00;">Operational</span></div>
    <div style="margin: 0.3rem 0;">✅ <strong>Blockchain:</strong> <span style="color: #00ff00;">Connected</span></div>
    <div style="margin: 0.3rem 0;">✅ <strong>Data Feed:</strong> <span style="color: #00ff00;">Live</span></div>
    <div style="margin: 0.3rem 0;">✅ <strong>Risk Limits:</strong> <span style="color: #00ff00;">Normal</span></div>
</div>
""", unsafe_allow_html=True)

# Performance snapshot
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Performance")
st.sidebar.markdown("""
<div class="sidebar-metric">
    <div style="font-size: 0.75rem; opacity: 0.8;">Sharpe Ratio</div>
    <div style="font-size: 1.3rem; font-weight: 700; color: #667eea;">2.14</div>
</div>
<div class="sidebar-metric">
    <div style="font-size: 0.75rem; opacity: 0.8;">Max Drawdown</div>
    <div style="font-size: 1.3rem; font-weight: 700; color: #f5576c;">-12.3%</div>
</div>
<div class="sidebar-metric">
    <div style="font-size: 0.75rem; opacity: 0.8;">Win Rate</div>
    <div style="font-size: 1.3rem; font-weight: 700; color: #00ff00;">67.8%</div>
</div>
""", unsafe_allow_html=True)

# Footer with timestamp
st.sidebar.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown(f"""
<div style='text-align: center; font-size: 0.75rem; opacity: 0.6; color: #cbd5e0;'>
    <div style="margin-bottom: 0.5rem;">⏰ Last Updated</div>
    <div style="font-weight: 600;">{current_time}</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.75rem; opacity: 0.7;">
    <div style="margin: 0.5rem 0;">
        <a href="https://github.com/mohin-io/AI-DAO-Hedge-Fund" target="_blank" style="color: #667eea; text-decoration: none;">GitHub</a> |
        <a href="https://github.com/mohin-io/AI-DAO-Hedge-Fund/blob/master/README.md" target="_blank" style="color: #667eea; text-decoration: none;">Docs</a>
    </div>
    <div style="margin-top: 0.5rem; opacity: 0.5;">v1.0.0</div>
</div>
""", unsafe_allow_html=True)

# Main content based on selected page
if page == "🏠 Home":
    from pages import home
    home.render()
elif page == "📊 Portfolio Dashboard":
    from pages import portfolio_dashboard
    portfolio_dashboard.render()
elif page == "🤖 AI Agents Control":
    from pages import agents_control
    agents_control.render()
elif page == "⛓️ DAO Governance":
    from pages import dao_governance
    dao_governance.render()
elif page == "🔍 Explainability (SHAP)":
    from pages import explainability
    explainability.render()
elif page == "🎮 Trading Simulator":
    from pages import trading_simulator
    trading_simulator.render()
elif page == "🔗 Blockchain Integration":
    from pages import blockchain_integration
    blockchain_integration.render()
elif page == "📈 Backtesting Results":
    from pages import backtesting_results
    backtesting_results.render()
