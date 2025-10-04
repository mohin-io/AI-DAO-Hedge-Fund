"""
AI DAO Hedge Fund - Streamlit Agentic Application
Main entry point for the Streamlit-based interactive dashboard
"""

import streamlit as st
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="AI DAO Hedge Fund",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/mohin-io/AI-DAO-Hedge-Fund',
        'Report a bug': "https://github.com/mohin-io/AI-DAO-Hedge-Fund/issues",
        'About': "# AI DAO Hedge Fund\nDecentralized Autonomous Hedge Fund powered by Multi-Agent RL and Blockchain DAO"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .agent-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🤖⛓️📈 AI DAO Navigation")
st.sidebar.markdown("---")

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
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**AI DAO Hedge Fund**

Multi-Agent Reinforcement Learning meets Decentralized Autonomous Organization

Built with:
- 🧠 PyTorch & Stable-Baselines3
- ⛓️ Solidity & Web3
- 📊 Streamlit
""")

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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown("**Version:** 1.0.0")
st.sidebar.markdown("[GitHub](https://github.com/mohin-io/AI-DAO-Hedge-Fund) | [Documentation](docs/PLAN.md)")
