# 📑 Streamlit App - File Index

Quick reference for all files in the Streamlit application.

## 🚀 Getting Started

**Start here**: [QUICK_START.md](QUICK_START.md)

---

## 📄 Main Files

| File | Purpose | Lines |
|------|---------|-------|
| [app.py](app.py) | Main application entry point | ~130 |
| [requirements.txt](requirements.txt) | Full Python dependencies | ~30 |
| [requirements-minimal.txt](requirements-minimal.txt) | Minimal deps for fast deploy | ~5 |

---

## 📖 Documentation

| File | Purpose | Word Count |
|------|---------|------------|
| [README.md](README.md) | Complete app documentation | ~2,500 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide | ~1,500 |
| [QUICK_START.md](QUICK_START.md) | 5-minute quick start | ~800 |
| [APP_OVERVIEW.md](APP_OVERVIEW.md) | Detailed feature docs | ~12,000 |

---

## 🖥️ Pages (Main Features)

| File | Page Name | Description | Lines |
|------|-----------|-------------|-------|
| [pages/home.py](pages/home.py) | 🏠 Home | System overview & quick stats | ~250 |
| [pages/portfolio_dashboard.py](pages/portfolio_dashboard.py) | 📊 Portfolio | Real-time monitoring | ~425 |
| [pages/agents_control.py](pages/agents_control.py) | 🤖 AI Agents | Control & configuration | ~490 |
| [pages/dao_governance.py](pages/dao_governance.py) | ⛓️ DAO | Governance & voting | ~450 |
| [pages/explainability.py](pages/explainability.py) | 🔍 SHAP | Trade explainability | ~450 |
| [pages/trading_simulator.py](pages/trading_simulator.py) | 🎮 Simulator | Backtesting & Monte Carlo | ~480 |
| [pages/blockchain_integration.py](pages/blockchain_integration.py) | 🔗 Blockchain | Smart contracts | ~180 |
| [pages/backtesting_results.py](pages/backtesting_results.py) | 📈 Results | Historical performance | ~230 |

**Total**: ~2,900 lines of page code

---

## ⚙️ Configuration

| File | Purpose |
|------|---------|
| [.streamlit/config.toml](.streamlit/config.toml) | Streamlit configuration |
| [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) | Secrets template |
| [packages.txt](packages.txt) | System dependencies |

---

## 🚀 Launchers

| File | Platform |
|------|----------|
| [run_local.bat](run_local.bat) | Windows |
| [run_local.sh](run_local.sh) | Mac/Linux |

---

## 📊 Page Features Summary

### 🏠 Home Dashboard
- System overview with metrics
- Architecture visualization
- Performance comparison
- Technology stack

### 📊 Portfolio Dashboard
- Real-time portfolio value
- Asset allocation pie chart
- Agent performance bars
- Risk metrics (VaR, Sharpe, DD)
- Market regime detection
- Recent trades table

### 🤖 AI Agents Control
- Individual agent cards
- Configuration panels
- Performance charts
- Training curves
- Hyperparameter tuning
- Action analysis

### ⛓️ DAO Governance
- Active proposals list
- Voting interface
- Create proposal form
- Voting analytics
- Member leaderboard
- Treasury metrics

### 🔍 Explainability (SHAP)
- Trade selection
- SHAP waterfall plots
- Feature importance
- Confidence gauges
- Risk assessment
- Alternative actions
- Detailed explanations

### 🎮 Trading Simulator
- **Historical Backtest**: Full configuration
- **Live Simulation**: Real-time with speed control
- **Monte Carlo**: 1000+ simulations
- Equity curves
- Drawdown analysis
- Trade statistics

### 🔗 Blockchain Integration
- Network status
- Smart contract tabs
- Read/Write functions
- Transaction history
- Gas analytics

### 📈 Backtesting Results
- Performance summary
- Agent comparison
- Cumulative returns
- Rolling Sharpe
- Underwater plot

---

## 🎯 Quick Navigation

**For First-Time Users**:
→ [QUICK_START.md](QUICK_START.md)

**To Deploy**:
→ [DEPLOYMENT.md](DEPLOYMENT.md)

**For Feature Details**:
→ [APP_OVERVIEW.md](APP_OVERVIEW.md)

**To Run Locally**:
→ [run_local.bat](run_local.bat) (Windows)
→ [run_local.sh](run_local.sh) (Mac/Linux)

---

## 📈 Statistics

- **Total Files**: 18
- **Total Lines of Code**: ~3,000
- **Total Documentation**: ~17,000 words
- **Pages**: 8 interactive
- **Charts**: 30+ visualizations
- **Features**: 50+ interactive elements

---

## 🔗 Related Files

**In Parent Directory**:
- [../README.md](../README.md) - Main project README
- [../STREAMLIT_APP_SUMMARY.md](../STREAMLIT_APP_SUMMARY.md) - Complete summary
- [../FINAL_DELIVERABLES.md](../FINAL_DELIVERABLES.md) - All deliverables

**Other Modules**:
- [../contracts/](../contracts/) - Smart contracts
- [../agents/](../agents/) - ML agents
- [../dashboard/frontend/](../dashboard/frontend/) - React dashboard

---

**Need help?** Start with [QUICK_START.md](QUICK_START.md) 🚀
