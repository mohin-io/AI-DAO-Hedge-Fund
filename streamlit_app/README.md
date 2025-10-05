# 🤖⛓️📈 Decentralized Autonomous Hedge Fund AI DAO - Streamlit App

Interactive Streamlit application for the Decentralized Autonomous Hedge Fund powered by Multi-Agent RL and Blockchain DAO.

## 🌟 Features

### 🏠 Home Dashboard
- System overview and quick stats
- Architecture visualization
- Performance comparison with benchmark
- Recent activity feed

### 📊 Portfolio Dashboard
- Real-time portfolio monitoring
- Asset allocation visualization
- Agent performance tracking
- Risk metrics and VaR analysis
- Live trade feed
- Market regime detection

### 🤖 AI Agents Control
- Individual agent monitoring and configuration
- Performance metrics per agent (PPO, DQN, SAC)
- Training curves and hyperparameter tuning
- Action distribution analysis
- Recent agent actions with explanations

### ⛓️ DAO Governance
- Active proposal viewing and voting
- Create new governance proposals
- Voting analytics and participation trends
- DAO member leaderboard
- Treasury management interface

### 🔍 Explainability (SHAP)
- SHAP waterfall plots for trade decisions
- Feature importance ranking
- Decision confidence breakdown
- Alternative actions considered
- SHAP summary plots across multiple trades

### 🎮 Trading Simulator
- **Historical Backtest**: Test strategies on historical data
- **Live Simulation**: Real-time strategy testing with simulated markets
- **Monte Carlo**: Run thousands of simulations for risk analysis
- Detailed performance metrics and visualizations

### 🔗 Blockchain Integration
- Smart contract interaction interface
- Read/write functions for all contracts
- Transaction history viewer
- Gas analytics
- Network status monitoring

### 📈 Backtesting Results
- Historical performance summary (2020-2025)
- Agent comparison metrics
- Cumulative returns visualization
- Rolling Sharpe ratio analysis
- Underwater plot (drawdown visualization)

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.9+
pip or conda
```

### Installation

```bash
# Navigate to streamlit_app directory
cd streamlit_app

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# From the streamlit_app directory
streamlit run app.py

# Or from project root
streamlit run streamlit_app/app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
streamlit_app/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── pages/                          # Page modules
    ├── __init__.py
    ├── home.py                     # Home dashboard
    ├── portfolio_dashboard.py      # Portfolio monitoring
    ├── agents_control.py           # AI agents control
    ├── dao_governance.py           # DAO governance interface
    ├── explainability.py           # SHAP explainability
    ├── trading_simulator.py        # Backtesting & simulation
    ├── blockchain_integration.py   # Smart contract interaction
    └── backtesting_results.py      # Historical results
```

## 🎨 Key Components

### Navigation
- Sidebar navigation with 8 main sections
- Real-time status indicators
- Quick access to documentation

### Visualizations
- **Plotly**: Interactive charts (portfolio, allocation, performance)
- **Metrics**: Real-time KPIs with delta indicators
- **Tables**: Sortable dataframes with custom formatting
- **Gauges**: Confidence scores and risk levels

### Interactivity
- Live data updates (configurable refresh rate)
- Form inputs for strategy configuration
- Blockchain transaction submission
- Downloadable reports and exports

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `streamlit_app` directory:

```env
# Optional: Configure API endpoints
API_URL=http://localhost:8000
WS_URL=ws://localhost:8000/ws

# Optional: Blockchain configuration
ETHEREUM_RPC_URL=https://sepolia.infura.io/v3/YOUR_API_KEY
PRIVATE_KEY=your_private_key_here

# Optional: Contract addresses
CONTRACT_DAO_GOVERNANCE=0x5FbDB2315678afecb367f032d93F642f64180aa3
CONTRACT_TREASURY_MANAGER=0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512
CONTRACT_AGENT_REGISTRY=0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0
```

### Streamlit Configuration

Create `.streamlit/config.toml` for custom theming:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
```

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set `streamlit_app/app.py` as the main file
5. Add secrets in dashboard settings
6. Deploy!

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t ai-dao-streamlit .
docker run -p 8501:8501 ai-dao-streamlit
```

### Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create ai-dao-hedge-fund
git push heroku main
```

## 🔧 Development

### Adding New Pages

1. Create new file in `pages/` directory:

```python
# pages/my_new_page.py
import streamlit as st

def render():
    st.title("My New Page")
    st.write("Content here...")
```

2. Add to navigation in `app.py`:

```python
elif page == "🆕 My New Page":
    from pages import my_new_page
    my_new_page.render()
```

### Custom Components

Add reusable components in `pages/`:

```python
# pages/components.py
import streamlit as st

def metric_card(title, value, delta):
    st.metric(title, value, delta=delta)
```

## 📊 Data Sources

The app currently uses **simulated data** for demonstration. To connect to real data:

1. **Portfolio Data**: Connect to your backend API
2. **Blockchain Data**: Use Web3 to read from deployed contracts
3. **Market Data**: Integrate yfinance or other market data providers
4. **Agent Models**: Load trained models from `models/` directory

## 🐛 Troubleshooting

### App Won't Start

```bash
# Clear Streamlit cache
streamlit cache clear

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Slow Performance

- Reduce refresh rate in dashboard settings
- Use `@st.cache_data` for expensive operations
- Limit historical data range

### Connection Issues

- Check firewall settings
- Verify RPC endpoints are accessible
- Ensure correct network (Sepolia vs Mainnet)

## 📚 Documentation

- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Python](https://plotly.com/python/)
- [Web3.py Docs](https://web3py.readthedocs.io)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](../LICENSE) for details

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mohin-io/AI-DAO-Hedge-Fund/issues)
- **Email**: mohinhasin999@gmail.com
- **Discussions**: [GitHub Discussions](https://github.com/mohin-io/AI-DAO-Hedge-Fund/discussions)

---

<div align="center">

**Built with ❤️ using Streamlit, Plotly, and Web3**

[🏠 Home](../README.md) | [📋 Plan](../docs/PLAN.md) | [📜 Smart Contracts](../contracts/)

</div>
