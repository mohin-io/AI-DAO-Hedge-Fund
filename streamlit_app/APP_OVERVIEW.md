# 🎮 Decentralized Autonomous Hedge Fund AI DAO - Streamlit Agentic App

## 📋 Overview

A comprehensive, production-ready Streamlit application providing full control and monitoring capabilities for the Decentralized Autonomous Hedge Fund powered by Multi-Agent Reinforcement Learning and Blockchain DAO.

## ✨ Key Features

### 1. 🏠 Home Dashboard
**Purpose**: System overview and quick navigation

**Features**:
- Real-time portfolio metrics (4 key indicators)
- System architecture visualization
- Performance vs benchmark comparison
- Recent activity feed (AI agents + DAO)
- Technology stack display
- Quick action buttons

**Use Case**: First landing page for all users, provides high-level system health at a glance

---

### 2. 📊 Portfolio Dashboard
**Purpose**: Real-time portfolio monitoring and risk management

**Features**:
- **Live Metrics**: Portfolio value, daily P&L, Sharpe ratio, max drawdown, win rate
- **Performance Chart**: Historical performance vs S&P 500 benchmark
- **Asset Allocation**: Interactive pie chart (Equities, Crypto, Options, Cash)
- **Agent Performance**: Bar chart showing P&L by agent (Momentum, Arbitrage, Hedging)
- **Dynamic Weight Allocation**: Stacked area chart showing agent weight changes over time
- **Risk Metrics**: Volatility, VaR (95%), Beta, Drawdown analysis
- **Recent Trades Table**: Last 5 trades with agent, action, asset, P&L, confidence
- **Market Regime Detection**: Current regime (Bull/Bear/Sideways/Volatile) with probability breakdown

**Use Case**: Day-to-day portfolio monitoring, risk assessment, performance tracking

---

### 3. 🤖 AI Agents Control
**Purpose**: Individual agent monitoring, configuration, and training

**Features**:
- **Agent Status Cards**: 3 gradient cards showing status, P&L, win rate, active positions
- **Agent Selection**: Detailed view for each agent (PPO, DQN, SAC)
- **Configuration Panel**:
  - Algorithm details (PPO, DQN, SAC)
  - Strategy description
  - Technical indicators used
  - Model path and training steps
  - Performance metrics (return, Sharpe, max DD, avg trade)
- **Agent Controls**:
  - Status toggle (Active/Paused/Stopped)
  - Risk parameters (max position size, stop loss)
  - Save/retrain/download options
- **Performance Charts**:
  - Cumulative P&L over time
  - Action distribution (BUY/SELL/HOLD/CLOSE)
  - Trade duration histogram
- **Training Metrics**:
  - Reward curve with moving average
  - Loss curve with moving average
- **Hyperparameter Tuning**: Full hyperparameter editor (learning rate, gamma, batch size, etc.)
- **Recent Actions Table**: Last 5 actions with timestamp, asset, confidence, reasoning

**Use Case**: Quant researchers, ML engineers configuring and monitoring individual agents

---

### 4. ⛓️ DAO Governance
**Purpose**: Decentralized governance and voting interface

**Features**:
- **Network Status**: DAO members, active proposals, votes cast, treasury balance
- **Blockchain Connection**:
  - Network selector (Mainnet, Sepolia, Polygon, Arbitrum)
  - Wallet address display
  - Connection status
  - Smart contract addresses with Etherscan links
- **Active Proposals**:
  - Expandable proposal cards with full details
  - Voting interface (For/Against with voting power slider)
  - Progress bars showing approval rate
  - Quorum status indicators
  - Execute button for passed proposals
- **Create New Proposal**:
  - Form with title, description, type
  - Voting period and quorum configuration
  - Submit to blockchain
- **Voting Analytics**:
  - Proposal outcomes pie chart (Passed/Rejected/Pending)
  - Proposal types bar chart
  - Participation trend line chart
- **DAO Leaderboard**: Top contributors by proposals, votes, voting power, reputation
- **Treasury Management**: Balance, fees, withdrawals, reserve ratio
- **Governance Parameters**: Voting, economic, and agent parameters

**Use Case**: DAO members voting on proposals, creating new proposals, monitoring governance

---

### 5. 🔍 Explainability (SHAP)
**Purpose**: AI decision transparency and explainability

**Features**:
- **Trade Selection**: Table of recent trades with ID, timestamp, agent, action, asset, P&L
- **SHAP Waterfall Plot**:
  - Visual breakdown of feature contributions
  - Base value → feature impacts → final decision
  - Color-coded (green = positive, red = negative)
- **Feature Importance Ranking**:
  - Horizontal bar chart sorted by absolute SHAP value
  - Top contributing features highlighted
  - Detailed explanation panel
- **Decision Confidence**:
  - Gauge chart showing overall confidence (0-100%)
  - Confidence factors breakdown (technical, regime, risk, historical)
  - Risk assessment with best/base/worst case scenarios
- **Alternative Actions**: Table showing why other actions weren't chosen
- **SHAP Summary Plot**: Box plots across last 100 trades showing feature importance distribution
- **Detailed Text Explanation**:
  - Full narrative of decision rationale
  - Primary and secondary signals
  - Caution signals
  - Risk management details
  - Expected outcomes
- **Export Options**: PDF report, CSV data, PNG visualizations

**Use Case**: Compliance, auditing, understanding individual trade decisions, regulatory reporting

---

### 6. 🎮 Trading Simulator
**Purpose**: Backtesting and simulation for strategy validation

**Modes**:

#### A. Historical Backtest
- **Configuration**:
  - Date range selector
  - Multi-asset selection
  - Initial capital
  - Transaction costs (bps)
  - Agent selection
  - Advanced settings (rebalance frequency, leverage, risk limits, stop loss)
- **Results**:
  - Performance metrics (return, Sharpe, max DD, win rate, total trades)
  - Equity curve vs benchmark
  - Drawdown analysis chart
  - Monthly returns heatmap
  - Trade statistics table
  - Win/loss distribution histogram

#### B. Live Simulation
- **Controls**:
  - Simulation speed selector (1x to 50x)
  - Market regime selector (Random, Bull, Bear, High/Low volatility)
  - Start/Pause/Reset buttons
- **Live Metrics**: Portfolio value, P&L, open positions, agent activity
- **Live Chart**: Real-time updating performance chart

#### C. Monte Carlo Simulation
- **Configuration**:
  - Number of simulations (100-10,000)
  - Time horizon (30-365 days)
  - Confidence level (90%, 95%, 99%)
  - Initial capital
- **Results**:
  - Simulation paths visualization (50 paths shown)
  - Percentile bands (5th, 50th, 95th)
  - Summary statistics (median, mean, percentiles)
  - Probability of profit
  - Expected return

**Use Case**: Strategy testing, risk scenario analysis, performance validation

---

### 7. 🔗 Blockchain Integration
**Purpose**: Direct smart contract interaction

**Features**:
- **Network Status**: Network, block number, gas price, your balance
- **Smart Contract Tabs**:

  **DAO Governance**:
  - Read: Total proposals, voting power, quorum threshold
  - Write: Cast vote (with proposal ID and For/Against selection)

  **Treasury Manager**:
  - Read: Total assets, share price
  - Write: Deposit ETH, withdraw shares

  **Agent Registry**:
  - Read: Registered agents count, agent performance by ID
  - Write: Register new agent (requires stake)

- **Transaction History**: Recent transactions with hash, function, status, block, gas, timestamp
- **Gas Analytics**: Total gas spent, average price, transaction count, failure rate

**Use Case**: Advanced users, DAO members, agent operators interacting directly with contracts

---

### 8. 📈 Backtesting Results
**Purpose**: Historical performance analysis and benchmarking

**Features**:
- **Performance Summary**: 5 key metrics with deltas vs benchmark
- **Agent Comparison Table**:
  - Ensemble vs individual agents vs S&P 500
  - Columns: Total return, Sharpe, max DD, win rate, volatility
- **Cumulative Returns Chart**:
  - Line chart with 5 series (Ensemble, Momentum, Arbitrage, Hedging, S&P 500)
  - Full 2020-2025 period
- **Rolling Sharpe Ratio**: 90-day rolling Sharpe with target line
- **Underwater Plot**: Drawdown from peak visualization
- **Export Options**: PDF report, CSV metrics, trade log

**Use Case**: Performance reporting, investor presentations, academic research

---

## 🎨 Design System

### Color Palette
- **Primary Gradient**: `#667eea` → `#764ba2` (Purple/Blue)
- **Secondary Gradient**: `#f093fb` → `#f5576c` (Pink/Red)
- **Accent Gradient**: `#fa709a` → `#fee140` (Pink/Yellow)
- **Success**: `#00ff00` (Green)
- **Warning**: `#ffff00` (Yellow)
- **Error**: `#ff0000` (Red)

### Typography
- **Font**: Sans serif (system default)
- **Headers**: Gradient text for main title
- **Metrics**: Bold, large numbers with delta indicators

### Components
- **Metric Cards**: Gradient backgrounds with white text
- **Charts**: Plotly interactive charts with dark gridlines
- **Tables**: Streamlit dataframes with custom column configs
- **Forms**: Streamlit native inputs with validation
- **Buttons**: Gradient backgrounds, full-width options

---

## 🔧 Technical Architecture

### Framework
- **Frontend**: Streamlit 1.29+
- **Visualization**: Plotly 5.18+
- **Data**: Pandas 2.1+, NumPy 1.26+
- **Blockchain**: Web3 6.11+ (optional)

### Data Flow
1. **Simulated Data**: Currently uses NumPy random data for demonstration
2. **Real Data Integration** (Future):
   - Backend API: FastAPI REST endpoints
   - WebSocket: Real-time updates
   - Blockchain: Web3 contract calls
   - Market Data: yfinance integration

### State Management
- **Session State**: Streamlit's built-in session state for simulation controls
- **Caching**: `@st.cache_data` for expensive computations (to be added)

### Security
- **Secrets Management**: `.streamlit/secrets.toml` for sensitive data
- **Input Validation**: Form validation on all user inputs
- **CORS**: Enabled for API integration

---

## 📊 Data Requirements

### Current (Simulated)
- All data generated via NumPy random functions
- Realistic distributions and correlations
- Time series with proper volatility

### Future (Production)
1. **Portfolio Data**: API endpoint `/api/portfolio`
2. **Agent Performance**: API endpoint `/api/agents/{agent_id}/performance`
3. **Blockchain Data**: Web3 contract reads
4. **Market Data**: yfinance historical data
5. **Trade History**: Database queries

---

## 🚀 Deployment Options

### 1. Streamlit Cloud (Recommended)
- **Pros**: Free, automatic SSL, easy updates
- **URL**: `https://ai-dao-hedge-fund.streamlit.app`
- **Setup**: Connect GitHub repo, set main file path

### 2. Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### 3. Heroku
```bash
echo "web: streamlit run app.py" > Procfile
heroku create ai-dao-hedge-fund
git push heroku main
```

### 4. Local Development
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 User Personas

### 1. **Portfolio Manager**
- **Primary Pages**: Portfolio Dashboard, Backtesting Results
- **Goals**: Monitor performance, assess risk, review historical returns
- **Frequency**: Daily

### 2. **Quant Researcher**
- **Primary Pages**: AI Agents Control, Trading Simulator, Explainability
- **Goals**: Configure agents, run simulations, understand decisions
- **Frequency**: Weekly (during development), Monthly (maintenance)

### 3. **DAO Member**
- **Primary Pages**: DAO Governance, Blockchain Integration
- **Goals**: Vote on proposals, monitor treasury, interact with contracts
- **Frequency**: Per proposal (weekly/monthly)

### 4. **Compliance Officer**
- **Primary Pages**: Explainability, Backtesting Results, Portfolio Dashboard
- **Goals**: Audit decisions, generate reports, verify transparency
- **Frequency**: Monthly/Quarterly

### 5. **Investor/Recruiter**
- **Primary Pages**: Home, Portfolio Dashboard, Backtesting Results
- **Goals**: Understand system, evaluate performance, assess technology
- **Frequency**: One-time (demo/evaluation)

---

## 📈 Metrics & Analytics

### Built-in Analytics
- **Portfolio Metrics**: Return, Sharpe, max DD, win rate, volatility, beta, VaR
- **Agent Metrics**: P&L, win rate, action distribution, trade duration
- **DAO Metrics**: Proposals, votes, participation rate, treasury balance
- **Blockchain Metrics**: Gas spent, transaction count, failure rate

### Export Capabilities
- **PDF Reports**: Full portfolio/SHAP reports
- **CSV Data**: Metrics, trade logs, feature values
- **PNG Images**: Chart visualizations

---

## 🔮 Future Enhancements

### Phase 1 (Near-term)
- [ ] Connect to real backend API
- [ ] Implement caching for performance
- [ ] Add user authentication
- [ ] Real-time WebSocket updates

### Phase 2 (Mid-term)
- [ ] Mobile-responsive design
- [ ] Custom alerts and notifications
- [ ] Advanced filtering and search
- [ ] Multi-language support

### Phase 3 (Long-term)
- [ ] Machine learning model upload
- [ ] Custom strategy builder
- [ ] Social features (sharing, comments)
- [ ] API for third-party integrations

---

## 📚 Documentation Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Python Docs](https://plotly.com/python/)
- [Web3.py Docs](https://web3py.readthedocs.io)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 🤝 Contributing

See main [README.md](../README.md) for contribution guidelines.

---

**Built with ❤️ for transparency, explainability, and decentralized finance**
