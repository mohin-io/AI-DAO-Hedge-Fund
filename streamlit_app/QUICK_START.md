# 🚀 Quick Start Guide - AI DAO Streamlit App

## ⚡ 5-Minute Setup

### Option 1: Run Locally (Fastest)

#### Windows
```bash
# Double-click this file:
run_local.bat
```

#### Mac/Linux
```bash
# Run this command:
chmod +x run_local.sh
./run_local.sh
```

#### Or manually:
```bash
cd streamlit_app
pip install -r requirements-minimal.txt
streamlit run app.py
```

**App opens at**: `http://localhost:8501`

---

### Option 2: Deploy to Streamlit Cloud (5 minutes)

1. **Fork/Clone to GitHub**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-DAO-Hedge-Fund.git
   ```

2. **Go to Streamlit Cloud**
   - Visit: [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Deploy**
   - Click "New app"
   - Repository: `YOUR_USERNAME/AI-DAO-Hedge-Fund`
   - Branch: `main`
   - Main file path: `streamlit_app/app.py`
   - Click "Deploy" ✨

4. **Live in 2-3 minutes!**
   - URL: `https://YOUR_USERNAME-ai-dao-hedge-fund.streamlit.app`

---

## 📱 What You'll See

### 8 Interactive Pages

1. **🏠 Home** - System overview & quick stats
2. **📊 Portfolio Dashboard** - Real-time monitoring
3. **🤖 AI Agents Control** - Configure PPO/DQN/SAC agents
4. **⛓️ DAO Governance** - Vote on proposals
5. **🔍 Explainability** - SHAP analysis for trades
6. **🎮 Trading Simulator** - Backtest & Monte Carlo
7. **🔗 Blockchain** - Smart contract interaction
8. **📈 Results** - Historical performance

---

## 🎯 Try These Features

### For Technical Evaluation
1. **AI Agents Control** → Explore ML configurations
2. **Explainability** → See SHAP waterfall plots
3. **Trading Simulator** → Run Monte Carlo simulations

### For DAO Demo
1. **DAO Governance** → Create/vote on proposals
2. **Blockchain Integration** → Interact with smart contracts

---

## 🔧 Troubleshooting

### "Module not found" error?
```bash
pip install -r requirements-minimal.txt
```

### Port 8501 already in use?
```bash
streamlit run app.py --server.port 8502
```

### App won't start?
```bash
# Clear cache
streamlit cache clear

# Verify installation
python -c "import streamlit; import plotly; import pandas; print('All good!')"
```

---

## 🌐 Live Demo

**Try it now without installing**:
👉 [https://ai-dao-hedge-fund.streamlit.app](https://ai-dao-hedge-fund.streamlit.app)

---

## 📊 Key Metrics to Explore

- **Portfolio Return**: +34.2% (vs S&P 500: +18.6%)
- **Sharpe Ratio**: 2.14 (institutional grade)
- **Max Drawdown**: -12.3% (38% better than benchmark)
- **Win Rate**: 58.3%
- **Active Agents**: 3 (Momentum PPO, Arbitrage DQN, Hedging SAC)

---

## 💡 Pro Tips

1. **Navigation**: Use sidebar to switch between pages
2. **Interactive Charts**: Hover for details, zoom/pan available
3. **Export**: Download reports from Explainability & Results pages
4. **Simulation**: Try Monte Carlo with 1000+ simulations
5. **DAO**: Create a test proposal in Governance page

---

## 📞 Need Help?

- **Issues**: [GitHub Issues](https://github.com/mohin-io/AI-DAO-Hedge-Fund/issues)
- **Docs**: See [README.md](README.md) and [DEPLOYMENT.md](DEPLOYMENT.md)
- **Email**: mohinhasin999@gmail.com

---

**Ready? Let's go! 🚀**

Choose Option 1 (local) or Option 2 (cloud) above and you'll be exploring the AI DAO in under 5 minutes!
