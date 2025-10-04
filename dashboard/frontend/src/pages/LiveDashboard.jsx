import { useState, useEffect, useRef } from 'react';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const LiveDashboard = () => {
  const [portfolioData, setPortfolioData] = useState([]);
  const [currentValue, setCurrentValue] = useState(100000);
  const [totalReturn, setTotalReturn] = useState(0);
  const [sharpeRatio, setSharpeRatio] = useState(0);
  const [maxDrawdown, setMaxDrawdown] = useState(0);
  const [agents, setAgents] = useState([
    { name: 'Momentum Trader', allocation: 40, pnl: 0, trades: 0, status: 'active' },
    { name: 'Arbitrage Hunter', allocation: 35, pnl: 0, trades: 0, status: 'active' },
    { name: 'Risk Hedger', allocation: 25, pnl: 0, trades: 0, status: 'active' }
  ]);
  const [recentTrades, setRecentTrades] = useState([]);
  const [isLive, setIsLive] = useState(true);
  const wsRef = useRef(null);

  // Simulate real-time market data
  useEffect(() => {
    const initialValue = 100000;
    const timestamps = [];
    const values = [];

    // Generate initial historical data
    for (let i = 60; i >= 0; i--) {
      const time = new Date(Date.now() - i * 60000);
      timestamps.push(time.toLocaleTimeString());
      const randomReturn = (Math.random() - 0.48) * 200; // Slight upward bias
      const value = i === 60 ? initialValue : values[values.length - 1] + randomReturn;
      values.push(value);
    }

    setPortfolioData(timestamps.map((time, idx) => ({
      time,
      value: values[idx]
    })));
    setCurrentValue(values[values.length - 1]);

    // Simulate live updates
    const interval = setInterval(() => {
      if (!isLive) return;

      const now = new Date();
      const newTime = now.toLocaleTimeString();

      // Simulate market movement with realistic volatility
      const volatility = 150;
      const trend = 0.02; // Slight upward trend
      const randomChange = (Math.random() - 0.5 + trend) * volatility;

      setPortfolioData(prev => {
        const lastValue = prev[prev.length - 1].value;
        const newValue = Math.max(lastValue + randomChange, 50000); // Don't go below 50k

        const updated = [...prev.slice(-59), { time: newTime, value: newValue }];

        // Update metrics
        setCurrentValue(newValue);
        const returnPct = ((newValue - initialValue) / initialValue) * 100;
        setTotalReturn(returnPct);

        // Calculate Sharpe ratio (simplified)
        const returns = updated.map((d, i) =>
          i === 0 ? 0 : (d.value - updated[i-1].value) / updated[i-1].value
        );
        const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const stdDev = Math.sqrt(returns.reduce((a, b) => a + Math.pow(b - avgReturn, 2), 0) / returns.length);
        setSharpeRatio(stdDev === 0 ? 0 : (avgReturn / stdDev * Math.sqrt(252)));

        // Calculate max drawdown
        let peak = updated[0].value;
        let maxDD = 0;
        updated.forEach(d => {
          if (d.value > peak) peak = d.value;
          const dd = (peak - d.value) / peak;
          if (dd > maxDD) maxDD = dd;
        });
        setMaxDrawdown(maxDD * 100);

        return updated;
      });

      // Randomly update agent performance
      if (Math.random() > 0.7) {
        const agentIdx = Math.floor(Math.random() * 3);
        setAgents(prev => {
          const updated = [...prev];
          const pnlChange = (Math.random() - 0.4) * 500;
          updated[agentIdx] = {
            ...updated[agentIdx],
            pnl: updated[agentIdx].pnl + pnlChange,
            trades: updated[agentIdx].trades + 1
          };
          return updated;
        });

        // Add a trade to recent trades
        const agentNames = ['Momentum Trader', 'Arbitrage Hunter', 'Risk Hedger'];
        const actions = ['BUY', 'SELL'];
        const assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTC', 'ETH'];

        setRecentTrades(prev => [{
          time: newTime,
          agent: agentNames[agentIdx],
          action: actions[Math.floor(Math.random() * 2)],
          asset: assets[Math.floor(Math.random() * assets.length)],
          amount: (Math.random() * 10000).toFixed(2),
          price: (Math.random() * 1000).toFixed(2)
        }, ...prev.slice(0, 9)]);
      }
    }, 2000); // Update every 2 seconds

    return () => clearInterval(interval);
  }, [isLive]);

  const portfolioChartData = {
    labels: portfolioData.map(d => d.time),
    datasets: [
      {
        label: 'Portfolio Value',
        data: portfolioData.map(d => d.value),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2
      }
    ]
  };

  const allocationChartData = {
    labels: agents.map(a => a.name),
    datasets: [{
      data: agents.map(a => a.allocation),
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)'
      ],
      borderColor: [
        'rgb(59, 130, 246)',
        'rgb(16, 185, 129)',
        'rgb(245, 158, 11)'
      ],
      borderWidth: 2
    }]
  };

  const agentPerformanceData = {
    labels: agents.map(a => a.name),
    datasets: [{
      label: 'PnL ($)',
      data: agents.map(a => a.pnl),
      backgroundColor: agents.map(a => a.pnl >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'),
      borderColor: agents.map(a => a.pnl >= 0 ? 'rgb(16, 185, 129)' : 'rgb(239, 68, 68)'),
      borderWidth: 2
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false
        }
      },
      y: {
        display: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
              AI DAO Hedge Fund
            </h1>
            <p className="text-gray-400">Multi-Agent RL × Blockchain Governance</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
              <span className="text-sm">{isLive ? 'LIVE' : 'PAUSED'}</span>
            </div>
            <button
              onClick={() => setIsLive(!isLive)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              {isLive ? 'Pause' : 'Resume'}
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl p-6 shadow-2xl">
          <div className="text-sm text-blue-200 mb-2">Portfolio Value</div>
          <div className="text-3xl font-bold">${currentValue.toLocaleString('en-US', { maximumFractionDigits: 0 })}</div>
          <div className={`text-sm mt-2 ${totalReturn >= 0 ? 'text-green-300' : 'text-red-300'}`}>
            {totalReturn >= 0 ? '↑' : '↓'} {Math.abs(totalReturn).toFixed(2)}%
          </div>
        </div>

        <div className="bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-xl p-6 shadow-2xl">
          <div className="text-sm text-emerald-200 mb-2">Sharpe Ratio</div>
          <div className="text-3xl font-bold">{sharpeRatio.toFixed(2)}</div>
          <div className="text-sm mt-2 text-emerald-300">Risk-adjusted return</div>
        </div>

        <div className="bg-gradient-to-br from-amber-600 to-amber-700 rounded-xl p-6 shadow-2xl">
          <div className="text-sm text-amber-200 mb-2">Max Drawdown</div>
          <div className="text-3xl font-bold">{maxDrawdown.toFixed(2)}%</div>
          <div className="text-sm mt-2 text-amber-300">Peak to trough</div>
        </div>

        <div className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl p-6 shadow-2xl">
          <div className="text-sm text-purple-200 mb-2">Active Agents</div>
          <div className="text-3xl font-bold">{agents.filter(a => a.status === 'active').length}/3</div>
          <div className="text-sm mt-2 text-purple-300">Multi-agent system</div>
        </div>
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Portfolio Performance */}
        <div className="lg:col-span-2 bg-gray-800 rounded-xl p-6 shadow-2xl">
          <h2 className="text-xl font-bold mb-4">Portfolio Performance (Live)</h2>
          <div className="h-80">
            <Line data={portfolioChartData} options={chartOptions} />
          </div>
        </div>

        {/* Agent Allocation */}
        <div className="bg-gray-800 rounded-xl p-6 shadow-2xl">
          <h2 className="text-xl font-bold mb-4">Agent Allocation</h2>
          <div className="h-80 flex items-center justify-center">
            <Doughnut
              data={allocationChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'bottom',
                    labels: {
                      color: 'white',
                      padding: 15,
                      font: { size: 12 }
                    }
                  }
                }
              }}
            />
          </div>
        </div>
      </div>

      {/* Agent Performance & Recent Trades */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Agent Performance */}
        <div className="bg-gray-800 rounded-xl p-6 shadow-2xl">
          <h2 className="text-xl font-bold mb-4">Agent Performance (PnL)</h2>
          <div className="h-64">
            <Bar data={agentPerformanceData} options={chartOptions} />
          </div>
          <div className="mt-4 space-y-2">
            {agents.map((agent, idx) => (
              <div key={idx} className="flex justify-between items-center p-3 bg-gray-700 rounded-lg">
                <div>
                  <div className="font-semibold">{agent.name}</div>
                  <div className="text-sm text-gray-400">{agent.trades} trades</div>
                </div>
                <div className={`text-right ${agent.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  <div className="font-bold">${agent.pnl.toFixed(2)}</div>
                  <div className="text-xs">{agent.allocation}% allocated</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Trades */}
        <div className="bg-gray-800 rounded-xl p-6 shadow-2xl">
          <h2 className="text-xl font-bold mb-4">Recent Trades (Live)</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {recentTrades.length === 0 ? (
              <div className="text-center text-gray-500 py-8">Waiting for trades...</div>
            ) : (
              recentTrades.map((trade, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-gray-700 rounded-lg flex justify-between items-center animate-fade-in"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        trade.action === 'BUY' ? 'bg-green-600' : 'bg-red-600'
                      }`}>
                        {trade.action}
                      </span>
                      <span className="font-semibold">{trade.asset}</span>
                    </div>
                    <div className="text-xs text-gray-400 mt-1">{trade.agent}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold">${trade.amount}</div>
                    <div className="text-xs text-gray-400">{trade.time}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* DAO Governance Stats */}
      <div className="bg-gray-800 rounded-xl p-6 shadow-2xl">
        <h2 className="text-xl font-bold mb-4">Blockchain DAO Governance</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 bg-gray-700 rounded-lg">
            <div className="text-gray-400 text-sm mb-1">Active Proposals</div>
            <div className="text-2xl font-bold">3</div>
          </div>
          <div className="p-4 bg-gray-700 rounded-lg">
            <div className="text-gray-400 text-sm mb-1">Total Votes Cast</div>
            <div className="text-2xl font-bold">1,247</div>
          </div>
          <div className="p-4 bg-gray-700 rounded-lg">
            <div className="text-gray-400 text-sm mb-1">DAO Members</div>
            <div className="text-2xl font-bold">89</div>
          </div>
          <div className="p-4 bg-gray-700 rounded-lg">
            <div className="text-gray-400 text-sm mb-1">Smart Contracts</div>
            <div className="text-2xl font-bold text-green-400">✓ Verified</div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateX(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

export default LiveDashboard;
