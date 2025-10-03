import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Shield } from 'lucide-react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'];

export default function Dashboard() {
  const [portfolioValue, setPortfolioValue] = useState(100000);
  const [dailyChange, setDailyChange] = useState(0);

  // Fetch portfolio data
  const { data: portfolio } = useQuery({
    queryKey: ['portfolio'],
    queryFn: async () => {
      const res = await axios.get(`${API_URL}/api/portfolio`);
      return res.data;
    },
    refetchInterval: 5000
  });

  // Fetch agents data
  const { data: agents } = useQuery({
    queryKey: ['agents'],
    queryFn: async () => {
      const res = await axios.get(`${API_URL}/api/agents`);
      return res.data;
    }
  });

  // Fetch performance summary
  const { data: performance } = useQuery({
    queryKey: ['performance'],
    queryFn: async () => {
      const res = await axios.get(`${API_URL}/api/performance/summary`);
      return res.data;
    }
  });

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/live`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'portfolio_update') {
        setPortfolioValue(data.portfolio_value);
        setDailyChange(data.change);
      }
    };

    return () => ws.close();
  }, []);

  // Mock historical data for charts
  const historicalData = Array.from({ length: 30 }, (_, i) => ({
    day: `Day ${i + 1}`,
    value: 100000 + Math.random() * 40000,
    benchmark: 100000 + Math.random() * 20000
  }));

  const agentAllocationData = agents?.map(agent => ({
    name: agent.name,
    value: agent.total_pnl || 0
  })) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">AI DAO Hedge Fund Dashboard</h1>
        <p className="text-gray-500 mt-2">Real-time autonomous trading performance</p>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Portfolio Value"
          value={`$${portfolioValue.toLocaleString('en-US', { maximumFractionDigits: 2 })}`}
          change={dailyChange}
          icon={<DollarSign className="w-6 h-6" />}
          color="blue"
        />
        <MetricCard
          title="Total Return"
          value={`${((portfolio?.total_return || 0) * 100).toFixed(2)}%`}
          change={portfolio?.total_return || 0}
          icon={<TrendingUp className="w-6 h-6" />}
          color="green"
        />
        <MetricCard
          title="Sharpe Ratio"
          value={(portfolio?.sharpe_ratio || 0).toFixed(2)}
          change={0.12}
          icon={<Activity className="w-6 h-6" />}
          color="purple"
        />
        <MetricCard
          title="Max Drawdown"
          value={`${((portfolio?.max_drawdown || 0) * 100).toFixed(2)}%`}
          change={-portfolio?.max_drawdown || 0}
          icon={<Shield className="w-6 h-6" />}
          color="red"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Performance Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Portfolio Performance</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#3b82f6" name="AI DAO Fund" strokeWidth={2} />
              <Line type="monotone" dataKey="benchmark" stroke="#9ca3af" name="S&P 500" strokeWidth={2} strokeDasharray="5 5" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Agent Allocation */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Agent Allocation</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={agentAllocationData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {agentAllocationData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Agent Performance Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold">AI Agent Performance</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategy</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trades</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Win Rate</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sharpe</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">PnL</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {agents?.map((agent) => (
                <tr key={agent.agent_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">{agent.name}</div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-500">{agent.strategy}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{agent.total_trades}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm ${agent.win_rate > 0.5 ? 'text-green-600' : 'text-red-600'}`}>
                      {(agent.win_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {agent.sharpe_ratio.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm font-medium ${agent.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${agent.total_pnl.toFixed(2)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, change, icon, color }) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
    red: 'bg-red-500'
  };

  const isPositive = change >= 0;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-2">{value}</p>
          <div className="flex items-center mt-2">
            {isPositive ? (
              <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
            )}
            <span className={`text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {(Math.abs(change) * 100).toFixed(2)}%
            </span>
          </div>
        </div>
        <div className={`${colorClasses[color]} p-3 rounded-full text-white`}>
          {icon}
        </div>
      </div>
    </div>
  );
}
