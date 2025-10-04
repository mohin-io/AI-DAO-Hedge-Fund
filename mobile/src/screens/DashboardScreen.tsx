/**
 * Dashboard Screen
 * Main overview with portfolio performance and AI agent status
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  RefreshControl,
  Dimensions,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Chip,
  DataTable,
  ActivityIndicator,
} from 'react-native-paper';
import { LineChart } from 'react-native-chart-kit';
import { Ionicons } from '@expo/vector-icons';
import { api } from '../services/api';

const screenWidth = Dimensions.get('window').width;

interface DashboardData {
  portfolio_value: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  total_return: number;
  sharpe_ratio: number;
  win_rate: number;
  active_agents: number;
  total_trades: number;
  equity_curve: number[];
  timestamps: string[];
}

export default function DashboardScreen() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [data, setData] = useState<DashboardData | null>(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await api.get('/api/dashboard');
      setData(response.data);
      setLoading(false);
      setRefreshing(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchDashboardData();
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#1976D2" />
      </View>
    );
  }

  if (!data) {
    return (
      <View style={styles.loadingContainer}>
        <Paragraph>Unable to load dashboard data</Paragraph>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Portfolio Value Card */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Portfolio Value</Title>
          <Paragraph style={styles.valueText}>
            ${data.portfolio_value.toLocaleString('en-US', {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
          </Paragraph>
          <View style={styles.pnlContainer}>
            <Ionicons
              name={data.daily_pnl >= 0 ? 'trending-up' : 'trending-down'}
              size={20}
              color={data.daily_pnl >= 0 ? '#4CAF50' : '#F44336'}
            />
            <Paragraph
              style={[
                styles.pnlText,
                { color: data.daily_pnl >= 0 ? '#4CAF50' : '#F44336' },
              ]}
            >
              {data.daily_pnl >= 0 ? '+' : ''}
              ${data.daily_pnl.toFixed(2)} ({data.daily_pnl_pct >= 0 ? '+' : ''}
              {data.daily_pnl_pct.toFixed(2)}%)
            </Paragraph>
          </View>
        </Card.Content>
      </Card>

      {/* Performance Chart */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Equity Curve</Title>
          <LineChart
            data={{
              labels: data.timestamps.slice(-7), // Last 7 data points
              datasets: [
                {
                  data: data.equity_curve.slice(-7),
                },
              ],
            }}
            width={screenWidth - 60}
            height={220}
            chartConfig={{
              backgroundColor: '#1976D2',
              backgroundGradientFrom: '#1976D2',
              backgroundGradientTo: '#64B5F6',
              decimalPlaces: 0,
              color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
              style: {
                borderRadius: 16,
              },
            }}
            bezier
            style={styles.chart}
          />
        </Card.Content>
      </Card>

      {/* Performance Metrics */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Performance Metrics</Title>
          <View style={styles.metricsGrid}>
            <View style={styles.metricItem}>
              <Paragraph style={styles.metricLabel}>Total Return</Paragraph>
              <Paragraph style={styles.metricValue}>
                {data.total_return >= 0 ? '+' : ''}
                {(data.total_return * 100).toFixed(2)}%
              </Paragraph>
            </View>
            <View style={styles.metricItem}>
              <Paragraph style={styles.metricLabel}>Sharpe Ratio</Paragraph>
              <Paragraph style={styles.metricValue}>
                {data.sharpe_ratio.toFixed(2)}
              </Paragraph>
            </View>
            <View style={styles.metricItem}>
              <Paragraph style={styles.metricLabel}>Win Rate</Paragraph>
              <Paragraph style={styles.metricValue}>
                {(data.win_rate * 100).toFixed(1)}%
              </Paragraph>
            </View>
            <View style={styles.metricItem}>
              <Paragraph style={styles.metricLabel}>Total Trades</Paragraph>
              <Paragraph style={styles.metricValue}>
                {data.total_trades}
              </Paragraph>
            </View>
          </View>
        </Card.Content>
      </Card>

      {/* AI Agents Status */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>AI Agents</Title>
          <View style={styles.agentsContainer}>
            <Chip
              icon={() => (
                <Ionicons name="checkmark-circle" size={18} color="#4CAF50" />
              )}
              style={styles.chip}
            >
              {data.active_agents} Active Agents
            </Chip>
            <Chip
              icon={() => <Ionicons name="trending-up" size={18} color="#2196F3" />}
              style={styles.chip}
            >
              Momentum Agent
            </Chip>
            <Chip
              icon={() => <Ionicons name="shuffle" size={18} color="#FF9800" />}
              style={styles.chip}
            >
              Arbitrage Agent
            </Chip>
            <Chip
              icon={() => <Ionicons name="shield" size={18} color="#9C27B0" />}
              style={styles.chip}
            >
              Hedging Agent
            </Chip>
          </View>
        </Card.Content>
      </Card>

      {/* Quick Stats */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Quick Stats</Title>
          <DataTable>
            <DataTable.Row>
              <DataTable.Cell>Active Positions</DataTable.Cell>
              <DataTable.Cell numeric>5</DataTable.Cell>
            </DataTable.Row>
            <DataTable.Row>
              <DataTable.Cell>Pending Orders</DataTable.Cell>
              <DataTable.Cell numeric>2</DataTable.Cell>
            </DataTable.Row>
            <DataTable.Row>
              <DataTable.Cell>Today's Trades</DataTable.Cell>
              <DataTable.Cell numeric>12</DataTable.Cell>
            </DataTable.Row>
            <DataTable.Row>
              <DataTable.Cell>Cash Available</DataTable.Cell>
              <DataTable.Cell numeric>$25,340.50</DataTable.Cell>
            </DataTable.Row>
          </DataTable>
        </Card.Content>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  card: {
    margin: 10,
    elevation: 4,
  },
  valueText: {
    fontSize: 32,
    fontWeight: 'bold',
    marginVertical: 10,
  },
  pnlContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 5,
  },
  pnlText: {
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 5,
  },
  chart: {
    marginVertical: 10,
    borderRadius: 16,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  metricItem: {
    width: '48%',
    marginBottom: 15,
  },
  metricLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  agentsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 10,
  },
  chip: {
    margin: 5,
  },
});
