/**
 * Home Screen - Dashboard Overview
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  Dimensions,
} from 'react-native';
import { Card, Title, Paragraph, Button } from 'react-native-paper';
import { LineChart } from 'react-native-chart-kit';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

const screenWidth = Dimensions.get('window').width;

export default function HomeScreen({ navigation }) {
  const [refreshing, setRefreshing] = useState(false);
  const [portfolioData, setPortfolioData] = useState({
    value: 1247893,
    dailyChange: 24567,
    dailyChangePercent: 2.01,
    totalReturn: 34.2,
    sharpeRatio: 2.14,
    maxDrawdown: -12.3,
    activeAgents: 3,
  });

  const chartData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        data: [100, 108, 112, 118, 125, 134.2],
        color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
        strokeWidth: 3,
      },
    ],
  };

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Portfolio Value Card */}
      <Card style={styles.mainCard}>
        <Card.Content>
          <Title style={styles.mainCardTitle}>Portfolio Value</Title>
          <Text style={styles.mainCardValue}>
            {formatCurrency(portfolioData.value)}
          </Text>
          <View style={styles.changeContainer}>
            <Icon
              name={portfolioData.dailyChange >= 0 ? 'arrow-up' : 'arrow-down'}
              size={20}
              color={portfolioData.dailyChange >= 0 ? '#10b981' : '#ef4444'}
            />
            <Text
              style={[
                styles.changeText,
                {
                  color:
                    portfolioData.dailyChange >= 0 ? '#10b981' : '#ef4444',
                },
              ]}
            >
              {formatCurrency(Math.abs(portfolioData.dailyChange))} (
              {formatPercent(portfolioData.dailyChangePercent)})
            </Text>
          </View>
          <Paragraph style={styles.subtext}>Today's Change</Paragraph>
        </Card.Content>
      </Card>

      {/* Performance Chart */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>6-Month Performance</Title>
          <LineChart
            data={chartData}
            width={screenWidth - 60}
            height={220}
            chartConfig={{
              backgroundColor: '#ffffff',
              backgroundGradientFrom: '#ffffff',
              backgroundGradientTo: '#ffffff',
              decimalPlaces: 1,
              color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
              style: {
                borderRadius: 16,
              },
              propsForDots: {
                r: '6',
                strokeWidth: '2',
                stroke: '#667eea',
              },
            }}
            bezier
            style={styles.chart}
          />
        </Card.Content>
      </Card>

      {/* Metrics Grid */}
      <View style={styles.metricsGrid}>
        <Card style={styles.metricCard}>
          <Card.Content>
            <Icon name="chart-line" size={32} color="#667eea" />
            <Text style={styles.metricValue}>
              {formatPercent(portfolioData.totalReturn)}
            </Text>
            <Text style={styles.metricLabel}>Total Return</Text>
          </Card.Content>
        </Card>

        <Card style={styles.metricCard}>
          <Card.Content>
            <Icon name="trending-up" size={32} color="#10b981" />
            <Text style={styles.metricValue}>{portfolioData.sharpeRatio}</Text>
            <Text style={styles.metricLabel}>Sharpe Ratio</Text>
          </Card.Content>
        </Card>

        <Card style={styles.metricCard}>
          <Card.Content>
            <Icon name="trending-down" size={32} color="#ef4444" />
            <Text style={styles.metricValue}>
              {formatPercent(portfolioData.maxDrawdown)}
            </Text>
            <Text style={styles.metricLabel}>Max Drawdown</Text>
          </Card.Content>
        </Card>

        <Card style={styles.metricCard}>
          <Card.Content>
            <Icon name="robot" size={32} color="#8b5cf6" />
            <Text style={styles.metricValue}>{portfolioData.activeAgents}</Text>
            <Text style={styles.metricLabel}>Active Agents</Text>
          </Card.Content>
        </Card>
      </View>

      {/* Quick Actions */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Quick Actions</Title>
          <View style={styles.buttonRow}>
            <Button
              mode="contained"
              icon="chart-line"
              onPress={() => navigation.navigate('Portfolio')}
              style={styles.actionButton}
            >
              View Details
            </Button>
            <Button
              mode="contained"
              icon="currency-usd"
              onPress={() => navigation.navigate('Trading')}
              style={styles.actionButton}
            >
              Trade
            </Button>
          </View>
          <View style={styles.buttonRow}>
            <Button
              mode="outlined"
              icon="robot"
              onPress={() => navigation.navigate('Agents')}
              style={styles.actionButton}
            >
              AI Agents
            </Button>
            <Button
              mode="outlined"
              icon="vote"
              onPress={() => navigation.navigate('DAO')}
              style={styles.actionButton}
            >
              Governance
            </Button>
          </View>
        </Card.Content>
      </Card>

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Powered by Multi-Agent RL & Blockchain DAO
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
  },
  mainCard: {
    margin: 15,
    marginTop: 10,
    elevation: 4,
    backgroundColor: '#667eea',
  },
  mainCardTitle: {
    color: '#ffffff',
    fontSize: 16,
    opacity: 0.9,
  },
  mainCardValue: {
    color: '#ffffff',
    fontSize: 36,
    fontWeight: 'bold',
    marginVertical: 10,
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  changeText: {
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 5,
  },
  subtext: {
    color: '#ffffff',
    opacity: 0.8,
    marginTop: 5,
  },
  card: {
    margin: 15,
    elevation: 2,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    paddingHorizontal: 15,
  },
  metricCard: {
    width: '48%',
    marginBottom: 15,
    elevation: 2,
  },
  metricValue: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 10,
    color: '#1f2937',
  },
  metricLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 5,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  actionButton: {
    flex: 1,
    marginHorizontal: 5,
  },
  footer: {
    padding: 20,
    alignItems: 'center',
  },
  footerText: {
    color: '#6b7280',
    fontSize: 12,
  },
});
