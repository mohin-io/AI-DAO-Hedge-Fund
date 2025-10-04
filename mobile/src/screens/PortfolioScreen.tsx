/**
 * Portfolio Screen
 * Display current holdings and positions
 */

import React, { useState, useEffect } from 'react';
import { View, ScrollView, StyleSheet, RefreshControl } from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  List,
  Divider,
  ActivityIndicator,
  Chip,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { api } from '../services/api';

interface Position {
  symbol: string;
  quantity: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  asset_type: string;
}

export default function PortfolioScreen() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [positions, setPositions] = useState<Position[]>([]);
  const [totalValue, setTotalValue] = useState(0);
  const [totalPnL, setTotalPnL] = useState(0);

  useEffect(() => {
    fetchPositions();
  }, []);

  const fetchPositions = async () => {
    try {
      const response = await api.get('/api/portfolio/positions');
      setPositions(response.data.positions);
      setTotalValue(response.data.total_value);
      setTotalPnL(response.data.total_pnl);
      setLoading(false);
      setRefreshing(false);
    } catch (error) {
      console.error('Error fetching positions:', error);
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchPositions();
  };

  const getAssetIcon = (assetType: string) => {
    switch (assetType) {
      case 'stock':
        return 'trending-up';
      case 'crypto':
        return 'logo-bitcoin';
      case 'option':
        return 'contract';
      default:
        return 'cash';
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#1976D2" />
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
      {/* Portfolio Summary */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Portfolio Summary</Title>
          <Paragraph style={styles.valueText}>
            ${totalValue.toLocaleString('en-US', {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
          </Paragraph>
          <View style={styles.pnlContainer}>
            <Ionicons
              name={totalPnL >= 0 ? 'arrow-up' : 'arrow-down'}
              size={20}
              color={totalPnL >= 0 ? '#4CAF50' : '#F44336'}
            />
            <Paragraph
              style={[
                styles.pnlText,
                { color: totalPnL >= 0 ? '#4CAF50' : '#F44336' },
              ]}
            >
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)} Total P&L
            </Paragraph>
          </View>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Paragraph style={styles.statLabel}>Positions</Paragraph>
              <Paragraph style={styles.statValue}>{positions.length}</Paragraph>
            </View>
            <View style={styles.statItem}>
              <Paragraph style={styles.statLabel}>Assets</Paragraph>
              <Paragraph style={styles.statValue}>
                {new Set(positions.map((p) => p.asset_type)).size}
              </Paragraph>
            </View>
          </View>
        </Card.Content>
      </Card>

      {/* Positions List */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Positions</Title>
        </Card.Content>
        <List.Section>
          {positions.map((position, index) => (
            <React.Fragment key={position.symbol}>
              <List.Item
                title={position.symbol}
                description={`${position.quantity} shares @ $${position.avg_entry_price.toFixed(
                  2
                )}`}
                left={(props) => (
                  <List.Icon
                    {...props}
                    icon={() => (
                      <Ionicons
                        name={getAssetIcon(position.asset_type)}
                        size={24}
                        color="#1976D2"
                      />
                    )}
                  />
                )}
                right={() => (
                  <View style={styles.positionRight}>
                    <Paragraph style={styles.positionValue}>
                      ${position.market_value.toFixed(2)}
                    </Paragraph>
                    <Chip
                      style={[
                        styles.pnlChip,
                        {
                          backgroundColor:
                            position.unrealized_pnl >= 0 ? '#E8F5E9' : '#FFEBEE',
                        },
                      ]}
                      textStyle={{
                        color: position.unrealized_pnl >= 0 ? '#4CAF50' : '#F44336',
                        fontSize: 12,
                      }}
                    >
                      {position.unrealized_pnl >= 0 ? '+' : ''}
                      {position.unrealized_pnl_pct.toFixed(2)}%
                    </Chip>
                  </View>
                )}
              />
              {index < positions.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List.Section>
      </Card>

      {/* Asset Allocation */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Asset Allocation</Title>
          <View style={styles.allocationContainer}>
            {positions.map((position) => {
              const allocation = (position.market_value / totalValue) * 100;
              return (
                <View key={position.symbol} style={styles.allocationItem}>
                  <View style={styles.allocationBar}>
                    <View
                      style={[
                        styles.allocationFill,
                        { width: `${allocation}%` },
                      ]}
                    />
                  </View>
                  <View style={styles.allocationLabel}>
                    <Paragraph style={styles.allocationSymbol}>
                      {position.symbol}
                    </Paragraph>
                    <Paragraph style={styles.allocationPercent}>
                      {allocation.toFixed(1)}%
                    </Paragraph>
                  </View>
                </View>
              );
            })}
          </View>
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
    marginBottom: 15,
  },
  pnlText: {
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 5,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 10,
  },
  statItem: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 5,
  },
  positionRight: {
    alignItems: 'flex-end',
    justifyContent: 'center',
  },
  positionValue: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 5,
  },
  pnlChip: {
    height: 24,
  },
  allocationContainer: {
    marginTop: 15,
  },
  allocationItem: {
    marginBottom: 15,
  },
  allocationBar: {
    height: 8,
    backgroundColor: '#E0E0E0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  allocationFill: {
    height: '100%',
    backgroundColor: '#1976D2',
  },
  allocationLabel: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 5,
  },
  allocationSymbol: {
    fontSize: 14,
    fontWeight: '600',
  },
  allocationPercent: {
    fontSize: 14,
    color: '#666',
  },
});
