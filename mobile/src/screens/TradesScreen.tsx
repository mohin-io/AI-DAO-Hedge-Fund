/**
 * Trades Screen
 * Display trade history and recent activity
 */

import React from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import { Card, Title, List, Chip, Divider } from 'react-native-paper';

export default function TradesScreen() {
  const trades = [
    { id: 1, symbol: 'AAPL', side: 'BUY', quantity: 10, price: 180.50, timestamp: '2025-10-04 09:30' },
    { id: 2, symbol: 'GOOGL', side: 'SELL', quantity: 5, price: 142.30, timestamp: '2025-10-04 10:15' },
    { id: 3, symbol: 'MSFT', side: 'BUY', quantity: 15, price: 378.20, timestamp: '2025-10-04 11:00' },
  ];

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Content>
          <Title>Recent Trades</Title>
        </Card.Content>
        <List.Section>
          {trades.map((trade, index) => (
            <React.Fragment key={trade.id}>
              <List.Item
                title={`${trade.symbol} - ${trade.side}`}
                description={`${trade.quantity} @ $${trade.price} | ${trade.timestamp}`}
                right={() => (
                  <Chip
                    style={{ backgroundColor: trade.side === 'BUY' ? '#E8F5E9' : '#FFEBEE' }}
                    textStyle={{ color: trade.side === 'BUY' ? '#4CAF50' : '#F44336' }}
                  >
                    {trade.side}
                  </Chip>
                )}
              />
              {index < trades.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List.Section>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  card: { margin: 10, elevation: 4 },
});
