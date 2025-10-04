/**
 * Agents Screen
 * Monitor AI trading agents
 */

import React from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import { Card, Title, Paragraph, ProgressBar, Chip } from 'react-native-paper';

export default function AgentsScreen() {
  const agents = [
    { name: 'Momentum Agent', status: 'active', performance: 0.85, trades: 45 },
    { name: 'Arbitrage Agent', status: 'active', performance: 0.72, trades: 128 },
    { name: 'Hedging Agent', status: 'active', performance: 0.91, trades: 23 },
  ];

  return (
    <ScrollView style={styles.container}>
      {agents.map((agent) => (
        <Card key={agent.name} style={styles.card}>
          <Card.Content>
            <View style={styles.header}>
              <Title>{agent.name}</Title>
              <Chip style={{ backgroundColor: '#E8F5E9' }} textStyle={{ color: '#4CAF50' }}>
                {agent.status.toUpperCase()}
              </Chip>
            </View>
            <Paragraph style={styles.metric}>
              Performance Score: {(agent.performance * 100).toFixed(1)}%
            </Paragraph>
            <ProgressBar progress={agent.performance} color="#1976D2" style={styles.progress} />
            <Paragraph style={styles.trades}>Total Trades: {agent.trades}</Paragraph>
          </Card.Content>
        </Card>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  card: { margin: 10, elevation: 4 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  metric: { marginTop: 10, fontSize: 14, color: '#666' },
  progress: { marginTop: 10, height: 8, borderRadius: 4 },
  trades: { marginTop: 15, fontSize: 14, fontWeight: '600' },
});
