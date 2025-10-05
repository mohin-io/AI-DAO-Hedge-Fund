import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { Card, Title } from 'react-native-paper';

export default function AgentsScreen() {
  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Content>
          <Title>AI Agents</Title>
          <Text>Agent monitoring and control coming soon...</Text>
        </Card.Content>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f3f4f6' },
  card: { margin: 15, elevation: 2 },
});
