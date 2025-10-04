/**
 * Settings Screen
 */

import React from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import { List, Switch, Divider } from 'react-native-paper';

export default function SettingsScreen() {
  const [notifications, setNotifications] = React.useState(true);
  const [darkMode, setDarkMode] = React.useState(false);

  return (
    <ScrollView style={styles.container}>
      <List.Section>
        <List.Subheader>Preferences</List.Subheader>
        <List.Item
          title="Push Notifications"
          right={() => <Switch value={notifications} onValueChange={setNotifications} />}
        />
        <Divider />
        <List.Item
          title="Dark Mode"
          right={() => <Switch value={darkMode} onValueChange={setDarkMode} />}
        />
      </List.Section>
      <List.Section>
        <List.Subheader>Account</List.Subheader>
        <List.Item title="Connected Wallet" description="0x1234...5678" />
        <Divider />
        <List.Item title="API Key" description="Configured" />
      </List.Section>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
});
