/**
 * AI DAO Hedge Fund Mobile App
 * React Native + Expo application for iOS and Android
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';

// Screens
import DashboardScreen from './src/screens/DashboardScreen';
import PortfolioScreen from './src/screens/PortfolioScreen';
import TradesScreen from './src/screens/TradesScreen';
import AgentsScreen from './src/screens/AgentsScreen';
import SettingsScreen from './src/screens/SettingsScreen';

// Theme
import { theme } from './src/theme';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <PaperProvider theme={theme}>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            tabBarIcon: ({ focused, color, size }) => {
              let iconName: keyof typeof Ionicons.glyphMap;

              if (route.name === 'Dashboard') {
                iconName = focused ? 'stats-chart' : 'stats-chart-outline';
              } else if (route.name === 'Portfolio') {
                iconName = focused ? 'wallet' : 'wallet-outline';
              } else if (route.name === 'Trades') {
                iconName = focused ? 'swap-horizontal' : 'swap-horizontal-outline';
              } else if (route.name === 'Agents') {
                iconName = focused ? 'cube' : 'cube-outline';
              } else if (route.name === 'Settings') {
                iconName = focused ? 'settings' : 'settings-outline';
              } else {
                iconName = 'help-circle-outline';
              }

              return <Ionicons name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: theme.colors.primary,
            tabBarInactiveTintColor: 'gray',
            headerStyle: {
              backgroundColor: theme.colors.primary,
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          })}
        >
          <Tab.Screen
            name="Dashboard"
            component={DashboardScreen}
            options={{ title: 'AI DAO Dashboard' }}
          />
          <Tab.Screen
            name="Portfolio"
            component={PortfolioScreen}
            options={{ title: 'Portfolio' }}
          />
          <Tab.Screen
            name="Trades"
            component={TradesScreen}
            options={{ title: 'Trades' }}
          />
          <Tab.Screen
            name="Agents"
            component={AgentsScreen}
            options={{ title: 'AI Agents' }}
          />
          <Tab.Screen
            name="Settings"
            component={SettingsScreen}
            options={{ title: 'Settings' }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
