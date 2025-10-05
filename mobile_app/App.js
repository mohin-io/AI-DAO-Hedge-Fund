/**
 * Decentralized Autonomous Hedge Fund AI DAO - Mobile App
 * Main Application Entry Point
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

// Import screens
import HomeScreen from './src/screens/HomeScreen';
import PortfolioScreen from './src/screens/PortfolioScreen';
import TradingScreen from './src/screens/TradingScreen';
import AgentsScreen from './src/screens/AgentsScreen';
import DAOScreen from './src/screens/DAOScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <PaperProvider>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            tabBarIcon: ({ focused, color, size }) => {
              let iconName;

              if (route.name === 'Home') {
                iconName = focused ? 'home' : 'home-outline';
              } else if (route.name === 'Portfolio') {
                iconName = focused ? 'chart-line' : 'chart-line-variant';
              } else if (route.name === 'Trading') {
                iconName = focused ? 'currency-usd' : 'currency-usd-off';
              } else if (route.name === 'Agents') {
                iconName = focused ? 'robot' : 'robot-outline';
              } else if (route.name === 'DAO') {
                iconName = focused ? 'account-group' : 'account-group-outline';
              }

              return <Icon name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: '#667eea',
            tabBarInactiveTintColor: 'gray',
            headerStyle: {
              backgroundColor: '#667eea',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          })}
        >
          <Tab.Screen
            name="Home"
            component={HomeScreen}
            options={{ title: 'Decentralized Autonomous Hedge Fund AI DAO' }}
          />
          <Tab.Screen
            name="Portfolio"
            component={PortfolioScreen}
            options={{ title: 'Portfolio' }}
          />
          <Tab.Screen
            name="Trading"
            component={TradingScreen}
            options={{ title: 'Trading' }}
          />
          <Tab.Screen
            name="Agents"
            component={AgentsScreen}
            options={{ title: 'AI Agents' }}
          />
          <Tab.Screen
            name="DAO"
            component={DAOScreen}
            options={{ title: 'Governance' }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
