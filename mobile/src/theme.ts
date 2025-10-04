/**
 * App Theme Configuration
 */

import { DefaultTheme } from 'react-native-paper';

export const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#1976D2',
    accent: '#FF9800',
    background: '#F5F5F5',
    surface: '#FFFFFF',
    error: '#F44336',
    text: '#212121',
    onSurface: '#000000',
    disabled: '#BDBDBD',
    placeholder: '#9E9E9E',
    backdrop: '#000000',
    notification: '#FF5252',
  },
};
