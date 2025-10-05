# Decentralized Autonomous Hedge Fund AI DAO - Mobile App

React Native mobile application for portfolio monitoring and trading.

## Features

### 📱 Screens
- **Home** - Dashboard with portfolio overview and performance charts
- **Portfolio** - Detailed portfolio analytics and holdings
- **Trading** - Place trades and monitor positions
- **AI Agents** - Configure and monitor ML trading agents
- **DAO Governance** - Vote on proposals and view governance

### 🎨 Key Features
- Real-time portfolio tracking
- Performance charts and analytics
- AI agent monitoring and control
- DAO voting interface
- Secure wallet integration (WalletConnect)
- Push notifications for trades and proposals
- Biometric authentication
- Dark mode support

## Tech Stack

- **Framework**: React Native (Expo)
- **Navigation**: React Navigation
- **UI Components**: React Native Paper
- **Charts**: React Native Chart Kit
- **Blockchain**: Web3.js, WalletConnect
- **State Management**: React Context API
- **API Client**: Axios

## Installation

### Prerequisites
- Node.js 18+
- npm or yarn
- Expo CLI
- iOS Simulator (Mac) or Android Emulator

### Setup

```bash
cd mobile_app

# Install dependencies
npm install

# Start development server
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android

# Run on web
npm run web
```

## Project Structure

```
mobile_app/
├── App.js                      # Main app entry point
├── package.json                # Dependencies
├── app.json                    # Expo configuration
├── src/
│   ├── screens/               # Screen components
│   │   ├── HomeScreen.js
│   │   ├── PortfolioScreen.js
│   │   ├── TradingScreen.js
│   │   ├── AgentsScreen.js
│   │   └── DAOScreen.js
│   ├── components/            # Reusable components
│   │   ├── PortfolioChart.js
│   │   ├── AgentCard.js
│   │   └── ProposalCard.js
│   ├── services/              # API and blockchain services
│   │   ├── api.js
│   │   ├── wallet.js
│   │   └── notifications.js
│   ├── context/               # React Context providers
│   │   ├── AuthContext.js
│   │   └── PortfolioContext.js
│   └── utils/                 # Utility functions
│       ├── formatters.js
│       └── constants.js
└── assets/                    # Images, fonts, icons
```

## Configuration

### Environment Variables

Create `.env` file:

```env
API_BASE_URL=http://localhost:8000
WALLET_CONNECT_PROJECT_ID=your_project_id
INFURA_API_KEY=your_infura_key
```

### Backend API

The app connects to a FastAPI backend at `API_BASE_URL`. Ensure the backend is running:

```bash
# In project root
cd dashboard/backend
uvicorn main:app --reload
```

## Features Detail

### Home Screen
- Portfolio value and daily P&L
- 6-month performance chart
- Key metrics (Sharpe, Drawdown, etc.)
- Quick action buttons

### Portfolio Screen
- Holdings breakdown
- Asset allocation pie chart
- Agent performance comparison
- Trade history

### Trading Screen
- Place market/limit orders
- View open positions
- Real-time price feeds
- Trade confirmations

### AI Agents Screen
- Monitor 3 RL agents (PPO, DQN, SAC)
- View agent metrics
- Enable/disable agents
- Configure hyperparameters

### DAO Governance Screen
- View active proposals
- Vote on proposals
- Create new proposals
- Governance statistics

## Deployment

### Production Build

```bash
# Build for iOS
expo build:ios

# Build for Android
expo build:android

# Publish update
expo publish
```

### App Stores

- **Apple App Store**: Requires Apple Developer account
- **Google Play Store**: Requires Google Play Developer account

## Testing

```bash
# Run tests
npm test

# Run linter
npm run lint

# Type check (TypeScript)
npm run type-check
```

## Security

- Secure storage for private keys
- Biometric authentication (Face ID/Fingerprint)
- API request signing
- SSL pinning
- No sensitive data in logs

## Performance

- Code splitting and lazy loading
- Image optimization
- React.memo for expensive renders
- Virtualized lists for long data
- Offline support with AsyncStorage

## Troubleshooting

### Common Issues

**Metro bundler error:**
```bash
npm start -- --reset-cache
```

**iOS build fails:**
```bash
cd ios && pod install && cd ..
```

**Android gradle issues:**
```bash
cd android && ./gradlew clean && cd ..
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License

MIT License - See LICENSE file

## Support

- **Issues**: GitHub Issues
- **Email**: mohinhasin999@gmail.com
- **Docs**: See main project README

---

**Status**: ✅ Complete MVP - Ready for development

Built with ❤️ using React Native and Expo
