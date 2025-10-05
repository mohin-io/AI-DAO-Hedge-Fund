# Deployment Guide - Decentralized Autonomous Hedge Fund AI DAO

Complete guide for deploying the Decentralized Autonomous Hedge Fund AI DAO system to production.

---

## 📋 Prerequisites

### Required Accounts & API Keys

1. **Blockchain (Choose one)**
   - [Infura](https://infura.io) - Ethereum node provider
   - [Alchemy](https://alchemy.com) - Alternative provider
   - [QuickNode](https://quicknode.com) - Another option

2. **Contract Verification**
   - [Etherscan API Key](https://etherscan.io/apis)
   - [Polygonscan API Key](https://polygonscan.com/apis) (if using Polygon)

3. **Market Data**
   - [Alpha Vantage API](https://www.alphavantage.co/) - Stock data
   - [Polygon.io](https://polygon.io/) - Real-time market data
   - [CoinGecko](https://www.coingecko.com/en/api) - Crypto data (free tier available)

4. **Cloud Services (for production)**
   - [AWS Account](https://aws.amazon.com) or [Google Cloud](https://cloud.google.com)
   - [Vercel](https://vercel.com) or [Netlify](https://netlify.com) for frontend

5. **Monitoring (optional)**
   - [Sentry](https://sentry.io) - Error tracking
   - [DataDog](https://datadoghq.com) - Monitoring

---

## 🔧 Local Development Setup

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/mohin-io/AI-DAO-Hedge-Fund.git
cd AI-DAO-Hedge-Fund

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies for contracts
cd contracts
npm install
cd ..
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

**Minimum required variables**:
```env
SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY
PRIVATE_KEY=your_wallet_private_key
ETHERSCAN_API_KEY=your_etherscan_key
```

### 3. Run Locally

```bash
# Terminal 1: Start backend
python -m uvicorn dashboard.backend.api:app --reload

# Terminal 2: Start frontend (optional)
cd dashboard/frontend
npm install
npm run dev

# Terminal 3: Run training (optional)
python simulations/backtest/run_multi_agent_training.py
```

Access:
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:3000

---

## ⛓️ Smart Contract Deployment

### Sepolia Testnet Deployment

1. **Get Testnet ETH**
   - Visit [Sepolia Faucet](https://sepoliafaucet.com/)
   - Enter your wallet address
   - Receive test ETH (may take a few minutes)

2. **Deploy Contracts**

```bash
cd contracts

# Compile contracts
npx hardhat compile

# Deploy to Sepolia
npm run deploy:sepolia

# Verify contracts
npx hardhat verify --network sepolia <CONTRACT_ADDRESS>
```

3. **Update Configuration**

After deployment, update `config/config.yaml`:

```yaml
blockchain:
  network: "sepolia"
  rpc_url: "https://sepolia.infura.io/v3/YOUR_KEY"
  dao_address: "0x..."  # From deployment output
  treasury_address: "0x..."
  agent_registry_address: "0x..."
```

### Mainnet Deployment (Production)

⚠️ **WARNING**: Mainnet deployment costs real ETH. Ensure thorough testing first!

```bash
# Audit contracts first!
npm install -g @consensys/mythril
myth analyze contracts/DAOGovernance.sol

# Deploy to mainnet (requires ~0.1 ETH for gas)
npm run deploy:mainnet

# Verify immediately
npx hardhat verify --network mainnet <CONTRACT_ADDRESS>
```

---

## 🐳 Docker Deployment

### Option 1: Docker Compose (All Services)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

### Option 2: Individual Containers

```bash
# Backend only
docker build -f Dockerfile.backend -t aidao-backend .
docker run -p 8000:8000 aidao-backend

# With environment variables
docker run -p 8000:8000 \
  -e SEPOLIA_RPC_URL=$SEPOLIA_RPC_URL \
  -e PRIVATE_KEY=$PRIVATE_KEY \
  aidao-backend
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup

```bash
# Launch EC2 instance (t2.large recommended)
# Amazon Linux 2 or Ubuntu 22.04

# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install dependencies
sudo yum update -y
sudo yum install python3 python3-pip git docker -y

# Clone and setup
git clone https://github.com/mohin-io/AI-DAO-Hedge-Fund.git
cd AI-DAO-Hedge-Fund
pip3 install -r requirements.txt
```

#### 2. Setup with PM2 (Process Manager)

```bash
# Install PM2
npm install -g pm2

# Start backend
pm2 start "uvicorn dashboard.backend.api:app --host 0.0.0.0 --port 8000" --name aidao-backend

# Save PM2 config
pm2 save
pm2 startup
```

#### 3. Nginx Reverse Proxy

```bash
# Install Nginx
sudo yum install nginx -y

# Configure Nginx
sudo nano /etc/nginx/conf.d/aidao.conf
```

**Nginx Configuration**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
    }
}
```

```bash
# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

#### 4. SSL Certificate (Let's Encrypt)

```bash
# Install Certbot
sudo yum install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d your-domain.com
```

### Vercel Deployment (Frontend)

```bash
cd dashboard/frontend

# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod

# Set environment variables in Vercel dashboard
# VITE_API_URL=https://api.your-domain.com
```

### Railway/Render Deployment (Backend)

1. Connect GitHub repository
2. Select `dashboard/backend/api.py` as entry point
3. Add environment variables
4. Deploy

---

## 📊 Monitoring Setup

### Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aidao-backend'
    static_configs:
      - targets: ['backend:8000']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Grafana Dashboard

1. Access Grafana: http://localhost:3001
2. Login: admin/admin
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboard from `monitoring/grafana/dashboard.json`

---

## 🔐 Security Checklist

### Pre-Deployment

- [ ] Audit smart contracts (Slither, Mythril, Manual review)
- [ ] Test on testnet for at least 1 week
- [ ] Set up multi-sig wallet for contract admin
- [ ] Enable rate limiting on API
- [ ] Set up WAF (Web Application Firewall)
- [ ] Configure CORS properly
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS/SSL
- [ ] Set up backup strategy

### Post-Deployment

- [ ] Monitor error rates
- [ ] Set up alerts (PagerDuty, Sentry)
- [ ] Regular security scans
- [ ] Backup database daily
- [ ] Monitor gas costs
- [ ] Review contract events
- [ ] Check API rate limits

---

## 📈 Scaling Considerations

### Database Optimization

```sql
-- Create indexes for faster queries
CREATE INDEX idx_trades_agent_id ON trades(agent_id);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_portfolio_date ON portfolio_values(date);
```

### Redis Caching

```python
# Cache expensive queries
@cache(expire=300)  # 5 minutes
async def get_portfolio_metrics():
    # Expensive calculation
    return metrics
```

### Load Balancing

```nginx
upstream aidao_backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    location / {
        proxy_pass http://aidao_backend;
    }
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Contract deployment fails**
```bash
# Check gas price
npx hardhat run scripts/check-gas.js --network sepolia

# Increase gas limit in hardhat.config.js
gas: 5000000
```

**2. Backend won't start**
```bash
# Check logs
docker-compose logs backend

# Verify environment variables
printenv | grep SEPOLIA_RPC_URL
```

**3. WebSocket connection fails**
```bash
# Check firewall
sudo ufw allow 8000/tcp

# Test WebSocket
wscat -c ws://localhost:8000/ws/live
```

**4. Out of memory during training**
```python
# Reduce batch size in config.yaml
training:
  batch_size: 64  # Reduce from 256
```

---

## 📞 Support

- **Issues**: https://github.com/mohin-io/AI-DAO-Hedge-Fund/issues
- **Discussions**: https://github.com/mohin-io/AI-DAO-Hedge-Fund/discussions
- **Email**: mohinhasin999@gmail.com

---

## ✅ Deployment Checklist

### Pre-Launch

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Smart contracts audited
- [ ] Environment variables configured
- [ ] SSL certificate installed
- [ ] Backup strategy implemented
- [ ] Monitoring dashboards configured
- [ ] Documentation updated

### Launch Day

- [ ] Deploy contracts to mainnet
- [ ] Deploy backend to cloud
- [ ] Deploy frontend to Vercel/Netlify
- [ ] Verify all services running
- [ ] Test end-to-end flow
- [ ] Monitor error rates
- [ ] Announce launch

### Post-Launch

- [ ] Daily monitoring
- [ ] Weekly performance review
- [ ] Monthly security audit
- [ ] Quarterly feature updates

---

**Good luck with your deployment! 🚀**
