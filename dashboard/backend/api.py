"""
FastAPI Backend Server for AI DAO Hedge Fund
Provides REST API endpoints for dashboard and system interaction
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.momentum_agent import MomentumAgent
from agents.arbitrage_agent import ArbitrageAgent
from agents.hedging_agent import HedgingAgent
from agents.multi_agent_coordinator import MultiAgentCoordinator
from utils.blockchain_interface import MockBlockchainInterface
from environment.data_loader import MarketDataLoader

# Initialize FastAPI app
app = FastAPI(
    title="AI DAO Hedge Fund API",
    description="REST API for decentralized autonomous hedge fund",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
class AppState:
    def __init__(self):
        self.agents = [
            MomentumAgent(agent_id=0),
            ArbitrageAgent(agent_id=1),
            HedgingAgent(agent_id=2)
        ]
        self.coordinator = MultiAgentCoordinator(agents=self.agents)
        self.blockchain = MockBlockchainInterface()
        self.portfolio_value = 100000.0
        self.active_websockets: List[WebSocket] = []

state = AppState()

# Pydantic models
class AgentMetrics(BaseModel):
    agent_id: int
    name: str
    strategy: str
    total_trades: int
    total_pnl: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float

class PortfolioStatus(BaseModel):
    total_value: float
    cash: float
    positions_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    daily_pnl: float

class ProposalCreate(BaseModel):
    description: str
    proposal_type: int
    data: str

class VoteRequest(BaseModel):
    proposal_id: int
    support: bool

class TradeRequest(BaseModel):
    agent_id: int
    asset: str
    action: str
    quantity: float
    price: float

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI DAO Hedge Fund API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "portfolio": "/api/portfolio",
            "agents": "/api/agents",
            "governance": "/api/governance",
            "docs": "/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_active": len(state.agents),
        "websockets_connected": len(state.active_websockets)
    }

@app.get("/api/portfolio", response_model=PortfolioStatus)
async def get_portfolio():
    """Get current portfolio status"""
    # Calculate metrics (in production, retrieve from database)
    return PortfolioStatus(
        total_value=state.portfolio_value,
        cash=state.portfolio_value * 0.3,
        positions_value=state.portfolio_value * 0.7,
        total_return=0.15,  # 15%
        sharpe_ratio=2.14,
        max_drawdown=0.123,
        daily_pnl=1250.50
    )

@app.get("/api/agents", response_model=List[AgentMetrics])
async def get_agents():
    """Get all agents and their metrics"""
    agent_metrics = []

    for agent in state.agents:
        metrics = agent.get_metrics()
        agent_metrics.append(AgentMetrics(
            agent_id=agent.agent_id,
            name=agent.name,
            strategy=agent.strategy,
            total_trades=metrics.get('total_trades', 0),
            total_pnl=metrics.get('total_pnl', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0)
        ))

    return agent_metrics

@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: int):
    """Get specific agent details"""
    if agent_id >= len(state.agents):
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = state.agents[agent_id]
    metrics = agent.get_metrics()

    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "strategy": agent.strategy,
        "metrics": metrics,
        "recent_trades": agent.get_trade_history()[-10:]  # Last 10 trades
    }

@app.get("/api/coordinator/allocations")
async def get_agent_allocations():
    """Get current agent allocation weights"""
    allocations = state.coordinator.get_agent_allocations()

    return {
        "allocations": allocations,
        "current_regime": state.coordinator.current_regime,
        "ensemble_method": state.coordinator.ensemble_method,
        "total_steps": state.coordinator.steps
    }

@app.get("/api/governance/proposals")
async def get_proposals():
    """Get all governance proposals"""
    proposals = []

    for proposal_id, proposal in state.blockchain.proposals.items():
        proposals.append({
            "id": proposal_id,
            "description": proposal['description'],
            "type": proposal['type'],
            "votes_for": proposal['votes_for'],
            "votes_against": proposal['votes_against'],
            "status": "active"
        })

    return {"proposals": proposals}

@app.post("/api/governance/proposals")
async def create_proposal(proposal: ProposalCreate):
    """Create a new governance proposal"""
    tx_hash = state.blockchain.create_proposal(
        "mock_key",
        proposal.description,
        proposal.proposal_type,
        proposal.data.encode()
    )

    return {
        "success": True,
        "tx_hash": tx_hash,
        "proposal_id": len(state.blockchain.proposals) - 1
    }

@app.post("/api/governance/vote")
async def cast_vote(vote: VoteRequest):
    """Cast a vote on a proposal"""
    tx_hash = state.blockchain.cast_vote(
        "mock_key",
        vote.proposal_id,
        vote.support
    )

    return {
        "success": True,
        "tx_hash": tx_hash,
        "proposal_id": vote.proposal_id,
        "support": vote.support
    }

@app.post("/api/trades")
async def record_trade(trade: TradeRequest):
    """Record a new trade"""
    # Record in blockchain
    pnl_bps = int((trade.price * trade.quantity) / state.portfolio_value * 10000)
    tx_hash = state.blockchain.record_trade("mock_key", trade.agent_id, pnl_bps)

    # Broadcast to WebSocket clients
    await broadcast_trade_update({
        "type": "new_trade",
        "agent_id": trade.agent_id,
        "asset": trade.asset,
        "action": trade.action,
        "quantity": trade.quantity,
        "price": trade.price,
        "timestamp": datetime.now().isoformat()
    })

    return {
        "success": True,
        "tx_hash": tx_hash,
        "trade_id": len(state.blockchain.trades)
    }

@app.get("/api/performance/summary")
async def get_performance_summary():
    """Get comprehensive performance summary"""
    return {
        "overview": {
            "total_return": 0.342,  # 34.2%
            "sharpe_ratio": 2.14,
            "sortino_ratio": 2.87,
            "max_drawdown": 0.123,
            "calmar_ratio": 2.78
        },
        "agent_performance": [
            {
                "name": "Momentum Trader",
                "return": 0.285,
                "sharpe": 1.87,
                "allocation": 0.40
            },
            {
                "name": "Arbitrage Hunter",
                "return": 0.193,
                "sharpe": 1.52,
                "allocation": 0.30
            },
            {
                "name": "Risk Hedger",
                "return": 0.151,
                "sharpe": 1.38,
                "allocation": 0.30
            }
        ],
        "benchmark_comparison": {
            "sp500_return": 0.186,
            "outperformance": 0.156,
            "information_ratio": 1.45
        }
    }

@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Get current risk metrics"""
    return {
        "volatility": 0.183,  # 18.3%
        "beta": 0.75,
        "var_95": 0.021,  # -2.1% daily VaR
        "cvar_95": 0.032,  # -3.2% CVaR
        "correlation_risk": 0.45,
        "tail_risk": "MEDIUM",
        "portfolio_concentration": {
            "top_5_holdings": 0.67,
            "herfindahl_index": 0.23
        }
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    state.active_websockets.append(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "connection_established",
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": state.portfolio_value
        })

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Echo back for now (in production, handle different message types)
            await websocket.send_json({
                "type": "echo",
                "data": message,
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        state.active_websockets.remove(websocket)

async def broadcast_trade_update(message: Dict):
    """Broadcast trade update to all connected WebSocket clients"""
    for websocket in state.active_websockets:
        try:
            await websocket.send_json(message)
        except:
            pass

# Background task for simulated live updates
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    asyncio.create_task(simulate_live_updates())

async def simulate_live_updates():
    """Simulate live portfolio updates (for demo)"""
    import random

    while True:
        await asyncio.sleep(5)  # Update every 5 seconds

        # Simulate portfolio value change
        change = random.uniform(-0.005, 0.005)  # ±0.5%
        state.portfolio_value *= (1 + change)

        # Broadcast to all WebSocket clients
        update = {
            "type": "portfolio_update",
            "portfolio_value": state.portfolio_value,
            "change": change,
            "timestamp": datetime.now().isoformat()
        }

        for websocket in state.active_websockets:
            try:
                await websocket.send_json(update)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
