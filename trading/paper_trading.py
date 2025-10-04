"""
Live Paper Trading System
Real-time simulation with live market data
"""

import asyncio
import websockets
import json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from threading import Thread, Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperAccount:
    """Paper trading account"""
    account_id: str
    initial_balance: float
    cash: float
    equity: float
    buying_power: float
    positions: Dict[str, 'PaperPosition']
    orders: List['PaperOrder']
    trades: List['PaperTrade']
    created_at: datetime


@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class PaperOrder:
    """Paper trading order"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: str  # 'pending', 'filled', 'cancelled'
    filled_price: Optional[float]
    filled_at: Optional[datetime]
    created_at: datetime


@dataclass
class PaperTrade:
    """Completed paper trade"""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime


class PaperTradingEngine:
    """
    Live paper trading engine with:
    - Real-time market data integration
    - Order management
    - Position tracking
    - Performance analytics
    - WebSocket updates
    """

    def __init__(
        self,
        initial_balance: float = 100000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        data_source: str = "yahoo"
    ):
        """
        Initialize paper trading engine

        Args:
            initial_balance: Starting cash balance
            commission_rate: Commission as fraction of trade value
            slippage_rate: Slippage as fraction of price
            data_source: Market data source ('yahoo', 'alpaca', 'binance')
        """
        self.account = PaperAccount(
            account_id=f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            initial_balance=initial_balance,
            cash=initial_balance,
            equity=initial_balance,
            buying_power=initial_balance,
            positions={},
            orders=[],
            trades=[],
            created_at=datetime.now()
        )

        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.data_source = data_source

        # State management
        self.lock = Lock()
        self.running = False
        self.market_data: Dict[str, float] = {}
        self.subscribers: List = []

        # Performance tracking
        self.equity_history: List[Dict] = []
        self.trade_history: List[Dict] = []

        self.order_counter = 0

    def start(self):
        """Start paper trading engine"""
        self.running = True
        logger.info("=" * 80)
        logger.info("PAPER TRADING ENGINE STARTED")
        logger.info("=" * 80)
        logger.info(f"Account ID: {self.account.account_id}")
        logger.info(f"Initial Balance: ${self.account.initial_balance:,.2f}")
        logger.info(f"Data Source: {self.data_source}")
        logger.info("=" * 80)

        # Start background tasks
        Thread(target=self._run_market_data_loop, daemon=True).start()
        Thread(target=self._run_order_processing_loop, daemon=True).start()

    def stop(self):
        """Stop paper trading engine"""
        self.running = False
        logger.info("Paper trading engine stopped")

    def _run_market_data_loop(self):
        """Background task to fetch market data"""
        while self.running:
            try:
                # Fetch latest prices for all symbols in positions and pending orders
                symbols = set(self.account.positions.keys())
                for order in self.account.orders:
                    if order.status == 'pending':
                        symbols.add(order.symbol)

                if symbols:
                    self._fetch_market_data(list(symbols))

                # Update positions
                self._update_positions()

                asyncio.run(self._broadcast_update())

            except Exception as e:
                logger.error(f"Error in market data loop: {str(e)}")

            asyncio.run(asyncio.sleep(1))  # Update every second

    def _run_order_processing_loop(self):
        """Background task to process pending orders"""
        while self.running:
            try:
                self._process_pending_orders()
            except Exception as e:
                logger.error(f"Error in order processing: {str(e)}")

            asyncio.run(asyncio.sleep(0.5))  # Check twice per second

    def _fetch_market_data(self, symbols: List[str]):
        """Fetch latest market prices"""
        # Simplified: In production, integrate with real data source
        # For demo, simulate price movements
        with self.lock:
            for symbol in symbols:
                if symbol not in self.market_data:
                    self.market_data[symbol] = 100.0  # Initial price

                # Simulate price movement (random walk)
                change = np.random.randn() * 0.005  # 0.5% std dev
                self.market_data[symbol] *= (1 + change)

    def _update_positions(self):
        """Update position values with latest prices"""
        with self.lock:
            for symbol, position in self.account.positions.items():
                if symbol in self.market_data:
                    position.current_price = self.market_data[symbol]
                    position.market_value = position.quantity * position.current_price
                    position.unrealized_pnl = (
                        position.current_price - position.avg_entry_price
                    ) * position.quantity
                    position.unrealized_pnl_pct = (
                        (position.current_price / position.avg_entry_price) - 1
                    ) * 100

            # Update account equity
            positions_value = sum(p.market_value for p in self.account.positions.values())
            self.account.equity = self.account.cash + positions_value

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = 'market',
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> PaperOrder:
        """
        Place a paper trading order

        Args:
            symbol: Asset symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: 'market', 'limit', or 'stop'
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            PaperOrder object
        """
        with self.lock:
            self.order_counter += 1

            order = PaperOrder(
                order_id=f"ORD_{self.order_counter:06d}",
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status='pending',
                filled_price=None,
                filled_at=None,
                created_at=datetime.now()
            )

            self.account.orders.append(order)

            logger.info(
                f"Order placed: {order.order_id} | {order.side.upper()} {order.quantity} "
                f"{order.symbol} @ {order_type.upper()}"
            )

            return order

    def _process_pending_orders(self):
        """Process pending orders"""
        with self.lock:
            for order in self.account.orders:
                if order.status != 'pending':
                    continue

                if order.symbol not in self.market_data:
                    continue

                current_price = self.market_data[order.symbol]
                should_execute = False

                # Check execution conditions
                if order.order_type == 'market':
                    should_execute = True
                    execution_price = current_price

                elif order.order_type == 'limit':
                    if order.side == 'buy' and current_price <= order.price:
                        should_execute = True
                        execution_price = order.price
                    elif order.side == 'sell' and current_price >= order.price:
                        should_execute = True
                        execution_price = order.price

                elif order.order_type == 'stop':
                    if order.side == 'buy' and current_price >= order.stop_price:
                        should_execute = True
                        execution_price = current_price
                    elif order.side == 'sell' and current_price <= order.stop_price:
                        should_execute = True
                        execution_price = current_price

                if should_execute:
                    self._execute_order(order, execution_price)

    def _execute_order(self, order: PaperOrder, execution_price: float):
        """Execute an order"""
        # Apply slippage
        if order.side == 'buy':
            execution_price *= (1 + self.slippage_rate)
        else:
            execution_price *= (1 - self.slippage_rate)

        # Calculate commission
        trade_value = execution_price * order.quantity
        commission = trade_value * self.commission_rate

        # Check buying power
        if order.side == 'buy':
            total_cost = trade_value + commission
            if total_cost > self.account.cash:
                order.status = 'rejected'
                logger.warning(f"Order {order.order_id} rejected: Insufficient cash")
                return

        # Update order
        order.status = 'filled'
        order.filled_price = execution_price
        order.filled_at = datetime.now()

        # Create trade record
        trade = PaperTrade(
            trade_id=f"TRD_{len(self.account.trades)+1:06d}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            timestamp=datetime.now()
        )
        self.account.trades.append(trade)
        self.trade_history.append(asdict(trade))

        # Update position
        self._update_position(order, execution_price, commission)

        logger.info(
            f"Order filled: {order.order_id} | {order.side.upper()} {order.quantity} "
            f"{order.symbol} @ ${execution_price:.2f}"
        )

    def _update_position(self, order: PaperOrder, execution_price: float, commission: float):
        """Update position after order execution"""
        symbol = order.symbol

        if order.side == 'buy':
            if symbol in self.account.positions:
                # Add to existing position
                pos = self.account.positions[symbol]
                total_quantity = pos.quantity + order.quantity
                avg_price = (
                    (pos.quantity * pos.avg_entry_price + order.quantity * execution_price)
                    / total_quantity
                )
                pos.quantity = total_quantity
                pos.avg_entry_price = avg_price
            else:
                # New position
                self.account.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    quantity=order.quantity,
                    avg_entry_price=execution_price,
                    current_price=execution_price,
                    market_value=order.quantity * execution_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0
                )

            # Deduct cash
            self.account.cash -= (order.quantity * execution_price + commission)

        else:  # sell
            if symbol in self.account.positions:
                pos = self.account.positions[symbol]
                pos.quantity -= order.quantity

                # Remove position if fully closed
                if pos.quantity <= 0:
                    del self.account.positions[symbol]

            # Add cash
            self.account.cash += (order.quantity * execution_price - commission)

        # Update buying power
        self.account.buying_power = self.account.cash

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        with self.lock:
            for order in self.account.orders:
                if order.order_id == order_id and order.status == 'pending':
                    order.status = 'cancelled'
                    logger.info(f"Order cancelled: {order_id}")
                    return True
        return False

    async def _broadcast_update(self):
        """Broadcast account update to subscribers"""
        update = {
            'type': 'account_update',
            'timestamp': datetime.now().isoformat(),
            'account': {
                'cash': self.account.cash,
                'equity': self.account.equity,
                'buying_power': self.account.buying_power,
                'positions': [asdict(p) for p in self.account.positions.values()],
                'pending_orders': len([o for o in self.account.orders if o.status == 'pending'])
            }
        }

        # Send to WebSocket subscribers
        for subscriber in self.subscribers:
            try:
                await subscriber.send(json.dumps(update))
            except:
                self.subscribers.remove(subscriber)

    def get_account_summary(self) -> Dict:
        """Get account summary"""
        with self.lock:
            total_pnl = self.account.equity - self.account.initial_balance
            total_return = (total_pnl / self.account.initial_balance) * 100

            return {
                'account_id': self.account.account_id,
                'initial_balance': self.account.initial_balance,
                'cash': self.account.cash,
                'equity': self.account.equity,
                'total_pnl': total_pnl,
                'total_return_pct': total_return,
                'positions_count': len(self.account.positions),
                'total_trades': len(self.account.trades),
                'pending_orders': len([o for o in self.account.orders if o.status == 'pending'])
            }

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.equity_history:
            return {}

        equity_df = pd.DataFrame(self.equity_history)
        equity_series = equity_df['equity']

        returns = equity_series.pct_change().dropna()

        # Basic metrics
        total_return = (self.account.equity / self.account.initial_balance) - 1
        annual_return = total_return  # Simplified, should annualize based on time

        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0

        # Drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = [t for t in self.account.trades if t.side == 'sell']  # Simplified
        win_rate = 0  # Would need to calculate actual P&L per trade

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.account.trades),
            'win_rate': win_rate
        }

    def export_results(self, output_dir: str = "paper_trading_results"):
        """Export paper trading results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export trades
        trades_df = pd.DataFrame(self.trade_history)
        trades_df.to_csv(output_path / 'trades.csv', index=False)

        # Export equity history
        equity_df = pd.DataFrame(self.equity_history)
        equity_df.to_csv(output_path / 'equity_history.csv', index=False)

        # Export summary
        summary = self.get_account_summary()
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"✓ Results exported to {output_path}")


if __name__ == "__main__":
    # Example usage
    engine = PaperTradingEngine(initial_balance=100000)
    engine.start()

    print("\n✓ Paper Trading Engine initialized!")
    print("  Features:")
    print("    - Real-time market data simulation")
    print("    - Order management (market, limit, stop)")
    print("    - Position tracking")
    print("    - Performance analytics")
    print("    - WebSocket updates")

    # Simulate some trades
    import time

    # Place buy order
    order1 = engine.place_order('AAPL', 'buy', 10, 'market')
    time.sleep(2)

    # Check account
    summary = engine.get_account_summary()
    print(f"\nAccount Summary:")
    print(f"  Equity: ${summary['equity']:,.2f}")
    print(f"  Cash: ${summary['cash']:,.2f}")
    print(f"  Positions: {summary['positions_count']}")

    engine.stop()
