"""
Comprehensive Backtesting Engine
Historical strategy simulation with performance analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    filled_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    status: str = "pending"


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    commission: float
    side: OrderSide
    duration: timedelta


class BacktestEngine:
    """
    Professional backtesting engine with:
    - Realistic order execution
    - Transaction costs and slippage
    - Position management
    - Performance analytics
    - Multi-strategy support
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,  # 0.05%
        max_position_size: float = 0.2,  # 20% per position
        risk_free_rate: float = 0.02
    ):
        """
        Initialize backtest engine

        Args:
            initial_capital: Starting capital
            commission_rate: Commission as fraction of trade value
            slippage_rate: Slippage as fraction of price
            max_position_size: Maximum position size as fraction of portfolio
            risk_free_rate: Annual risk-free rate
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate

        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        self.current_time: Optional[datetime] = None

        # Performance tracking
        self.equity_curve = []
        self.returns = []
        self.drawdowns = []

    def reset(self):
        """Reset backtest state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.portfolio_history = []
        self.equity_curve = []
        self.returns = []
        self.drawdowns = []
        self.current_time = None

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        Place a trading order

        Args:
            symbol: Asset symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: Type of order
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)

        Returns:
            Order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            timestamp=self.current_time
        )

        self.orders.append(order)
        return order

    def execute_order(self, order: Order, current_price: float) -> bool:
        """
        Execute an order if conditions are met

        Args:
            order: Order to execute
            current_price: Current market price

        Returns:
            True if executed, False otherwise
        """
        if order.status != "pending":
            return False

        # Check order type conditions
        should_execute = False

        if order.order_type == OrderType.MARKET:
            should_execute = True
            execution_price = current_price
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                should_execute = True
                execution_price = order.price
            elif order.side == OrderSide.SELL and current_price >= order.price:
                should_execute = True
                execution_price = order.price
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                should_execute = True
                execution_price = current_price
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                should_execute = True
                execution_price = current_price

        if not should_execute:
            return False

        # Apply slippage
        if order.side == OrderSide.BUY:
            execution_price *= (1 + self.slippage_rate)
        else:
            execution_price *= (1 - self.slippage_rate)

        # Calculate commission
        trade_value = execution_price * order.quantity
        commission = trade_value * self.commission_rate

        # Check if we have enough cash (for BUY orders)
        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission
            if total_cost > self.cash:
                order.status = "rejected"
                logger.warning(f"Insufficient cash for order: {order.symbol}")
                return False

        # Execute the trade
        order.filled_price = execution_price
        order.commission = commission
        order.slippage = abs(execution_price - current_price)
        order.status = "filled"

        self._update_position(order)

        return True

    def _update_position(self, order: Order):
        """Update position after order execution"""
        symbol = order.symbol

        if order.side == OrderSide.BUY:
            # Open or add to position
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_quantity = pos.quantity + order.quantity
                avg_price = (
                    (pos.quantity * pos.entry_price + order.quantity * order.filled_price)
                    / total_quantity
                )
                pos.quantity = total_quantity
                pos.entry_price = avg_price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=order.quantity,
                    entry_price=order.filled_price,
                    current_price=order.filled_price,
                    entry_time=self.current_time
                )

            # Deduct cash
            self.cash -= (order.quantity * order.filled_price + order.commission)

        else:  # SELL
            if symbol not in self.positions:
                logger.warning(f"No position to sell: {symbol}")
                return

            pos = self.positions[symbol]

            # Calculate realized P&L
            pnl = (order.filled_price - pos.entry_price) * order.quantity
            pnl -= order.commission

            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_time=pos.entry_time,
                exit_time=self.current_time,
                entry_price=pos.entry_price,
                exit_price=order.filled_price,
                quantity=order.quantity,
                pnl=pnl,
                pnl_percent=(order.filled_price / pos.entry_price - 1) * 100,
                commission=order.commission,
                side=OrderSide.SELL,
                duration=self.current_time - pos.entry_time
            )
            self.trades.append(trade)

            # Update position
            pos.quantity -= order.quantity
            pos.realized_pnl += pnl

            # Add cash
            self.cash += (order.quantity * order.filled_price - order.commission)

            # Close position if quantity is zero
            if pos.quantity <= 0:
                del self.positions[symbol]

    def update_positions(self, prices: Dict[str, float]):
        """Update current prices and unrealized P&L"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity

    def run_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> Dict:
        """
        Run backtest with a trading strategy

        Args:
            strategy: Strategy function that returns signals
            data: Historical market data with MultiIndex (timestamp, symbol)
            symbols: List of symbols to trade

        Returns:
            Dictionary with backtest results
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Period: {data.index[0]} to {data.index[-1]}")
        logger.info("=" * 80)

        self.reset()

        # Get unique timestamps
        timestamps = data.index.get_level_values(0).unique()

        for i, timestamp in enumerate(timestamps):
            self.current_time = timestamp

            # Get current prices
            current_prices = {}
            for symbol in symbols:
                try:
                    price = data.loc[(timestamp, symbol), 'close']
                    current_prices[symbol] = price
                except KeyError:
                    continue

            # Update positions with current prices
            self.update_positions(current_prices)

            # Execute pending orders
            for order in self.orders:
                if order.status == "pending" and order.symbol in current_prices:
                    self.execute_order(order, current_prices[order.symbol])

            # Get strategy signals
            signals = strategy(data, timestamp, self.positions, self.cash)

            # Process signals
            for signal in signals:
                self.place_order(**signal)

            # Record portfolio state
            portfolio_value = self.get_portfolio_value()
            self.equity_curve.append(portfolio_value)

            if i > 0:
                daily_return = (portfolio_value / self.equity_curve[i-1]) - 1
                self.returns.append(daily_return)

            self.portfolio_history.append({
                'timestamp': timestamp,
                'cash': self.cash,
                'portfolio_value': portfolio_value,
                'positions': len(self.positions),
                'num_trades': len(self.trades)
            })

            # Progress logging
            if (i + 1) % 50 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(timestamps)} | "
                    f"Portfolio: ${portfolio_value:,.2f} | "
                    f"Trades: {len(self.trades)}"
                )

        # Calculate final metrics
        results = self.calculate_metrics()

        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Annual Return: {results['annual_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info("=" * 80)

        return results

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        final_value = self.get_portfolio_value()
        total_return = (final_value / self.initial_capital) - 1

        # Convert to pandas Series for easier calculation
        returns_series = pd.Series(self.returns)
        equity_series = pd.Series(self.equity_curve)

        # Annualized metrics
        n_days = len(self.returns)
        annual_factor = 252 / n_days if n_days > 0 else 0
        annual_return = (1 + total_return) ** annual_factor - 1 if total_return > -1 else -1
        annual_vol = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0

        # Sharpe ratio
        excess_returns = returns_series - (self.risk_free_rate / 252)
        sharpe_ratio = (
            excess_returns.mean() / returns_series.std() * np.sqrt(252)
            if len(returns_series) > 0 and returns_series.std() > 0
            else 0
        )

        # Drawdown calculation
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = (
            abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades))
            if losing_trades and sum(t.pnl for t in losing_trades) != 0
            else 0
        )

        return {
            'initial_value': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission': sum(o.commission for o in self.orders if o.status == "filled")
        }

    def export_results(self, output_dir: str = "backtest_results"):
        """Export backtest results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export trades
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'duration_days': t.duration.days
            }
            for t in self.trades
        ])
        trades_df.to_csv(output_path / 'trades.csv', index=False)

        # Export portfolio history
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.to_csv(output_path / 'portfolio_history.csv', index=False)

        # Export metrics
        metrics = self.calculate_metrics()
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"✓ Results exported to {output_path}")


if __name__ == "__main__":
    # Example: Simple moving average crossover strategy
    def sma_crossover_strategy(data, timestamp, positions, cash):
        """Simple SMA crossover strategy"""
        signals = []

        # Example logic (simplified)
        # In real implementation, you'd calculate SMAs and generate signals

        return signals

    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL']

    # Generate sample price data
    np.random.seed(42)
    data_list = []
    for date in dates:
        for symbol in symbols:
            data_list.append({
                'timestamp': date,
                'symbol': symbol,
                'close': 100 + np.random.randn() * 10
            })

    df = pd.DataFrame(data_list)
    df.set_index(['timestamp', 'symbol'], inplace=True)

    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    # results = engine.run_backtest(sma_crossover_strategy, df, symbols)

    print("\n✓ Backtesting engine initialized!")
    print("  Features:")
    print("    - Realistic order execution")
    print("    - Transaction costs and slippage")
    print("    - Position management")
    print("    - Comprehensive performance metrics")
    print("    - Export to CSV/JSON")
