"""
Pre-built Trading Strategies for Backtesting
Collection of common algorithmic trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from backtesting.engine import OrderSide, OrderType


class MovingAverageCrossover:
    """
    Simple Moving Average Crossover Strategy
    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50, position_size: float = 0.1):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size

    def __call__(self, data: pd.DataFrame, timestamp, positions: Dict, cash: float) -> List[Dict]:
        """Generate trading signals"""
        signals = []

        # Get all symbols
        symbols = data.index.get_level_values(1).unique()

        for symbol in symbols:
            try:
                # Get historical data for this symbol
                symbol_data = data.xs(symbol, level=1)
                symbol_data = symbol_data[symbol_data.index <= timestamp]

                if len(symbol_data) < self.slow_period:
                    continue

                # Calculate moving averages
                fast_ma = symbol_data['close'].rolling(window=self.fast_period).mean()
                slow_ma = symbol_data['close'].rolling(window=self.slow_period).mean()

                # Get current and previous values
                current_fast = fast_ma.iloc[-1]
                current_slow = slow_ma.iloc[-1]
                prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else current_fast
                prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else current_slow

                current_price = symbol_data['close'].iloc[-1]

                # Check for crossover
                bullish_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
                bearish_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

                # Generate signals
                if bullish_cross and symbol not in positions:
                    # Buy signal
                    quantity = (cash * self.position_size) / current_price
                    signals.append({
                        'symbol': symbol,
                        'side': OrderSide.BUY,
                        'quantity': quantity,
                        'order_type': OrderType.MARKET
                    })

                elif bearish_cross and symbol in positions:
                    # Sell signal
                    quantity = positions[symbol].quantity
                    signals.append({
                        'symbol': symbol,
                        'side': OrderSide.SELL,
                        'quantity': quantity,
                        'order_type': OrderType.MARKET
                    })

            except Exception as e:
                continue

        return signals


class MeanReversion:
    """
    Mean Reversion Strategy using Bollinger Bands
    Buy when price touches lower band
    Sell when price touches upper band
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        position_size: float = 0.1,
        stop_loss_pct: float = 0.05
    ):
        self.period = period
        self.num_std = num_std
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct

    def __call__(self, data: pd.DataFrame, timestamp, positions: Dict, cash: float) -> List[Dict]:
        """Generate trading signals"""
        signals = []

        symbols = data.index.get_level_values(1).unique()

        for symbol in symbols:
            try:
                symbol_data = data.xs(symbol, level=1)
                symbol_data = symbol_data[symbol_data.index <= timestamp]

                if len(symbol_data) < self.period:
                    continue

                # Calculate Bollinger Bands
                sma = symbol_data['close'].rolling(window=self.period).mean()
                std = symbol_data['close'].rolling(window=self.period).std()
                upper_band = sma + (std * self.num_std)
                lower_band = sma - (std * self.num_std)

                current_price = symbol_data['close'].iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]

                # Buy at lower band
                if current_price <= current_lower and symbol not in positions:
                    quantity = (cash * self.position_size) / current_price
                    signals.append({
                        'symbol': symbol,
                        'side': OrderSide.BUY,
                        'quantity': quantity,
                        'order_type': OrderType.MARKET
                    })

                # Sell at upper band or stop loss
                elif symbol in positions:
                    pos = positions[symbol]
                    pnl_pct = (current_price / pos.entry_price) - 1

                    if current_price >= current_upper or pnl_pct <= -self.stop_loss_pct:
                        signals.append({
                            'symbol': symbol,
                            'side': OrderSide.SELL,
                            'quantity': pos.quantity,
                            'order_type': OrderType.MARKET
                        })

            except Exception:
                continue

        return signals


class MomentumStrategy:
    """
    Momentum Strategy using RSI
    Buy when RSI crosses above oversold level
    Sell when RSI crosses below overbought level
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        position_size: float = 0.1
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_size = position_size

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def __call__(self, data: pd.DataFrame, timestamp, positions: Dict, cash: float) -> List[Dict]:
        """Generate trading signals"""
        signals = []

        symbols = data.index.get_level_values(1).unique()

        for symbol in symbols:
            try:
                symbol_data = data.xs(symbol, level=1)
                symbol_data = symbol_data[symbol_data.index <= timestamp]

                if len(symbol_data) < self.rsi_period + 1:
                    continue

                # Calculate RSI
                rsi = self.calculate_rsi(symbol_data['close'], self.rsi_period)

                current_rsi = rsi.iloc[-1]
                prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
                current_price = symbol_data['close'].iloc[-1]

                # Buy signal: RSI crosses above oversold
                if prev_rsi <= self.oversold and current_rsi > self.oversold and symbol not in positions:
                    quantity = (cash * self.position_size) / current_price
                    signals.append({
                        'symbol': symbol,
                        'side': OrderSide.BUY,
                        'quantity': quantity,
                        'order_type': OrderType.MARKET
                    })

                # Sell signal: RSI crosses below overbought
                elif prev_rsi >= self.overbought and current_rsi < self.overbought and symbol in positions:
                    quantity = positions[symbol].quantity
                    signals.append({
                        'symbol': symbol,
                        'side': OrderSide.SELL,
                        'quantity': quantity,
                        'order_type': OrderType.MARKET
                    })

            except Exception:
                continue

        return signals


class TrendFollowing:
    """
    Trend Following Strategy using ADX and Moving Averages
    Enter when trend is strong (high ADX) and price confirms direction
    Exit when trend weakens
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25,
        ma_period: int = 20,
        position_size: float = 0.15
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.ma_period = ma_period
        self.position_size = position_size

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ADX (simplified version)"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed indicators
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def __call__(self, data: pd.DataFrame, timestamp, positions: Dict, cash: float) -> List[Dict]:
        """Generate trading signals"""
        signals = []

        symbols = data.index.get_level_values(1).unique()

        for symbol in symbols:
            try:
                symbol_data = data.xs(symbol, level=1)
                symbol_data = symbol_data[symbol_data.index <= timestamp]

                if len(symbol_data) < self.adx_period * 2:
                    continue

                # Calculate indicators
                adx = self.calculate_adx(
                    symbol_data['high'],
                    symbol_data['low'],
                    symbol_data['close'],
                    self.adx_period
                )
                ma = symbol_data['close'].rolling(window=self.ma_period).mean()

                current_adx = adx.iloc[-1]
                current_price = symbol_data['close'].iloc[-1]
                current_ma = ma.iloc[-1]

                # Strong trend detected
                if current_adx > self.adx_threshold:
                    # Uptrend: Buy
                    if current_price > current_ma and symbol not in positions:
                        quantity = (cash * self.position_size) / current_price
                        signals.append({
                            'symbol': symbol,
                            'side': OrderSide.BUY,
                            'quantity': quantity,
                            'order_type': OrderType.MARKET
                        })

                    # Downtrend or weak trend: Sell
                    elif current_price < current_ma and symbol in positions:
                        quantity = positions[symbol].quantity
                        signals.append({
                            'symbol': symbol,
                            'side': OrderSide.SELL,
                            'quantity': quantity,
                            'order_type': OrderType.MARKET
                        })

                # Weak trend: Exit positions
                elif current_adx < self.adx_threshold and symbol in positions:
                    quantity = positions[symbol].quantity
                    signals.append({
                        'symbol': symbol,
                        'side': OrderSide.SELL,
                        'quantity': quantity,
                        'order_type': OrderType.MARKET
                    })

            except Exception:
                continue

        return signals


class PairsTradingStrategy:
    """
    Statistical Arbitrage / Pairs Trading
    Trade mean-reverting spread between correlated assets
    """

    def __init__(
        self,
        pair: tuple,
        lookback_period: int = 30,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        position_size: float = 0.1
    ):
        self.pair = pair  # e.g., ('AAPL', 'MSFT')
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold  # Z-score threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size

    def __call__(self, data: pd.DataFrame, timestamp, positions: Dict, cash: float) -> List[Dict]:
        """Generate trading signals"""
        signals = []

        try:
            symbol1, symbol2 = self.pair

            # Get data for both symbols
            data1 = data.xs(symbol1, level=1)
            data2 = data.xs(symbol2, level=1)

            data1 = data1[data1.index <= timestamp]
            data2 = data2[data2.index <= timestamp]

            if len(data1) < self.lookback_period or len(data2) < self.lookback_period:
                return signals

            # Align data
            prices1 = data1['close']
            prices2 = data2['close']

            # Calculate spread
            spread = prices1 - prices2
            spread_mean = spread.rolling(window=self.lookback_period).mean()
            spread_std = spread.rolling(window=self.lookback_period).std()

            # Z-score
            z_score = (spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1]

            current_price1 = prices1.iloc[-1]
            current_price2 = prices2.iloc[-1]

            # Entry signals
            if abs(z_score) > self.entry_threshold:
                quantity = (cash * self.position_size) / (current_price1 + current_price2)

                if z_score > 0:  # Spread too high
                    # Short symbol1, long symbol2
                    if symbol1 in positions:
                        signals.append({
                            'symbol': symbol1,
                            'side': OrderSide.SELL,
                            'quantity': quantity,
                            'order_type': OrderType.MARKET
                        })
                    if symbol2 not in positions:
                        signals.append({
                            'symbol': symbol2,
                            'side': OrderSide.BUY,
                            'quantity': quantity,
                            'order_type': OrderType.MARKET
                        })

                else:  # z_score < 0, spread too low
                    # Long symbol1, short symbol2
                    if symbol1 not in positions:
                        signals.append({
                            'symbol': symbol1,
                            'side': OrderSide.BUY,
                            'quantity': quantity,
                            'order_type': OrderType.MARKET
                        })
                    if symbol2 in positions:
                        signals.append({
                            'symbol': symbol2,
                            'side': OrderSide.SELL,
                            'quantity': positions[symbol2].quantity,
                            'order_type': OrderType.MARKET
                        })

            # Exit signals (spread reverts to mean)
            elif abs(z_score) < self.exit_threshold:
                if symbol1 in positions:
                    signals.append({
                        'symbol': symbol1,
                        'side': OrderSide.SELL,
                        'quantity': positions[symbol1].quantity,
                        'order_type': OrderType.MARKET
                    })
                if symbol2 in positions:
                    signals.append({
                        'symbol': symbol2,
                        'side': OrderSide.SELL,
                        'quantity': positions[symbol2].quantity,
                        'order_type': OrderType.MARKET
                    })

        except Exception:
            pass

        return signals


if __name__ == "__main__":
    print("=" * 80)
    print("AVAILABLE BACKTESTING STRATEGIES")
    print("=" * 80)
    print("\n1. Moving Average Crossover")
    print("   - Fast/Slow MA crossover signals")
    print("   - Simple trend-following approach")
    print("\n2. Mean Reversion (Bollinger Bands)")
    print("   - Buy at lower band, sell at upper band")
    print("   - Includes stop-loss protection")
    print("\n3. Momentum (RSI)")
    print("   - RSI-based entry/exit signals")
    print("   - Overbought/oversold levels")
    print("\n4. Trend Following (ADX)")
    print("   - Trades strong trends only")
    print("   - Exits when trend weakens")
    print("\n5. Pairs Trading")
    print("   - Statistical arbitrage")
    print("   - Mean-reverting spread trading")
    print("\n" + "=" * 80)
