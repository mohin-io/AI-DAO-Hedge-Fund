"""
Real Market Data Loader for Transformer Fine-tuning
Supports multiple data sources: Yahoo Finance, Alpha Vantage, Binance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataLoader:
    """
    Loads real market data from multiple sources for Transformer training
    """

    def __init__(self, data_dir: str = "data/market_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=1)

    def fetch_stock_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance

        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'GOOGL'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            Dictionary mapping ticker -> DataFrame with OHLCV data
        """
        logger.info(f"Fetching stock data for {len(tickers)} tickers from {start_date} to {end_date}")

        data = {}
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )

                if not df.empty:
                    # Flatten multi-index columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    data[ticker] = df
                    logger.info(f"✓ {ticker}: {len(df)} rows")
                else:
                    logger.warning(f"✗ {ticker}: No data available")

            except Exception as e:
                logger.error(f"✗ {ticker}: {str(e)}")

        return data

    def fetch_crypto_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch cryptocurrency data from Yahoo Finance (uses -USD suffix)

        Args:
            symbols: List of crypto symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval

        Returns:
            Dictionary mapping symbol -> DataFrame with OHLCV data
        """
        # Convert to Yahoo Finance format (BTC -> BTC-USD)
        tickers = [f"{sym}-USD" for sym in symbols]

        logger.info(f"Fetching crypto data for {len(symbols)} symbols")

        raw_data = self.fetch_stock_data(tickers, start_date, end_date, interval)

        # Remove -USD suffix from keys
        data = {}
        for ticker, df in raw_data.items():
            symbol = ticker.replace('-USD', '')
            data[symbol] = df

        return data

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to price data

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with added technical indicator columns
        """
        df = df.copy()

        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Price momentum
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std()

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and targets for Transformer training

        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences (lookback window)
            prediction_horizon: How many steps ahead to predict
            feature_columns: List of feature column names to use

        Returns:
            Tuple of (X, y) where X is (n_samples, sequence_length, n_features)
            and y is (n_samples, prediction_horizon)
        """
        if feature_columns is None:
            # Use all numeric columns except the target
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove NaN rows
        df = df.dropna()

        X, y = [], []

        for i in range(len(df) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X.append(df[feature_columns].iloc[i:i+sequence_length].values)

            # Target (future returns)
            future_price = df['Close'].iloc[i+sequence_length+prediction_horizon-1]
            current_price = df['Close'].iloc[i+sequence_length-1]
            future_return = (future_price - current_price) / current_price
            y.append(future_return)

        return np.array(X), np.array(y)

    def prepare_training_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        test_split: float = 0.2
    ) -> Dict[str, np.ndarray]:
        """
        Comprehensive data preparation pipeline

        Returns:
            Dictionary with keys: 'X_train', 'X_test', 'y_train', 'y_test'
        """
        logger.info("=" * 60)
        logger.info("MARKET DATA PREPARATION PIPELINE")
        logger.info("=" * 60)

        # Fetch data
        data = self.fetch_stock_data(tickers, start_date, end_date)

        if not data:
            raise ValueError("No data fetched. Check tickers and date range.")

        # Process each ticker
        all_X, all_y = [], []

        for ticker, df in data.items():
            logger.info(f"\nProcessing {ticker}...")

            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Create sequences
            X, y = self.create_sequences(
                df,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon
            )

            all_X.append(X)
            all_y.append(y)

            logger.info(f"  Generated {len(X)} sequences with {X.shape[2]} features")

        # Combine all tickers
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        # Train/test split
        split_idx = int(len(X_combined) * (1 - test_split))

        result = {
            'X_train': X_combined[:split_idx],
            'X_test': X_combined[split_idx:],
            'y_train': y_combined[:split_idx],
            'y_test': y_combined[split_idx:],
        }

        logger.info("\n" + "=" * 60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(result['X_train'])}")
        logger.info(f"Test samples: {len(result['X_test'])}")
        logger.info(f"Sequence length: {sequence_length}")
        logger.info(f"Features per timestep: {X_combined.shape[2]}")
        logger.info(f"Prediction horizon: {prediction_horizon} step(s)")
        logger.info("=" * 60)

        return result


if __name__ == "__main__":
    # Example usage
    loader = MarketDataLoader()

    # Prepare data for major tech stocks
    data = loader.prepare_training_data(
        tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        sequence_length=60,
        prediction_horizon=1,
        test_split=0.2
    )

    print(f"\n✓ Data prepared successfully!")
    print(f"  Training set: {data['X_train'].shape}")
    print(f"  Test set: {data['X_test'].shape}")
