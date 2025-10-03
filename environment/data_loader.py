"""
Market Data Loader
Fetches and prepares financial data for training
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """Load and preprocess market data for trading environment"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data loader with configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['data']
        self.assets = self.config['assets']
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']
        self.timeframe = self.config['timeframe']

    def load_data(self, cache: bool = True) -> pd.DataFrame:
        """
        Load market data for all configured assets

        Args:
            cache: Whether to cache data locally

        Returns:
            DataFrame with OHLCV data for all assets
        """
        cache_path = Path("data/market_data.csv")

        # Try to load from cache
        if cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        logger.info(f"Downloading data for {len(self.assets)} assets...")

        all_data = {}

        for asset in self.assets:
            try:
                logger.info(f"Fetching {asset}...")
                ticker = yf.Ticker(asset)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.timeframe
                )

                if df.empty:
                    logger.warning(f"No data for {asset}, skipping...")
                    continue

                # Select relevant columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                # Rename columns with asset prefix
                df.columns = [f'{asset}_{col.lower()}' for col in df.columns]

                all_data[asset] = df

            except Exception as e:
                logger.error(f"Failed to fetch {asset}: {e}")

        # Combine all asset data
        if not all_data:
            raise ValueError("No data was successfully loaded")

        combined_df = pd.concat(all_data.values(), axis=1)

        # Forward fill missing values
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')

        # Cache the data
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(cache_path)
            logger.info(f"Cached data to {cache_path}")

        logger.info(f"Loaded data shape: {combined_df.shape}")
        logger.info(f"Date range: {combined_df.index[0]} to {combined_df.index[-1]}")

        return combined_df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe

        Indicators:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Moving Averages (SMA 20, 50)
        - ATR (Average True Range)
        """
        result_df = df.copy()

        for asset in self.assets:
            if asset not in df.columns[0]:
                continue

            close_col = f'{asset}_close'
            high_col = f'{asset}_high'
            low_col = f'{asset}_low'

            if close_col not in df.columns:
                continue

            # Calculate indicators
            prices = df[close_col]

            # RSI
            result_df[f'{asset}_rsi'] = self._calculate_rsi(prices)

            # MACD
            macd, signal = self._calculate_macd(prices)
            result_df[f'{asset}_macd'] = macd
            result_df[f'{asset}_macd_signal'] = signal

            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
            result_df[f'{asset}_bb_upper'] = bb_upper
            result_df[f'{asset}_bb_lower'] = bb_lower

            # Moving Averages
            result_df[f'{asset}_sma_20'] = prices.rolling(window=20).mean()
            result_df[f'{asset}_sma_50'] = prices.rolling(window=50).mean()

            # ATR
            if high_col in df.columns and low_col in df.columns:
                result_df[f'{asset}_atr'] = self._calculate_atr(
                    df[high_col], df[low_col], df[close_col]
                )

        # Fill NaN values from indicator calculations
        result_df = result_df.fillna(method='bfill')

        return result_df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()

        return macd, signal_line

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band, lower_band

    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def prepare_for_environment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for trading environment

        Normalizes data and structures it properly
        """
        # Add technical indicators
        df_with_indicators = self.add_technical_indicators(df)

        # Normalize prices (percentage change from start)
        for asset in self.assets:
            for col_type in ['close', 'open', 'high', 'low']:
                col = f'{asset}_{col_type}'
                if col in df_with_indicators.columns:
                    # Use percentage change instead of raw prices
                    df_with_indicators[col] = df_with_indicators[col].pct_change().fillna(0)

        # Normalize volumes (z-score)
        for asset in self.assets:
            vol_col = f'{asset}_volume'
            if vol_col in df_with_indicators.columns:
                mean_vol = df_with_indicators[vol_col].mean()
                std_vol = df_with_indicators[vol_col].std()
                df_with_indicators[vol_col] = (df_with_indicators[vol_col] - mean_vol) / (std_vol + 1e-6)

        return df_with_indicators

    def split_train_test(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets"""
        split_idx = int(len(df) * train_ratio)

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        logger.info(f"Train set: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
        logger.info(f"Test set: {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")

        return train_df, test_df

    def get_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Get a small sample of data for quick testing"""
        full_data = self.load_data()
        sample = full_data.iloc[:n_samples]
        return self.prepare_for_environment(sample)


def generate_synthetic_data(
    n_assets: int = 3,
    n_days: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing

    Uses geometric Brownian motion
    """
    np.random.seed(seed)

    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    data = {}

    for i in range(n_assets):
        # Parameters for GBM
        mu = 0.0005  # drift
        sigma = 0.02  # volatility
        S0 = 100.0  # initial price

        # Generate price series
        returns = np.random.normal(mu, sigma, n_days)
        prices = S0 * np.exp(np.cumsum(returns))

        # Generate OHLCV
        close = prices
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        open_price = np.roll(close, 1)
        open_price[0] = S0
        volume = np.random.lognormal(15, 1, n_days)

        data[f'asset_{i}_open'] = open_price
        data[f'asset_{i}_high'] = high
        data[f'asset_{i}_low'] = low
        data[f'asset_{i}_close'] = close
        data[f'asset_{i}_volume'] = volume

    df = pd.DataFrame(data, index=dates)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with synthetic data
    logger.info("Generating synthetic data...")
    synthetic_df = generate_synthetic_data(n_assets=3, n_days=500)
    logger.info(f"Synthetic data shape: {synthetic_df.shape}")
    logger.info(f"\n{synthetic_df.head()}")

    # Test real data loader (commented out to avoid API calls in example)
    # loader = MarketDataLoader()
    # real_df = loader.load_data()
    # prepared_df = loader.prepare_for_environment(real_df)
    # train_df, test_df = loader.split_train_test(prepared_df)
