"""
Sentiment Analysis Integration for Market Prediction
Sources: Twitter/X, Reddit, News APIs
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from textblob import TextBlob
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis for market-related social media and news

    Supports:
    - Twitter/X API (requires API key)
    - Reddit API (via PRAW)
    - News APIs (NewsAPI, Alpha Vantage)
    - Custom text sentiment
    """

    def __init__(
        self,
        twitter_api_key: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        news_api_key: Optional[str] = None
    ):
        self.twitter_api_key = twitter_api_key
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.news_api_key = news_api_key

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using TextBlob

        Args:
            text: Text to analyze

        Returns:
            Dictionary with 'polarity' (-1 to 1) and 'subjectivity' (0 to 1)
        """
        # Clean text
        text = self._clean_text(text)

        # Analyze with TextBlob
        blob = TextBlob(text)

        return {
            'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
            'subjectivity': blob.sentiment.subjectivity,  # 0 (objective) to 1 (subjective)
            'compound_score': blob.sentiment.polarity * (1 - blob.sentiment.subjectivity)
        }

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)

        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.lower()

    def fetch_reddit_sentiment(
        self,
        subreddits: List[str],
        keywords: List[str],
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch and analyze Reddit sentiment (using PRAW)

        Args:
            subreddits: List of subreddit names (e.g., ['wallstreetbets', 'stocks'])
            keywords: Keywords to search for
            limit: Number of posts to fetch per subreddit

        Returns:
            DataFrame with sentiment scores
        """
        if not self.reddit_client_id or not self.reddit_client_secret:
            logger.warning("Reddit API credentials not provided. Using mock data.")
            return self._generate_mock_reddit_data(subreddits, keywords, limit)

        try:
            import praw

            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='AI_DAO_Hedge_Fund/1.0'
            )

            results = []

            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)

                for submission in subreddit.hot(limit=limit):
                    # Check if any keyword is in title or selftext
                    text = f"{submission.title} {submission.selftext}"

                    if any(keyword.lower() in text.lower() for keyword in keywords):
                        sentiment = self.analyze_text(text)

                        results.append({
                            'source': 'reddit',
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'text': text[:200],  # Truncate
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': datetime.fromtimestamp(submission.created_utc),
                            **sentiment
                        })

            df = pd.DataFrame(results)
            logger.info(f"✓ Fetched {len(df)} Reddit posts with sentiment")

            return df

        except ImportError:
            logger.error("praw not installed. Install with: pip install praw")
            return self._generate_mock_reddit_data(subreddits, keywords, limit)

        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
            return self._generate_mock_reddit_data(subreddits, keywords, limit)

    def fetch_news_sentiment(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch and analyze news sentiment (using NewsAPI)

        Args:
            query: Search query (e.g., 'Tesla OR TSLA')
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with sentiment scores
        """
        if not self.news_api_key:
            logger.warning("NewsAPI key not provided. Using mock data.")
            return self._generate_mock_news_data(query)

        try:
            from_date = from_date or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = to_date or datetime.now().strftime('%Y-%m-%d')

            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'sortBy': 'popularity',
                'apiKey': self.news_api_key,
                'language': 'en',
                'pageSize': 100
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            articles = response.json().get('articles', [])

            results = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_text(text)

                results.append({
                    'source': 'news',
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'url': article.get('url'),
                    'published_at': article.get('publishedAt'),
                    'source_name': article.get('source', {}).get('name'),
                    **sentiment
                })

            df = pd.DataFrame(results)
            logger.info(f"✓ Fetched {len(df)} news articles with sentiment")

            return df

        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return self._generate_mock_news_data(query)

    def aggregate_sentiment(
        self,
        df: pd.DataFrame,
        time_window: str = '1D'
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores over time windows

        Args:
            df: DataFrame with sentiment scores
            time_window: Pandas time window (e.g., '1D', '1H', '4H')

        Returns:
            Time-aggregated sentiment DataFrame
        """
        if df.empty:
            return pd.DataFrame()

        # Determine time column
        time_col = None
        for col in ['created_utc', 'published_at', 'timestamp']:
            if col in df.columns:
                time_col = col
                break

        if not time_col:
            logger.warning("No time column found in DataFrame")
            return df

        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

        # Aggregate
        agg_df = df.resample(time_window).agg({
            'polarity': ['mean', 'std', 'count'],
            'subjectivity': 'mean',
            'compound_score': 'mean'
        })

        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()

        return agg_df

    def get_sentiment_features(
        self,
        ticker: str,
        lookback_days: int = 7
    ) -> Dict[str, float]:
        """
        Get comprehensive sentiment features for a ticker

        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back

        Returns:
            Dictionary of sentiment features
        """
        # Fetch data from multiple sources
        reddit_df = self.fetch_reddit_sentiment(
            subreddits=['wallstreetbets', 'stocks', 'investing'],
            keywords=[ticker],
            limit=50
        )

        news_df = self.fetch_news_sentiment(
            query=ticker,
            from_date=(datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        )

        # Combine
        all_sentiment = pd.concat([reddit_df, news_df], ignore_index=True)

        if all_sentiment.empty:
            return {
                'sentiment_mean': 0.0,
                'sentiment_std': 0.0,
                'sentiment_trend': 0.0,
                'sentiment_volume': 0
            }

        # Calculate features
        features = {
            'sentiment_mean': all_sentiment['polarity'].mean(),
            'sentiment_std': all_sentiment['polarity'].std(),
            'sentiment_trend': self._calculate_trend(all_sentiment),
            'sentiment_volume': len(all_sentiment),
            'sentiment_positive_ratio': (all_sentiment['polarity'] > 0).mean(),
            'sentiment_negative_ratio': (all_sentiment['polarity'] < 0).mean(),
            'subjectivity_mean': all_sentiment['subjectivity'].mean()
        }

        return features

    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate sentiment trend (linear regression slope)"""
        if len(df) < 2:
            return 0.0

        # Sort by time
        time_col = 'created_utc' if 'created_utc' in df.columns else 'published_at'
        df = df.sort_values(time_col)

        # Simple linear trend
        x = np.arange(len(df))
        y = df['polarity'].values

        if len(x) == 0 or len(y) == 0:
            return 0.0

        slope = np.polyfit(x, y, 1)[0]

        return float(slope)

    def _generate_mock_reddit_data(
        self,
        subreddits: List[str],
        keywords: List[str],
        limit: int
    ) -> pd.DataFrame:
        """Generate mock Reddit data for testing"""
        data = []

        for subreddit in subreddits:
            for i in range(limit):
                keyword = np.random.choice(keywords)

                data.append({
                    'source': 'reddit',
                    'subreddit': subreddit,
                    'title': f"Mock post about {keyword}",
                    'text': f"This is a mock Reddit post discussing {keyword}",
                    'score': np.random.randint(1, 1000),
                    'num_comments': np.random.randint(0, 100),
                    'created_utc': datetime.now() - timedelta(hours=np.random.randint(0, 168)),
                    'polarity': np.random.uniform(-0.5, 0.5),
                    'subjectivity': np.random.uniform(0.3, 0.7),
                    'compound_score': np.random.uniform(-0.3, 0.3)
                })

        return pd.DataFrame(data)

    def _generate_mock_news_data(self, query: str) -> pd.DataFrame:
        """Generate mock news data for testing"""
        data = []

        for i in range(20):
            sentiment_polarity = np.random.uniform(-0.3, 0.3)

            data.append({
                'source': 'news',
                'title': f"Mock news article about {query}",
                'description': f"This is a mock news article discussing {query}",
                'url': f"https://example.com/article-{i}",
                'published_at': (datetime.now() - timedelta(hours=np.random.randint(0, 168))).isoformat(),
                'source_name': f"News Source {i % 5}",
                'polarity': sentiment_polarity,
                'subjectivity': np.random.uniform(0.2, 0.6),
                'compound_score': sentiment_polarity * 0.8
            })

        return pd.DataFrame(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    analyzer = SentimentAnalyzer()

    # Test single text
    text = "Tesla stock is soaring! Amazing performance this quarter. Very bullish!"
    result = analyzer.analyze_text(text)
    print(f"\nSentiment Analysis:")
    print(f"  Text: {text}")
    print(f"  Polarity: {result['polarity']:.3f}")
    print(f"  Subjectivity: {result['subjectivity']:.3f}")
    print(f"  Compound Score: {result['compound_score']:.3f}")

    # Test ticker sentiment
    print("\nFetching sentiment for AAPL...")
    features = analyzer.get_sentiment_features('AAPL', lookback_days=7)

    print("\nSentiment Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    print("\n✓ Sentiment analysis complete!")
