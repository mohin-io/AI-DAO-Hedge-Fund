"""
Sentiment Analysis for Market Prediction
Analyzes social media sentiment from Twitter/Reddit
"""

import requests
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyze market sentiment from social media

    Data sources:
    - Twitter (via API)
    - Reddit (r/wallstreetbets, r/stocks)
    - News headlines
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Sentiment keywords
        self.bullish_keywords = [
            'moon', 'bullish', 'buy', 'long', 'calls', 'rocket', 'gain',
            'profit', 'up', 'surge', 'rally', 'breakout', 'strong'
        ]

        self.bearish_keywords = [
            'crash', 'bearish', 'sell', 'short', 'puts', 'loss', 'down',
            'fall', 'drop', 'decline', 'weak', 'dump', 'bubble'
        ]

        # API endpoints
        self.reddit_api = "https://www.reddit.com/r/{}/top.json?limit={}"
        self.news_api = "https://newsapi.org/v2/everything"

        # Cache
        self.sentiment_cache = {}

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text

        Args:
            text: Text to analyze

        Returns:
            Dict with sentiment score and classification
        """
        text = text.lower()

        # Count keywords
        bullish_count = sum(1 for word in self.bullish_keywords if word in text)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text)

        # Calculate sentiment score (-1 to 1)
        total = bullish_count + bearish_count
        if total == 0:
            score = 0.0
            sentiment = 'neutral'
        else:
            score = (bullish_count - bearish_count) / total

            if score > 0.3:
                sentiment = 'bullish'
            elif score < -0.3:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'

        return {
            'score': score,
            'sentiment': sentiment,
            'bullish_keywords': bullish_count,
            'bearish_keywords': bearish_count
        }

    def get_reddit_sentiment(
        self,
        subreddits: List[str] = ['wallstreetbets', 'stocks'],
        limit: int = 100
    ) -> Dict:
        """
        Get sentiment from Reddit

        Args:
            subreddits: List of subreddits to analyze
            limit: Number of posts to fetch per subreddit

        Returns:
            Aggregated sentiment analysis
        """
        all_texts = []
        post_data = []

        for subreddit in subreddits:
            try:
                url = self.reddit_api.format(subreddit, limit)
                headers = {'User-Agent': 'AI-DAO-Hedge-Fund/1.0'}

                response = requests.get(url, headers=headers, timeout=10)
                data = response.json()

                posts = data.get('data', {}).get('children', [])

                for post in posts:
                    post_data_item = post.get('data', {})
                    title = post_data_item.get('title', '')
                    selftext = post_data_item.get('selftext', '')
                    score = post_data_item.get('score', 0)
                    num_comments = post_data_item.get('num_comments', 0)

                    text = f"{title} {selftext}"
                    all_texts.append(text)

                    post_data.append({
                        'text': text,
                        'score': score,
                        'comments': num_comments,
                        'subreddit': subreddit
                    })

                logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")

            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit}: {e}")

        # Analyze all texts
        sentiments = [self.analyze_text(text) for text in all_texts]

        # Aggregate
        avg_score = np.mean([s['score'] for s in sentiments])
        sentiment_counts = defaultdict(int)

        for s in sentiments:
            sentiment_counts[s['sentiment']] += 1

        # Weighted by post engagement
        weighted_scores = []
        for post, sent in zip(post_data, sentiments):
            weight = np.log1p(post['score'] + post['comments'])
            weighted_scores.append(sent['score'] * weight)

        weighted_avg = np.mean(weighted_scores) if weighted_scores else 0.0

        return {
            'average_score': avg_score,
            'weighted_score': weighted_avg,
            'sentiment_distribution': dict(sentiment_counts),
            'total_posts': len(all_texts),
            'timestamp': datetime.now().isoformat()
        }

    def get_asset_sentiment(self, ticker: str, hours: int = 24) -> Dict:
        """
        Get sentiment for a specific asset

        Args:
            ticker: Stock/crypto ticker (e.g., 'AAPL', 'BTC')
            hours: Look back period in hours

        Returns:
            Sentiment analysis for the asset
        """
        cache_key = f"{ticker}_{hours}"

        # Check cache
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            cache_time = datetime.fromisoformat(cached['timestamp'])

            if datetime.now() - cache_time < timedelta(hours=1):
                logger.info(f"Using cached sentiment for {ticker}")
                return cached

        # Search Reddit for ticker mentions
        ticker_pattern = re.compile(rf'\b{ticker}\b', re.IGNORECASE)
        relevant_posts = []

        try:
            # Search r/wallstreetbets
            url = self.reddit_api.format('wallstreetbets', 100)
            headers = {'User-Agent': 'AI-DAO-Hedge-Fund/1.0'}

            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()

            posts = data.get('data', {}).get('children', [])

            for post in posts:
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')

                text = f"{title} {selftext}"

                if ticker_pattern.search(text):
                    relevant_posts.append({
                        'text': text,
                        'score': post_data.get('score', 0)
                    })

            logger.info(f"Found {len(relevant_posts)} posts mentioning {ticker}")

        except Exception as e:
            logger.error(f"Error searching for {ticker}: {e}")

        # Analyze sentiment
        if not relevant_posts:
            return {
                'ticker': ticker,
                'sentiment_score': 0.0,
                'sentiment': 'neutral',
                'mention_count': 0,
                'timestamp': datetime.now().isoformat()
            }

        sentiments = [self.analyze_text(post['text']) for post in relevant_posts]

        # Weight by post score
        weighted_scores = [
            s['score'] * np.log1p(post['score'])
            for s, post in zip(sentiments, relevant_posts)
        ]

        avg_score = np.mean(weighted_scores) if weighted_scores else 0.0

        if avg_score > 0.3:
            overall_sentiment = 'bullish'
        elif avg_score < -0.3:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'

        result = {
            'ticker': ticker,
            'sentiment_score': avg_score,
            'sentiment': overall_sentiment,
            'mention_count': len(relevant_posts),
            'timestamp': datetime.now().isoformat()
        }

        # Cache result
        self.sentiment_cache[cache_key] = result

        return result

    def get_market_fear_greed_index(self) -> float:
        """
        Calculate market-wide fear & greed index

        Returns:
            Score from 0 (extreme fear) to 100 (extreme greed)
        """
        # Analyze overall market sentiment
        reddit_sentiment = self.get_reddit_sentiment(
            subreddits=['wallstreetbets', 'stocks', 'investing'],
            limit=100
        )

        # Convert sentiment score (-1 to 1) to fear/greed (0 to 100)
        score = reddit_sentiment['weighted_score']
        fear_greed = (score + 1) * 50  # Map [-1, 1] to [0, 100]

        return max(0, min(100, fear_greed))

    def get_trending_tickers(self, limit: int = 10) -> List[Dict]:
        """
        Get trending tickers from social media

        Args:
            limit: Number of tickers to return

        Returns:
            List of trending tickers with mention counts
        """
        ticker_mentions = defaultdict(int)
        ticker_sentiments = defaultdict(list)

        try:
            url = self.reddit_api.format('wallstreetbets', 100)
            headers = {'User-Agent': 'AI-DAO-Hedge-Fund/1.0'}

            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()

            posts = data.get('data', {}).get('children', [])

            # Common stock ticker pattern (1-5 uppercase letters)
            ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

            for post in posts:
                post_data = post.get('data', {})
                text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"

                # Extract tickers
                tickers = ticker_pattern.findall(text)

                for ticker in tickers:
                    # Filter out common words
                    if ticker not in ['CEO', 'IPO', 'WSB', 'DD', 'YOLO', 'ATH']:
                        ticker_mentions[ticker] += 1

                        # Analyze sentiment
                        sentiment = self.analyze_text(text)
                        ticker_sentiments[ticker].append(sentiment['score'])

        except Exception as e:
            logger.error(f"Error getting trending tickers: {e}")

        # Sort by mentions
        trending = [
            {
                'ticker': ticker,
                'mentions': count,
                'avg_sentiment': np.mean(ticker_sentiments[ticker])
            }
            for ticker, count in sorted(
                ticker_mentions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
        ]

        return trending


class SentimentSignalGenerator:
    """Generate trading signals from sentiment analysis"""

    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer

    def get_signal(self, ticker: str) -> Dict:
        """
        Get trading signal based on sentiment

        Args:
            ticker: Asset ticker

        Returns:
            Trading signal with confidence
        """
        sentiment = self.analyzer.get_asset_sentiment(ticker)

        score = sentiment['sentiment_score']
        mentions = sentiment['mention_count']

        # Signal strength based on sentiment and volume
        if mentions < 5:
            confidence = 'low'
            signal = 'neutral'
        elif score > 0.5 and mentions > 20:
            signal = 'strong_buy'
            confidence = 'high'
        elif score > 0.3:
            signal = 'buy'
            confidence = 'medium'
        elif score < -0.5 and mentions > 20:
            signal = 'strong_sell'
            confidence = 'high'
        elif score < -0.3:
            signal = 'sell'
            confidence = 'medium'
        else:
            signal = 'neutral'
            confidence = 'low'

        return {
            'ticker': ticker,
            'signal': signal,
            'confidence': confidence,
            'sentiment_score': score,
            'mention_count': mentions,
            'timestamp': sentiment['timestamp']
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    analyzer = SentimentAnalyzer()

    # Reddit sentiment
    print("\n=== Reddit Sentiment ===")
    reddit_sent = analyzer.get_reddit_sentiment(limit=50)
    print(f"Average Score: {reddit_sent['average_score']:.3f}")
    print(f"Weighted Score: {reddit_sent['weighted_score']:.3f}")
    print(f"Distribution: {reddit_sent['sentiment_distribution']}")

    # Asset-specific sentiment
    print("\n=== AAPL Sentiment ===")
    aapl_sent = analyzer.get_asset_sentiment('AAPL')
    print(f"Sentiment: {aapl_sent['sentiment']}")
    print(f"Score: {aapl_sent['sentiment_score']:.3f}")
    print(f"Mentions: {aapl_sent['mention_count']}")

    # Fear & Greed Index
    print("\n=== Fear & Greed Index ===")
    fg_index = analyzer.get_market_fear_greed_index()
    print(f"Fear & Greed: {fg_index:.1f}/100")

    # Trending tickers
    print("\n=== Trending Tickers ===")
    trending = analyzer.get_trending_tickers(limit=5)
    for item in trending:
        print(f"{item['ticker']}: {item['mentions']} mentions, "
              f"sentiment: {item['avg_sentiment']:.3f}")

    # Trading signal
    print("\n=== Trading Signal ===")
    signal_gen = SentimentSignalGenerator(analyzer)
    signal = signal_gen.get_signal('TSLA')
    print(f"Ticker: {signal['ticker']}")
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']}")
