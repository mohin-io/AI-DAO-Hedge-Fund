"""
Advanced Portfolio Optimization using Black-Litterman Model
Combines market equilibrium with investor views for optimal allocation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlackLittermanOptimizer:
    """
    Black-Litterman Portfolio Optimization Model

    Combines:
    1. Market equilibrium (CAPM-based prior)
    2. Investor views (agent predictions)
    3. Optimal portfolio weights
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        market_risk_premium: float = 0.08,
        tau: float = 0.025
    ):
        """
        Initialize Black-Litterman optimizer

        Args:
            risk_free_rate: Annual risk-free rate
            market_risk_premium: Expected market risk premium
            tau: Uncertainty scalar (typically 0.01 - 0.05)
        """
        self.risk_free_rate = risk_free_rate
        self.market_risk_premium = market_risk_premium
        self.tau = tau

    def calculate_implied_returns(
        self,
        market_caps: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 2.5
    ) -> np.ndarray:
        """
        Calculate implied equilibrium returns using reverse optimization

        Args:
            market_caps: Market capitalization for each asset
            covariance_matrix: Asset return covariance matrix
            risk_aversion: Risk aversion coefficient (typically 2-4)

        Returns:
            Implied equilibrium returns (Pi)
        """
        # Calculate market weights
        market_weights = market_caps / np.sum(market_caps)

        # Implied returns: Pi = lambda * Sigma * w_mkt
        implied_returns = risk_aversion * covariance_matrix @ market_weights

        return implied_returns

    def black_litterman(
        self,
        implied_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        views_matrix: np.ndarray,
        views_returns: np.ndarray,
        views_uncertainty: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Black-Litterman model calculation

        Args:
            implied_returns: Equilibrium returns (Pi)
            covariance_matrix: Asset covariance matrix (Sigma)
            views_matrix: Views matrix (P) - each row is a view
            views_returns: Expected returns from views (Q)
            views_uncertainty: Uncertainty matrix (Omega)

        Returns:
            Tuple of (posterior_returns, posterior_covariance)
        """
        n_assets = len(implied_returns)
        n_views = len(views_returns)

        # Default uncertainty: proportional to view variance
        if views_uncertainty is None:
            views_uncertainty = np.diag(
                np.diag(views_matrix @ (self.tau * covariance_matrix) @ views_matrix.T)
            )

        # Posterior return calculation
        # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]

        tau_sigma = self.tau * covariance_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)

        omega_inv = np.linalg.inv(views_uncertainty)

        # Posterior precision matrix
        posterior_precision = tau_sigma_inv + views_matrix.T @ omega_inv @ views_matrix

        # Posterior covariance
        posterior_covariance = np.linalg.inv(posterior_precision)

        # Posterior returns
        posterior_returns = posterior_covariance @ (
            tau_sigma_inv @ implied_returns + views_matrix.T @ omega_inv @ views_returns
        )

        return posterior_returns, posterior_covariance

    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        target_return: Optional[float] = None,
        allow_short: bool = False
    ) -> Dict[str, any]:
        """
        Mean-variance portfolio optimization

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            target_return: Target portfolio return (if None, maximize Sharpe)
            allow_short: Allow short positions

        Returns:
            Dictionary with optimal weights and portfolio statistics
        """
        n_assets = len(expected_returns)

        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return weights @ covariance_matrix @ weights

        # Objective for Sharpe maximization: minimize negative Sharpe ratio
        def negative_sharpe(weights):
            port_return = weights @ expected_returns
            port_vol = np.sqrt(weights @ covariance_matrix @ weights)
            return -(port_return - self.risk_free_rate) / port_vol

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        if target_return is not None:
            # Add return constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ expected_returns - target_return
            })

        # Bounds
        if allow_short:
            bounds = [(-1, 1) for _ in range(n_assets)]
        else:
            bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        if target_return is None:
            # Maximize Sharpe ratio
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            # Minimize variance with target return
            result = minimize(
                portfolio_variance,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

        # Calculate portfolio statistics
        optimal_weights = result.x
        port_return = optimal_weights @ expected_returns
        port_variance = optimal_weights @ covariance_matrix @ optimal_weights
        port_vol = np.sqrt(port_variance)
        sharpe_ratio = (port_return - self.risk_free_rate) / port_vol

        return {
            'weights': optimal_weights,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }

    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        n_points: int = 100,
        allow_short: bool = False
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            n_points: Number of points on frontier
            allow_short: Allow short positions

        Returns:
            DataFrame with frontier points
        """
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)

        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_points = []

        for target_ret in target_returns:
            try:
                result = self.optimize_portfolio(
                    expected_returns,
                    covariance_matrix,
                    target_return=target_ret,
                    allow_short=allow_short
                )

                if result['success']:
                    frontier_points.append({
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe': result['sharpe_ratio']
                    })
            except:
                continue

        return pd.DataFrame(frontier_points)


class RiskParityOptimizer:
    """
    Risk Parity Portfolio Optimization
    Each asset contributes equally to portfolio risk
    """

    @staticmethod
    def calculate_risk_contributions(
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk contribution of each asset

        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix

        Returns:
            Risk contributions for each asset
        """
        portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
        marginal_contrib = covariance_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol

        return risk_contrib

    @staticmethod
    def optimize_risk_parity(covariance_matrix: np.ndarray) -> Dict[str, any]:
        """
        Optimize for risk parity (equal risk contribution)

        Args:
            covariance_matrix: Asset covariance matrix

        Returns:
            Dictionary with optimal weights and statistics
        """
        n_assets = covariance_matrix.shape[0]

        # Objective: minimize variance of risk contributions
        def objective(weights):
            risk_contrib = RiskParityOptimizer.calculate_risk_contributions(
                weights, covariance_matrix
            )
            target_contrib = 1 / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds: long-only
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        risk_contrib = RiskParityOptimizer.calculate_risk_contributions(
            optimal_weights, covariance_matrix
        )

        return {
            'weights': optimal_weights,
            'risk_contributions': risk_contrib,
            'success': result.success
        }


class PortfolioAnalyzer:
    """
    Portfolio performance analysis and attribution
    """

    @staticmethod
    def calculate_performance_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics

        Args:
            returns: Portfolio returns (daily)
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary of performance metrics
        """
        # Annualization factor (assuming daily returns)
        annualization = 252

        # Basic statistics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (annualization / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(annualization)

        # Sharpe ratio
        excess_returns = returns - risk_free_rate / annualization
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(annualization)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(annualization)
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)

        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
    print("=" * 80)

    # Example: 5 assets
    n_assets = 5
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

    # Market capitalization (example)
    market_caps = np.array([3000, 1800, 2800, 1600, 800])  # Billions

    # Covariance matrix (example - annualized)
    np.random.seed(42)
    random_cov = np.random.randn(n_assets, n_assets)
    covariance_matrix = random_cov @ random_cov.T * 0.01

    # Initialize optimizer
    optimizer = BlackLittermanOptimizer()

    # Calculate implied returns
    implied_returns = optimizer.calculate_implied_returns(
        market_caps,
        covariance_matrix
    )

    print("\nImplied Equilibrium Returns:")
    for name, ret in zip(asset_names, implied_returns):
        print(f"  {name}: {ret:.2%}")

    # Investor views
    # View 1: AAPL will outperform MSFT by 5%
    # View 2: TSLA will return 15%
    views_matrix = np.array([
        [1, 0, -1, 0, 0],  # AAPL - MSFT
        [0, 0, 0, 0, 1]    # TSLA
    ])

    views_returns = np.array([0.05, 0.15])

    # Black-Litterman optimization
    posterior_returns, posterior_cov = optimizer.black_litterman(
        implied_returns,
        covariance_matrix,
        views_matrix,
        views_returns
    )

    print("\nPosterior Returns (after incorporating views):")
    for name, ret in zip(asset_names, posterior_returns):
        print(f"  {name}: {ret:.2%}")

    # Optimize portfolio
    optimal = optimizer.optimize_portfolio(
        posterior_returns,
        posterior_cov
    )

    print("\nOptimal Portfolio:")
    print(f"  Expected Return: {optimal['expected_return']:.2%}")
    print(f"  Volatility: {optimal['volatility']:.2%}")
    print(f"  Sharpe Ratio: {optimal['sharpe_ratio']:.2f}")

    print("\n  Asset Allocation:")
    for name, weight in zip(asset_names, optimal['weights']):
        if weight > 0.01:  # Only show significant positions
            print(f"    {name}: {weight:.1%}")

    # Risk Parity
    print("\n" + "=" * 80)
    print("RISK PARITY OPTIMIZATION")
    print("=" * 80)

    rp_optimizer = RiskParityOptimizer()
    rp_result = rp_optimizer.optimize_risk_parity(covariance_matrix)

    print("\nRisk Parity Weights:")
    for name, weight in zip(asset_names, rp_result['weights']):
        print(f"  {name}: {weight:.1%}")

    print("\nRisk Contributions:")
    for name, contrib in zip(asset_names, rp_result['risk_contributions']):
        print(f"  {name}: {contrib:.4f}")

    print("\n✓ Portfolio optimization complete!")
