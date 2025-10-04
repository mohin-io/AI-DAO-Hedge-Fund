"""
Advanced Risk Analytics
Comprehensive risk measurement and monitoring
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RiskAnalytics:
    """
    Advanced risk analytics for portfolio management

    Metrics calculated:
    - Value at Risk (VaR) - Historical, Parametric, Monte Carlo
    - Conditional VaR (CVaR/Expected Shortfall)
    - Maximum Drawdown & Underwater Duration
    - Volatility (realized, implied, GARCH)
    - Beta, Correlation, Tracking Error
    - Tail Risk measures
    - Stress Testing
    """

    def __init__(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize risk analytics

        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns for comparison
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns

    def calculate_var(
        self,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', or 'monte_carlo'

        Returns:
            VaR value (positive number representing loss)
        """
        if method == 'historical':
            var = -np.percentile(self.returns, (1 - confidence_level) * 100)

        elif method == 'parametric':
            mean = np.mean(self.returns)
            std = np.std(self.returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)

        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            mean = np.mean(self.returns)
            std = np.std(self.returns)

            simulated_returns = np.random.normal(mean, std, n_simulations)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)

        else:
            raise ValueError(f"Unknown method: {method}")

        return float(var)

    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)

        Args:
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        var = self.calculate_var(confidence_level, method='historical')
        # Average of returns below VaR
        cvar = -np.mean(self.returns[self.returns < -var])

        return float(cvar)

    def calculate_max_drawdown(self) -> Dict:
        """
        Calculate maximum drawdown and related metrics

        Returns:
            Dict with max_drawdown, max_drawdown_duration, current_drawdown
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()

        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # Find drawdown duration
        if max_dd_idx in drawdown.index:
            # Find start of drawdown period
            start_idx = running_max[:max_dd_idx].idxmax()
            duration = (max_dd_idx - start_idx).days if hasattr(max_dd_idx, 'days') else len(drawdown[start_idx:max_dd_idx])
        else:
            duration = 0

        current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0

        return {
            'max_drawdown': float(max_dd),
            'max_drawdown_date': str(max_dd_idx),
            'max_drawdown_duration': int(duration),
            'current_drawdown': float(current_dd),
            'is_underwater': current_dd < 0
        }

    def calculate_volatility_metrics(self) -> Dict:
        """
        Calculate various volatility metrics

        Returns:
            Dict with realized_vol, downside_vol, upside_vol
        """
        # Annualized volatility
        realized_vol = np.std(self.returns) * np.sqrt(252)

        # Downside volatility (Sortino)
        downside_returns = self.returns[self.returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Upside volatility
        upside_returns = self.returns[self.returns > 0]
        upside_vol = np.std(upside_returns) * np.sqrt(252) if len(upside_returns) > 0 else 0

        return {
            'realized_volatility': float(realized_vol),
            'downside_volatility': float(downside_vol),
            'upside_volatility': float(upside_vol),
            'volatility_skew': float(upside_vol / downside_vol) if downside_vol > 0 else 0
        }

    def calculate_beta(self) -> Optional[float]:
        """
        Calculate portfolio beta against benchmark

        Returns:
            Beta value or None if no benchmark
        """
        if self.benchmark_returns is None:
            return None

        # Align returns
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()

        if len(aligned) < 2:
            return None

        covariance = np.cov(aligned['portfolio'], aligned['benchmark'])[0, 1]
        benchmark_var = np.var(aligned['benchmark'])

        beta = covariance / benchmark_var if benchmark_var > 0 else 0

        return float(beta)

    def calculate_tracking_error(self) -> Optional[float]:
        """
        Calculate tracking error against benchmark

        Returns:
            Annualized tracking error or None
        """
        if self.benchmark_returns is None:
            return None

        # Active returns
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()

        active_returns = aligned['portfolio'] - aligned['benchmark']
        tracking_error = np.std(active_returns) * np.sqrt(252)

        return float(tracking_error)

    def calculate_tail_risk_metrics(self) -> Dict:
        """
        Calculate tail risk measures

        Returns:
            Dict with skewness, kurtosis, tail_ratio
        """
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)

        # Tail ratio (95th percentile / 5th percentile)
        p95 = np.percentile(self.returns, 95)
        p5 = np.percentile(self.returns, 5)
        tail_ratio = abs(p95 / p5) if p5 != 0 else 0

        return {
            'skewness': float(skewness),
            'excess_kurtosis': float(kurtosis),
            'tail_ratio': float(tail_ratio),
            'fat_tails': kurtosis > 3  # Indicates fat tails
        }

    def stress_test(self, scenarios: Dict[str, float]) -> Dict:
        """
        Perform stress testing

        Args:
            scenarios: Dict of scenario_name -> return shock

        Returns:
            Impact of each scenario on portfolio
        """
        results = {}

        current_value = (1 + self.returns).prod()

        for scenario_name, shock in scenarios.items():
            shocked_value = current_value * (1 + shock)
            impact = (shocked_value - current_value) / current_value

            results[scenario_name] = {
                'shock': float(shock),
                'portfolio_impact': float(impact),
                'shocked_value': float(shocked_value)
            }

        return results

    def calculate_risk_adjusted_metrics(self) -> Dict:
        """
        Calculate risk-adjusted performance metrics

        Returns:
            Dict with Sharpe, Sortino, Calmar, Omega ratios
        """
        mean_return = np.mean(self.returns) * 252  # Annualized
        risk_free_rate = 0.02  # 2% assumed

        # Sharpe Ratio
        vol = np.std(self.returns) * np.sqrt(252)
        sharpe = (mean_return - risk_free_rate) / vol if vol > 0 else 0

        # Sortino Ratio
        downside_vol = self.calculate_volatility_metrics()['downside_volatility']
        sortino = (mean_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Calmar Ratio
        max_dd = abs(self.calculate_max_drawdown()['max_drawdown'])
        calmar = mean_return / max_dd if max_dd > 0 else 0

        # Omega Ratio (threshold = 0)
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        omega = gains / losses if losses > 0 else 0

        return {
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'omega_ratio': float(omega)
        }

    def generate_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report

        Returns:
            Complete risk analysis report
        """
        report = {
            'var_95': self.calculate_var(0.95, 'historical'),
            'var_99': self.calculate_var(0.99, 'historical'),
            'cvar_95': self.calculate_cvar(0.95),
            'cvar_99': self.calculate_cvar(0.99),
            **self.calculate_max_drawdown(),
            **self.calculate_volatility_metrics(),
            **self.calculate_tail_risk_metrics(),
            **self.calculate_risk_adjusted_metrics(),
            'beta': self.calculate_beta(),
            'tracking_error': self.calculate_tracking_error()
        }

        # Stress test scenarios
        stress_scenarios = {
            '2008_crisis': -0.40,  # -40% market crash
            'flash_crash': -0.20,  # -20% rapid decline
            'volatility_spike': -0.15,  # Vol spike scenario
            'bull_rally': 0.30  # +30% rally
        }

        report['stress_tests'] = self.stress_test(stress_scenarios)

        return report


class RealTimeRiskMonitor:
    """Real-time risk monitoring and alerts"""

    def __init__(self, risk_limits: Dict):
        """
        Initialize risk monitor

        Args:
            risk_limits: Dict of risk metric -> limit value
        """
        self.risk_limits = risk_limits
        self.alerts = []

    def check_limits(self, current_metrics: Dict) -> List[Dict]:
        """
        Check if any risk limits are breached

        Args:
            current_metrics: Current risk metrics

        Returns:
            List of alerts
        """
        alerts = []

        for metric, limit in self.risk_limits.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]

                # Check if limit breached
                if isinstance(limit, dict):
                    if 'max' in limit and current_value > limit['max']:
                        alerts.append({
                            'metric': metric,
                            'severity': 'high',
                            'message': f"{metric} ({current_value:.4f}) exceeds max limit ({limit['max']})",
                            'timestamp': pd.Timestamp.now()
                        })

                    if 'min' in limit and current_value < limit['min']:
                        alerts.append({
                            'metric': metric,
                            'severity': 'high',
                            'message': f"{metric} ({current_value:.4f}) below min limit ({limit['min']})",
                            'timestamp': pd.Timestamp.now()
                        })

        self.alerts.extend(alerts)
        return alerts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
    benchmark = pd.Series(np.random.normal(0.0008, 0.015, 252))

    # Initialize analytics
    risk = RiskAnalytics(returns, benchmark)

    # Generate risk report
    print("=== Risk Analytics Report ===\n")
    report = risk.generate_risk_report()

    print(f"VaR (95%): {report['var_95']:.4f}")
    print(f"CVaR (95%): {report['cvar_95']:.4f}")
    print(f"Max Drawdown: {report['max_drawdown']*100:.2f}%")
    print(f"Volatility: {report['realized_volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
    print(f"Beta: {report['beta']:.2f}")

    print("\n=== Stress Test Results ===\n")
    for scenario, result in report['stress_tests'].items():
        print(f"{scenario}: {result['portfolio_impact']*100:+.2f}%")

    # Risk monitoring
    print("\n=== Risk Monitoring ===\n")
    limits = {
        'var_95': {'max': 0.03},
        'max_drawdown': {'max': -0.20},
        'realized_volatility': {'max': 0.25}
    }

    monitor = RealTimeRiskMonitor(limits)
    alerts = monitor.check_limits(report)

    if alerts:
        print("⚠️ ALERTS:")
        for alert in alerts:
            print(f"  - {alert['message']}")
    else:
        print("✅ All risk metrics within limits")
