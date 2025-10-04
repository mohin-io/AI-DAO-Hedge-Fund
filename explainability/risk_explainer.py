"""
Risk Explainer - Portfolio Risk Breakdown and Scenario Analysis
Breaks down portfolio risk by agent and provides scenario analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class RiskExplainer:
    """
    Provides detailed explanations of portfolio risk decomposition
    and scenario analysis for the AI DAO Hedge Fund
    """

    def __init__(self, portfolio_value: float = 1000000):
        """
        Initialize Risk Explainer

        Args:
            portfolio_value: Total portfolio value in dollars
        """
        self.portfolio_value = portfolio_value
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level for VaR (default 95%)

        Returns:
            VaR value as a percentage
        """
        return np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR)

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level for CVaR

        Returns:
            CVaR value as a percentage
        """
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def decompose_risk_by_agent(
        self,
        agent_positions: Dict[str, float],
        agent_volatilities: Dict[str, float],
        agent_correlations: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Decompose portfolio risk by individual agent contributions

        Args:
            agent_positions: Dictionary of agent names to position sizes (as % of portfolio)
            agent_volatilities: Dictionary of agent names to historical volatilities
            agent_correlations: Correlation matrix between agents

        Returns:
            Dictionary containing risk decomposition metrics
        """
        agents = list(agent_positions.keys())
        positions = np.array([agent_positions[agent] for agent in agents])
        vols = np.array([agent_volatilities[agent] for agent in agents])

        # Portfolio volatility
        portfolio_variance = 0
        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                correlation = agent_correlations.loc[agent_i, agent_j]
                portfolio_variance += (
                    positions[i] * positions[j] *
                    vols[i] * vols[j] * correlation
                )

        portfolio_volatility = np.sqrt(portfolio_variance)

        # Marginal contribution to risk
        risk_contributions = {}
        for i, agent in enumerate(agents):
            marginal_var = 0
            for j, other_agent in enumerate(agents):
                correlation = agent_correlations.loc[agent, other_agent]
                marginal_var += positions[j] * vols[j] * vols[i] * correlation

            marginal_contribution = marginal_var / portfolio_volatility
            risk_contribution_pct = (positions[i] * marginal_contribution) / portfolio_volatility

            risk_contributions[agent] = {
                'position_weight': positions[i],
                'volatility': vols[i],
                'marginal_contribution': marginal_contribution,
                'risk_contribution_pct': risk_contribution_pct,
                'diversification_ratio': risk_contribution_pct / positions[i]
            }

        return {
            'portfolio_volatility': portfolio_volatility,
            'agent_contributions': risk_contributions
        }

    def scenario_analysis(
        self,
        current_positions: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Perform scenario analysis on portfolio

        Args:
            current_positions: Current agent positions (% of portfolio)
            scenarios: Dictionary of scenario names to agent return expectations

        Returns:
            DataFrame with scenario analysis results
        """
        results = []

        for scenario_name, agent_returns in scenarios.items():
            portfolio_return = sum(
                current_positions.get(agent, 0) * agent_returns.get(agent, 0)
                for agent in set(list(current_positions.keys()) + list(agent_returns.keys()))
            )

            portfolio_value_change = self.portfolio_value * portfolio_return
            new_portfolio_value = self.portfolio_value + portfolio_value_change

            results.append({
                'Scenario': scenario_name,
                'Portfolio Return (%)': portfolio_return * 100,
                'Value Change ($)': portfolio_value_change,
                'New Portfolio Value ($)': new_portfolio_value
            })

        return pd.DataFrame(results)

    def stress_test(
        self,
        agent_positions: Dict[str, float],
        shock_magnitude: float = -0.20  # -20% market crash
    ) -> Dict[str, float]:
        """
        Perform stress test on portfolio

        Args:
            agent_positions: Current agent positions
            shock_magnitude: Size of market shock (default: -20%)

        Returns:
            Dictionary with stress test results
        """
        # Assume different agent sensitivities to market shock
        agent_betas = {
            'momentum': 1.2,  # More sensitive to market
            'arbitrage': 0.5,  # Less sensitive
            'hedging': -0.3   # Negative correlation (protective)
        }

        portfolio_loss = 0
        agent_losses = {}

        for agent, position in agent_positions.items():
            agent_key = agent.lower().split()[0]  # Extract agent type
            beta = agent_betas.get(agent_key, 1.0)
            agent_return = shock_magnitude * beta
            agent_loss = position * agent_return * self.portfolio_value

            agent_losses[agent] = agent_loss
            portfolio_loss += agent_loss

        return {
            'total_portfolio_loss': portfolio_loss,
            'portfolio_loss_pct': (portfolio_loss / self.portfolio_value) * 100,
            'agent_losses': agent_losses,
            'surviving_value': self.portfolio_value + portfolio_loss
        }

    def generate_risk_report(
        self,
        agent_positions: Dict[str, float],
        historical_returns: Dict[str, np.ndarray]
    ) -> str:
        """
        Generate comprehensive risk report

        Args:
            agent_positions: Current agent positions
            historical_returns: Historical returns for each agent

        Returns:
            Formatted risk report string
        """
        report = []
        report.append("=" * 70)
        report.append("PORTFOLIO RISK REPORT")
        report.append("=" * 70)
        report.append(f"\nPortfolio Value: ${self.portfolio_value:,.2f}")
        report.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Calculate overall portfolio risk
        report.append("-" * 70)
        report.append("OVERALL PORTFOLIO RISK")
        report.append("-" * 70)

        # Combine returns
        portfolio_returns = sum(
            agent_positions.get(agent, 0) * returns
            for agent, returns in historical_returns.items()
        )

        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        var_95 = self.calculate_var(portfolio_returns, 0.95)
        var_99 = self.calculate_var(portfolio_returns, 0.99)
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)

        report.append(f"  Annualized Volatility: {volatility*100:.2f}%")
        report.append(f"  VaR (95%): {var_95*100:.2f}% (${var_95*self.portfolio_value:,.2f})")
        report.append(f"  VaR (99%): {var_99*100:.2f}% (${var_99*self.portfolio_value:,.2f})")
        report.append(f"  CVaR (95%): {cvar_95*100:.2f}% (${cvar_95*self.portfolio_value:,.2f})")

        # Agent-level risk
        report.append("\n" + "-" * 70)
        report.append("AGENT-LEVEL RISK BREAKDOWN")
        report.append("-" * 70)

        for agent, position in agent_positions.items():
            if agent in historical_returns:
                returns = historical_returns[agent]
                agent_vol = np.std(returns) * np.sqrt(252)
                agent_var = self.calculate_var(returns, 0.95)

                report.append(f"\n  {agent}:")
                report.append(f"    Position: {position*100:.1f}% (${position*self.portfolio_value:,.2f})")
                report.append(f"    Volatility: {agent_vol*100:.2f}%")
                report.append(f"    VaR (95%): {agent_var*100:.2f}%")
                report.append(f"    Risk Contribution: {(position * agent_vol / volatility)*100:.1f}%")

        # Stress test scenarios
        report.append("\n" + "-" * 70)
        report.append("STRESS TEST SCENARIOS")
        report.append("-" * 70)

        for shock_name, shock_size in [("Moderate Correction", -0.10), ("Market Crash", -0.20), ("Black Swan", -0.30)]:
            stress_result = self.stress_test(agent_positions, shock_size)
            report.append(f"\n  {shock_name} ({shock_size*100:.0f}% shock):")
            report.append(f"    Portfolio Loss: ${stress_result['total_portfolio_loss']:,.2f} ({stress_result['portfolio_loss_pct']:.2f}%)")
            report.append(f"    Surviving Value: ${stress_result['surviving_value']:,.2f}")

        # Recommendations
        report.append("\n" + "-" * 70)
        report.append("RISK MANAGEMENT RECOMMENDATIONS")
        report.append("-" * 70)

        if volatility > 0.20:  # >20% volatility
            report.append("  [HIGH RISK] Portfolio volatility exceeds 20%. Consider:")
            report.append("    - Increase hedging agent allocation")
            report.append("    - Reduce position sizes")
            report.append("    - Add defensive assets")
        elif volatility > 0.15:
            report.append("  [MODERATE RISK] Portfolio volatility is moderate. Consider:")
            report.append("    - Monitor volatility trends")
            report.append("    - Maintain hedging positions")
        else:
            report.append("  [LOW RISK] Portfolio volatility is within acceptable range.")

        if abs(var_95) > 0.05:  # VaR > 5%
            report.append("\n  [WARNING] Daily VaR exceeds 5%. Risk mitigation suggested.")

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def plot_risk_decomposition(
        self,
        agent_positions: Dict[str, float],
        agent_volatilities: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Create visualization of risk decomposition

        Args:
            agent_positions: Agent positions
            agent_volatilities: Agent volatilities
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Position allocation
        agents = list(agent_positions.keys())
        positions = [agent_positions[agent] * 100 for agent in agents]

        ax1.bar(agents, positions, color=['#667eea', '#f093fb', '#fa709a'])
        ax1.set_title('Agent Position Allocation', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Position (%)', fontsize=12)
        ax1.set_xlabel('Agent', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)

        # Risk contribution
        vols = [agent_volatilities[agent] * 100 for agent in agents]
        risk_contribs = [positions[i] * vols[i] / sum([positions[j] * vols[j] for j in range(len(agents))]) * 100
                         for i in range(len(agents))]

        ax2.bar(agents, risk_contribs, color=['#667eea', '#f093fb', '#fa709a'])
        ax2.set_title('Risk Contribution by Agent', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Risk Contribution (%)', fontsize=12)
        ax2.set_xlabel('Agent', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize risk explainer
    explainer = RiskExplainer(portfolio_value=1_000_000)

    # Example data
    agent_positions = {
        'Momentum Agent': 0.40,
        'Arbitrage Agent': 0.35,
        'Hedging Agent': 0.25
    }

    agent_volatilities = {
        'Momentum Agent': 0.25,
        'Arbitrage Agent': 0.15,
        'Hedging Agent': 0.10
    }

    # Generate historical returns (simulated)
    np.random.seed(42)
    historical_returns = {
        'Momentum Agent': np.random.normal(0.001, 0.015, 252),
        'Arbitrage Agent': np.random.normal(0.0007, 0.010, 252),
        'Hedging Agent': np.random.normal(0.0005, 0.007, 252)
    }

    # Generate risk report
    report = explainer.generate_risk_report(agent_positions, historical_returns)
    print(report)

    # Scenario analysis
    scenarios = {
        'Bull Market': {'Momentum Agent': 0.30, 'Arbitrage Agent': 0.15, 'Hedging Agent': 0.05},
        'Bear Market': {'Momentum Agent': -0.20, 'Arbitrage Agent': -0.05, 'Hedging Agent': 0.10},
        'Sideways': {'Momentum Agent': 0.02, 'Arbitrage Agent': 0.08, 'Hedging Agent': 0.03}
    }

    scenario_results = explainer.scenario_analysis(agent_positions, scenarios)
    print("\n\nSCENARIO ANALYSIS:")
    print(scenario_results.to_string(index=False))
