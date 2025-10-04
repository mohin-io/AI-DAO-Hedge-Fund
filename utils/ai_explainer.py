"""
GPT-Based Trade Explanation System
Natural language explanations for trading decisions using LLM
"""

import openai
import json
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AITradeExplainer:
    """
    Generate human-readable explanations for trading decisions using GPT-4
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize AI Trade Explainer

        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        if api_key:
            openai.api_key = api_key

        self.model = model
        self.conversation_history = []

    def explain_trade_decision(
        self,
        action: str,
        asset: str,
        quantity: float,
        price: float,
        market_data: Dict,
        technical_indicators: Dict,
        agent_confidence: float,
        shap_values: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive trade explanation

        Args:
            action: Trade action (BUY, SELL, HOLD)
            asset: Asset symbol
            quantity: Trade quantity
            price: Execution price
            market_data: Current market conditions
            technical_indicators: Technical indicator values
            agent_confidence: Agent's confidence score (0-1)
            shap_values: SHAP feature importance values

        Returns:
            Natural language explanation
        """
        # Build context for GPT
        context = self._build_trade_context(
            action, asset, quantity, price,
            market_data, technical_indicators,
            agent_confidence, shap_values
        )

        # Create prompt
        prompt = f"""You are an expert quantitative trader providing explanations for automated trading decisions.

TRADE DETAILS:
{context}

Please provide a comprehensive explanation that includes:
1. **Executive Summary**: Brief 1-2 sentence overview of the decision
2. **Market Analysis**: Current market conditions and key factors
3. **Technical Reasoning**: Which technical indicators influenced this decision and why
4. **Risk Assessment**: Potential risks and how they're being managed
5. **Expected Outcome**: What we expect to happen and our confidence level

Make the explanation clear, professional, and suitable for both technical and non-technical stakeholders.
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional quantitative trading analyst explaining automated trading decisions in clear, concise language."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )

            explanation = response.choices[0].message.content.strip()

            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "trade": {
                    "action": action,
                    "asset": asset,
                    "quantity": quantity,
                    "price": price
                },
                "explanation": explanation
            })

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return self._generate_fallback_explanation(
                action, asset, quantity, price, agent_confidence
            )

    def _build_trade_context(
        self,
        action: str,
        asset: str,
        quantity: float,
        price: float,
        market_data: Dict,
        technical_indicators: Dict,
        agent_confidence: float,
        shap_values: Optional[Dict]
    ) -> str:
        """Build formatted context string for GPT"""

        context_parts = [
            f"Action: {action}",
            f"Asset: {asset}",
            f"Quantity: {quantity:.2f}",
            f"Price: ${price:.2f}",
            f"Confidence: {agent_confidence:.1%}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "MARKET DATA:",
        ]

        # Add market data
        for key, value in market_data.items():
            if isinstance(value, (int, float)):
                context_parts.append(f"  {key}: {value:.4f}")
            else:
                context_parts.append(f"  {key}: {value}")

        context_parts.append("\nTECHNICAL INDICATORS:")

        # Add technical indicators
        for key, value in technical_indicators.items():
            if isinstance(value, (int, float)):
                context_parts.append(f"  {key}: {value:.4f}")
            else:
                context_parts.append(f"  {key}: {value}")

        # Add SHAP values if available
        if shap_values:
            context_parts.append("\nFEATURE IMPORTANCE (SHAP):")
            sorted_features = sorted(
                shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]  # Top 5 features

            for feature, importance in sorted_features:
                context_parts.append(f"  {feature}: {importance:.4f}")

        return "\n".join(context_parts)

    def _generate_fallback_explanation(
        self,
        action: str,
        asset: str,
        quantity: float,
        price: float,
        confidence: float
    ) -> str:
        """Generate fallback explanation if GPT fails"""

        return f"""
**Trade Execution Summary**

**Decision**: {action} {quantity:.2f} {asset} @ ${price:.2f}

**Confidence**: {confidence:.1%}

**Rationale**: Our AI trading model has identified a {action.lower()} opportunity
based on current market conditions and technical indicators. The model's confidence
level of {confidence:.1%} suggests {'strong' if confidence > 0.7 else 'moderate'}
conviction in this decision.

**Note**: Detailed GPT-based explanation temporarily unavailable.
This is a rule-based fallback explanation.
"""

    def explain_portfolio_rebalance(
        self,
        changes: List[Dict],
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        reason: str
    ) -> str:
        """
        Explain portfolio rebalancing decision

        Args:
            changes: List of trade changes
            current_allocation: Current portfolio weights
            target_allocation: Target portfolio weights
            reason: Reason for rebalancing

        Returns:
            Natural language explanation
        """
        prompt = f"""Explain this portfolio rebalancing decision:

REASON: {reason}

CURRENT ALLOCATION:
{json.dumps(current_allocation, indent=2)}

TARGET ALLOCATION:
{json.dumps(target_allocation, indent=2)}

CHANGES:
{json.dumps(changes, indent=2)}

Provide a clear explanation of:
1. Why this rebalancing is necessary
2. How the new allocation improves the portfolio
3. Expected impact on risk and returns
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a portfolio manager explaining rebalancing decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=600
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating rebalancing explanation: {str(e)}")
            return f"Portfolio rebalanced due to: {reason}"

    def explain_risk_event(
        self,
        event_type: str,
        severity: str,
        metrics: Dict[str, float],
        recommended_action: str
    ) -> str:
        """
        Explain risk events and mitigation strategies

        Args:
            event_type: Type of risk event (drawdown, volatility, etc.)
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            metrics: Relevant risk metrics
            recommended_action: Recommended mitigation action

        Returns:
            Natural language explanation
        """
        prompt = f"""Explain this risk event and recommended action:

EVENT TYPE: {event_type}
SEVERITY: {severity}

RISK METRICS:
{json.dumps(metrics, indent=2)}

RECOMMENDED ACTION: {recommended_action}

Provide:
1. What this risk event means
2. Why it's important
3. How the recommended action mitigates the risk
4. What stakeholders should know
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a risk management expert explaining risk events in clear terms."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=600
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating risk explanation: {str(e)}")
            return f"Risk Event: {event_type} ({severity}). Action: {recommended_action}"

    def generate_daily_summary(
        self,
        date: str,
        trades: List[Dict],
        performance_metrics: Dict[str, float],
        market_conditions: Dict[str, any]
    ) -> str:
        """
        Generate daily trading summary

        Args:
            date: Trading date
            trades: List of executed trades
            performance_metrics: Daily performance metrics
            market_conditions: Market condition summary

        Returns:
            Natural language daily summary
        """
        prompt = f"""Generate a professional daily trading summary for {date}:

TRADES EXECUTED: {len(trades)}
{json.dumps(trades[:10], indent=2)}  # First 10 trades

PERFORMANCE METRICS:
{json.dumps(performance_metrics, indent=2)}

MARKET CONDITIONS:
{json.dumps(market_conditions, indent=2)}

Create a concise executive summary covering:
1. Overall performance
2. Key trading activities
3. Notable market conditions
4. Outlook for tomorrow
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a hedge fund manager writing daily performance summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating daily summary: {str(e)}")
            return f"Daily Summary for {date}: {len(trades)} trades executed."

    def get_conversation_history(self) -> List[Dict]:
        """Get all explanations from this session"""
        return self.conversation_history

    def export_history(self, filepath: str):
        """Export conversation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

        logger.info(f"Exported conversation history to {filepath}")


class ExplainabilityDashboard:
    """
    Interactive dashboard for AI explanations
    """

    def __init__(self, explainer: AITradeExplainer):
        self.explainer = explainer

    def create_explanation_report(
        self,
        trades: List[Dict],
        output_file: str = "explanation_report.html"
    ) -> str:
        """
        Create HTML report with all trade explanations

        Args:
            trades: List of trade dictionaries
            output_file: Output HTML file path

        Returns:
            Path to generated report
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Trade Explanations Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .trade-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .trade-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .action-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .buy {{ background-color: #27ae60; }}
        .sell {{ background-color: #e74c3c; }}
        .hold {{ background-color: #95a5a6; }}
        .explanation {{
            line-height: 1.6;
            color: #34495e;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🤖 AI Trade Explanations Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        for i, trade in enumerate(trades, 1):
            action_class = trade.get('action', 'HOLD').lower()

            html_content += f"""
    <div class="trade-card">
        <div class="trade-header">
            <h2>Trade #{i}: {trade.get('asset', 'N/A')}</h2>
            <span class="action-badge {action_class}">{trade.get('action', 'N/A')}</span>
        </div>
        <div class="explanation">
            {trade.get('explanation', 'No explanation available').replace('\n', '<br>')}
        </div>
        <p class="timestamp">
            Quantity: {trade.get('quantity', 0)} |
            Price: ${trade.get('price', 0):.2f} |
            Time: {trade.get('timestamp', 'N/A')}
        </p>
    </div>
"""

        html_content += """
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html_content)

        logger.info(f"Created explanation report: {output_file}")
        return output_file


if __name__ == "__main__":
    # Example usage (requires OpenAI API key)
    print("=" * 80)
    print("AI TRADE EXPLAINER - DEMO")
    print("=" * 80)

    # Initialize explainer (Note: Requires valid OpenAI API key)
    # explainer = AITradeExplainer(api_key="your-api-key-here")

    # Demo without actual API call
    print("\n✓ AITradeExplainer initialized")
    print("✓ Features:")
    print("  - Natural language trade explanations")
    print("  - Portfolio rebalancing explanations")
    print("  - Risk event explanations")
    print("  - Daily summary generation")
    print("  - Interactive HTML reports")
    print("\nNote: Requires OpenAI API key for actual GPT-based explanations")
