"""
Alert and Notification System
Multi-channel notifications (Email, Telegram, Webhook)
"""

import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from enum import Enum
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NotificationChannel:
    """Base class for notification channels"""

    def send(self, alert: Dict) -> bool:
        raise NotImplementedError


class EmailNotifier(NotificationChannel):
    """Send email notifications"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str]
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send(self, alert: Dict) -> bool:
        """
        Send email alert

        Args:
            alert: Alert dictionary with title, message, severity

        Returns:
            True if sent successfully
        """
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert['severity'].upper()}] {alert['title']}"
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)

            # Create HTML content
            html = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert['severity'] == 'critical' else 'orange' if alert['severity'] == 'warning' else 'blue'};">
                    {alert['title']}
                </h2>
                <p><strong>Severity:</strong> {alert['severity'].upper()}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
                <hr>
                <p>{alert['message']}</p>

                {self._format_metrics(alert.get('metrics', {}))}

                <hr>
                <p style="color: gray; font-size: 12px;">
                    AI DAO Hedge Fund - Automated Alert System
                </p>
            </body>
            </html>
            """

            msg.attach(MIMEText(html, 'html'))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert['title']}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics as HTML table"""
        if not metrics:
            return ""

        html = "<h3>Metrics:</h3><table border='1' style='border-collapse: collapse;'>"
        for key, value in metrics.items():
            html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
        html += "</table>"

        return html


class TelegramNotifier(NotificationChannel):
    """Send Telegram notifications"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(self, alert: Dict) -> bool:
        """
        Send Telegram alert

        Args:
            alert: Alert dictionary

        Returns:
            True if sent successfully
        """
        try:
            # Emoji based on severity
            emoji = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'critical': '🚨'
            }.get(alert['severity'], '📢')

            # Format message
            message = f"{emoji} <b>{alert['title']}</b>\n\n"
            message += f"<b>Severity:</b> {alert['severity'].upper()}\n"
            message += f"<b>Time:</b> {alert['timestamp']}\n\n"
            message += f"{alert['message']}\n"

            if 'metrics' in alert:
                message += "\n<b>Metrics:</b>\n"
                for key, value in alert['metrics'].items():
                    message += f"• {key}: {value}\n"

            # Send request
            response = requests.post(
                self.api_url,
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                },
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Telegram alert sent: {alert['title']}")
                return True
            else:
                logger.error(f"Telegram API error: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False


class WebhookNotifier(NotificationChannel):
    """Send webhook notifications"""

    def __init__(self, webhook_url: str, headers: Optional[Dict] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}

    def send(self, alert: Dict) -> bool:
        """
        Send webhook notification

        Args:
            alert: Alert dictionary

        Returns:
            True if sent successfully
        """
        try:
            response = requests.post(
                self.webhook_url,
                json=alert,
                headers=self.headers,
                timeout=10
            )

            if response.status_code in [200, 201, 202]:
                logger.info(f"Webhook alert sent: {alert['title']}")
                return True
            else:
                logger.error(f"Webhook error: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False


class SlackNotifier(NotificationChannel):
    """Send Slack notifications"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, alert: Dict) -> bool:
        """Send Slack notification"""
        try:
            color = {
                'info': '#36a64f',  # Green
                'warning': '#ff9800',  # Orange
                'critical': '#f44336'  # Red
            }.get(alert['severity'], '#2196f3')  # Blue default

            payload = {
                "attachments": [{
                    "color": color,
                    "title": alert['title'],
                    "text": alert['message'],
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert['severity'].upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert['timestamp'],
                            "short": True
                        }
                    ],
                    "footer": "AI DAO Hedge Fund",
                    "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png"
                }]
            }

            # Add metrics as fields
            if 'metrics' in alert:
                for key, value in alert['metrics'].items():
                    payload["attachments"][0]["fields"].append({
                        "title": key,
                        "value": str(value),
                        "short": True
                    })

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert['title']}")
                return True
            else:
                logger.error(f"Slack error: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False


class AlertManager:
    """Manage alerts across multiple channels"""

    def __init__(self):
        self.channels: List[NotificationChannel] = []
        self.alert_history: List[Dict] = []
        self.alert_rules: Dict = {}

    def add_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.channels.append(channel)

    def add_rule(self, rule_name: str, condition: callable, alert_template: Dict):
        """
        Add alert rule

        Args:
            rule_name: Name of the rule
            condition: Callable that returns True if alert should fire
            alert_template: Template for alert message
        """
        self.alert_rules[rule_name] = {
            'condition': condition,
            'template': alert_template
        }

    def check_rules(self, metrics: Dict) -> List[Dict]:
        """
        Check all rules and send alerts if triggered

        Args:
            metrics: Current system metrics

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](metrics):
                    alert = self._create_alert(rule['template'], metrics)
                    self.send_alert(alert)
                    triggered_alerts.append(alert)

            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {e}")

        return triggered_alerts

    def _create_alert(self, template: Dict, metrics: Dict) -> Dict:
        """Create alert from template"""
        alert = template.copy()
        alert['timestamp'] = datetime.now().isoformat()
        alert['metrics'] = metrics

        return alert

    def send_alert(self, alert: Dict) -> Dict[str, bool]:
        """
        Send alert through all channels

        Args:
            alert: Alert dictionary

        Returns:
            Dict of channel -> success status
        """
        results = {}

        for i, channel in enumerate(self.channels):
            channel_name = type(channel).__name__
            try:
                success = channel.send(alert)
                results[channel_name] = success

            except Exception as e:
                logger.error(f"Error sending to {channel_name}: {e}")
                results[channel_name] = False

        # Store in history
        self.alert_history.append({
            **alert,
            'sent_to': results
        })

        return results

    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:]


# Pre-defined alert templates
ALERT_TEMPLATES = {
    'high_drawdown': {
        'title': 'High Drawdown Alert',
        'severity': 'critical',
        'message': 'Portfolio drawdown exceeds threshold'
    },
    'high_volatility': {
        'title': 'High Volatility Alert',
        'severity': 'warning',
        'message': 'Portfolio volatility is elevated'
    },
    'large_loss': {
        'title': 'Large Loss Alert',
        'severity': 'critical',
        'message': 'Significant loss detected'
    },
    'agent_failure': {
        'title': 'Agent Failure Alert',
        'severity': 'critical',
        'message': 'Trading agent has failed or stopped responding'
    },
    'profitable_trade': {
        'title': 'Profitable Trade',
        'severity': 'info',
        'message': 'Successful trade executed'
    }
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    manager = AlertManager()

    # Add email channel (example - use real credentials)
    # email = EmailNotifier(
    #     smtp_host="smtp.gmail.com",
    #     smtp_port=587,
    #     username="your_email@gmail.com",
    #     password="your_app_password",
    #     from_addr="your_email@gmail.com",
    #     to_addrs=["recipient@example.com"]
    # )
    # manager.add_channel(email)

    # Add Telegram channel (example)
    # telegram = TelegramNotifier(
    #     bot_token="YOUR_BOT_TOKEN",
    #     chat_id="YOUR_CHAT_ID"
    # )
    # manager.add_channel(telegram)

    # Add webhook channel
    webhook = WebhookNotifier("https://webhook.site/your-unique-url")
    manager.add_channel(webhook)

    # Define alert rules
    manager.add_rule(
        'drawdown_alert',
        condition=lambda m: m.get('max_drawdown', 0) < -0.15,  # 15% drawdown
        alert_template=ALERT_TEMPLATES['high_drawdown']
    )

    manager.add_rule(
        'volatility_alert',
        condition=lambda m: m.get('volatility', 0) > 0.30,  # 30% vol
        alert_template=ALERT_TEMPLATES['high_volatility']
    )

    # Test with sample metrics
    test_metrics = {
        'max_drawdown': -0.18,  # Will trigger drawdown alert
        'volatility': 0.25,
        'sharpe_ratio': 1.5,
        'portfolio_value': 105000
    }

    print("\n=== Testing Alert System ===\n")
    triggered = manager.check_rules(test_metrics)

    print(f"Triggered {len(triggered)} alerts")

    # Manual alert
    manual_alert = {
        'title': 'Test Alert',
        'severity': 'info',
        'message': 'This is a test alert from AI DAO Hedge Fund',
        'metrics': test_metrics
    }

    results = manager.send_alert(manual_alert)
    print(f"\nSent to channels: {results}")

    # View history
    print(f"\nAlert history: {len(manager.get_alert_history())} alerts")
