#!/usr/bin/env python3
"""
Production Error Monitoring & Alerting System
Provides real-time error tracking, alerting, and reporting for BEM system
"""

import os
import json
import logging
import traceback
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import asyncio
import aiohttp
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/error_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ErrorEvent:
    """Structured error event data"""
    error_id: str
    timestamp: str
    severity: str  # CRITICAL, ERROR, WARNING, INFO
    service: str
    error_type: str
    message: str
    stack_trace: Optional[str]
    user_id: Optional[str]
    request_id: Optional[str]
    endpoint: Optional[str]
    metadata: Dict[str, Any]

class ErrorMonitor:
    """Production error monitoring and alerting system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_config()
        self.error_store = []
        self.alert_rules = self._load_alert_rules()
        self.notification_channels = self._setup_notification_channels()
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
    def _load_config(self) -> Dict:
        """Load error monitoring configuration"""
        return {
            'max_stored_errors': 10000,
            'alert_thresholds': {
                'critical_errors_per_minute': 5,
                'error_rate_threshold': 0.1,  # 10% error rate
                'response_time_threshold': 5000  # 5 seconds
            },
            'retention_days': 30,
            'email_alerts': {
                'enabled': os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true',
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'sender_email': os.getenv('ALERT_SENDER_EMAIL'),
                'sender_password': os.getenv('ALERT_SENDER_PASSWORD'),
                'recipients': os.getenv('ALERT_RECIPIENTS', '').split(',')
            },
            'slack_webhook': os.getenv('SLACK_WEBHOOK_URL'),
            'sentry_dsn': os.getenv('SENTRY_DSN')
        }
    
    def _load_alert_rules(self) -> List[Dict]:
        """Load alerting rules configuration"""
        return [
            {
                'name': 'Critical Error Spike',
                'condition': 'critical_errors_per_minute > 5',
                'severity': 'CRITICAL',
                'channels': ['email', 'slack']
            },
            {
                'name': 'High Error Rate',
                'condition': 'error_rate > 0.1',
                'severity': 'ERROR',
                'channels': ['email']
            },
            {
                'name': 'WebSocket Connection Failures',
                'condition': 'websocket_errors > 10',
                'severity': 'WARNING',
                'channels': ['slack']
            },
            {
                'name': 'Database Connection Issues',
                'condition': 'db_connection_errors > 3',
                'severity': 'CRITICAL',
                'channels': ['email', 'slack']
            }
        ]
    
    def _setup_notification_channels(self) -> Dict:
        """Setup notification channels"""
        channels = {}
        
        # Email notifications
        if self.config['email_alerts']['enabled']:
            channels['email'] = self._send_email_alert
        
        # Slack notifications
        if self.config.get('slack_webhook'):
            channels['slack'] = self._send_slack_alert
        
        # Sentry integration
        if self.config.get('sentry_dsn'):
            try:
                import sentry_sdk
                sentry_sdk.init(dsn=self.config['sentry_dsn'])
                channels['sentry'] = self._send_sentry_alert
                logger.info("Sentry integration initialized")
            except ImportError:
                logger.warning("Sentry SDK not installed, skipping Sentry integration")
        
        return channels
    
    def capture_error(self, 
                     error: Exception, 
                     service: str,
                     severity: str = 'ERROR',
                     context: Optional[Dict] = None) -> str:
        """Capture and process an error event"""
        
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(error)) % 10000:04d}"
        
        error_event = ErrorEvent(
            error_id=error_id,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            service=service,
            error_type=type(error).__name__,
            message=str(error),
            stack_trace=traceback.format_exc(),
            user_id=context.get('user_id') if context else None,
            request_id=context.get('request_id') if context else None,
            endpoint=context.get('endpoint') if context else None,
            metadata=context or {}
        )
        
        # Store error
        self.error_store.append(error_event)
        self._cleanup_old_errors()
        
        # Log structured error
        logger.error(f"Error captured: {error_event.error_id}", extra={
            'error_id': error_event.error_id,
            'service': service,
            'severity': severity,
            'error_type': error_event.error_type
        })
        
        # Check alert conditions
        asyncio.create_task(self._check_alert_conditions(error_event))
        
        # Send to external services
        if 'sentry' in self.notification_channels:
            self.notification_channels['sentry'](error_event)
        
        return error_id
    
    def _cleanup_old_errors(self):
        """Remove old errors based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
        
        self.error_store = [
            error for error in self.error_store
            if datetime.fromisoformat(error.timestamp) > cutoff_date
        ]
        
        # Keep only max_stored_errors most recent
        if len(self.error_store) > self.config['max_stored_errors']:
            self.error_store = self.error_store[-self.config['max_stored_errors']:]
    
    async def _check_alert_conditions(self, error_event: ErrorEvent):
        """Check if error triggers any alert rules"""
        current_time = datetime.now()
        one_minute_ago = current_time - timedelta(minutes=1)
        
        # Count recent errors by severity
        recent_errors = [
            e for e in self.error_store
            if datetime.fromisoformat(e.timestamp) > one_minute_ago
        ]
        
        critical_errors = len([e for e in recent_errors if e.severity == 'CRITICAL'])
        error_count = len([e for e in recent_errors if e.severity in ['ERROR', 'CRITICAL']])
        
        # Check alert rules
        for rule in self.alert_rules:
            should_alert = False
            
            if 'critical_errors_per_minute' in rule['condition'] and critical_errors > 5:
                should_alert = True
            elif 'error_rate' in rule['condition'] and error_count > 10:
                should_alert = True
            elif error_event.service == 'websocket' and 'websocket_errors' in rule['condition']:
                websocket_errors = len([e for e in recent_errors if e.service == 'websocket'])
                if websocket_errors > 10:
                    should_alert = True
            elif 'database' in error_event.message.lower() and 'db_connection_errors' in rule['condition']:
                db_errors = len([e for e in recent_errors if 'database' in e.message.lower()])
                if db_errors > 3:
                    should_alert = True
            
            if should_alert:
                await self._trigger_alert(rule, error_event, recent_errors)
    
    async def _trigger_alert(self, rule: Dict, error_event: ErrorEvent, recent_errors: List[ErrorEvent]):
        """Trigger alert notifications"""
        alert_data = {
            'rule_name': rule['name'],
            'severity': rule['severity'],
            'error_event': error_event,
            'recent_error_count': len(recent_errors),
            'timestamp': datetime.now().isoformat()
        }
        
        for channel in rule['channels']:
            if channel in self.notification_channels:
                try:
                    await self.notification_channels[channel](alert_data)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
    
    async def _send_email_alert(self, alert_data: Dict):
        """Send email alert"""
        if not self.config['email_alerts']['enabled']:
            return
        
        config = self.config['email_alerts']
        
        subject = f"🚨 BEM System Alert: {alert_data['rule_name']}"
        
        body = f"""
        Alert: {alert_data['rule_name']}
        Severity: {alert_data['severity']}
        Time: {alert_data['timestamp']}
        
        Latest Error:
        - Service: {alert_data['error_event'].service}
        - Error: {alert_data['error_event'].message}
        - Error ID: {alert_data['error_event'].error_id}
        
        Recent Error Count: {alert_data['recent_error_count']}
        
        Please investigate immediately.
        
        BEM Error Monitoring System
        """
        
        msg = MimeMultipart()
        msg['From'] = config['sender_email']
        msg['Subject'] = subject
        msg.attach(MimeText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            
            for recipient in config['recipients']:
                if recipient.strip():
                    msg['To'] = recipient.strip()
                    server.send_message(msg)
            
            server.quit()
            logger.info(f"Email alert sent for: {alert_data['rule_name']}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert_data: Dict):
        """Send Slack alert"""
        if not self.config.get('slack_webhook'):
            return
        
        severity_emoji = {
            'CRITICAL': '🔴',
            'ERROR': '🟠', 
            'WARNING': '🟡',
            'INFO': '🔵'
        }
        
        emoji = severity_emoji.get(alert_data['severity'], '⚪')
        
        payload = {
            'text': f"{emoji} BEM System Alert",
            'attachments': [
                {
                    'color': 'danger' if alert_data['severity'] == 'CRITICAL' else 'warning',
                    'fields': [
                        {
                            'title': 'Alert',
                            'value': alert_data['rule_name'],
                            'short': True
                        },
                        {
                            'title': 'Severity',
                            'value': alert_data['severity'],
                            'short': True
                        },
                        {
                            'title': 'Service',
                            'value': alert_data['error_event'].service,
                            'short': True
                        },
                        {
                            'title': 'Error Count',
                            'value': str(alert_data['recent_error_count']),
                            'short': True
                        },
                        {
                            'title': 'Latest Error',
                            'value': alert_data['error_event'].message[:200],
                            'short': False
                        }
                    ],
                    'ts': int(datetime.now().timestamp())
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config['slack_webhook'], json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for: {alert_data['rule_name']}")
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_sentry_alert(self, error_event: ErrorEvent):
        """Send error to Sentry"""
        try:
            import sentry_sdk
            
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("service", error_event.service)
                scope.set_tag("severity", error_event.severity)
                scope.set_context("error_details", {
                    "error_id": error_event.error_id,
                    "endpoint": error_event.endpoint,
                    "user_id": error_event.user_id
                })
                
                sentry_sdk.capture_message(
                    error_event.message,
                    level=error_event.severity.lower()
                )
            
            logger.info(f"Error sent to Sentry: {error_event.error_id}")
            
        except Exception as e:
            logger.error(f"Failed to send to Sentry: {e}")
    
    def get_error_stats(self) -> Dict:
        """Get error statistics"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_errors = [
            e for e in self.error_store
            if datetime.fromisoformat(e.timestamp) > last_hour
        ]
        
        daily_errors = [
            e for e in self.error_store
            if datetime.fromisoformat(e.timestamp) > last_day
        ]
        
        return {
            'total_errors': len(self.error_store),
            'errors_last_hour': len(recent_errors),
            'errors_last_day': len(daily_errors),
            'critical_errors_last_hour': len([e for e in recent_errors if e.severity == 'CRITICAL']),
            'top_services_with_errors': self._get_top_error_services(daily_errors),
            'error_rate_last_hour': len(recent_errors) / 60 if recent_errors else 0,
            'most_recent_error': self.error_store[-1] if self.error_store else None
        }
    
    def _get_top_error_services(self, errors: List[ErrorEvent]) -> List[Dict]:
        """Get services with most errors"""
        service_counts = {}
        for error in errors:
            service_counts[error.service] = service_counts.get(error.service, 0) + 1
        
        return sorted([
            {'service': service, 'count': count}
            for service, count in service_counts.items()
        ], key=lambda x: x['count'], reverse=True)[:5]
    
    def export_errors(self, start_date: str, end_date: str) -> List[Dict]:
        """Export errors for analysis"""
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        filtered_errors = [
            asdict(error) for error in self.error_store
            if start <= datetime.fromisoformat(error.timestamp) <= end
        ]
        
        return filtered_errors

# Global error monitor instance
error_monitor = ErrorMonitor()

def capture_error(error: Exception, service: str, severity: str = 'ERROR', **context):
    """Convenience function to capture errors"""
    return error_monitor.capture_error(error, service, severity, context)

# Flask/FastAPI integration decorator
def error_tracking(service: str):
    """Decorator for automatic error tracking in endpoints"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_id = capture_error(e, service, context={
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                })
                # Re-raise the exception after logging
                raise e
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test the error monitoring system
    monitor = ErrorMonitor()
    
    # Simulate some errors
    try:
        raise ValueError("Test database connection error")
    except Exception as e:
        monitor.capture_error(e, "database", "CRITICAL", {
            'user_id': 'test_user',
            'endpoint': '/api/test'
        })
    
    # Print stats
    stats = monitor.get_error_stats()
    print(json.dumps(stats, indent=2, default=str)) 