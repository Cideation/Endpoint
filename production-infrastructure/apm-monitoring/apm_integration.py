"""
APM (Application Performance Monitoring) Integration
Provides comprehensive performance monitoring with multiple provider support
"""

import time
import functools
import threading
import psutil
import logging
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import os
from contextlib import contextmanager

# APM Provider Integrations
try:
    import newrelic.agent as newrelic
    NEWRELIC_AVAILABLE = True
except ImportError:
    NEWRELIC_AVAILABLE = False

try:
    from datadog import initialize, statsd
    import datadog
    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False

try:
    from elasticapm import Client as ElasticAPMClient
    from elasticapm.contrib.flask import ElasticAPM
    ELASTIC_APM_AVAILABLE = True
except ImportError:
    ELASTIC_APM_AVAILABLE = False

class APMMetrics:
    """Core APM metrics collection and management"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.timers = defaultdict(deque)
        self.alerts = deque(maxlen=1000)
        self.start_time = datetime.now()
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            'response_time_ms': 1000,  # 1 second
            'memory_usage_mb': 512,    # 512MB
            'cpu_usage_percent': 80,   # 80%
            'error_rate_percent': 5,   # 5%
            'db_query_time_ms': 500    # 500ms
        }
        
        # Initialize system monitoring
        self._init_system_monitoring()
    
    def _init_system_monitoring(self):
        """Initialize system resource monitoring"""
        def monitor_system():
            while True:
                try:
                    # CPU Usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_metric('system.cpu_usage', cpu_percent)
                    
                    # Memory Usage
                    memory = psutil.virtual_memory()
                    memory_mb = memory.used / (1024 * 1024)
                    self.record_metric('system.memory_usage_mb', memory_mb)
                    self.record_metric('system.memory_percent', memory.percent)
                    
                    # Disk Usage
                    disk = psutil.disk_usage('/')
                    disk_percent = (disk.used / disk.total) * 100
                    self.record_metric('system.disk_usage_percent', disk_percent)
                    
                    # Check thresholds
                    self._check_thresholds({
                        'cpu_usage_percent': cpu_percent,
                        'memory_usage_mb': memory_mb
                    })
                    
                except Exception as e:
                    logging.error(f"System monitoring error: {e}")
                
                time.sleep(30)  # Monitor every 30 seconds
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value"""
        with self._lock:
            timestamp = datetime.now()
            metric_data = {
                'timestamp': timestamp,
                'value': value,
                'tags': tags or {}
            }
            
            # Keep only last 1000 entries per metric
            if len(self.metrics[name]) >= 1000:
                self.metrics[name].popleft()
            
            self.metrics[name].append(metric_data)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self._lock:
            full_name = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
            self.counters[full_name] += value
            
            # Also record as time series
            self.record_metric(name, self.counters[full_name], tags)
    
    def record_timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record timing metric"""
        self.record_metric(f"{name}.duration", duration_ms, tags)
        
        # Keep timing history for percentile calculation
        with self._lock:
            if len(self.timers[name]) >= 1000:
                self.timers[name].popleft()
            self.timers[name].append(duration_ms)
    
    def _check_thresholds(self, current_values: Dict[str, float]):
        """Check if any metrics exceed thresholds"""
        for metric, value in current_values.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                if value > threshold:
                    alert = {
                        'timestamp': datetime.now(),
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'severity': 'high' if value > threshold * 1.5 else 'medium'
                    }
                    self.alerts.append(alert)
                    logging.warning(f"APM Alert: {metric} = {value} exceeds threshold {threshold}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            summary = {
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'total_metrics': len(self.metrics),
                'total_counters': len(self.counters),
                'recent_alerts': list(self.alerts)[-10:],
                'current_stats': {}
            }
            
            # Calculate current statistics
            for metric_name, metric_data in self.metrics.items():
                if metric_data:
                    recent_values = [m['value'] for m in metric_data[-10:]]
                    summary['current_stats'][metric_name] = {
                        'current': recent_values[-1] if recent_values else 0,
                        'avg_10': sum(recent_values) / len(recent_values) if recent_values else 0,
                        'min_10': min(recent_values) if recent_values else 0,
                        'max_10': max(recent_values) if recent_values else 0
                    }
            
            return summary

class APMProvider:
    """Base APM provider interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', False)
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric"""
        pass
    
    def record_transaction(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a transaction"""
        pass
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error"""
        pass

class NewRelicProvider(APMProvider):
    """New Relic APM provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if self.enabled and NEWRELIC_AVAILABLE:
            self.initialized = True
            logging.info("New Relic APM provider initialized")
        else:
            self.initialized = False
            logging.warning("New Relic APM not available or disabled")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        if self.initialized:
            newrelic.record_custom_metric(f"Custom/{name}", value)
    
    def record_transaction(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        if self.initialized:
            newrelic.record_custom_metric(f"Custom/Transaction/{name}", duration_ms)
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        if self.initialized:
            newrelic.record_exception(error, context)

class DataDogProvider(APMProvider):
    """DataDog APM provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if self.enabled and DATADOG_AVAILABLE:
            datadog.initialize(
                api_key=config.get('api_key'),
                app_key=config.get('app_key')
            )
            self.initialized = True
            logging.info("DataDog APM provider initialized")
        else:
            self.initialized = False
            logging.warning("DataDog APM not available or disabled")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        if self.initialized:
            tag_list = [f"{k}:{v}" for k, v in (tags or {}).items()]
            statsd.gauge(name, value, tags=tag_list)
    
    def record_transaction(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        if self.initialized:
            tag_list = [f"{k}:{v}" for k, v in (tags or {}).items()]
            statsd.timing(f"transaction.{name}", duration_ms, tags=tag_list)
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        if self.initialized:
            statsd.increment('errors.total', tags=[f"error_type:{type(error).__name__}"])

class ElasticAPMProvider(APMProvider):
    """Elastic APM provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if self.enabled and ELASTIC_APM_AVAILABLE:
            self.client = ElasticAPMClient(config)
            self.initialized = True
            logging.info("Elastic APM provider initialized")
        else:
            self.initialized = False
            logging.warning("Elastic APM not available or disabled")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        if self.initialized:
            self.client.capture_message(f"Metric {name}: {value}")
    
    def record_transaction(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        if self.initialized:
            transaction = self.client.begin_transaction('custom')
            transaction.name = name
            transaction.duration = duration_ms / 1000.0  # Convert to seconds
            self.client.end_transaction(name, 'success')
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        if self.initialized:
            self.client.capture_exception(exc_info=(type(error), error, error.__traceback__))

class APMManager:
    """Central APM management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_config()
        self.metrics = APMMetrics()
        self.providers = []
        self._init_providers()
        
        logging.info(f"APM Manager initialized with {len(self.providers)} providers")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load APM configuration from environment and config files"""
        config = {
            'newrelic': {
                'enabled': os.getenv('NEWRELIC_ENABLED', 'false').lower() == 'true',
                'license_key': os.getenv('NEWRELIC_LICENSE_KEY'),
                'app_name': os.getenv('NEWRELIC_APP_NAME', 'Endpoint-1')
            },
            'datadog': {
                'enabled': os.getenv('DATADOG_ENABLED', 'false').lower() == 'true',
                'api_key': os.getenv('DATADOG_API_KEY'),
                'app_key': os.getenv('DATADOG_APP_KEY')
            },
            'elastic_apm': {
                'enabled': os.getenv('ELASTIC_APM_ENABLED', 'false').lower() == 'true',
                'service_name': os.getenv('ELASTIC_APM_SERVICE_NAME', 'endpoint-1'),
                'server_url': os.getenv('ELASTIC_APM_SERVER_URL', 'http://localhost:8200'),
                'secret_token': os.getenv('ELASTIC_APM_SECRET_TOKEN')
            },
            'internal': {
                'enabled': True,  # Always enable internal metrics
                'retention_hours': int(os.getenv('APM_RETENTION_HOURS', '24'))
            }
        }
        return config
    
    def _init_providers(self):
        """Initialize all configured APM providers"""
        # New Relic
        if self.config['newrelic']['enabled']:
            self.providers.append(NewRelicProvider(self.config['newrelic']))
        
        # DataDog
        if self.config['datadog']['enabled']:
            self.providers.append(DataDogProvider(self.config['datadog']))
        
        # Elastic APM
        if self.config['elastic_apm']['enabled']:
            self.providers.append(ElasticAPMProvider(self.config['elastic_apm']))
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record metric to all providers"""
        # Always record to internal metrics
        self.metrics.record_metric(name, value, tags)
        
        # Record to external providers
        for provider in self.providers:
            try:
                provider.record_metric(name, value, tags)
            except Exception as e:
                logging.error(f"APM provider error recording metric: {e}")
    
    def record_transaction(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record transaction to all providers"""
        self.metrics.record_timing(name, duration_ms, tags)
        
        for provider in self.providers:
            try:
                provider.record_transaction(name, duration_ms, tags)
            except Exception as e:
                logging.error(f"APM provider error recording transaction: {e}")
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record error to all providers"""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        self.metrics.increment_counter('errors.total', tags={
            'error_type': type(error).__name__
        })
        
        for provider in self.providers:
            try:
                provider.record_error(error, context)
            except Exception as e:
                logging.error(f"APM provider error recording error: {e}")
    
    @contextmanager
    def trace_operation(self, name: str, tags: Dict[str, str] = None):
        """Context manager for tracing operations"""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            self.record_error(e, {'operation': name, 'tags': tags})
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_transaction(name, duration_ms, tags)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get APM system health status"""
        summary = self.metrics.get_metrics_summary()
        
        return {
            'apm_status': 'healthy',
            'providers_active': len(self.providers),
            'internal_metrics': summary,
            'recent_alerts': summary.get('recent_alerts', []),
            'uptime_seconds': summary.get('uptime_seconds', 0)
        }

# Global APM instance
apm_manager = None

def init_apm(config: Dict[str, Any] = None) -> APMManager:
    """Initialize global APM manager"""
    global apm_manager
    apm_manager = APMManager(config)
    return apm_manager

def get_apm() -> APMManager:
    """Get global APM manager instance"""
    global apm_manager
    if apm_manager is None:
        apm_manager = init_apm()
    return apm_manager

# Decorators for easy integration
def apm_trace(operation_name: str = None, tags: Dict[str, str] = None):
    """Decorator to automatically trace function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            apm = get_apm()
            
            with apm.trace_operation(name, tags):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def apm_counter(metric_name: str = None, tags: Dict[str, str] = None):
    """Decorator to automatically count function calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}.calls"
            apm = get_apm()
            apm.metrics.increment_counter(name, tags=tags)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Flask integration helpers
def init_flask_apm(app, config: Dict[str, Any] = None):
    """Initialize APM for Flask application"""
    apm = init_apm(config)
    
    @app.before_request
    def before_request():
        from flask import request, g
        g.apm_start_time = time.time()
        g.apm_endpoint = request.endpoint or 'unknown'
    
    @app.after_request
    def after_request(response):
        from flask import g
        if hasattr(g, 'apm_start_time'):
            duration_ms = (time.time() - g.apm_start_time) * 1000
            apm.record_transaction(
                f"http.{g.apm_endpoint}",
                duration_ms,
                tags={
                    'method': request.method,
                    'status_code': str(response.status_code),
                    'endpoint': g.apm_endpoint
                }
            )
        return response
    
    @app.errorhandler(Exception)
    def handle_error(error):
        apm.record_error(error, {
            'endpoint': getattr(g, 'apm_endpoint', 'unknown'),
            'method': request.method,
            'url': request.url
        })
        return error
    
    # Add APM health endpoint
    @app.route('/apm/health')
    def apm_health():
        return apm.get_health_status()
    
    @app.route('/apm/metrics')
    def apm_metrics():
        return apm.metrics.get_metrics_summary()
    
    return apm 