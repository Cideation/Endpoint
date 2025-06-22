#!/usr/bin/env python3
"""
BEM Emergence Performance Monitor
Real-time monitoring for VaaS, PaaS, P2P financial modes
Tracks emergence generation, routing efficiency, and billing performance
"""

import asyncio
import time
import logging
import json
import psutil
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import redis
import threading
from collections import deque, defaultdict
import websockets
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

class EmergencePerformanceMonitor:
    """
    Real-time performance monitoring for BEM emergence financial system
    Monitors VaaS billing, PaaS pool management, P2P exchanges, and system health
    """
    
    def __init__(self, redis_client=None, monitoring_interval=10):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        self.monitoring_interval = monitoring_interval
        
        # Performance data storage
        self.metrics_buffer = deque(maxlen=10000)  # Last 10k metrics
        self.alerts = deque(maxlen=1000)  # Last 1k alerts
        
        # Real-time counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # System thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 1000.0,
            'error_rate_percent': 5.0,
            'queue_size': 100,
            'cache_hit_rate_percent': 70.0
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_tasks = []
        
        # WebSocket connections for real-time updates
        self.websocket_clients = set()
        
        logger.info("Emergence Performance Monitor initialized")
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitor_tasks = [
            asyncio.create_task(self._monitor_system_resources()),
            asyncio.create_task(self._monitor_emergence_routing()),
            asyncio.create_task(self._monitor_financial_transactions()),
            asyncio.create_task(self._monitor_database_performance()),
            asyncio.create_task(self._monitor_cache_performance()),
            asyncio.create_task(self._process_alerts()),
            asyncio.create_task(self._broadcast_metrics())
        ]
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        self.monitoring_active = False
        
        for task in self.monitor_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitor_tasks, return_exceptions=True)
        self.monitor_tasks.clear()
        
        logger.info("Performance monitoring stopped")
    
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        self.metrics_buffer.append(metric)
        
        # Store in Redis for persistence
        redis_key = f"metrics:{metric_name}:{int(time.time())}"
        self.redis_client.setex(redis_key, 3600, json.dumps(asdict(metric), default=str))
        
        # Check for alerts
        self._check_threshold_alert(metric)
    
    def record_timer(self, timer_name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record timing information"""
        self.timers[timer_name].append(duration_ms)
        
        # Keep only last 1000 measurements
        if len(self.timers[timer_name]) > 1000:
            self.timers[timer_name] = self.timers[timer_name][-1000:]
        
        # Record as metric
        self.record_metric(f"timer.{timer_name}", duration_ms, tags)
    
    def increment_counter(self, counter_name: str, tags: Dict[str, str] = None):
        """Increment a counter"""
        self.counters[counter_name] += 1
        self.record_metric(f"counter.{counter_name}", self.counters[counter_name], tags)
    
    async def _monitor_system_resources(self):
        """Monitor system CPU, memory, disk usage"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric("system.cpu_percent", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric("system.memory_percent", memory.percent)
                self.record_metric("system.memory_available_gb", memory.available / (1024**3))
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.record_metric("system.disk_percent", disk_percent)
                
                # Network I/O
                network = psutil.net_io_counters()
                self.record_metric("system.network_bytes_sent", network.bytes_sent)
                self.record_metric("system.network_bytes_recv", network.bytes_recv)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_emergence_routing(self):
        """Monitor emergence routing performance"""
        while self.monitoring_active:
            try:
                # Get routing metrics from Redis
                routing_metrics = self._get_routing_metrics()
                
                for metric_name, value in routing_metrics.items():
                    self.record_metric(f"routing.{metric_name}", value)
                
                # Monitor queue sizes
                queue_sizes = self._get_queue_sizes()
                for queue_name, size in queue_sizes.items():
                    self.record_metric(f"queue.{queue_name}_size", size)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Routing monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_financial_transactions(self):
        """Monitor VaaS, PaaS, P2P transaction performance"""
        while self.monitoring_active:
            try:
                # VaaS transaction metrics
                vaas_metrics = await self._get_vaas_metrics()
                for metric_name, value in vaas_metrics.items():
                    self.record_metric(f"vaas.{metric_name}", value)
                
                # PaaS pool metrics
                paas_metrics = await self._get_paas_metrics()
                for metric_name, value in paas_metrics.items():
                    self.record_metric(f"paas.{metric_name}", value)
                
                # P2P exchange metrics
                p2p_metrics = await self._get_p2p_metrics()
                for metric_name, value in p2p_metrics.items():
                    self.record_metric(f"p2p.{metric_name}", value)
                
                await asyncio.sleep(self.monitoring_interval * 2)  # Less frequent for financial data
                
            except Exception as e:
                logger.error(f"Financial monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_database_performance(self):
        """Monitor database query performance"""
        while self.monitoring_active:
            try:
                # Get database metrics
                db_metrics = self._get_database_metrics()
                
                for metric_name, value in db_metrics.items():
                    self.record_metric(f"database.{metric_name}", value)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Database monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_cache_performance(self):
        """Monitor Redis cache performance"""
        while self.monitoring_active:
            try:
                # Redis info
                redis_info = self.redis_client.info()
                
                self.record_metric("cache.used_memory_mb", redis_info.get('used_memory', 0) / (1024*1024))
                self.record_metric("cache.connected_clients", redis_info.get('connected_clients', 0))
                self.record_metric("cache.total_commands_processed", redis_info.get('total_commands_processed', 0))
                self.record_metric("cache.keyspace_hits", redis_info.get('keyspace_hits', 0))
                self.record_metric("cache.keyspace_misses", redis_info.get('keyspace_misses', 0))
                
                # Calculate hit rate
                hits = redis_info.get('keyspace_hits', 0)
                misses = redis_info.get('keyspace_misses', 0)
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0
                self.record_metric("cache.hit_rate_percent", hit_rate)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Cache monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def _get_routing_metrics(self) -> Dict[str, float]:
        """Get emergence routing metrics from Redis"""
        try:
            # Get stored routing statistics
            routing_stats = self.redis_client.get("routing_stats")
            if routing_stats:
                stats = json.loads(routing_stats)
                return {
                    'total_requests': stats.get('total_requests', 0),
                    'vaas_routed': stats.get('vaas_routed', 0),
                    'paas_routed': stats.get('paas_routed', 0),
                    'p2p_routed': stats.get('p2p_routed', 0),
                    'held_requests': stats.get('held_requests', 0),
                    'failed_requests': stats.get('failed_requests', 0),
                    'average_processing_time_ms': stats.get('average_processing_time', 0) * 1000
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting routing metrics: {e}")
            return {}
    
    def _get_queue_sizes(self) -> Dict[str, int]:
        """Get queue sizes from Redis"""
        try:
            queue_sizes = {}
            for queue_name in ['vaas_queue', 'paas_queue', 'p2p_queue']:
                size = self.redis_client.llen(queue_name)
                queue_sizes[queue_name] = size
            return queue_sizes
        except Exception as e:
            logger.error(f"Error getting queue sizes: {e}")
            return {}
    
    async def _get_vaas_metrics(self) -> Dict[str, float]:
        """Get VaaS transaction metrics"""
        try:
            # This would integrate with your actual VaaS system
            # For now, return mock metrics
            return {
                'transactions_per_minute': 5.2,
                'average_transaction_amount': 89.50,
                'success_rate_percent': 98.5,
                'payment_processing_time_ms': 450.0
            }
        except Exception as e:
            logger.error(f"Error getting VaaS metrics: {e}")
            return {}
    
    async def _get_paas_metrics(self) -> Dict[str, float]:
        """Get PaaS pool metrics"""
        try:
            return {
                'active_pools': 12,
                'fulfilled_pools_today': 3,
                'average_pool_size': 5500.0,
                'contribution_processing_time_ms': 250.0
            }
        except Exception as e:
            logger.error(f"Error getting PaaS metrics: {e}")
            return {}
    
    async def _get_p2p_metrics(self) -> Dict[str, float]:
        """Get P2P exchange metrics"""
        try:
            return {
                'exchanges_per_hour': 15.8,
                'average_trust_score': 0.82,
                'successful_exchanges_percent': 95.2,
                'exchange_processing_time_ms': 120.0
            }
        except Exception as e:
            logger.error(f"Error getting P2P metrics: {e}")
            return {}
    
    def _get_database_metrics(self) -> Dict[str, float]:
        """Get database performance metrics"""
        try:
            # Get stored database statistics
            db_stats = self.redis_client.get("db_performance_stats")
            if db_stats:
                stats = json.loads(db_stats)
                return {
                    'query_count': stats.get('total_queries', 0),
                    'average_query_time_ms': stats.get('average_query_time_ms', 0),
                    'slow_queries': stats.get('slow_queries', 0),
                    'cache_hit_rate_percent': stats.get('cache_hit_rate_percent', 0),
                    'connection_pool_usage': stats.get('connection_pool_usage', 0)
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting database metrics: {e}")
            return {}
    
    def _check_threshold_alert(self, metric: PerformanceMetric):
        """Check if metric exceeds threshold and create alert"""
        threshold_key = metric.metric_name.replace('system.', '').replace('database.', '').replace('cache.', '')
        
        if threshold_key in self.thresholds:
            threshold = self.thresholds[threshold_key]
            
            if metric.value > threshold:
                alert = {
                    'type': 'threshold_exceeded',
                    'metric': metric.metric_name,
                    'value': metric.value,
                    'threshold': threshold,
                    'timestamp': metric.timestamp.isoformat(),
                    'severity': self._calculate_severity(metric.value, threshold),
                    'tags': metric.tags
                }
                
                self.alerts.append(alert)
                logger.warning(f"Alert: {metric.metric_name} = {metric.value} exceeds threshold {threshold}")
    
    def _calculate_severity(self, value: float, threshold: float) -> str:
        """Calculate alert severity based on how much threshold is exceeded"""
        ratio = value / threshold
        
        if ratio >= 2.0:
            return 'critical'
        elif ratio >= 1.5:
            return 'high'
        elif ratio >= 1.2:
            return 'medium'
        else:
            return 'low'
    
    async def _process_alerts(self):
        """Process and send alerts"""
        while self.monitoring_active:
            try:
                # Process recent alerts
                recent_alerts = [alert for alert in self.alerts 
                               if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(minutes=5)]
                
                if recent_alerts:
                    # Send alerts (email, Slack, etc.)
                    await self._send_alerts(recent_alerts)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def _send_alerts(self, alerts: List[Dict]):
        """Send alerts to configured channels"""
        try:
            # This would integrate with your alerting system
            # For now, just log the alerts
            for alert in alerts:
                if alert['severity'] in ['critical', 'high']:
                    logger.critical(f"ALERT: {alert['metric']} = {alert['value']}")
                else:
                    logger.warning(f"Alert: {alert['metric']} = {alert['value']}")
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    
    async def _broadcast_metrics(self):
        """Broadcast real-time metrics to WebSocket clients"""
        while self.monitoring_active:
            try:
                if self.websocket_clients:
                    # Get recent metrics
                    recent_metrics = [
                        asdict(metric) for metric in list(self.metrics_buffer)[-50:]
                    ]
                    
                    # Broadcast to all connected clients
                    message = json.dumps({
                        'type': 'metrics_update',
                        'metrics': recent_metrics,
                        'timestamp': datetime.now().isoformat()
                    }, default=str)
                    
                    disconnected_clients = set()
                    for client in self.websocket_clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                    
                    # Remove disconnected clients
                    self.websocket_clients -= disconnected_clients
                
                await asyncio.sleep(5)  # Broadcast every 5 seconds
                
            except Exception as e:
                logger.error(f"Metrics broadcast error: {e}")
                await asyncio.sleep(5)
    
    def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        # Filter recent metrics
        recent_metrics = [
            metric for metric in self.metrics_buffer
            if metric.timestamp > cutoff_time
        ]
        
        # Group metrics by name
        grouped_metrics = defaultdict(list)
        for metric in recent_metrics:
            grouped_metrics[metric.metric_name].append(metric.value)
        
        # Calculate statistics
        summary = {}
        for metric_name, values in grouped_metrics.items():
            if values:
                summary[metric_name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'median': statistics.median(values),
                    'latest': values[-1]
                }
        
        # Add system overview
        summary['_overview'] = {
            'period_minutes': minutes,
            'total_metrics': len(recent_metrics),
            'unique_metrics': len(grouped_metrics),
            'alerts_count': len([alert for alert in self.alerts 
                               if datetime.fromisoformat(alert['timestamp']) > cutoff_time]),
            'generated_at': datetime.now().isoformat()
        }
        
        return summary
    
    def get_alerts(self, minutes: int = 60) -> List[Dict]:
        """Get alerts from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
    
    async def handle_websocket_client(self, websocket, path):
        """Handle WebSocket client connections for real-time monitoring"""
        self.websocket_clients.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            # Send initial data
            initial_data = {
                'type': 'initial_data',
                'performance_summary': self.get_performance_summary(15),
                'recent_alerts': self.get_alerts(15)
            }
            await websocket.send(json.dumps(initial_data, default=str))
            
            # Keep connection alive
            async for message in websocket:
                # Handle client messages if needed
                pass
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")

# Global monitor instance
performance_monitor = EmergencePerformanceMonitor()

# Convenience functions for integration
def record_emergence_timing(emergence_type: str, duration_ms: float, mode: str):
    """Record emergence generation timing"""
    performance_monitor.record_timer(
        f"emergence.{emergence_type.lower()}_generation",
        duration_ms,
        {'mode': mode}
    )

def record_routing_decision(mode: str, processing_time_ms: float):
    """Record routing decision timing"""
    performance_monitor.record_timer(
        "routing.decision_time",
        processing_time_ms,
        {'mode': mode}
    )
    performance_monitor.increment_counter(f"routing.{mode}_routed")

def record_financial_transaction(transaction_type: str, amount: float, success: bool):
    """Record financial transaction"""
    performance_monitor.record_metric(
        f"financial.{transaction_type}_amount",
        amount,
        {'success': str(success)}
    )
    performance_monitor.increment_counter(
        f"financial.{transaction_type}_{'success' if success else 'failure'}"
    )

if __name__ == "__main__":
    # Test the performance monitor
    async def test_monitor():
        monitor = EmergencePerformanceMonitor()
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Record some test metrics
        monitor.record_metric("test.cpu_usage", 45.2)
        monitor.record_timer("test.api_response", 250.5)
        monitor.increment_counter("test.requests")
        
        # Wait a bit for monitoring to collect data
        await asyncio.sleep(30)
        
        # Get performance summary
        summary = monitor.get_performance_summary(5)
        print(f"Performance Summary: {json.dumps(summary, indent=2, default=str)}")
        
        # Get alerts
        alerts = monitor.get_alerts(5)
        print(f"Recent Alerts: {json.dumps(alerts, indent=2, default=str)}")
        
        # Stop monitoring
        await monitor.stop_monitoring()
    
    asyncio.run(test_monitor())
