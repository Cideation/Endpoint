#!/usr/bin/env python3
"""
Production Health Monitoring System
Provides comprehensive health checks, dependency monitoring, and system status
"""

import os
import time
import json
import psutil
import asyncio
import aiohttp
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import redis
import psycopg2
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 10
    critical: bool = True
    dependencies: List[str] = None

@dataclass
class HealthResult:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    timestamp: str
    response_time: float
    details: Dict[str, Any] = None

class HealthMonitor:
    """Production health monitoring system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_default_config()
        self.health_checks = {}
        self.results_history = {}
        self.system_status = HealthStatus.UNKNOWN
        self.monitoring_active = False
        
        self._register_default_checks()
        self._start_monitoring()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'history_retention_hours': 24,
            'max_results_per_check': 1000,
            'alert_thresholds': {
                'consecutive_failures': 3,
                'failure_rate_threshold': 0.8,
                'response_time_threshold': 5.0
            },
            'database': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'bem_production'),
                'user': os.getenv('DB_USER', 'bem_user'),
                'password': os.getenv('DB_PASSWORD', '')
            },
            'redis': {
                'url': os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            },
            'services': {
                'ecm_gateway': {'url': 'http://localhost:8765/health', 'critical': True},
                'dual_ac_api': {'url': 'http://localhost:8002/health', 'critical': True},
                'behavior_ac': {'url': 'http://localhost:8003/health', 'critical': False},
                'frontend': {'url': 'http://localhost:80/health', 'critical': False}
            }
        }
    
    def _register_default_checks(self):
        """Register default health checks"""
        
        # System resource checks
        self.register_check(HealthCheck(
            name="system_cpu",
            check_function=self._check_cpu_usage,
            interval_seconds=30,
            critical=False
        ))
        
        self.register_check(HealthCheck(
            name="system_memory",
            check_function=self._check_memory_usage,
            interval_seconds=30,
            critical=True
        ))
        
        self.register_check(HealthCheck(
            name="system_disk",
            check_function=self._check_disk_usage,
            interval_seconds=60,
            critical=True
        ))
        
        # Database connectivity
        self.register_check(HealthCheck(
            name="database_connection",
            check_function=self._check_database,
            interval_seconds=30,
            critical=True
        ))
        
        # Redis connectivity
        self.register_check(HealthCheck(
            name="redis_connection",
            check_function=self._check_redis,
            interval_seconds=30,
            critical=False
        ))
        
        # File system checks
        self.register_check(HealthCheck(
            name="file_system",
            check_function=self._check_file_system,
            interval_seconds=60,
            critical=True
        ))
        
        # Service endpoint checks
        for service_name, service_config in self.config['services'].items():
            self.register_check(HealthCheck(
                name=f"service_{service_name}",
                check_function=lambda config=service_config: self._check_service_endpoint(config),
                interval_seconds=30,
                critical=service_config['critical']
            ))
    
    def register_check(self, health_check: HealthCheck):
        """Register a health check"""
        self.health_checks[health_check.name] = health_check
        self.results_history[health_check.name] = []
        logger.info(f"Registered health check: {health_check.name}")
    
    async def _check_cpu_usage(self) -> HealthResult:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 70:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthResult(
                name="system_cpu",
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                response_time=1.0,
                details={
                    'cpu_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count(),
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            )
            
        except Exception as e:
            return HealthResult(
                name="system_cpu",
                status=HealthStatus.CRITICAL,
                message=f"CPU check failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=0.0
            )
    
    async def _check_memory_usage(self) -> HealthResult:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory.percent:.1f}%"
            elif memory.percent > 80:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            return HealthResult(
                name="system_memory",
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                response_time=0.1,
                details={
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'memory_total': memory.total,
                    'swap_percent': swap.percent,
                    'swap_used': swap.used,
                    'swap_total': swap.total
                }
            )
            
        except Exception as e:
            return HealthResult(
                name="system_memory",
                status=HealthStatus.CRITICAL,
                message=f"Memory check failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=0.0
            )
    
    async def _check_disk_usage(self) -> HealthResult:
        """Check disk usage"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent > 80:
                status = HealthStatus.WARNING
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthResult(
                name="system_disk",
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                response_time=0.1,
                details={
                    'disk_percent': disk_percent,
                    'disk_free': disk_usage.free,
                    'disk_used': disk_usage.used,
                    'disk_total': disk_usage.total
                }
            )
            
        except Exception as e:
            return HealthResult(
                name="system_disk",
                status=HealthStatus.CRITICAL,
                message=f"Disk check failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=0.0
            )
    
    async def _check_database(self) -> HealthResult:
        """Check database connectivity"""
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(
                host=self.config['database']['host'],
                port=self.config['database']['port'],
                database=self.config['database']['database'],
                user=self.config['database']['user'],
                password=self.config['database']['password'],
                connect_timeout=5
            )
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT version(), now()")
                result = cursor.fetchone()
                
                if result:
                    version = result[0]
                    timestamp = result[1]
                    
                    conn.close()
                    response_time = time.time() - start_time
                    
                    return HealthResult(
                        name="database_connection",
                        status=HealthStatus.HEALTHY,
                        message="Database connection successful",
                        timestamp=datetime.now().isoformat(),
                        response_time=response_time,
                        details={
                            'database_version': version,
                            'database_time': str(timestamp),
                            'host': self.config['database']['host'],
                            'database': self.config['database']['database']
                        }
                    )
            
        except psycopg2.OperationalError as e:
            return HealthResult(
                name="database_connection",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
        except Exception as e:
            return HealthResult(
                name="database_connection",
                status=HealthStatus.CRITICAL,
                message=f"Database check error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
    
    async def _check_redis(self) -> HealthResult:
        """Check Redis connectivity"""
        start_time = time.time()
        
        try:
            redis_client = redis.from_url(self.config['redis']['url'])
            
            # Test basic operations
            redis_client.ping()
            redis_client.set('health_check', 'test', ex=60)
            value = redis_client.get('health_check')
            
            if value == b'test':
                response_time = time.time() - start_time
                
                # Get Redis info
                info = redis_client.info()
                
                return HealthResult(
                    name="redis_connection",
                    status=HealthStatus.HEALTHY,
                    message="Redis connection successful",
                    timestamp=datetime.now().isoformat(),
                    response_time=response_time,
                    details={
                        'redis_version': info.get('redis_version'),
                        'used_memory': info.get('used_memory'),
                        'connected_clients': info.get('connected_clients'),
                        'uptime_in_seconds': info.get('uptime_in_seconds')
                    }
                )
            else:
                return HealthResult(
                    name="redis_connection",
                    status=HealthStatus.WARNING,
                    message="Redis read/write test failed",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time
                )
            
        except redis.ConnectionError as e:
            return HealthResult(
                name="redis_connection",
                status=HealthStatus.CRITICAL,
                message=f"Redis connection failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
        except Exception as e:
            return HealthResult(
                name="redis_connection",
                status=HealthStatus.WARNING,
                message=f"Redis check error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
    
    async def _check_file_system(self) -> HealthResult:
        """Check file system health"""
        try:
            issues = []
            details = {}
            
            # Check critical directories
            critical_dirs = ['logs', 'backups', 'Final_Phase', 'MICROSERVICE_ENGINES']
            
            for directory in critical_dirs:
                if not os.path.exists(directory):
                    issues.append(f"Critical directory missing: {directory}")
                else:
                    # Check write permissions
                    test_file = os.path.join(directory, '.health_check_write_test')
                    try:
                        with open(test_file, 'w') as f:
                            f.write('test')
                        os.remove(test_file)
                        details[f"{directory}_writable"] = True
                    except Exception:
                        issues.append(f"Directory not writable: {directory}")
                        details[f"{directory}_writable"] = False
            
            # Check log file sizes
            log_files = ['logs/error_monitoring.log', 'ecm_log.txt', 'error.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    size_mb = os.path.getsize(log_file) / (1024 * 1024)
                    details[f"{log_file}_size_mb"] = size_mb
                    
                    if size_mb > 1000:  # 1GB
                        issues.append(f"Large log file: {log_file} ({size_mb:.1f}MB)")
            
            if issues:
                status = HealthStatus.WARNING
                message = f"File system issues found: {', '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "File system healthy"
            
            return HealthResult(
                name="file_system",
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                response_time=0.5,
                details=details
            )
            
        except Exception as e:
            return HealthResult(
                name="file_system",
                status=HealthStatus.CRITICAL,
                message=f"File system check failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=0.0
            )
    
    async def _check_service_endpoint(self, service_config: Dict) -> HealthResult:
        """Check service endpoint health"""
        start_time = time.time()
        url = service_config['url']
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                            return HealthResult(
                                name=f"service_{url.split('/')[2]}",
                                status=HealthStatus.HEALTHY,
                                message=f"Service endpoint healthy: {url}",
                                timestamp=datetime.now().isoformat(),
                                response_time=response_time,
                                details=data if isinstance(data, dict) else {'response': str(data)}
                            )
                        except:
                            return HealthResult(
                                name=f"service_{url.split('/')[2]}",
                                status=HealthStatus.HEALTHY,
                                message=f"Service endpoint responding: {url}",
                                timestamp=datetime.now().isoformat(),
                                response_time=response_time
                            )
                    else:
                        return HealthResult(
                            name=f"service_{url.split('/')[2]}",
                            status=HealthStatus.WARNING,
                            message=f"Service endpoint returned {response.status}: {url}",
                            timestamp=datetime.now().isoformat(),
                            response_time=response_time
                        )
                        
        except asyncio.TimeoutError:
            return HealthResult(
                name=f"service_{url.split('/')[2]}",
                status=HealthStatus.CRITICAL,
                message=f"Service endpoint timeout: {url}",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
        except Exception as e:
            return HealthResult(
                name=f"service_{url.split('/')[2]}",
                status=HealthStatus.CRITICAL,
                message=f"Service endpoint failed: {url} - {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
    
    async def run_single_check(self, check_name: str) -> Optional[HealthResult]:
        """Run a single health check"""
        if check_name not in self.health_checks:
            return None
        
        check = self.health_checks[check_name]
        
        try:
            result = await asyncio.wait_for(
                check.check_function(),
                timeout=check.timeout_seconds
            )
            
            # Store result in history
            self.results_history[check_name].append(result)
            
            # Limit history size
            max_results = self.config['max_results_per_check']
            if len(self.results_history[check_name]) > max_results:
                self.results_history[check_name] = self.results_history[check_name][-max_results:]
            
            return result
            
        except asyncio.TimeoutError:
            result = HealthResult(
                name=check_name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timeout after {check.timeout_seconds}s",
                timestamp=datetime.now().isoformat(),
                response_time=check.timeout_seconds
            )
            
            self.results_history[check_name].append(result)
            return result
            
        except Exception as e:
            result = HealthResult(
                name=check_name,
                status=HealthStatus.CRITICAL,
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                response_time=0.0
            )
            
            self.results_history[check_name].append(result)
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthResult]:
        """Run all registered health checks"""
        tasks = []
        
        for check_name in self.health_checks:
            task = asyncio.create_task(self.run_single_check(check_name))
            tasks.append((check_name, task))
        
        results = {}
        for check_name, task in tasks:
            try:
                result = await task
                if result:
                    results[check_name] = result
            except Exception as e:
                logger.error(f"Failed to run health check {check_name}: {e}")
        
        # Update overall system status
        self._update_system_status(results)
        
        return results
    
    def _update_system_status(self, results: Dict[str, HealthResult]):
        """Update overall system status based on check results"""
        critical_failures = 0
        warnings = 0
        
        for check_name, result in results.items():
            check = self.health_checks[check_name]
            
            if result.status == HealthStatus.CRITICAL and check.critical:
                critical_failures += 1
            elif result.status == HealthStatus.WARNING:
                warnings += 1
        
        if critical_failures > 0:
            self.system_status = HealthStatus.CRITICAL
        elif warnings > 2:
            self.system_status = HealthStatus.WARNING
        else:
            self.system_status = HealthStatus.HEALTHY
    
    def get_system_status(self) -> Dict:
        """Get overall system health status"""
        latest_results = {}
        
        for check_name, history in self.results_history.items():
            if history:
                latest_results[check_name] = asdict(history[-1])
        
        return {
            'overall_status': self.system_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': latest_results,
            'summary': self._get_status_summary(latest_results)
        }
    
    def _get_status_summary(self, results: Dict) -> Dict:
        """Generate status summary"""
        total_checks = len(results)
        healthy = len([r for r in results.values() if r['status'] == HealthStatus.HEALTHY.value])
        warnings = len([r for r in results.values() if r['status'] == HealthStatus.WARNING.value])
        critical = len([r for r in results.values() if r['status'] == HealthStatus.CRITICAL.value])
        
        return {
            'total_checks': total_checks,
            'healthy': healthy,
            'warnings': warnings,
            'critical': critical,
            'health_percentage': (healthy / total_checks * 100) if total_checks > 0 else 0
        }
    
    def _start_monitoring(self):
        """Start continuous health monitoring"""
        def monitor_loop():
            self.monitoring_active = True
            logger.info("Health monitoring started")
            
            while self.monitoring_active:
                try:
                    # Run health checks asynchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    results = loop.run_until_complete(self.run_all_checks())
                    
                    # Log overall status
                    status_summary = self._get_status_summary({k: asdict(v) for k, v in results.items()})
                    logger.info(f"Health check complete: {status_summary['healthy']}/{status_summary['total_checks']} healthy")
                    
                    loop.close()
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                
                # Wait before next check cycle
                time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("Health monitoring stopped")

# Global health monitor instance
health_monitor = HealthMonitor()

# Flask integration
def create_health_endpoints(app):
    """Create Flask health check endpoints"""
    
    @app.route('/health')
    def basic_health():
        """Basic health check endpoint"""
        status = health_monitor.get_system_status()
        
        if status['overall_status'] == HealthStatus.HEALTHY.value:
            return {'status': 'healthy', 'timestamp': status['timestamp']}, 200
        else:
            return {'status': status['overall_status'], 'timestamp': status['timestamp']}, 503
    
    @app.route('/health/detailed')
    def detailed_health():
        """Detailed health check endpoint"""
        return health_monitor.get_system_status()
    
    @app.route('/health/check/<check_name>')
    async def single_health_check(check_name):
        """Run single health check"""
        result = await health_monitor.run_single_check(check_name)
        
        if result:
            return asdict(result)
        else:
            return {'error': f'Unknown health check: {check_name}'}, 404

if __name__ == "__main__":
    # Test the health monitor
    async def test_health_monitor():
        # Run all checks
        results = await health_monitor.run_all_checks()
        
        print("Health Check Results:")
        print("=" * 50)
        
        for check_name, result in results.items():
            status_icon = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.WARNING: "⚠️",
                HealthStatus.CRITICAL: "❌",
                HealthStatus.UNKNOWN: "❓"
            }.get(result.status, "❓")
            
            print(f"{status_icon} {check_name}: {result.message} ({result.response_time:.3f}s)")
        
        # Get system status
        system_status = health_monitor.get_system_status()
        print(f"\nOverall System Status: {system_status['overall_status']}")
        print(f"Summary: {system_status['summary']}")
    
    # Run test
    asyncio.run(test_health_monitor())
    
    # Keep monitor running for a bit
    time.sleep(5)
    health_monitor.stop_monitoring() 