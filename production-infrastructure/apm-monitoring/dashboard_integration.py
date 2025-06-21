"""
APM Dashboard Integration
Real-time performance monitoring dashboard with web interface
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request
import threading
import logging
from collections import defaultdict, deque
import psutil

from .apm_integration import get_apm
from .performance_profiler import get_profiler

class APMDashboard:
    """Real-time APM dashboard with web interface"""
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.real_time_data = defaultdict(deque)
        self.alert_history = deque(maxlen=1000)
        self.dashboard_config = {
            'refresh_interval': 5,  # seconds
            'chart_history_minutes': 30,
            'alert_retention_hours': 24
        }
        
        # Start real-time data collection
        self._start_data_collection()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize dashboard with Flask app"""
        self.app = app
        self._register_routes()
        logging.info("APM Dashboard initialized with Flask app")
    
    def _start_data_collection(self):
        """Start background data collection for real-time dashboard"""
        def collect_data():
            while True:
                try:
                    timestamp = datetime.now()
                    
                    # System metrics
                    process = psutil.Process()
                    cpu_percent = psutil.cpu_percent()
                    memory_info = process.memory_info()
                    
                    # APM metrics
                    apm = get_apm()
                    apm_health = apm.get_health_status()
                    
                    # Performance metrics
                    profiler = get_profiler()
                    perf_report = profiler.get_performance_report(hours_back=1)
                    
                    # Store real-time data
                    self._store_real_time_data('cpu_usage', cpu_percent, timestamp)
                    self._store_real_time_data('memory_usage_mb', memory_info.rss / (1024 * 1024), timestamp)
                    self._store_real_time_data('response_time', self._get_avg_response_time(), timestamp)
                    self._store_real_time_data('error_rate', self._get_error_rate(), timestamp)
                    self._store_real_time_data('throughput', self._get_throughput(), timestamp)
                    
                    time.sleep(self.dashboard_config['refresh_interval'])
                    
                except Exception as e:
                    logging.error(f"Data collection error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=collect_data, daemon=True)
        thread.start()
    
    def _store_real_time_data(self, metric: str, value: float, timestamp: datetime):
        """Store real-time metric data"""
        data_point = {
            'timestamp': timestamp.isoformat(),
            'value': value
        }
        
        # Keep only last 30 minutes of data
        max_age = datetime.now() - timedelta(minutes=self.dashboard_config['chart_history_minutes'])
        
        self.real_time_data[metric].append(data_point)
        
        # Remove old data
        while (self.real_time_data[metric] and 
               datetime.fromisoformat(self.real_time_data[metric][0]['timestamp']) < max_age):
            self.real_time_data[metric].popleft()
    
    def _get_avg_response_time(self) -> float:
        """Calculate average response time from recent requests"""
        apm = get_apm()
        recent_transactions = []
        
        # Get recent transaction data
        for metric_name, metric_data in apm.metrics.metrics.items():
            if 'duration' in metric_name and metric_data:
                recent_values = [m['value'] for m in metric_data[-10:]]
                if recent_values:
                    recent_transactions.extend(recent_values)
        
        return sum(recent_transactions) / len(recent_transactions) if recent_transactions else 0
    
    def _get_error_rate(self) -> float:
        """Calculate error rate percentage"""
        apm = get_apm()
        
        # Get error count and total requests
        error_count = 0
        total_requests = 0
        
        for counter_name, count in apm.metrics.counters.items():
            if 'errors.total' in counter_name:
                error_count += count
            elif 'requests.total' in counter_name or 'calls' in counter_name:
                total_requests += count
        
        return (error_count / total_requests * 100) if total_requests > 0 else 0
    
    def _get_throughput(self) -> float:
        """Calculate current throughput (requests per minute)"""
        apm = get_apm()
        
        # Count transactions in last minute
        cutoff_time = datetime.now() - timedelta(minutes=1)
        transaction_count = 0
        
        for metric_name, metric_data in apm.metrics.metrics.items():
            if 'duration' in metric_name:
                recent_transactions = [
                    m for m in metric_data 
                    if m['timestamp'] > cutoff_time
                ]
                transaction_count += len(recent_transactions)
        
        return transaction_count
    
    def _register_routes(self):
        """Register dashboard routes with Flask app"""
        
        @self.app.route('/apm/dashboard')
        def dashboard():
            """Main dashboard page"""
            return render_template_string(DASHBOARD_TEMPLATE)
        
        @self.app.route('/apm/api/realtime')
        def realtime_data():
            """Real-time data API endpoint"""
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'metrics': dict(self.real_time_data)
            })
        
        @self.app.route('/apm/api/summary')
        def summary_data():
            """Summary statistics API endpoint"""
            apm = get_apm()
            profiler = get_profiler()
            
            # Get comprehensive summary
            apm_health = apm.get_health_status()
            perf_report = profiler.get_performance_report(hours_back=1)
            
            return jsonify({
                'apm_health': apm_health,
                'performance_report': perf_report,
                'system_info': self._get_system_info(),
                'alerts': list(self.alert_history)[-10:]  # Last 10 alerts
            })
        
        @self.app.route('/apm/api/metrics/<metric_name>')
        def metric_detail(metric_name):
            """Detailed metric data"""
            hours = request.args.get('hours', 1, type=int)
            apm = get_apm()
            
            metric_data = apm.metrics.metrics.get(metric_name, [])
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_data = [
                {
                    'timestamp': m['timestamp'].isoformat(),
                    'value': m['value'],
                    'tags': m.get('tags', {})
                }
                for m in metric_data
                if m['timestamp'] > cutoff_time
            ]
            
            return jsonify({
                'metric_name': metric_name,
                'data_points': len(recent_data),
                'time_range_hours': hours,
                'data': recent_data
            })
        
        @self.app.route('/apm/api/profile')
        def profile_data():
            """Performance profile data"""
            profiler = get_profiler()
            hours = request.args.get('hours', 1, type=int)
            
            report = profiler.get_performance_report(hours_back=hours)
            
            return jsonify(report)
        
        @self.app.route('/apm/api/alerts')
        def alerts_data():
            """Alert history data"""
            return jsonify({
                'alerts': list(self.alert_history),
                'total_alerts': len(self.alert_history)
            })
        
        @self.app.route('/apm/api/export')
        def export_data():
            """Export all APM data"""
            apm = get_apm()
            profiler = get_profiler()
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'apm_metrics': apm.metrics.get_metrics_summary(),
                'performance_data': profiler.get_performance_report(hours_back=24),
                'real_time_data': dict(self.real_time_data),
                'alert_history': list(self.alert_history)
            }
            
            return jsonify(export_data)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        process = psutil.Process()
        
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'process_id': process.pid,
            'process_threads': process.num_threads(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'uptime_seconds': time.time() - process.create_time()
        }
    
    def add_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Add alert to dashboard"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alert_history.append(alert)
        logging.info(f"APM Alert: {alert_type} - {message}")

# Dashboard HTML template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>APM Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #0066cc;
        }
        .metric-unit {
            font-size: 14px;
            color: #666;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 15px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy { background-color: #4CAF50; }
        .status-warning { background-color: #FF9800; }
        .status-error { background-color: #F44336; }
        .alerts-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #ddd;
            background: #f9f9f9;
        }
        .alert-error { border-left-color: #F44336; }
        .alert-warning { border-left-color: #FF9800; }
        .alert-info { border-left-color: #2196F3; }
        .refresh-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>🚀 APM Performance Dashboard</h1>
            <p>Real-time application performance monitoring</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">
                    <span class="status-indicator status-healthy"></span>
                    System Status
                </div>
                <div class="metric-value" id="system-status">Healthy</div>
                <div class="chart-container">
                    <canvas id="cpu-chart"></canvas>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Memory Usage</div>
                <div class="metric-value">
                    <span id="memory-value">0</span>
                    <span class="metric-unit">MB</span>
                </div>
                <div class="chart-container">
                    <canvas id="memory-chart"></canvas>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Response Time</div>
                <div class="metric-value">
                    <span id="response-time-value">0</span>
                    <span class="metric-unit">ms</span>
                </div>
                <div class="chart-container">
                    <canvas id="response-chart"></canvas>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Throughput</div>
                <div class="metric-value">
                    <span id="throughput-value">0</span>
                    <span class="metric-unit">req/min</span>
                </div>
                <div class="chart-container">
                    <canvas id="throughput-chart"></canvas>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Error Rate</div>
                <div class="metric-value">
                    <span id="error-rate-value">0</span>
                    <span class="metric-unit">%</span>
                </div>
                <div class="chart-container">
                    <canvas id="error-chart"></canvas>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Active Connections</div>
                <div class="metric-value">
                    <span id="connections-value">0</span>
                </div>
                <div style="margin-top: 15px; font-size: 14px; color: #666;">
                    <div>Uptime: <span id="uptime-value">0</span> hours</div>
                    <div>APM Providers: <span id="providers-value">0</span></div>
                </div>
            </div>
        </div>
        
        <div class="alerts-section">
            <h3>Recent Alerts</h3>
            <div id="alerts-container">
                <p>No recent alerts</p>
            </div>
        </div>
    </div>
    
    <div class="refresh-status" id="refresh-status">
        Last updated: Never
    </div>
    
    <script>
        // Chart configurations
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                elements: {
                    point: {
                        radius: 0
                    }
                }
            }
        };
        
        // Initialize charts
        const charts = {
            cpu: new Chart(document.getElementById('cpu-chart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: '#FF6384',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        fill: true
                    }]
                }
            }),
            memory: new Chart(document.getElementById('memory-chart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Memory MB',
                        data: [],
                        borderColor: '#36A2EB',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: true
                    }]
                }
            }),
            response: new Chart(document.getElementById('response-chart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Response Time ms',
                        data: [],
                        borderColor: '#FFCE56',
                        backgroundColor: 'rgba(255, 206, 86, 0.1)',
                        fill: true
                    }]
                }
            }),
            throughput: new Chart(document.getElementById('throughput-chart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Requests/min',
                        data: [],
                        borderColor: '#4BC0C0',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        fill: true
                    }]
                }
            }),
            error: new Chart(document.getElementById('error-chart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Error Rate %',
                        data: [],
                        borderColor: '#FF6384',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        fill: true
                    }]
                }
            })
        };
        
        // Update dashboard data
        function updateDashboard() {
            Promise.all([
                fetch('/apm/api/realtime').then(r => r.json()),
                fetch('/apm/api/summary').then(r => r.json())
            ]).then(([realtime, summary]) => {
                updateMetrics(realtime.metrics);
                updateSummary(summary);
                updateCharts(realtime.metrics);
                updateRefreshStatus();
            }).catch(console.error);
        }
        
        function updateMetrics(metrics) {
            // Update current values
            const latest = (metric) => {
                const data = metrics[metric];
                return data && data.length > 0 ? data[data.length - 1].value : 0;
            };
            
            document.getElementById('memory-value').textContent = Math.round(latest('memory_usage_mb'));
            document.getElementById('response-time-value').textContent = Math.round(latest('response_time'));
            document.getElementById('throughput-value').textContent = Math.round(latest('throughput'));
            document.getElementById('error-rate-value').textContent = latest('error_rate').toFixed(1);
        }
        
        function updateSummary(summary) {
            if (summary.apm_health) {
                document.getElementById('uptime-value').textContent = 
                    Math.round(summary.apm_health.uptime_seconds / 3600);
                document.getElementById('providers-value').textContent = 
                    summary.apm_health.providers_active;
            }
            
            // Update alerts
            const alertsContainer = document.getElementById('alerts-container');
            if (summary.alerts && summary.alerts.length > 0) {
                alertsContainer.innerHTML = summary.alerts.map(alert => 
                    `<div class="alert-item alert-${alert.severity}">
                        <strong>${alert.type}</strong>: ${alert.message}
                        <small style="float: right;">${new Date(alert.timestamp).toLocaleTimeString()}</small>
                    </div>`
                ).join('');
            } else {
                alertsContainer.innerHTML = '<p>No recent alerts</p>';
            }
        }
        
        function updateCharts(metrics) {
            Object.keys(charts).forEach(chartKey => {
                const metricKey = chartKey === 'cpu' ? 'cpu_usage' : 
                                chartKey === 'response' ? 'response_time' :
                                chartKey + '_usage_mb';
                
                if (metrics[metricKey]) {
                    const data = metrics[metricKey];
                    const chart = charts[chartKey];
                    
                    chart.data.labels = data.map(d => new Date(d.timestamp).toLocaleTimeString());
                    chart.data.datasets[0].data = data.map(d => d.value);
                    chart.update('none');
                }
            });
        }
        
        function updateRefreshStatus() {
            document.getElementById('refresh-status').textContent = 
                `Last updated: ${new Date().toLocaleTimeString()}`;
        }
        
        // Start auto-refresh
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""

# Global dashboard instance
apm_dashboard = None

def init_dashboard(app: Flask = None) -> APMDashboard:
    """Initialize APM dashboard"""
    global apm_dashboard
    apm_dashboard = APMDashboard(app)
    return apm_dashboard

def get_dashboard() -> APMDashboard:
    """Get global dashboard instance"""
    global apm_dashboard
    if apm_dashboard is None:
        apm_dashboard = init_dashboard()
    return apm_dashboard 