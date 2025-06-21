"""
APM System Demo
Demonstrates comprehensive APM monitoring capabilities
"""

import time
import random
import threading
from flask import Flask, jsonify
import logging

# Import APM components
from apm_integration import init_apm, get_apm, apm_trace, apm_counter, init_flask_apm
from performance_profiler import init_profiler, get_profiler, profile_performance, profile_memory
from dashboard_integration import init_dashboard, get_dashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# APM Configuration
APM_CONFIG = {
    'newrelic': {
        'enabled': False,  # Set to True with real credentials
        'license_key': 'your_license_key',
        'app_name': 'Endpoint-1-Demo'
    },
    'datadog': {
        'enabled': False,  # Set to True with real credentials
        'api_key': 'your_api_key',
        'app_key': 'your_app_key'
    },
    'elastic_apm': {
        'enabled': False,  # Set to True with real setup
        'service_name': 'endpoint-1-demo',
        'server_url': 'http://localhost:8200'
    },
    'profiling': {
        'memory_profiling': True,
        'cpu_profiling': True,
        'line_profiling': False,
        'profile_interval': 30  # 30 seconds for demo
    }
}

class DemoBusinessLogic:
    """Demo business logic with APM instrumentation"""
    
    def __init__(self):
        self.user_sessions = {}
        self.order_queue = []
        self.apm = get_apm()
    
    @apm_trace('business.user_login')
    @apm_counter('user_events.login')
    def user_login(self, user_id: str, source: str = 'web'):
        """Simulate user login with APM tracking"""
        # Simulate login processing time
        processing_time = random.uniform(0.1, 0.5)
        time.sleep(processing_time)
        
        # Record business metrics
        self.apm.record_metric('user_sessions.active', len(self.user_sessions) + 1)
        self.apm.record_metric('login.processing_time_ms', processing_time * 1000, 
                             tags={'source': source})
        
        # Simulate occasional errors
        if random.random() < 0.05:  # 5% error rate
            error = Exception(f"Login failed for user {user_id}")
            self.apm.record_error(error, {'user_id': user_id, 'source': source})
            raise error
        
        self.user_sessions[user_id] = {
            'login_time': time.time(),
            'source': source,
            'requests': 0
        }
        
        logger.info(f"User {user_id} logged in from {source}")
        return {'status': 'success', 'user_id': user_id}

def create_demo_flask_app():
    """Create Flask app with APM integration"""
    app = Flask(__name__)
    
    # Initialize APM with Flask
    apm = init_flask_apm(app, APM_CONFIG)
    dashboard = init_dashboard(app)
    
    business = DemoBusinessLogic()
    
    @app.route('/')
    def home():
        """Home page with links to APM dashboard"""
        return '''
        <h1>APM Demo Application</h1>
        <h2>Available Endpoints:</h2>
        <ul>
            <li><a href="/apm/dashboard">APM Dashboard</a> - Real-time monitoring</li>
            <li><a href="/apm/health">Health Check</a> - System health status</li>
            <li><a href="/apm/metrics">Metrics Summary</a> - APM metrics data</li>
        </ul>
        <h2>APM Features Demonstrated:</h2>
        <ul>
            <li>Real-time system monitoring (CPU, Memory, Disk)</li>
            <li>Application performance tracking</li>
            <li>Error monitoring and alerting</li>
            <li>Custom business metrics</li>
            <li>Performance profiling</li>
            <li>Interactive dashboard</li>
        </ul>
        '''
    
    return app

def main():
    """Main demo function"""
    print("🚀 Starting APM Demo Application")
    print("=" * 50)
    
    # Initialize APM system
    print("📊 Initializing APM system...")
    apm = init_apm(APM_CONFIG)
    profiler = init_profiler(APM_CONFIG.get('profiling', {}))
    
    print(f"✅ APM initialized with {len(apm.providers)} providers")
    print(f"✅ Performance profiler initialized")
    
    # Create and start Flask app
    print("🌐 Starting Flask application...")
    app = create_demo_flask_app()
    
    print("\n" + "=" * 50)
    print("🎯 APM Demo Ready!")
    print("📈 Access dashboard at: http://localhost:5000/apm/dashboard")
    print("🏠 Home page at: http://localhost:5000/")
    print("📊 Health check at: http://localhost:5000/apm/health")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down APM demo...")

if __name__ == '__main__':
    main()
