#!/usr/bin/env python3
"""
Start Project-Aware BEM System
Launches all required services for the project-aware agent console
"""

import subprocess
import time
import sys
import os
import signal
import webbrowser
from pathlib import Path

class ProjectAwareBEMLauncher:
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent.parent
        self.frontend_dir = Path(__file__).parent
        
    def log(self, message, level="INFO"):
        """Log with timestamp and level"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def start_process(self, name, command, cwd=None, wait_time=2):
        """Start a process and track it"""
        self.log(f"Starting {name}...")
        try:
            if cwd is None:
                cwd = self.base_dir
                
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            self.processes.append({
                'name': name,
                'process': process,
                'command': command
            })
            
            self.log(f"✅ {name} started (PID: {process.pid})")
            time.sleep(wait_time)
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to start {name}: {e}", "ERROR")
            return False
    
    def check_ports(self):
        """Check if required ports are available"""
        import socket
        
        required_ports = {
            8765: "ECM WebSocket Gateway",
            8080: "GraphQL API Server", 
            5000: "DAG Alpha Engine",
            5001: "Functor Types Engine",
            5002: "Callback Engine",
            5003: "SFDE Engine",
            5004: "Graph Runtime Engine",
            6379: "Redis Cache"
        }
        
        unavailable = []
        
        for port, service in required_ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                unavailable.append(f"Port {port} ({service})")
        
        if unavailable:
            self.log("⚠️  Warning: Some ports are already in use:", "WARN")
            for service in unavailable:
                self.log(f"   - {service}", "WARN")
            self.log("System may still work if services are already running", "WARN")
        
        return len(unavailable) == 0
    
    def start_infrastructure(self):
        """Start infrastructure services"""
        self.log("🚀 Starting Infrastructure Services...")
        
        # Start Redis for real-time subscriptions
        if not self.start_process(
            "Redis Cache", 
            "redis-server --port 6379 --daemonize yes",
            wait_time=1
        ):
            self.log("Redis may already be running or using system service", "WARN")
        
        # Start microservices via Docker Compose
        self.log("Starting microservice containers...")
        docker_success = self.start_process(
            "Microservice Containers",
            "docker-compose -f MICROSERVICE_ENGINES/docker-compose.optimized.yml up -d",
            wait_time=5
        )
        
        if not docker_success:
            self.log("⚠️  Microservices failed to start via Docker", "WARN")
            self.log("Attempting to start individual services...", "INFO")
            self.start_individual_microservices()
        
        return True
    
    def start_individual_microservices(self):
        """Start microservices individually if Docker fails"""
        microservices = [
            {
                'name': 'DAG Alpha Engine',
                'command': 'python main.py',
                'cwd': 'MICROSERVICE_ENGINES/ne-dag-alpha'
            },
            {
                'name': 'Functor Types Engine', 
                'command': 'python main.py',
                'cwd': 'MICROSERVICE_ENGINES/ne-functor-types'
            },
            {
                'name': 'Callback Engine',
                'command': 'python main.py', 
                'cwd': 'MICROSERVICE_ENGINES/ne-callback-engine'
            },
            {
                'name': 'SFDE Engine',
                'command': 'python main.py',
                'cwd': 'MICROSERVICE_ENGINES/sfde'
            },
            {
                'name': 'Graph Runtime Engine',
                'command': 'python main.py',
                'cwd': 'MICROSERVICE_ENGINES/ne-graph-runtime-engine'
            }
        ]
        
        for service in microservices:
            cwd = self.base_dir / service['cwd']
            self.start_process(
                service['name'],
                service['command'],
                cwd=cwd,
                wait_time=1
            )
    
    def start_core_services(self):
        """Start core BEM services"""
        self.log("🎯 Starting Core BEM Services...")
        
        # Start ECM Gateway
        ecm_success = self.start_process(
            "ECM WebSocket Gateway",
            "python Final_Phase/ecm_gateway.py",
            wait_time=2
        )
        
        # Start GraphQL API Server
        graphql_success = self.start_process(
            "GraphQL API Server",
            "python frontend/graphql_server.py",
            wait_time=3
        )
        
        return ecm_success and graphql_success
    
    def verify_services(self):
        """Verify that services are responding"""
        self.log("🔍 Verifying service health...")
        
        import requests
        import socket
        
        checks = [
            {
                'name': 'GraphQL API',
                'url': 'http://localhost:8080/health',
                'timeout': 5
            },
            {
                'name': 'ECM WebSocket',
                'host': 'localhost',
                'port': 8765,
                'type': 'socket'
            }
        ]
        
        all_healthy = True
        
        for check in checks:
            try:
                if check.get('type') == 'socket':
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(check.get('timeout', 3))
                    result = sock.connect_ex((check['host'], check['port']))
                    sock.close()
                    
                    if result == 0:
                        self.log(f"✅ {check['name']} is healthy")
                    else:
                        self.log(f"❌ {check['name']} is not responding", "ERROR")
                        all_healthy = False
                else:
                    response = requests.get(check['url'], timeout=check.get('timeout', 5))
                    if response.status_code == 200:
                        self.log(f"✅ {check['name']} is healthy")
                    else:
                        self.log(f"❌ {check['name']} returned status {response.status_code}", "ERROR")
                        all_healthy = False
                        
            except Exception as e:
                self.log(f"❌ {check['name']} health check failed: {e}", "ERROR")
                all_healthy = False
        
        return all_healthy
    
    def launch_interface(self):
        """Launch the project-aware agent console"""
        self.log("🌐 Launching Project-Aware Agent Console...")
        
        interface_file = self.frontend_dir / "agent_console_project_aware.html"
        
        if interface_file.exists():
            try:
                # Convert to file:// URL
                interface_url = f"file://{interface_file.absolute()}"
                
                webbrowser.open(interface_url)
                self.log(f"✅ Interface opened at: {interface_url}")
                
                # Also show GraphQL playground
                graphql_url = "http://localhost:8080/graphql"
                self.log(f"🔧 GraphQL Playground: {graphql_url}")
                
            except Exception as e:
                self.log(f"❌ Failed to open interface: {e}", "ERROR")
                self.log(f"📝 Manually open: {interface_file.absolute()}", "INFO")
        else:
            self.log(f"❌ Interface file not found: {interface_file}", "ERROR")
    
    def display_status(self):
        """Display system status and URLs"""
        self.log("📊 System Status:")
        self.log("=" * 50)
        
        services = [
            ("Project-Aware Agent Console", f"file://{self.frontend_dir}/agent_console_project_aware.html"),
            ("GraphQL API", "http://localhost:8080/graphql"),
            ("GraphQL Health", "http://localhost:8080/health"),
            ("ECM WebSocket", "ws://localhost:8765"),
            ("DAG Alpha Engine", "http://localhost:5000/health"),
            ("SFDE Engine", "http://localhost:5003/health"),
            ("Graph Runtime Engine", "http://localhost:5004/health"),
            ("Functor Types Engine", "http://localhost:5001/health"),
            ("Callback Engine", "http://localhost:5002/health")
        ]
        
        for name, url in services:
            self.log(f"🔗 {name:<30} {url}")
        
        self.log("=" * 50)
        self.log("💡 Tips:")
        self.log("   - Use project selector to assign agent_project_tag")
        self.log("   - Select nodes and run data affinity analysis")
        self.log("   - Monitor real-time pulse events")
        self.log("   - Press Ctrl+C to stop all services")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.log("\n🛑 Received shutdown signal, stopping services...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def cleanup(self):
        """Stop all started processes"""
        self.log("🧹 Cleaning up processes...")
        
        for proc_info in reversed(self.processes):
            try:
                process = proc_info['process']
                name = proc_info['name']
                
                if process.poll() is None:  # Process is still running
                    self.log(f"Stopping {name} (PID: {process.pid})")
                    
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    else:
                        process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.log(f"Force killing {name}")
                        if hasattr(os, 'killpg'):
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        else:
                            process.kill()
                        
            except Exception as e:
                self.log(f"Error stopping {proc_info['name']}: {e}", "ERROR")
        
        # Stop Docker containers
        try:
            subprocess.run([
                "docker-compose", "-f", "MICROSERVICE_ENGINES/docker-compose.optimized.yml", "down"
            ], cwd=self.base_dir, timeout=30, capture_output=True)
            self.log("✅ Docker containers stopped")
        except Exception as e:
            self.log(f"Warning: Error stopping Docker containers: {e}", "WARN")
        
        self.log("✅ Cleanup completed")
    
    def run(self):
        """Main execution flow"""
        self.log("🎯 Starting Project-Aware BEM System")
        self.log("=" * 50)
        
        self.setup_signal_handlers()
        
        # Check port availability
        self.check_ports()
        
        try:
            # Start infrastructure
            if not self.start_infrastructure():
                self.log("❌ Failed to start infrastructure", "ERROR")
                return False
            
            # Start core services  
            if not self.start_core_services():
                self.log("❌ Failed to start core services", "ERROR")
                return False
            
            # Wait for services to fully initialize
            self.log("⏳ Waiting for services to initialize...")
            time.sleep(5)
            
            # Verify service health
            if not self.verify_services():
                self.log("⚠️  Some services are not healthy, but continuing...", "WARN")
            
            # Launch interface
            self.launch_interface()
            
            # Display status
            self.display_status()
            
            # Keep running
            self.log("🚀 System ready! Press Ctrl+C to stop.")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            self.log(f"❌ Unexpected error: {e}", "ERROR")
            return False
        finally:
            self.cleanup()
        
        return True

def main():
    """Entry point"""
    launcher = ProjectAwareBEMLauncher()
    success = launcher.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 