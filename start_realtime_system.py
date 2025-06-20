#!/usr/bin/env python3
"""
Real-Time GraphQL System Launcher
⚡ No Cosmetic Delays - Immediate Backend State Synchronization
"""

import asyncio
import subprocess
import sys
import os
import time
import signal
import webbrowser
from pathlib import Path

# Configuration
SERVICES = {
    'graphql_engine': {
        'script': 'frontend/graphql_realtime_engine.py',
        'port': 8004,
        'name': 'Real-Time GraphQL Engine',
        'health_endpoint': '/health'
    },
    'web_server': {
        'command': ['python', '-m', 'http.server', '8005'],
        'port': 8005,
        'name': 'Frontend Web Server',
        'cwd': 'frontend'
    }
}

class RealTimeSystemLauncher:
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def log(self, message, level='INFO'):
        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
    
    async def check_port(self, port, timeout=30):
        """Check if a port is available and responsive"""
        import socket
        import asyncio
        
        for _ in range(timeout):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', port))
                    if result == 0:
                        return True
            except:
                pass
            await asyncio.sleep(1)
        return False
    
    async def start_service(self, service_name, config):
        """Start a service process"""
        try:
            self.log(f"🚀 Starting {config['name']}...")
            
            if 'script' in config:
                # Python script
                cmd = [sys.executable, config['script']]
                cwd = None
            else:
                # Command
                cmd = config['command']
                cwd = config.get('cwd')
            
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[service_name] = {
                'process': process,
                'config': config
            }
            
            # Wait for service to be ready
            if await self.check_port(config['port']):
                self.log(f"✅ {config['name']} started on port {config['port']}")
                return True
            else:
                self.log(f"❌ {config['name']} failed to start", 'ERROR')
                return False
                
        except Exception as e:
            self.log(f"❌ Error starting {config['name']}: {e}", 'ERROR')
            return False
    
    async def stop_service(self, service_name):
        """Stop a service process"""
        if service_name in self.processes:
            process_info = self.processes[service_name]
            process = process_info['process']
            config = process_info['config']
            
            try:
                self.log(f"🛑 Stopping {config['name']}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.log(f"⚠️ Force killing {config['name']}")
                    process.kill()
                    process.wait()
                
                self.log(f"✅ {config['name']} stopped")
                del self.processes[service_name]
                
            except Exception as e:
                self.log(f"❌ Error stopping {config['name']}: {e}", 'ERROR')
    
    async def start_all_services(self):
        """Start all services in sequence"""
        self.log("🚀 Starting Real-Time GraphQL System...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.log("❌ Python 3.8+ required", 'ERROR')
            return False
        
        # Install dependencies if needed
        await self.check_dependencies()
        
        # Start services
        success_count = 0
        for service_name, config in SERVICES.items():
            if await self.start_service(service_name, config):
                success_count += 1
            else:
                self.log(f"❌ Failed to start {service_name}", 'ERROR')
        
        if success_count == len(SERVICES):
            self.running = True
            self.log("✅ All services started successfully!")
            self.show_startup_info()
            return True
        else:
            self.log(f"❌ Only {success_count}/{len(SERVICES)} services started", 'ERROR')
            await self.stop_all_services()
            return False
    
    async def stop_all_services(self):
        """Stop all running services"""
        self.log("🛑 Stopping all services...")
        
        for service_name in list(self.processes.keys()):
            await self.stop_service(service_name)
        
        self.running = False
        self.log("✅ All services stopped")
    
    async def check_dependencies(self):
        """Check and install dependencies"""
        try:
            import strawberry
            import fastapi
            import uvicorn
            self.log("✅ Core dependencies available")
        except ImportError as e:
            self.log(f"⚠️ Missing dependency: {e}")
            self.log("📦 Installing dependencies...")
            
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', '-r', 'requirements_realtime.txt'
                ])
                self.log("✅ Dependencies installed")
            except subprocess.CalledProcessError:
                self.log("❌ Failed to install dependencies", 'ERROR')
    
    def show_startup_info(self):
        """Display startup information"""
        self.log("=" * 60)
        self.log("🎉 Real-Time GraphQL System Running!")
        self.log("⚡ No Cosmetic Delays - Immediate State Sync")
        self.log("=" * 60)
        
        for service_name, process_info in self.processes.items():
            config = process_info['config']
            self.log(f"🔗 {config['name']}: http://localhost:{config['port']}")
        
        self.log("")
        self.log("📋 Available Endpoints:")
        self.log("   GraphQL Playground: http://localhost:8004/graphql")
        self.log("   WebSocket Endpoint: ws://localhost:8004/ws/realtime")
        self.log("   Health Check: http://localhost:8004/health")
        self.log("   Real-time Interface: http://localhost:8005/realtime_graph_interface.html")
        self.log("")
        self.log("🎮 Controls:")
        self.log("   Ctrl+C to stop all services")
        self.log("   Visit the interface to see real-time graph updates")
        self.log("=" * 60)
        
        # Auto-open browser
        try:
            webbrowser.open('http://localhost:8005/realtime_graph_interface.html')
            self.log("🌐 Browser opened automatically")
        except:
            self.log("⚠️ Could not auto-open browser")
    
    async def monitor_services(self):
        """Monitor running services and restart if needed"""
        while self.running:
            for service_name, process_info in list(self.processes.items()):
                process = process_info['process']
                config = process_info['config']
                
                if process.poll() is not None:
                    self.log(f"⚠️ {config['name']} has stopped unexpectedly", 'WARNING')
                    
                    # Try to restart
                    await self.stop_service(service_name)
                    if await self.start_service(service_name, config):
                        self.log(f"✅ {config['name']} restarted successfully")
                    else:
                        self.log(f"❌ Failed to restart {config['name']}", 'ERROR')
                        self.running = False
                        break
            
            await asyncio.sleep(5)
    
    async def run(self):
        """Main run loop"""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Start all services
            if await self.start_all_services():
                # Monitor services
                await self.monitor_services()
            
        except KeyboardInterrupt:
            self.log("👋 Shutdown requested by user")
        except Exception as e:
            self.log(f"❌ Unexpected error: {e}", 'ERROR')
        finally:
            await self.stop_all_services()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.log("👋 Shutdown signal received")
        self.running = False

# Standalone functions for individual service management
async def start_graphql_only():
    """Start only the GraphQL engine"""
    launcher = RealTimeSystemLauncher()
    config = SERVICES['graphql_engine']
    
    if await launcher.start_service('graphql_engine', config):
        launcher.log("✅ GraphQL Engine running standalone")
        launcher.log(f"🔗 GraphQL Playground: http://localhost:{config['port']}/graphql")
        launcher.log(f"🔌 WebSocket: ws://localhost:{config['port']}/ws/realtime")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await launcher.stop_service('graphql_engine')

async def test_system():
    """Test the real-time system"""
    import httpx
    
    print("🧪 Testing Real-Time System...")
    
    try:
        # Test GraphQL health
        async with httpx.AsyncClient() as client:
            response = await client.get('http://localhost:8004/health')
            if response.status_code == 200:
                print("✅ GraphQL Engine: Healthy")
                data = response.json()
                print(f"   Connected clients: {data.get('connected_clients', 0)}")
                print(f"   Graph version: {data.get('graph_version', 0)}")
            else:
                print("❌ GraphQL Engine: Not responding")
        
        # Test WebSocket
        import websockets
        try:
            async with websockets.connect('ws://localhost:8004/ws/realtime') as websocket:
                await websocket.send('{"type": "ping"}')
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print("✅ WebSocket: Connected and responsive")
        except:
            print("❌ WebSocket: Connection failed")
        
        # Test frontend
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get('http://localhost:8005/realtime_graph_interface.html')
                if response.status_code == 200:
                    print("✅ Frontend: Available")
                else:
                    print("❌ Frontend: Not available")
        except:
            print("❌ Frontend: Connection failed")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time GraphQL System Launcher')
    parser.add_argument('--graphql-only', action='store_true', 
                       help='Start only the GraphQL engine')
    parser.add_argument('--test', action='store_true', 
                       help='Test the running system')
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_system())
    elif args.graphql_only:
        asyncio.run(start_graphql_only())
    else:
        launcher = RealTimeSystemLauncher()
        asyncio.run(launcher.run()) 