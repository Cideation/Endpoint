#!/usr/bin/env python3
"""
Complete Backend Launcher
Starts both GraphQL Admin API and SocketIO Dashboard Server
"""

import subprocess
import sys
import time
import os

def start_services():
    print("🚀 Starting Complete BEM Backend Services...")
    print("=" * 60)
    
    processes = []
    
    # Start GraphQL Admin API
    print("📊 Starting GraphQL Admin API (Port 8001)...")
    graphql_process = subprocess.Popen([
        sys.executable, "engine/graphql_admin_api.py"
    ])
    processes.append(graphql_process)
    
    time.sleep(2)
    
    # Start SocketIO Dashboard Server
    print("🎛️ Starting SocketIO Dashboard Server (Port 5000)...")
    socketio_process = subprocess.Popen([
        sys.executable, "engine/socketio_dashboard_server.py"
    ])
    processes.append(socketio_process)
    
    time.sleep(3)
    
    print("✅ Backend Services Started!")
    print("=" * 60)
    print("📊 GraphQL Admin API:     http://localhost:8001/graphql")
    print("🎛️ SocketIO Dashboard:    http://localhost:5000/socket.io/")
    print("🔧 Health Check (GraphQL): http://localhost:8001/health")
    print("🔧 Health Check (SocketIO): http://localhost:5000/health")
    print("=" * 60)
    print("🎯 Use Cases:")
    print("   • GraphQL: System config, schema management, functor settings")
    print("   • SocketIO: Real-time agent states, pulse rendering, live updates")
    print("=" * 60)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        for process in processes:
            process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    start_services()
