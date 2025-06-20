#!/usr/bin/env python3
"""
BEM GraphQL Server Startup Script
Handles environment setup and launches the GraphQL server
"""

import os
import sys
import subprocess
import json
import signal
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "strawberry-graphql",
        "fastapi", 
        "uvicorn",
        "psycopg2",
        "asyncpg",
        "redis"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + missing_packages, check=True)
        print("✅ Dependencies installed successfully")
    
    print()

def setup_environment():
    """Setup environment variables with defaults"""
    print("🔧 Setting up environment...")
    
    # Default environment variables
    env_defaults = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432", 
        "DB_NAME": "bem_production",
        "DB_USER": "bem_user",
        "DB_PASSWORD": "bem_secure_pass_2024",
        "REDIS_URL": "redis://localhost:6379",
        "ENVIRONMENT": "development",
        "LOG_LEVEL": "INFO"
    }
    
    # Set defaults if not already set
    for key, default_value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = default_value
            print(f"📝 Set {key}={default_value}")
        else:
            print(f"✅ Using existing {key}={os.environ[key]}")
    
    print()

def check_external_services():
    """Check if external services (Redis, PostgreSQL) are available"""
    print("🌐 Checking external services...")
    
    # Check Redis
    try:
        import redis
        redis_client = redis.Redis.from_url(os.environ["REDIS_URL"])
        redis_client.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("   You can still run the server, but real-time features may not work")
    
    # Check PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ["DB_HOST"],
            port=os.environ["DB_PORT"],
            database=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"]
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
    except Exception as e:
        print(f"⚠️  PostgreSQL connection failed: {e}")
        print("   Server will run with limited functionality")
    
    print()

def create_output_directories():
    """Create necessary output directories"""
    print("📁 Creating output directories...")
    
    directories = [
        "outputs",
        "outputs/gcode",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📂 {directory}")
    
    print()

def start_development_server():
    """Start the GraphQL server in development mode"""
    print("🚀 Starting BEM GraphQL Server...")
    print("=" * 50)
    
    try:
        # Import and run the server
        import uvicorn
        from graphql_server import app
        
        print("🌟 Server starting on http://localhost:8000")
        print("📊 GraphQL endpoint: http://localhost:8000/graphql")
        print("🔍 Health check: http://localhost:8000/health")
        print("🌐 WebSocket: ws://localhost:8000/ws/cytoscape")
        print("=" * 50)
        
        # Start server with hot reload
        uvicorn.run(
            "graphql_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level=os.environ.get("LOG_LEVEL", "info").lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

def start_with_docker():
    """Start server using Docker Compose"""
    print("🐳 Starting with Docker Compose...")
    
    if not Path("docker-compose.graphql.yml").exists():
        print("❌ docker-compose.graphql.yml not found")
        sys.exit(1)
    
    try:
        subprocess.run([
            "docker-compose", 
            "-f", "docker-compose.graphql.yml",
            "up", "--build"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Compose failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping containers...")
        subprocess.run([
            "docker-compose",
            "-f", "docker-compose.graphql.yml", 
            "down"
        ])

def show_usage():
    """Show usage information"""
    print("""
🚀 BEM GraphQL Server Startup Script

Usage:
    python start_graphql_server.py [OPTIONS]

Options:
    --docker    Start using Docker Compose
    --test      Run test suite after starting
    --help      Show this help message

Environment Variables:
    DB_HOST         PostgreSQL host (default: localhost)
    DB_PORT         PostgreSQL port (default: 5432)
    DB_NAME         Database name (default: bem_production)
    DB_USER         Database user (default: bem_user)
    DB_PASSWORD     Database password (default: bem_secure_pass_2024)
    REDIS_URL       Redis URL (default: redis://localhost:6379)

Examples:
    python start_graphql_server.py                # Start development server
    python start_graphql_server.py --docker       # Start with Docker
    python start_graphql_server.py --test         # Start and run tests
    """)

def run_tests():
    """Run the test suite"""
    print("🧪 Running GraphQL test suite...")
    try:
        subprocess.run([sys.executable, "test_graphql_demo.py"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️  Some tests failed, but server should still work")
    except FileNotFoundError:
        print("❌ test_graphql_demo.py not found")

def main():
    """Main startup function"""
    args = sys.argv[1:]
    
    if "--help" in args:
        show_usage()
        return
    
    print("🎯 BEM GraphQL Server Startup")
    print("=" * 40)
    
    # Setup phase
    check_dependencies()
    setup_environment()
    create_output_directories()
    check_external_services()
    
    # Start server
    if "--docker" in args:
        start_with_docker()
    else:
        # Start in background if testing
        if "--test" in args:
            print("🚀 Starting server in background for testing...")
            import threading
            import time
            
            server_thread = threading.Thread(target=start_development_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Wait for server to start
            time.sleep(3)
            run_tests()
        else:
            start_development_server()

if __name__ == "__main__":
    main() 