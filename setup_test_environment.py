#!/usr/bin/env python3
"""
BEM System Test Environment Setup
Installs dependencies and prepares environment for testing
"""

import subprocess
import sys
import os
from pathlib import Path

def install_test_dependencies():
    """Install required test dependencies"""
    
    test_requirements = [
        "httpx>=0.25.0",
        "pytest>=7.0.0", 
        "pytest-asyncio>=0.21.0",
        "docker>=6.0.0",
        "psutil>=5.9.0",
        "asyncpg>=0.29.0",
        "redis>=4.0.0",
        "aioredis>=2.0.0"
    ]
    
    print("📦 Installing test dependencies...")
    
    for package in test_requirements:
        print(f"  Installing {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_test_config():
    """Create test configuration files"""
    
    # Create pytest configuration
    pytest_ini = """
[tool:pytest]
testpaths = .
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
timeout = 300
markers =
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests
    quick: Quick tests
"""
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_ini.strip())
    
    print("✅ Created pytest.ini configuration")
    
    # Create test environment file
    test_env = """
# Test Environment Configuration
VAAS_BILLING_URL=http://localhost:8004
VAAS_CRM_URL=http://localhost:8005  
VAAS_WEBHOOK_URL=http://localhost:8006
FABRICATION_BUNDLE_URL=http://localhost:8007
ECM_GATEWAY_URL=http://localhost:8765
AUTOMATED_ADMIN_URL=http://localhost:8003

# Database configurations
VAAS_POSTGRES_HOST=localhost
VAAS_POSTGRES_PORT=5433
VAAS_POSTGRES_DB=bem_vaas
VAAS_POSTGRES_USER=bem_user
VAAS_POSTGRES_PASSWORD=bem_password

MAIN_POSTGRES_HOST=localhost
MAIN_POSTGRES_PORT=5432
MAIN_POSTGRES_DB=bem_production

# Redis configurations
VAAS_REDIS_HOST=localhost
VAAS_REDIS_PORT=6380

MAIN_REDIS_HOST=localhost
MAIN_REDIS_PORT=6379

# Test settings
TEST_TIMEOUT=300
PARALLEL_TESTS=10
QUICK_MODE=false
"""
    
    with open(".env.test", "w") as f:
        f.write(test_env.strip())
    
    print("✅ Created .env.test configuration")

def check_docker_setup():
    """Check if Docker is properly set up"""
    
    try:
        result = subprocess.run(
            ["docker", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ Docker version: {result.stdout.strip()}")
        
        # Check if Docker is running
        subprocess.run(
            ["docker", "ps"], 
            capture_output=True, 
            check=True
        )
        print("✅ Docker daemon is running")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not available or not running")
        print("Please install Docker and ensure it's running:")
        print("  - macOS: Docker Desktop for Mac")
        print("  - Linux: sudo apt install docker.io")
        print("  - Windows: Docker Desktop for Windows")
        return False

def create_test_data():
    """Create sample test data"""
    
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Sample emergence test data
    emergence_samples = {
        "blueprint_emergence.json": {
            "project_id": "sample-blueprint-project",
            "node_id": "blueprint-node-001", 
            "fit_score": 0.95,
            "node_data": {
                "finalized": True,
                "emergence_status": "ready",
                "geometry_resolved": True,
                "specs_complete": True,
                "wall_count": 4,
                "total_area": 120.5
            }
        },
        "bom_emergence.json": {
            "project_id": "sample-bom-project",
            "node_id": "bom-node-001",
            "fit_score": 0.92,
            "node_data": {
                "finalized": True,
                "emergence_status": "ready", 
                "materials_resolved": True,
                "components_sourced": True,
                "material_list": ["concrete_block", "steel_rebar", "cement"]
            }
        },
        "investment_emergence.json": {
            "project_id": "sample-investment-project",
            "node_id": "investment-node-001",
            "fit_score": 0.94,
            "node_data": {
                "finalized": True,
                "emergence_status": "ready",
                "investment_analysis_complete": True,
                "roi_calculated": True,
                "roi_percentage": 15.2,
                "payback_years": 6.8
            }
        }
    }
    
    import json
    
    for filename, data in emergence_samples.items():
        with open(test_data_dir / filename, "w") as f:
            json.dump(data, f, indent=2)
    
    print(f"✅ Created test data in {test_data_dir}")

def setup_service_monitoring():
    """Set up basic service monitoring for tests"""
    
    monitoring_script = """#!/bin/bash
# Service Health Monitor for Tests

echo "🔍 Checking BEM services..."

services=(
    "bem-vaas-billing:8004"
    "bem-vaas-crm:8005"
    "bem-vaas-webhook:8006"
    "bem-postgres-vaas:5433"
    "bem-redis-vaas:6380"
)

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if docker ps | grep -q $name; then
        echo "✅ $name (container running)"
        
        if nc -z localhost $port 2>/dev/null; then
            echo "✅ $name (port $port accessible)"
        else
            echo "⚠️ $name (port $port not accessible)"
        fi
    else
        echo "❌ $name (container not running)"
    fi
done

echo ""
echo "To start services:"
echo "cd MICROSERVICE_ENGINES/admin-devops && ./start-devops.sh"
"""
    
    with open("check_services.sh", "w") as f:
        f.write(monitoring_script)
    
    os.chmod("check_services.sh", 0o755)
    print("✅ Created service monitoring script (check_services.sh)")

def main():
    """Main setup function"""
    
    print("🚀 BEM System Test Environment Setup")
    print("=" * 50)
    
    steps = [
        ("Checking Docker setup", check_docker_setup),
        ("Installing test dependencies", install_test_dependencies),
        ("Creating test configuration", create_test_config),
        ("Creating test data", create_test_data),
        ("Setting up service monitoring", setup_service_monitoring)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ {step_name} failed!")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 Test environment setup complete!")
    print("")
    print("Next steps:")
    print("1. Start BEM services:")
    print("   cd MICROSERVICE_ENGINES/admin-devops")
    print("   ./start-devops.sh")
    print("   Select option 3: Start complete stack")
    print("")
    print("2. Run tests:")
    print("   python run_system_tests.py")
    print("")
    print("3. Quick health check:")
    print("   python run_system_tests.py --quick")
    print("")
    print("4. Check service status:")
    print("   ./check_services.sh")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 