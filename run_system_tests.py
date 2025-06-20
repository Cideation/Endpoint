#!/usr/bin/env python3
"""
BEM System Test Runner
Executes comprehensive tests for all recent implementations
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "httpx", "pytest", "docker", "psutil", "asyncpg", "redis"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_services_running():
    """Check if required services are running"""
    import docker
    
    client = docker.from_env()
    running_containers = [c.name for c in client.containers.list()]
    
    required_services = [
        "bem-vaas-billing",
        "bem-vaas-crm", 
        "bem-vaas-webhook",
        "bem-postgres-vaas",
        "bem-redis-vaas"
    ]
    
    missing_services = []
    for service in required_services:
        if service not in running_containers:
            missing_services.append(service)
    
    if missing_services:
        print(f"❌ Required services not running: {', '.join(missing_services)}")
        print("Start services with: cd MICROSERVICE_ENGINES/admin-devops && ./start-devops.sh")
        return False
    
    print("✅ All required services are running")
    return True

def run_tests():
    """Run the complete test suite"""
    
    print("🚀 BEM System Integration Test Runner")
    print("=" * 50)
    
    # Check dependencies
    print("📋 Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Check services
    print("🔍 Checking services...")
    if not check_services_running():
        print("\n💡 To start services:")
        print("   cd MICROSERVICE_ENGINES/admin-devops")
        print("   ./start-devops.sh")
        print("   Select option 3: Start complete stack")
        return False
    
    # Run main test suite
    print("🧪 Running integration tests...")
    
    try:
        # Import and run the test suite
        from test_complete_system_integration import run_complete_system_test
        success = run_complete_system_test()
        return success
    except ImportError:
        print("❌ Test module not found. Running with subprocess...")
        result = subprocess.run([
            sys.executable, "test_complete_system_integration.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0

def run_specific_tests():
    """Run specific test categories"""
    
    test_categories = {
        "1": ("Infrastructure Health", "TestInfrastructureHealth"),
        "2": ("VaaS Billing System", "TestVaaSBillingSystem"), 
        "3": ("Fabrication Bundle Generator", "TestFabricationBundleGenerator"),
        "4": ("Microservice Integration", "TestMicroserviceIntegration"),
        "5": ("End-to-End Workflows", "TestEndToEndWorkflows"),
        "6": ("Performance & Load", "TestPerformanceAndLoad"),
        "7": ("Data Consistency", "TestDataConsistency")
    }
    
    print("\nSelect test category:")
    for key, (name, _) in test_categories.items():
        print(f"  {key}) {name}")
    print("  8) Run all tests")
    print("  9) Exit")
    
    choice = input("\nEnter choice (1-9): ").strip()
    
    if choice == "9":
        return True
    elif choice == "8":
        return run_tests()
    elif choice in test_categories:
        category_name, test_class = test_categories[choice]
        print(f"\n🧪 Running {category_name} tests...")
        
        # Run specific test class with pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "test_complete_system_integration.py", 
            f"-k", test_class,
            "-v"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    else:
        print("❌ Invalid choice")
        return run_specific_tests()

def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Quick infrastructure check only
            print("🏃 Quick health check...")
            if check_dependencies() and check_services_running():
                print("✅ Quick check passed!")
                return True
            else:
                print("❌ Quick check failed!")
                return False
        elif sys.argv[1] == "--specific":
            return run_specific_tests()
    
    # Default: run all tests
    success = run_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("System is ready for production use.")
    else:
        print("\n❌ Some tests failed.")
        print("Please check the output above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 