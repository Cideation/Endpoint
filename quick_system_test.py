#!/usr/bin/env python3
"""
Quick BEM System Test
Simple test to verify all recent implementations are working
"""

import requests
import json
import time
import sys
from datetime import datetime

def test_service_health(service_name, url, timeout=10):
    """Test if a service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print(f"✅ {service_name}: Healthy")
                return True
            else:
                print(f"⚠️ {service_name}: Responding but reports unhealthy")
                return False
        else:
            print(f"❌ {service_name}: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {service_name}: Connection refused (service not running)")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ {service_name}: Request timeout")
        return False
    except Exception as e:
        print(f"❌ {service_name}: Error - {e}")
        return False

def test_vaas_billing():
    """Test VaaS billing system"""
    print("\n🧪 Testing VaaS Billing System...")
    
    # Test emergence detection
    try:
        emergence_data = {
            "project_id": "quick-test-project",
            "node_id": "quick-test-node",
            "fit_score": 0.95,
            "node_data": {
                "finalized": True,
                "emergence_status": "ready",
                "geometry_resolved": True,
                "specs_complete": True
            }
        }
        
        response = requests.post(
            "http://localhost:8004/emergence/check",
            json=emergence_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("triggered") and data.get("tier_id") == "blueprint_package":
                print("✅ Emergence detection: Working")
                return True
            else:
                print(f"⚠️ Emergence detection: Unexpected response - {data}")
                return False
        else:
            print(f"❌ Emergence detection: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Emergence detection: Error - {e}")
        return False

def test_billing_tiers():
    """Test billing tiers endpoint"""
    print("\n🧪 Testing Billing Tiers...")
    
    try:
        response = requests.get("http://localhost:8004/tiers", timeout=10)
        
        if response.status_code == 200:
            tiers = response.json()
            if len(tiers) == 5:
                tier_ids = [tier["id"] for tier in tiers]
                expected_tiers = [
                    "blueprint_package", "bom_with_suppliers", 
                    "compliance_docs", "investment_packet", "full_emergence_bundle"
                ]
                
                if all(tier in tier_ids for tier in expected_tiers):
                    print("✅ Billing tiers: All 5 tiers configured correctly")
                    return True
                else:
                    print(f"⚠️ Billing tiers: Missing expected tiers")
                    return False
            else:
                print(f"⚠️ Billing tiers: Expected 5 tiers, got {len(tiers)}")
                return False
        else:
            print(f"❌ Billing tiers: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Billing tiers: Error - {e}")
        return False

def test_fabrication_bundle():
    """Test fabrication bundle generator"""
    print("\n🧪 Testing Fabrication Bundle Generator...")
    
    try:
        # Test health first
        response = requests.get("http://localhost:8007/health", timeout=10)
        if response.status_code != 200:
            print("❌ Fabrication service not responding")
            return False
        
        # Test blueprint generation
        geometry_data = {
            "project_id": "quick-fab-test",
            "geometry": {
                "walls": [
                    {"start": [0, 0], "end": [10, 0], "height": 3},
                    {"start": [10, 0], "end": [10, 8], "height": 3},
                    {"start": [10, 8], "end": [0, 8], "height": 3},
                    {"start": [0, 8], "end": [0, 0], "height": 3}
                ]
            },
            "specifications": {
                "wall_material": "concrete_block",
                "foundation_type": "slab_on_grade"
            }
        }
        
        response = requests.post(
            "http://localhost:8007/generate/blueprint",
            json=geometry_data,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success" and "blueprint_id" in data:
                print("✅ Fabrication bundle: Blueprint generation working")
                return True
            else:
                print(f"⚠️ Fabrication bundle: Unexpected response - {data}")
                return False
        else:
            print(f"❌ Fabrication bundle: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Fabrication bundle: Service not running")
        return False
    except Exception as e:
        print(f"❌ Fabrication bundle: Error - {e}")
        return False

def test_webhook_integration():
    """Test webhook integration"""
    print("\n🧪 Testing Webhook Integration...")
    
    try:
        node_update = {
            "node_id": "quick-webhook-test",
            "project_id": "quick-webhook-project",
            "customer_id": "quick-webhook-customer",
            "node_type": "test_node",
            "fit_score": 0.93,
            "finalized": True,
            "emergence_status": "ready",
            "node_data": {
                "geometry_resolved": True,
                "specs_complete": True
            }
        }
        
        response = requests.post(
            "http://localhost:8006/webhook/node-update",
            json=node_update,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Webhook integration: Working")
            return True
        else:
            print(f"❌ Webhook integration: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Webhook integration: Service not running")
        return False
    except Exception as e:
        print(f"❌ Webhook integration: Error - {e}")
        return False

def test_crm_dashboard():
    """Test CRM dashboard APIs"""
    print("\n🧪 Testing CRM Dashboard...")
    
    try:
        # Test metrics endpoint
        response = requests.get("http://localhost:8005/api/metrics", timeout=10)
        
        if response.status_code == 200:
            metrics = response.json()
            required_fields = ["daily_revenue", "monthly_revenue", "total_customers"]
            
            if all(field in metrics for field in required_fields):
                print("✅ CRM dashboard: Metrics API working")
                return True
            else:
                print(f"⚠️ CRM dashboard: Missing required metrics fields")
                return False
        else:
            print(f"❌ CRM dashboard: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ CRM dashboard: Service not running")
        return False
    except Exception as e:
        print(f"❌ CRM dashboard: Error - {e}")
        return False

def check_docker_services():
    """Check if Docker services are running"""
    print("\n🐳 Checking Docker Services...")
    
    try:
        import docker
        client = docker.from_env()
        
        required_containers = [
            "bem-vaas-billing",
            "bem-vaas-crm",
            "bem-vaas-webhook", 
            "bem-postgres-vaas",
            "bem-redis-vaas"
        ]
        
        running_containers = [c.name for c in client.containers.list()]
        
        all_running = True
        for container in required_containers:
            if container in running_containers:
                print(f"✅ {container}: Running")
            else:
                print(f"❌ {container}: Not running")
                all_running = False
        
        return all_running
        
    except ImportError:
        print("⚠️ Docker package not available, skipping container check")
        return True
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Quick BEM System Test")
    print("=" * 40)
    print(f"Started at: {datetime.now()}")
    print("=" * 40)
    
    # Service health checks
    services = {
        "VaaS Billing": "http://localhost:8004",
        "VaaS CRM": "http://localhost:8005", 
        "VaaS Webhook": "http://localhost:8006",
        "Fabrication Bundle": "http://localhost:8007"
    }
    
    print("\n🔍 Service Health Checks...")
    health_results = []
    for name, url in services.items():
        result = test_service_health(name, url)
        health_results.append(result)
    
    # Functional tests
    functional_tests = [
        ("VaaS Billing", test_vaas_billing),
        ("Billing Tiers", test_billing_tiers),
        ("Fabrication Bundle", test_fabrication_bundle),
        ("Webhook Integration", test_webhook_integration),
        ("CRM Dashboard", test_crm_dashboard)
    ]
    
    print("\n🧪 Functional Tests...")
    functional_results = []
    for test_name, test_func in functional_tests:
        try:
            result = test_func()
            functional_results.append(result)
        except Exception as e:
            print(f"❌ {test_name}: Exception - {e}")
            functional_results.append(False)
    
    # Docker services check
    docker_result = check_docker_services()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    total_health = len(health_results)
    passed_health = sum(health_results)
    print(f"Health Checks: {passed_health}/{total_health} passed")
    
    total_functional = len(functional_results)
    passed_functional = sum(functional_results)
    print(f"Functional Tests: {passed_functional}/{total_functional} passed")
    
    if docker_result:
        print("Docker Services: ✅ All containers running")
    else:
        print("Docker Services: ⚠️ Some containers missing")
    
    overall_success = (
        passed_health == total_health and 
        passed_functional == total_functional and 
        docker_result
    )
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("System is operational and ready for use.")
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        print("Please check the output above for details.")
        
        if not docker_result:
            print("\n💡 To start services:")
            print("   cd MICROSERVICE_ENGINES/admin-devops")
            print("   ./start-devops.sh")
            print("   Select option 3: Start complete stack")
    
    print(f"\nCompleted at: {datetime.now()}")
    return overall_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1) 