#!/usr/bin/env python3
"""
Phase 2 Container Orchestration Test
Tests real container communication and orchestration
"""

import asyncio
import json
import sys
import time
import requests
from datetime import datetime

# Add neon to path
sys.path.append('.')

def test_container_services():
    """Test individual container services"""
    print("🐳 Testing Individual Container Services")
    
    # Container endpoints
    containers = {
        "ne-dag-alpha": "http://localhost:5000",
        "ne-functor-types": "http://localhost:5001", 
        "ne-callback-engine": "http://localhost:5002",
        "sfde": "http://localhost:5003",
        "ne-graph-runtime-engine": "http://localhost:5004",
        "ne-optimization-engine": "http://localhost:5005"
    }
    
    # Test data
    test_data = {
        "data": {
            "components": [
                {"id": "comp_001", "type": "beam", "primary_functor": "structural", "status": "processed"},
                {"id": "comp_002", "type": "column", "primary_functor": "structural", "status": "completed"},
                {"id": "comp_003", "type": "slab", "primary_functor": "cost", "status": "processed"}
            ],
            "node_sequence": ["comp_001", "comp_002", "comp_003"],
            "phase": "phase_2",
            "affinity_types": ["structural", "cost", "energy"]
        },
        "timestamp": datetime.now().isoformat()
    }
    
    results = {}
    
    for service_name, base_url in containers.items():
        try:
            # Test health endpoint
            print(f"  Testing {service_name} health...")
            health_response = requests.get(f"{base_url}/health", timeout=5)
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"    ✅ {service_name}: {health_data.get('status', 'unknown')}")
                
                # Test process endpoint
                print(f"  Testing {service_name} processing...")
                process_response = requests.post(
                    f"{base_url}/process", 
                    json=test_data, 
                    timeout=10
                )
                
                if process_response.status_code == 200:
                    process_data = process_response.json()
                    print(f"    ✅ {service_name} processing: {process_data.get('status', 'unknown')}")
                    results[service_name] = {
                        "health": health_data,
                        "process": process_data,
                        "status": "success"
                    }
                else:
                    print(f"    ❌ {service_name} processing failed: {process_response.status_code}")
                    results[service_name] = {
                        "status": "process_failed",
                        "error": f"HTTP {process_response.status_code}"
                    }
            else:
                print(f"    ❌ {service_name} health failed: {health_response.status_code}")
                results[service_name] = {
                    "status": "health_failed",
                    "error": f"HTTP {health_response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            print(f"    ❌ {service_name} connection failed: {str(e)}")
            results[service_name] = {
                "status": "connection_failed",
                "error": str(e)
            }
    
    return results

async def test_orchestrator():
    """Test the orchestrator with real container communication"""
    print("\n🎯 Testing Orchestrator with Real Containers")
    
    try:
        from neon.orchestrator import Orchestrator
        from neon.schemas import ExecutionPhase
        
        # Create orchestrator
        orchestrator = Orchestrator()
        
        # Test components
        test_components = [
            {
                "id": "test_beam_001",
                "type": "structural_beam",
                "primary_functor": "structural",
                "material": "steel",
                "length": 10.0,
                "status": "processed"
            },
            {
                "id": "test_column_002", 
                "type": "structural_column",
                "primary_functor": "structural",
                "material": "concrete",
                "height": 3.5,
                "status": "completed"
            },
            {
                "id": "test_slab_003",
                "type": "floor_slab", 
                "primary_functor": "cost",
                "material": "concrete",
                "area": 100.0,
                "status": "processed"
            }
        ]
        
        print(f"  Executing pipeline with {len(test_components)} components...")
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(
            components=test_components,
            phase=ExecutionPhase.CROSS_PHASE
        )
        
        print(f"  ✅ Pipeline completed: {results.get('status', 'unknown')}")
        print(f"  📊 Containers executed: {results.get('containers_executed', 0)}")
        print(f"  ⏱️  Execution time: {results.get('execution_time_seconds', 0):.2f}s")
        
        # Check container health
        health_results = await orchestrator.get_container_health()
        print(f"  🏥 Container health status:")
        for container, status in health_results.get('container_status', {}).items():
            print(f"    {container}: {status}")
        
        return {
            "status": "success",
            "pipeline_results": results,
            "health_results": health_results
        }
        
    except Exception as e:
        print(f"  ❌ Orchestrator test failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

def test_container_client():
    """Test the ContainerClient directly"""
    print("\n📡 Testing ContainerClient Direct Communication")
    
    try:
        from neon.container_client import ContainerClient, ContainerType
        
        # Create client
        client = ContainerClient()
        
        # Test data
        test_data = {
            "components": [
                {"id": "client_test_001", "type": "beam", "primary_functor": "structural"}
            ],
            "phase": "phase_2"
        }
        
        # Test each container type
        container_types = [
            ContainerType.DAG_ALPHA,
            ContainerType.FUNCTOR_TYPES,
            ContainerType.CALLBACK_ENGINE,
            ContainerType.SFDE_ENGINE,
            ContainerType.GRAPH_RUNTIME
        ]
        
        results = {}
        
        for container_type in container_types:
            try:
                print(f"  Testing {container_type.value}...")
                
                response = client.call_container(container_type, test_data)
                
                if response.get('status') == 'success':
                    print(f"    ✅ {container_type.value}: Success")
                    results[container_type.value] = {"status": "success", "response": response}
                else:
                    print(f"    ⚠️  {container_type.value}: {response.get('status', 'unknown')}")
                    results[container_type.value] = {"status": "warning", "response": response}
                    
            except Exception as e:
                print(f"    ❌ {container_type.value}: {str(e)}")
                results[container_type.value] = {"status": "error", "error": str(e)}
        
        return results
        
    except Exception as e:
        print(f"  ❌ ContainerClient test failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

def test_json_transformer():
    """Test the JSON transformer for different container formats"""
    print("\n🔄 Testing JSON Transformer")
    
    try:
        from neon.json_transformer import JSONTransformer
        
        transformer = JSONTransformer()
        
        # Test components
        test_components = [
            {"id": "transform_001", "type": "beam", "primary_functor": "structural"},
            {"id": "transform_002", "type": "column", "primary_functor": "cost"}
        ]
        
        # Test all transformer methods
        transformations = {
            "dag_alpha": transformer.transform_for_dag_alpha(test_components),
            "functor_types": transformer.transform_for_functor_types(test_components),
            "callback_engine": transformer.transform_for_callback_engine(test_components, phase="phase_2"),
            "sfde_engine": transformer.transform_for_sfde_engine(test_components, ["structural", "cost"]),
            "graph_runtime": transformer.transform_for_graph_runtime(test_components),
            "optimization": transformer.transform_for_optimization_engine(test_components)
        }
        
        for transform_name, result in transformations.items():
            if result:
                print(f"  ✅ {transform_name}: {len(result)} items transformed")
            else:
                print(f"  ❌ {transform_name}: Failed to transform")
        
        return {
            "status": "success",
            "transformations": transformations
        }
        
    except Exception as e:
        print(f"  ❌ JSON Transformer test failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

def test_execution_history():
    """Test execution history tracking"""
    print("\n📚 Testing Execution History")
    
    try:
        from neon.orchestrator import Orchestrator
        
        orchestrator = Orchestrator()
        
        # Check initial history
        history = orchestrator.get_execution_history()
        initial_count = len(history)
        
        print(f"  Initial execution history: {initial_count} entries")
        
        if history:
            latest = history[-1]
            print(f"  Latest execution:")
            print(f"    Pipeline ID: {latest.get('pipeline_id', 'unknown')}")
            print(f"    Status: {latest.get('status', 'unknown')}")
            print(f"    Containers: {latest.get('containers_executed', 0)}")
        
        return {
            "status": "success",
            "history_count": initial_count,
            "latest_execution": history[-1] if history else None
        }
        
    except Exception as e:
        print(f"  ❌ Execution history test failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

async def main():
    """Run all container orchestration tests"""
    print("🚀 PHASE 2: REAL CONTAINER ORCHESTRATION TESTS")
    print("=" * 60)
    
    # Test results
    test_results = {}
    
    # 1. Test individual container services
    test_results["container_services"] = test_container_services()
    
    # 2. Test orchestrator
    test_results["orchestrator"] = await test_orchestrator()
    
    # 3. Test container client
    test_results["container_client"] = test_container_client()
    
    # 4. Test JSON transformer
    test_results["json_transformer"] = test_json_transformer()
    
    # 5. Test execution history
    test_results["execution_history"] = test_execution_history()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        if isinstance(result, dict) and result.get('status') == 'success':
            print(f"  ✅ {test_name}: PASSED")
            successful_tests += 1
        elif isinstance(result, dict):
            # Check individual container results
            if test_name == "container_services":
                container_successes = sum(1 for r in result.values() if r.get('status') == 'success')
                container_total = len(result)
                print(f"  🔄 {test_name}: {container_successes}/{container_total} containers responding")
                if container_successes > 0:
                    successful_tests += 1
            else:
                print(f"  ❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            print(f"  ⚠️  {test_name}: UNKNOWN RESULT")
    
    print(f"\n🎯 OVERALL RESULT: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Real container orchestration is working!")
    elif successful_tests > 0:
        print("⚠️  PARTIAL SUCCESS - Some containers are responding")
        print("   💡 TIP: Start missing containers with: docker-compose up -d")
    else:
        print("❌ ALL TESTS FAILED - Check container availability")
        print("   💡 TIP: Start containers with: cd MICROSERVICE_ENGINES && docker-compose up -d")
    
    # Save results to file
    with open('container_orchestration_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: container_orchestration_test_results.json")
    
    return test_results

if __name__ == "__main__":
    asyncio.run(main()) 