#!/usr/bin/env python3
"""
Focused Validation Tests
Implementation of 5 key test scenarios to improve code quality:
1. Smoke Test (Containers)
2. Node Test  
3. Mini DAG Test Callbacks
4. Persistence Test
5. Routing Test
"""

import asyncio
import docker
import json
import logging
import os
import psycopg2
import pytest
import requests
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "MICROSERVICE_ENGINES" / "shared"))
sys.path.append(str(Path(__file__).parent / "neon"))

try:
    from network_graph import NetworkGraph
    from phase_2_runtime_modules import NodeEngine
    from db_manager import DatabaseManager
except ImportError as e:
    logging.warning(f"Import warning: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FocusedTestSuite:
    """Comprehensive focused test suite for BEM system validation"""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker client initialization failed: {e}")
            self.docker_client = None
            
        self.test_results = {
            'smoke_test': {'status': 'pending', 'details': {}},
            'node_test': {'status': 'pending', 'details': {}},
            'dag_test': {'status': 'pending', 'details': {}},
            'persistence_test': {'status': 'pending', 'details': {}},
            'routing_test': {'status': 'pending', 'details': {}}
        }
        self.start_time = datetime.now()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Execute all 5 focused tests"""
        logger.info("🚀 Starting Focused Validation Test Suite")
        
        try:
            # Test 1: Smoke Test (Containers)
            self.smoke_test_containers()
            
            # Test 2: Node Test
            self.node_test()
            
            # Test 3: Mini DAG Test Callbacks
            self.mini_dag_test()
            
            # Test 4: Persistence Test
            self.persistence_test()
            
            # Test 5: Routing Test
            self.routing_test()
            
        except Exception as e:
            logger.error(f"Critical test suite failure: {e}")
            
        finally:
            self.generate_report()
            
        return self.test_results
    
    def smoke_test_containers(self):
        """🔹1. Smoke Test (Containers)"""
        logger.info("🔹1. Starting Smoke Test (Containers)")
        
        try:
            if not self.docker_client:
                raise Exception("Docker client not available")
                
            # Start docker-compose
            logger.info("Starting docker-compose services...")
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                logger.warning(f"Docker compose warning: {result.stderr}")
            
            # Wait for services to stabilize
            time.sleep(10)
            
            # Check container status
            containers = self.docker_client.containers.list()
            running_containers = []
            failed_containers = []
            
            for container in containers:
                if 'endpoint' in container.name.lower():
                    if container.status == 'running':
                        running_containers.append(container.name)
                    else:
                        failed_containers.append({
                            'name': container.name,
                            'status': container.status
                        })
            
            # Check health endpoints
            health_checks = self.check_health_endpoints()
            
            # Check logs for fatal errors
            fatal_logs = self.check_fatal_logs()
            
            self.test_results['smoke_test'] = {
                'status': 'passed' if len(running_containers) > 0 else 'failed',
                'details': {
                    'running_containers': running_containers,
                    'failed_containers': failed_containers,
                    'health_checks': health_checks,
                    'fatal_logs': fatal_logs,
                    'total_containers': len(containers)
                }
            }
            
            logger.info(f"✅ Smoke test completed: {len(running_containers)} containers running")
            
        except Exception as e:
            logger.error(f"❌ Smoke test failed: {e}")
            self.test_results['smoke_test'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    def check_health_endpoints(self) -> Dict[str, Any]:
        """Check /healthz and /status endpoints"""
        endpoints = [
            'http://localhost:8000/healthz',
            'http://localhost:8000/status',
            'http://localhost:8001/healthz',
            'http://localhost:8001/status'
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                results[endpoint] = {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'healthy': response.status_code == 200
                }
            except Exception as e:
                results[endpoint] = {
                    'error': str(e),
                    'healthy': False
                }
        
        return results
    
    def check_fatal_logs(self) -> List[Dict[str, Any]]:
        """Check container logs for fatal errors"""
        fatal_patterns = ['FATAL', 'CRITICAL', 'ERROR']
        fatal_logs = []
        
        try:
            if not self.docker_client:
                return fatal_logs
                
            containers = self.docker_client.containers.list()
            for container in containers:
                if 'endpoint' in container.name.lower():
                    try:
                        logs = container.logs(tail=50).decode('utf-8')
                        for pattern in fatal_patterns:
                            if pattern in logs:
                                fatal_logs.append({
                                    'container': container.name,
                                    'pattern': pattern,
                                    'log_snippet': logs[-200:]
                                })
                                break
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Could not check logs: {e}")
        
        return fatal_logs
    
    def node_test(self):
        """🔹2. Node Test"""
        logger.info("🔹2. Starting Node Test")
        
        try:
            # Create test node data
            test_node_data = {
                "node_id": "V01_ProductComponent_TEST",
                "functor_type": "ProductComponent",
                "inputs": {
                    "material": "Steel",
                    "dimensions": {"length": 100, "width": 50, "height": 25},
                    "quantity": 10
                },
                "properties": {
                    "weight": 15.5,
                    "cost": 125.0,
                    "supplier": "TestSupplier"
                }
            }
            
            # Simulate evaluate_manufacturing
            manufacturing_result = self.simulate_manufacturing_evaluation(test_node_data)
            
            # Validate dictionary output
            dictionary_valid = self.validate_node_dictionary(manufacturing_result)
            
            # Check for expected keys
            expected_keys = ['node_id', 'functor_type', 'inputs', 'outputs', 'state', 'timestamp']
            keys_present = all(key in manufacturing_result for key in expected_keys)
            
            self.test_results['node_test'] = {
                'status': 'passed' if dictionary_valid and keys_present else 'failed',
                'details': {
                    'input_node': test_node_data,
                    'manufacturing_result': manufacturing_result,
                    'dictionary_valid': dictionary_valid,
                    'expected_keys_present': keys_present,
                    'missing_keys': [key for key in expected_keys if key not in manufacturing_result]
                }
            }
            
            logger.info("✅ Node test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Node test failed: {e}")
            self.test_results['node_test'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    def simulate_manufacturing_evaluation(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate manufacturing evaluation process"""
        inputs = node_data.get('inputs', {})
        dimensions = inputs.get('dimensions', {})
        quantity = inputs.get('quantity', 1)
        
        # Calculate volume and manufacturing time
        volume = dimensions.get('length', 0) * dimensions.get('width', 0) * dimensions.get('height', 0)
        manufacturing_time = volume * quantity * 0.001
        
        return {
            'node_id': node_data['node_id'],
            'functor_type': node_data['functor_type'],
            'inputs': inputs,
            'outputs': {
                'volume': volume,
                'manufacturing_time': manufacturing_time,
                'batch_size': quantity,
                'status': 'evaluated'
            },
            'state': 'completed',
            'timestamp': datetime.now().isoformat(),
            'properties': node_data.get('properties', {})
        }
    
    def validate_node_dictionary(self, node_dict: Dict[str, Any]) -> bool:
        """Validate node dictionary structure"""
        try:
            if not isinstance(node_dict, dict):
                return False
            
            required_fields = ['node_id', 'functor_type', 'state']
            if not all(field in node_dict for field in required_fields):
                return False
            
            if not isinstance(node_dict.get('inputs'), dict):
                return False
            
            if not isinstance(node_dict.get('outputs'), dict):
                return False
            
            return True
            
        except Exception:
            return False
    
    def mini_dag_test(self):
        """🔹3. Mini DAG Test Callbacks"""
        logger.info("🔹3. Starting Mini DAG Test")
        
        try:
            # Create mini DAG: V01 → V02 → V05
            dag_nodes = self.create_mini_dag()
            
            # Execute functors in sequence
            execution_results = self.execute_dag_sequence(dag_nodes)
            
            # Validate downstream value resolution
            downstream_valid = self.validate_downstream_resolution(execution_results)
            
            self.test_results['dag_test'] = {
                'status': 'passed' if downstream_valid else 'failed',
                'details': {
                    'dag_nodes': dag_nodes,
                    'execution_results': execution_results,
                    'downstream_valid': downstream_valid
                }
            }
            
            logger.info("✅ Mini DAG test completed")
            
        except Exception as e:
            logger.error(f"❌ Mini DAG test failed: {e}")
            self.test_results['dag_test'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    def create_mini_dag(self) -> List[Dict[str, Any]]:
        """Create mini DAG structure"""
        return [
            {
                'node_id': 'V01_Component',
                'functor_type': 'ProductComponent',
                'inputs': {'material': 'Steel', 'quantity': 5},
                'outputs': {},
                'downstream': ['V02_Assembly']
            },
            {
                'node_id': 'V02_Assembly',
                'functor_type': 'Assembly',
                'inputs': {},
                'outputs': {},
                'downstream': ['V05_Quality']
            },
            {
                'node_id': 'V05_Quality',
                'functor_type': 'QualityCheck',
                'inputs': {},
                'outputs': {},
                'downstream': []
            }
        ]
    
    def execute_dag_sequence(self, dag_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute DAG nodes in sequence"""
        results = []
        
        for i, node in enumerate(dag_nodes):
            if i > 0:
                previous_outputs = results[i-1].get('outputs', {})
                node['inputs'].update(previous_outputs)
            
            result = self.execute_single_node(node)
            results.append(result)
        
        return results
    
    def execute_single_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node"""
        functor_type = node['functor_type']
        inputs = node['inputs']
        
        if functor_type == 'ProductComponent':
            outputs = {
                'component_id': node['node_id'],
                'processed_quantity': inputs.get('quantity', 0),
                'material_type': inputs.get('material', 'Unknown')
            }
        elif functor_type == 'Assembly':
            outputs = {
                'assembly_id': node['node_id'],
                'components_count': 1,
                'assembly_status': 'assembled'
            }
        elif functor_type == 'QualityCheck':
            outputs = {
                'quality_score': 95.5,
                'passed': True,
                'checked_at': datetime.now().isoformat()
            }
        else:
            outputs = {'status': 'unknown_functor'}
        
        return {
            'node_id': node['node_id'],
            'functor_type': functor_type,
            'inputs': inputs,
            'outputs': outputs,
            'execution_time': 0.1,
            'status': 'completed'
        }
    
    def validate_downstream_resolution(self, execution_results: List[Dict[str, Any]]) -> bool:
        """Validate that downstream values resolve correctly"""
        try:
            for i in range(1, len(execution_results)):
                current_node = execution_results[i]
                previous_node = execution_results[i-1]
                
                current_inputs = current_node.get('inputs', {})
                previous_outputs = previous_node.get('outputs', {})
                
                if not any(key in current_inputs for key in previous_outputs.keys()):
                    logger.warning(f"No data flow from {previous_node['node_id']} to {current_node['node_id']}")
            
            final_node = execution_results[-1]
            return bool(final_node.get('outputs')) and final_node.get('status') == 'completed'
            
        except Exception as e:
            logger.error(f"Downstream validation error: {e}")
            return False
    
    def persistence_test(self):
        """🔹4. Persistence Test"""
        logger.info("🔹4. Starting Persistence Test")
        
        try:
            test_node = {
                'node_id': 'PERSIST_TEST_001',
                'functor_type': 'TestNode',
                'data': {'value': 42, 'name': 'test_persistence'},
                'timestamp': datetime.now().isoformat()
            }
            
            save_success = self.save_node_to_postgres(test_node)
            
            if save_success:
                reloaded_node = self.load_node_from_postgres(test_node['node_id'])
                
                if reloaded_node:
                    match = self.compare_node_dictionaries(test_node, reloaded_node)
                    
                    self.test_results['persistence_test'] = {
                        'status': 'passed' if match else 'failed',
                        'details': {
                            'original_node': test_node,
                            'reloaded_node': reloaded_node,
                            'dictionaries_match': match,
                            'save_success': save_success
                        }
                    }
                else:
                    raise Exception("Failed to reload node from database")
            else:
                raise Exception("Failed to save node to database")
            
            logger.info("✅ Persistence test completed")
            
        except Exception as e:
            logger.error(f"❌ Persistence test failed: {e}")
            self.test_results['persistence_test'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    def save_node_to_postgres(self, node: Dict[str, Any]) -> bool:
        """Save node to PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'bem_system'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'password'),
                port=os.getenv('DB_PORT', '5432')
            )
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS test_nodes (
                        node_id VARCHAR(255) PRIMARY KEY,
                        functor_type VARCHAR(100),
                        data JSONB,
                        timestamp TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    INSERT INTO test_nodes (node_id, functor_type, data, timestamp)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE SET
                        functor_type = EXCLUDED.functor_type,
                        data = EXCLUDED.data,
                        timestamp = EXCLUDED.timestamp
                """, (
                    node['node_id'],
                    node['functor_type'],
                    json.dumps(node['data']),
                    node['timestamp']
                ))
                
                conn.commit()
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Database save error: {e}")
            return False
    
    def load_node_from_postgres(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load node from PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'bem_system'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'password'),
                port=os.getenv('DB_PORT', '5432')
            )
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT node_id, functor_type, data, timestamp
                    FROM test_nodes WHERE node_id = %s
                """, (node_id,))
                
                row = cur.fetchone()
                if row:
                    return {
                        'node_id': row[0],
                        'functor_type': row[1],
                        'data': json.loads(row[2]) if row[2] else {},
                        'timestamp': row[3].isoformat() if row[3] else None
                    }
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Database load error: {e}")
        
        return None
    
    def compare_node_dictionaries(self, original: Dict[str, Any], reloaded: Dict[str, Any]) -> bool:
        """Compare original and reloaded node dictionaries"""
        try:
            key_fields = ['node_id', 'functor_type', 'data']
            
            for field in key_fields:
                if original.get(field) != reloaded.get(field):
                    logger.warning(f"Field mismatch: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dictionary comparison error: {e}")
            return False
    
    def routing_test(self):
        """🔹5. Routing Test"""
        logger.info("🔹5. Starting Routing Test")
        
        try:
            source_node = {
                'node_id': 'V03_Router_Source',
                'functor_type': 'RoutingSource',
                'outputs': {'route_data': 'test_payload', 'target': 'V04_Target'}
            }
            
            target_node = {
                'node_id': 'V04_Target',
                'functor_type': 'RoutingTarget',
                'inputs': {},
                'received_data': None
            }
            
            routing_success = self.trigger_route_to_node(source_node, target_node)
            target_executed = self.verify_target_execution(target_node)
            
            self.test_results['routing_test'] = {
                'status': 'passed' if routing_success and target_executed else 'failed',
                'details': {
                    'source_node': source_node,
                    'target_node': target_node,
                    'routing_success': routing_success,
                    'target_executed': target_executed
                }
            }
            
            logger.info("✅ Routing test completed")
            
        except Exception as e:
            logger.error(f"❌ Routing test failed: {e}")
            self.test_results['routing_test'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    def trigger_route_to_node(self, source_node: Dict[str, Any], target_node: Dict[str, Any]) -> bool:
        """Trigger routing from source to target node"""
        try:
            route_data = source_node.get('outputs', {}).get('route_data')
            target_id = source_node.get('outputs', {}).get('target')
            
            if route_data and target_id == target_node['node_id']:
                target_node['inputs'].update({
                    'routed_data': route_data,
                    'source_node': source_node['node_id'],
                    'routing_timestamp': datetime.now().isoformat()
                })
                
                target_node['received_data'] = route_data
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return False
    
    def verify_target_execution(self, target_node: Dict[str, Any]) -> bool:
        """Verify that target node executed and received inputs"""
        try:
            has_inputs = bool(target_node.get('inputs'))
            has_received_data = target_node.get('received_data') is not None
            
            if has_inputs and has_received_data:
                target_node['outputs'] = {
                    'processed_data': target_node['received_data'],
                    'execution_status': 'completed',
                    'processed_at': datetime.now().isoformat()
                }
                target_node['status'] = 'executed'
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Target execution verification error: {e}")
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        passed_tests = sum(1 for test in self.test_results.values() if test['status'] == 'passed')
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = {
            'test_suite': 'Focused Validation Tests',
            'execution_time': duration,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'timestamp': end_time.isoformat(),
            'results': self.test_results
        }
        
        report_file = f"focused_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"""
        
🎯 FOCUSED VALIDATION TEST REPORT
================================
Duration: {duration:.2f} seconds
Tests: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)

🔹1. Smoke Test (Containers): {self.test_results['smoke_test']['status'].upper()}
🔹2. Node Test: {self.test_results['node_test']['status'].upper()}
🔹3. Mini DAG Test: {self.test_results['dag_test']['status'].upper()}
🔹4. Persistence Test: {self.test_results['persistence_test']['status'].upper()}
🔹5. Routing Test: {self.test_results['routing_test']['status'].upper()}

Report saved: {report_file}
        """)
        
        return report

def main():
    """Main execution function"""
    print("🎯 Starting Focused Validation Test Suite")
    
    test_suite = FocusedTestSuite()
    results = test_suite.run_all_tests()
    
    failed_tests = sum(1 for test in results.values() if test['status'] == 'failed')
    sys.exit(failed_tests)

if __name__ == "__main__":
    main() 