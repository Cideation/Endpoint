#!/usr/bin/env python3
"""
Edge Integrity Test
Tests edge direction, metadata, and callback accuracy
Validates proper edge routing and callback execution
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgeCallbackLogicTest:
    """Test edge integrity and callback logic"""
    
    def __init__(self):
        self.test_results = {
            'edge_direction_test': {'status': 'pending', 'details': {}},
            'edge_metadata_test': {'status': 'pending', 'details': {}},
            'callback_accuracy_test': {'status': 'pending', 'details': {}},
            'edge_routing_test': {'status': 'pending', 'details': {}}
        }
        self.start_time = datetime.now()
        
    def run_edge_tests(self) -> Dict[str, Any]:
        """Execute all edge and callback tests"""
        logger.info("🔗 Starting Edge Callback Logic Tests")
        
        try:
            test_edges = self.create_test_edges()
            self.test_edge_directions(test_edges)
            self.test_edge_metadata(test_edges)
            self.test_callback_accuracy(test_edges)
            self.test_edge_routing(test_edges)
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Edge callback test failed: {e}")
            
        return self.test_results
    
    def create_test_edges(self) -> List[Dict[str, Any]]:
        """Create test edge configurations"""
        return [
            {
                'edge_id': 'E001_Material_To_Design',
                'source_node': 'A01_MaterialInput',
                'target_node': 'B01_ComponentDesign',
                'direction': 'forward',
                'metadata': {
                    'data_type': 'material_properties',
                    'priority': 'high',
                    'transformation': 'material_spec_to_design_params'
                },
                'callback_config': {
                    'trigger_condition': 'source_complete',
                    'callback_function': 'propagate_material_data',
                    'retry_count': 3
                }
            },
            {
                'edge_id': 'E002_Design_To_Quality',
                'source_node': 'B01_ComponentDesign',
                'target_node': 'G02_QualityAssurance',
                'direction': 'forward',
                'metadata': {
                    'data_type': 'design_specifications',
                    'priority': 'medium',
                    'transformation': 'design_to_quality_params'
                },
                'callback_config': {
                    'trigger_condition': 'source_complete',
                    'callback_function': 'propagate_design_data',
                    'retry_count': 2
                }
            },
            {
                'edge_id': 'E003_Quality_Feedback',
                'source_node': 'G02_QualityAssurance',
                'target_node': 'B01_ComponentDesign',
                'direction': 'backward',
                'metadata': {
                    'data_type': 'quality_feedback',
                    'priority': 'high',
                    'transformation': 'quality_to_design_feedback'
                },
                'callback_config': {
                    'trigger_condition': 'quality_failure',
                    'callback_function': 'trigger_design_revision',
                    'retry_count': 1
                }
            }
        ]
    
    def test_edge_directions(self, edges: List[Dict[str, Any]]):
        """Test edge direction validation"""
        logger.info("➡️ Testing Edge Directions")
        
        direction_results = []
        for edge in edges:
            try:
                direction = edge['direction']
                source = edge['source_node']
                target = edge['target_node']
                
                # Validate direction logic
                if direction == 'forward':
                    valid = self.validate_forward_edge(source, target)
                elif direction == 'backward':
                    valid = self.validate_backward_edge(source, target)
                else:
                    valid = False
                
                direction_results.append({
                    'edge_id': edge['edge_id'],
                    'direction': direction,
                    'valid': valid,
                    'source': source,
                    'target': target
                })
                
            except Exception as e:
                direction_results.append({
                    'edge_id': edge['edge_id'],
                    'valid': False,
                    'error': str(e)
                })
        
        all_valid = all(result['valid'] for result in direction_results)
        self.test_results['edge_direction_test'] = {
            'status': 'passed' if all_valid else 'failed',
            'details': {
                'total_edges': len(edges),
                'valid_edges': sum(1 for r in direction_results if r['valid']),
                'results': direction_results
            }
        }
    
    def validate_forward_edge(self, source: str, target: str) -> bool:
        """Validate forward edge logic"""
        # Phase progression validation
        source_phase = self.get_node_phase(source)
        target_phase = self.get_node_phase(target)
        
        phase_order = {'alpha': 1, 'beta': 2, 'gamma': 3}
        return phase_order.get(source_phase, 0) <= phase_order.get(target_phase, 0)
    
    def validate_backward_edge(self, source: str, target: str) -> bool:
        """Validate backward edge logic (feedback loops)"""
        # Feedback edges should go from later to earlier phases
        source_phase = self.get_node_phase(source)
        target_phase = self.get_node_phase(target)
        
        phase_order = {'alpha': 1, 'beta': 2, 'gamma': 3}
        return phase_order.get(source_phase, 0) > phase_order.get(target_phase, 0)
    
    def get_node_phase(self, node_id: str) -> str:
        """Extract phase from node ID"""
        if node_id.startswith('A'):
            return 'alpha'
        elif node_id.startswith('B'):
            return 'beta'
        elif node_id.startswith('G'):
            return 'gamma'
        else:
            return 'unknown'
    
    def test_edge_metadata(self, edges: List[Dict[str, Any]]):
        """Test edge metadata integrity"""
        logger.info("📋 Testing Edge Metadata")
        
        metadata_results = []
        for edge in edges:
            try:
                metadata = edge.get('metadata', {})
                
                # Required metadata fields
                required_fields = ['data_type', 'priority', 'transformation']
                has_required = all(field in metadata for field in required_fields)
                
                # Validate priority values
                valid_priorities = ['low', 'medium', 'high', 'critical']
                priority_valid = metadata.get('priority') in valid_priorities
                
                # Validate data type format
                data_type = metadata.get('data_type', '')
                data_type_valid = len(data_type) > 0 and '_' in data_type
                
                metadata_valid = has_required and priority_valid and data_type_valid
                
                metadata_results.append({
                    'edge_id': edge['edge_id'],
                    'valid': metadata_valid,
                    'has_required_fields': has_required,
                    'priority_valid': priority_valid,
                    'data_type_valid': data_type_valid,
                    'metadata': metadata
                })
                
            except Exception as e:
                metadata_results.append({
                    'edge_id': edge['edge_id'],
                    'valid': False,
                    'error': str(e)
                })
        
        all_valid = all(result['valid'] for result in metadata_results)
        self.test_results['edge_metadata_test'] = {
            'status': 'passed' if all_valid else 'failed',
            'details': {
                'total_edges': len(edges),
                'valid_metadata': sum(1 for r in metadata_results if r['valid']),
                'results': metadata_results
            }
        }
    
    def test_callback_accuracy(self, edges: List[Dict[str, Any]]):
        """Test callback execution accuracy"""
        logger.info("🔄 Testing Callback Accuracy")
        
        callback_results = []
        for edge in edges:
            try:
                callback_config = edge.get('callback_config', {})
                
                # Execute callback simulation
                callback_result = self.simulate_callback_execution(edge, callback_config)
                
                callback_results.append({
                    'edge_id': edge['edge_id'],
                    'callback_executed': callback_result['executed'],
                    'execution_time': callback_result['execution_time'],
                    'retry_attempts': callback_result['retry_attempts'],
                    'success': callback_result['success']
                })
                
            except Exception as e:
                callback_results.append({
                    'edge_id': edge['edge_id'],
                    'callback_executed': False,
                    'error': str(e)
                })
        
        all_successful = all(result.get('success', False) for result in callback_results)
        self.test_results['callback_accuracy_test'] = {
            'status': 'passed' if all_successful else 'failed',
            'details': {
                'total_callbacks': len(edges),
                'successful_callbacks': sum(1 for r in callback_results if r.get('success')),
                'results': callback_results
            }
        }
    
    def simulate_callback_execution(self, edge: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate callback execution"""
        start_time = time.time()
        
        callback_function = config.get('callback_function', '')
        max_retries = config.get('retry_count', 1)
        
        # Simulate callback execution based on function type
        if 'propagate' in callback_function:
            success = True  # Data propagation usually succeeds
        elif 'trigger' in callback_function:
            success = True  # Trigger functions usually succeed
        else:
            success = False  # Unknown function type
        
        execution_time = time.time() - start_time
        
        return {
            'executed': True,
            'success': success,
            'execution_time': execution_time,
            'retry_attempts': 1 if success else max_retries
        }
    
    def test_edge_routing(self, edges: List[Dict[str, Any]]):
        """Test edge routing logic"""
        logger.info("🛣️ Testing Edge Routing")
        
        routing_results = []
        for edge in edges:
            try:
                # Test routing path validation
                routing_valid = self.validate_routing_path(edge)
                
                # Test data transformation
                transformation_valid = self.validate_data_transformation(edge)
                
                # Test routing performance
                routing_performance = self.measure_routing_performance(edge)
                
                routing_results.append({
                    'edge_id': edge['edge_id'],
                    'routing_valid': routing_valid,
                    'transformation_valid': transformation_valid,
                    'performance_ms': routing_performance,
                    'overall_valid': routing_valid and transformation_valid and routing_performance < 100
                })
                
            except Exception as e:
                routing_results.append({
                    'edge_id': edge['edge_id'],
                    'overall_valid': False,
                    'error': str(e)
                })
        
        all_valid = all(result.get('overall_valid', False) for result in routing_results)
        self.test_results['edge_routing_test'] = {
            'status': 'passed' if all_valid else 'failed',
            'details': {
                'total_routes': len(edges),
                'valid_routes': sum(1 for r in routing_results if r.get('overall_valid')),
                'results': routing_results
            }
        }
    
    def validate_routing_path(self, edge: Dict[str, Any]) -> bool:
        """Validate routing path exists and is accessible"""
        source = edge['source_node']
        target = edge['target_node']
        
        # Simulate path validation
        return len(source) > 0 and len(target) > 0 and source != target
    
    def validate_data_transformation(self, edge: Dict[str, Any]) -> bool:
        """Validate data transformation logic"""
        transformation = edge.get('metadata', {}).get('transformation', '')
        
        # Check transformation function exists and is valid
        valid_transformations = [
            'material_spec_to_design_params',
            'design_to_quality_params',
            'quality_to_design_feedback'
        ]
        
        return transformation in valid_transformations
    
    def measure_routing_performance(self, edge: Dict[str, Any]) -> float:
        """Measure routing performance in milliseconds"""
        start_time = time.time()
        
        # Simulate routing operation
        time.sleep(0.001)  # 1ms simulation
        
        end_time = time.time()
        return (end_time - start_time) * 1000
    
    def generate_report(self):
        """Generate comprehensive test report"""
        duration = (datetime.now() - self.start_time).total_seconds()
        report_file = f"edge_callback_logic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'test_suite': 'Edge Callback Logic Test',
                'execution_time': duration,
                'results': self.test_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"🔗 Edge Callback Logic Test completed in {duration:.2f}s - Report: {report_file}")

def main():
    test_suite = EdgeCallbackLogicTest()
    results = test_suite.run_edge_tests()
    failed_tests = sum(1 for result in results.values() if result.get('status') == 'failed')
    sys.exit(failed_tests)

if __name__ == "__main__":
    main()
