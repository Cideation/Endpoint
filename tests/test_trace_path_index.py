#!/usr/bin/env python3
"""
Output Tracing Test
Tests final output trace to source inputs (component ID path)
Validates end-to-end traceability and data lineage
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TracePathIndexTest:
    """Test output tracing and path indexing"""
    
    def __init__(self):
        self.test_results = {
            'path_tracing': {'status': 'pending', 'details': {}},
            'component_lineage': {'status': 'pending', 'details': {}},
            'data_provenance': {'status': 'pending', 'details': {}},
            'trace_index_integrity': {'status': 'pending', 'details': {}}
        }
        self.start_time = datetime.now()
        self.trace_index = {}
        
    def run_trace_tests(self) -> Dict[str, Any]:
        """Execute all tracing tests"""
        logger.info("🔍 Starting Output Tracing Tests")
        
        try:
            # Create test execution graph
            execution_graph = self.create_execution_graph()
            
            # Build trace index
            self.build_trace_index(execution_graph)
            
            # Test path tracing
            self.test_path_tracing(execution_graph)
            
            # Test component lineage
            self.test_component_lineage(execution_graph)
            
            # Test data provenance
            self.test_data_provenance(execution_graph)
            
            # Test trace index integrity
            self.test_trace_index_integrity()
            
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Trace test failed: {e}")
            
        return self.test_results
    
    def create_execution_graph(self) -> Dict[str, Any]:
        """Create test execution graph with complete data flow"""
        return {
            'nodes': [
                {
                    'node_id': 'A01_RawMaterial',
                    'phase': 'alpha',
                    'inputs': {
                        'material_spec': 'STEEL_A36_001',
                        'supplier_id': 'SUP_001',
                        'batch_number': 'BATCH_20240101_001'
                    },
                    'outputs': {
                        'material_id': 'MAT_STEEL_A36_001',
                        'material_properties': {
                            'yield_strength': 36000,
                            'tensile_strength': 58000,
                            'density': 7850
                        },
                        'quality_cert': 'QC_20240101_001'
                    },
                    'execution_order': 1,
                    'timestamp': '2024-01-01T10:00:00Z'
                },
                {
                    'node_id': 'A02_DesignRequirements',
                    'phase': 'alpha',
                    'inputs': {
                        'project_id': 'PROJ_BEM_001',
                        'load_requirements': '50kN',
                        'safety_factor': 2.5
                    },
                    'outputs': {
                        'design_spec_id': 'DESIGN_SPEC_001',
                        'structural_requirements': {
                            'max_load': 125000,
                            'deflection_limit': 5,
                            'safety_margin': 2.5
                        }
                    },
                    'execution_order': 2,
                    'timestamp': '2024-01-01T10:05:00Z'
                },
                {
                    'node_id': 'B01_ComponentDesign',
                    'phase': 'beta',
                    'inputs': {
                        'material_id': 'MAT_STEEL_A36_001',
                        'design_spec_id': 'DESIGN_SPEC_001',
                        'material_properties': {
                            'yield_strength': 36000,
                            'tensile_strength': 58000
                        }
                    },
                    'outputs': {
                        'component_id': 'COMP_BEAM_001',
                        'component_design': {
                            'length': 6000,
                            'width': 300,
                            'height': 400,
                            'weight': 567.0
                        },
                        'cad_file_id': 'CAD_BEAM_001'
                    },
                    'execution_order': 3,
                    'timestamp': '2024-01-01T10:30:00Z'
                },
                {
                    'node_id': 'B02_ManufacturingPlan',
                    'phase': 'beta',
                    'inputs': {
                        'component_id': 'COMP_BEAM_001',
                        'component_design': {
                            'length': 6000,
                            'width': 300,
                            'height': 400
                        }
                    },
                    'outputs': {
                        'manufacturing_plan_id': 'MFG_PLAN_001',
                        'process_sequence': ['cutting', 'welding', 'finishing'],
                        'estimated_time': 480,
                        'resource_requirements': {
                            'workers': 3,
                            'machines': ['CNC_001', 'WELD_002']
                        }
                    },
                    'execution_order': 4,
                    'timestamp': '2024-01-01T11:00:00Z'
                },
                {
                    'node_id': 'G01_QualityControl',
                    'phase': 'gamma',
                    'inputs': {
                        'component_id': 'COMP_BEAM_001',
                        'manufacturing_plan_id': 'MFG_PLAN_001',
                        'quality_cert': 'QC_20240101_001'
                    },
                    'outputs': {
                        'quality_report_id': 'QR_BEAM_001',
                        'inspection_results': {
                            'dimensional_accuracy': 99.2,
                            'surface_quality': 98.5,
                            'material_integrity': 99.8
                        },
                        'certification_status': 'CERTIFIED'
                    },
                    'execution_order': 5,
                    'timestamp': '2024-01-01T14:00:00Z'
                },
                {
                    'node_id': 'G02_FinalAssembly',
                    'phase': 'gamma',
                    'inputs': {
                        'component_id': 'COMP_BEAM_001',
                        'quality_report_id': 'QR_BEAM_001',
                        'certification_status': 'CERTIFIED'
                    },
                    'outputs': {
                        'final_product_id': 'PROD_STRUCTURE_001',
                        'assembly_record': {
                            'components': ['COMP_BEAM_001'],
                            'assembly_date': '2024-01-01',
                            'quality_score': 99.0
                        },
                        'delivery_package': {
                            'documentation': ['QR_BEAM_001', 'CAD_BEAM_001'],
                            'certifications': ['QC_20240101_001']
                        }
                    },
                    'execution_order': 6,
                    'timestamp': '2024-01-01T16:00:00Z'
                }
            ],
            'edges': [
                {'source': 'A01_RawMaterial', 'target': 'B01_ComponentDesign', 'data_flow': ['material_id', 'material_properties']},
                {'source': 'A02_DesignRequirements', 'target': 'B01_ComponentDesign', 'data_flow': ['design_spec_id']},
                {'source': 'B01_ComponentDesign', 'target': 'B02_ManufacturingPlan', 'data_flow': ['component_id', 'component_design']},
                {'source': 'B01_ComponentDesign', 'target': 'G01_QualityControl', 'data_flow': ['component_id']},
                {'source': 'B02_ManufacturingPlan', 'target': 'G01_QualityControl', 'data_flow': ['manufacturing_plan_id']},
                {'source': 'A01_RawMaterial', 'target': 'G01_QualityControl', 'data_flow': ['quality_cert']},
                {'source': 'B01_ComponentDesign', 'target': 'G02_FinalAssembly', 'data_flow': ['component_id']},
                {'source': 'G01_QualityControl', 'target': 'G02_FinalAssembly', 'data_flow': ['quality_report_id', 'certification_status']}
            ]
        }
    
    def build_trace_index(self, execution_graph: Dict[str, Any]):
        """Build comprehensive trace index"""
        logger.info("📊 Building Trace Index")
        
        nodes = execution_graph['nodes']
        edges = execution_graph['edges']
        
        # Initialize trace index
        self.trace_index = {
            'node_lineage': {},
            'data_lineage': {},
            'component_paths': {},
            'reverse_lookup': {}
        }
        
        # Build node lineage
        for node in nodes:
            node_id = node['node_id']
            self.trace_index['node_lineage'][node_id] = {
                'phase': node['phase'],
                'execution_order': node['execution_order'],
                'timestamp': node['timestamp'],
                'upstream_nodes': [],
                'downstream_nodes': []
            }
        
        # Build edge relationships
        for edge in edges:
            source = edge['source']
            target = edge['target']
            
            if source in self.trace_index['node_lineage']:
                self.trace_index['node_lineage'][source]['downstream_nodes'].append(target)
            
            if target in self.trace_index['node_lineage']:
                self.trace_index['node_lineage'][target]['upstream_nodes'].append(source)
        
        # Build data lineage
        for node in nodes:
            node_id = node['node_id']
            
            # Track input data sources
            for input_key, input_value in node['inputs'].items():
                if isinstance(input_value, str) and ('_' in input_value or input_value.startswith(('MAT_', 'COMP_', 'QC_'))):
                    if input_value not in self.trace_index['data_lineage']:
                        self.trace_index['data_lineage'][input_value] = {
                            'created_by': None,
                            'used_by': [],
                            'data_type': input_key
                        }
                    self.trace_index['data_lineage'][input_value]['used_by'].append(node_id)
            
            # Track output data creation
            for output_key, output_value in node['outputs'].items():
                if isinstance(output_value, str) and ('_' in output_value or output_value.startswith(('MAT_', 'COMP_', 'QC_'))):
                    if output_value not in self.trace_index['data_lineage']:
                        self.trace_index['data_lineage'][output_value] = {
                            'created_by': node_id,
                            'used_by': [],
                            'data_type': output_key
                        }
                    else:
                        self.trace_index['data_lineage'][output_value]['created_by'] = node_id
        
        # Build component paths
        self.build_component_paths(nodes)
    
    def build_component_paths(self, nodes: List[Dict[str, Any]]):
        """Build component traceability paths"""
        
        for node in nodes:
            # Look for component IDs in outputs
            for output_key, output_value in node['outputs'].items():
                if isinstance(output_value, str) and output_value.startswith('COMP_'):
                    component_id = output_value
                    
                    # Trace component path
                    component_path = self.trace_component_path(component_id, nodes)
                    
                    self.trace_index['component_paths'][component_id] = {
                        'creation_node': node['node_id'],
                        'full_path': component_path,
                        'material_sources': self.extract_material_sources(component_path, nodes),
                        'design_sources': self.extract_design_sources(component_path, nodes)
                    }
    
    def trace_component_path(self, component_id: str, nodes: List[Dict[str, Any]]) -> List[str]:
        """Trace complete path for a component"""
        path = []
        
        # Find creation node
        creation_node = None
        for node in nodes:
            if any(output_value == component_id for output_value in node['outputs'].values() 
                  if isinstance(output_value, str)):
                creation_node = node
                break
        
        if creation_node:
            # Trace backwards through dependencies
            path = self.trace_backwards(creation_node['node_id'], nodes)
            
            # Trace forwards through usage
            forward_path = self.trace_forwards(creation_node['node_id'], nodes)
            
            # Combine paths
            path.extend(forward_path[1:])  # Exclude duplicate node
        
        return path
    
    def trace_backwards(self, node_id: str, nodes: List[Dict[str, Any]]) -> List[str]:
        """Trace backwards through node dependencies"""
        visited = set()
        path = []
        
        def dfs_backwards(current_node_id):
            if current_node_id in visited:
                return
            
            visited.add(current_node_id)
            
            # Find upstream dependencies
            upstream_nodes = self.trace_index['node_lineage'].get(current_node_id, {}).get('upstream_nodes', [])
            
            for upstream in upstream_nodes:
                dfs_backwards(upstream)
            
            path.append(current_node_id)
        
        dfs_backwards(node_id)
        return path
    
    def trace_forwards(self, node_id: str, nodes: List[Dict[str, Any]]) -> List[str]:
        """Trace forwards through node usage"""
        visited = set()
        path = []
        
        def dfs_forwards(current_node_id):
            if current_node_id in visited:
                return
            
            visited.add(current_node_id)
            path.append(current_node_id)
            
            # Find downstream dependencies
            downstream_nodes = self.trace_index['node_lineage'].get(current_node_id, {}).get('downstream_nodes', [])
            
            for downstream in downstream_nodes:
                dfs_forwards(downstream)
        
        dfs_forwards(node_id)
        return path
    
    def extract_material_sources(self, path: List[str], nodes: List[Dict[str, Any]]) -> List[str]:
        """Extract material sources from component path"""
        material_sources = []
        
        for node_id in path:
            node = next((n for n in nodes if n['node_id'] == node_id), None)
            if node:
                for output_key, output_value in node['outputs'].items():
                    if isinstance(output_value, str) and output_value.startswith('MAT_'):
                        material_sources.append(output_value)
        
        return material_sources
    
    def extract_design_sources(self, path: List[str], nodes: List[Dict[str, Any]]) -> List[str]:
        """Extract design sources from component path"""
        design_sources = []
        
        for node_id in path:
            node = next((n for n in nodes if n['node_id'] == node_id), None)
            if node:
                for output_key, output_value in node['outputs'].items():
                    if isinstance(output_value, str) and ('DESIGN_' in output_value or 'CAD_' in output_value):
                        design_sources.append(output_value)
        
        return design_sources
    
    def test_path_tracing(self, execution_graph: Dict[str, Any]):
        """Test path tracing functionality"""
        logger.info("🛤️ Testing Path Tracing")
        
        tracing_results = []
        
        # Test tracing for each final output
        final_nodes = [node for node in execution_graph['nodes'] if node['phase'] == 'gamma']
        
        for final_node in final_nodes:
            try:
                node_id = final_node['node_id']
                
                # Trace complete path
                complete_path = self.trace_backwards(node_id, execution_graph['nodes'])
                
                # Validate path completeness
                path_complete = len(complete_path) > 1 and complete_path[-1] == node_id
                
                # Validate path order
                path_ordered = self.validate_path_order(complete_path, execution_graph['nodes'])
                
                # Validate data flow
                data_flow_valid = self.validate_data_flow_in_path(complete_path, execution_graph)
                
                tracing_results.append({
                    'final_node': node_id,
                    'complete_path': complete_path,
                    'path_length': len(complete_path),
                    'path_complete': path_complete,
                    'path_ordered': path_ordered,
                    'data_flow_valid': data_flow_valid,
                    'tracing_successful': path_complete and path_ordered and data_flow_valid
                })
                
            except Exception as e:
                tracing_results.append({
                    'final_node': final_node['node_id'],
                    'tracing_successful': False,
                    'error': str(e)
                })
        
        successful_traces = sum(1 for result in tracing_results if result.get('tracing_successful'))
        
        self.test_results['path_tracing'] = {
            'status': 'passed' if successful_traces > 0 else 'failed',
            'details': {
                'total_traces': len(tracing_results),
                'successful_traces': successful_traces,
                'tracing_results': tracing_results
            }
        }
    
    def validate_path_order(self, path: List[str], nodes: List[Dict[str, Any]]) -> bool:
        """Validate that path follows execution order"""
        try:
            path_orders = []
            for node_id in path:
                node = next((n for n in nodes if n['node_id'] == node_id), None)
                if node:
                    path_orders.append(node['execution_order'])
            
            # Check if orders are in ascending sequence
            return all(path_orders[i] <= path_orders[i+1] for i in range(len(path_orders)-1))
            
        except Exception:
            return False
    
    def validate_data_flow_in_path(self, path: List[str], execution_graph: Dict[str, Any]) -> bool:
        """Validate data flow exists between path nodes"""
        try:
            edges = execution_graph['edges']
            
            # Check if consecutive nodes in path have data flow
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                
                # Check if edge exists (direct or indirect)
                edge_exists = any(
                    edge['source'] == current_node and edge['target'] == next_node
                    for edge in edges
                )
                
                if not edge_exists:
                    # Check for indirect connection through data
                    indirect_connection = self.check_indirect_data_connection(
                        current_node, next_node, execution_graph['nodes']
                    )
                    if not indirect_connection:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def check_indirect_data_connection(self, source_node: str, target_node: str, 
                                     nodes: List[Dict[str, Any]]) -> bool:
        """Check for indirect data connection between nodes"""
        try:
            source = next((n for n in nodes if n['node_id'] == source_node), None)
            target = next((n for n in nodes if n['node_id'] == target_node), None)
            
            if not source or not target:
                return False
            
            # Check if any source output appears in target input
            source_outputs = set()
            for output_value in source['outputs'].values():
                if isinstance(output_value, str):
                    source_outputs.add(output_value)
            
            target_inputs = set()
            for input_value in target['inputs'].values():
                if isinstance(input_value, str):
                    target_inputs.add(input_value)
            
            return len(source_outputs.intersection(target_inputs)) > 0
            
        except Exception:
            return False
    
    def test_component_lineage(self, execution_graph: Dict[str, Any]):
        """Test component lineage tracking"""
        logger.info("🔗 Testing Component Lineage")
        
        lineage_results = []
        
        for component_id, component_info in self.trace_index['component_paths'].items():
            try:
                # Validate lineage completeness
                lineage_complete = len(component_info['full_path']) > 0
                
                # Validate material traceability
                material_traceable = len(component_info['material_sources']) > 0
                
                # Validate design traceability
                design_traceable = len(component_info['design_sources']) > 0
                
                # Test lineage query performance
                start_time = time.time()
                lineage_query_result = self.query_component_lineage(component_id)
                query_time = time.time() - start_time
                
                lineage_results.append({
                    'component_id': component_id,
                    'lineage_complete': lineage_complete,
                    'material_traceable': material_traceable,
                    'design_traceable': design_traceable,
                    'query_time_ms': query_time * 1000,
                    'lineage_valid': lineage_complete and material_traceable,
                    'lineage_details': component_info
                })
                
            except Exception as e:
                lineage_results.append({
                    'component_id': component_id,
                    'lineage_valid': False,
                    'error': str(e)
                })
        
        valid_lineages = sum(1 for result in lineage_results if result.get('lineage_valid'))
        
        self.test_results['component_lineage'] = {
            'status': 'passed' if valid_lineages > 0 else 'failed',
            'details': {
                'total_components': len(lineage_results),
                'valid_lineages': valid_lineages,
                'lineage_results': lineage_results
            }
        }
    
    def query_component_lineage(self, component_id: str) -> Dict[str, Any]:
        """Query component lineage from trace index"""
        if component_id in self.trace_index['component_paths']:
            return self.trace_index['component_paths'][component_id]
        else:
            return {}
    
    def test_data_provenance(self, execution_graph: Dict[str, Any]):
        """Test data provenance tracking"""
        logger.info("📋 Testing Data Provenance")
        
        provenance_results = []
        
        for data_id, data_info in self.trace_index['data_lineage'].items():
            try:
                # Validate provenance completeness
                has_creator = data_info['created_by'] is not None
                has_users = len(data_info['used_by']) > 0
                
                # Test provenance query
                provenance_chain = self.build_provenance_chain(data_id)
                chain_complete = len(provenance_chain) > 0
                
                provenance_results.append({
                    'data_id': data_id,
                    'has_creator': has_creator,
                    'has_users': has_users,
                    'chain_complete': chain_complete,
                    'usage_count': len(data_info['used_by']),
                    'provenance_valid': has_creator and chain_complete,
                    'provenance_chain': provenance_chain
                })
                
            except Exception as e:
                provenance_results.append({
                    'data_id': data_id,
                    'provenance_valid': False,
                    'error': str(e)
                })
        
        valid_provenance = sum(1 for result in provenance_results if result.get('provenance_valid'))
        
        self.test_results['data_provenance'] = {
            'status': 'passed' if valid_provenance > 0 else 'failed',
            'details': {
                'total_data_items': len(provenance_results),
                'valid_provenance': valid_provenance,
                'provenance_results': provenance_results
            }
        }
    
    def build_provenance_chain(self, data_id: str) -> List[Dict[str, Any]]:
        """Build complete provenance chain for data item"""
        chain = []
        
        if data_id in self.trace_index['data_lineage']:
            data_info = self.trace_index['data_lineage'][data_id]
            
            # Add creation event
            if data_info['created_by']:
                chain.append({
                    'event': 'created',
                    'node': data_info['created_by'],
                    'data_type': data_info['data_type']
                })
            
            # Add usage events
            for user_node in data_info['used_by']:
                chain.append({
                    'event': 'used',
                    'node': user_node,
                    'data_type': data_info['data_type']
                })
        
        return chain
    
    def test_trace_index_integrity(self):
        """Test trace index integrity and consistency"""
        logger.info("🔍 Testing Trace Index Integrity")
        
        integrity_tests = []
        
        try:
            # Test 1: Node lineage consistency
            node_consistency = self.check_node_lineage_consistency()
            integrity_tests.append({
                'test': 'node_lineage_consistency',
                'passed': node_consistency,
                'description': 'Upstream/downstream relationships are consistent'
            })
            
            # Test 2: Data lineage completeness
            data_completeness = self.check_data_lineage_completeness()
            integrity_tests.append({
                'test': 'data_lineage_completeness',
                'passed': data_completeness,
                'description': 'All data items have complete lineage'
            })
            
            # Test 3: Component path validity
            component_validity = self.check_component_path_validity()
            integrity_tests.append({
                'test': 'component_path_validity',
                'passed': component_validity,
                'description': 'Component paths are valid and traceable'
            })
            
            # Test 4: Index performance
            index_performance = self.check_index_performance()
            integrity_tests.append({
                'test': 'index_performance',
                'passed': index_performance,
                'description': 'Index queries perform within acceptable limits'
            })
            
            all_passed = all(test['passed'] for test in integrity_tests)
            
            self.test_results['trace_index_integrity'] = {
                'status': 'passed' if all_passed else 'failed',
                'details': {
                    'total_tests': len(integrity_tests),
                    'passed_tests': sum(1 for test in integrity_tests if test['passed']),
                    'integrity_tests': integrity_tests
                }
            }
            
        except Exception as e:
            self.test_results['trace_index_integrity'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    def check_node_lineage_consistency(self) -> bool:
        """Check node lineage consistency"""
        try:
            for node_id, lineage in self.trace_index['node_lineage'].items():
                # Check that downstream nodes reference this node as upstream
                for downstream in lineage['downstream_nodes']:
                    if downstream in self.trace_index['node_lineage']:
                        downstream_upstream = self.trace_index['node_lineage'][downstream]['upstream_nodes']
                        if node_id not in downstream_upstream:
                            return False
            
            return True
            
        except Exception:
            return False
    
    def check_data_lineage_completeness(self) -> bool:
        """Check data lineage completeness"""
        try:
            for data_id, lineage in self.trace_index['data_lineage'].items():
                # Each data item should have either a creator or users
                if not lineage['created_by'] and len(lineage['used_by']) == 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def check_component_path_validity(self) -> bool:
        """Check component path validity"""
        try:
            for component_id, path_info in self.trace_index['component_paths'].items():
                # Each component should have a valid path
                if len(path_info['full_path']) == 0:
                    return False
                
                # Component should have material sources
                if len(path_info['material_sources']) == 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def check_index_performance(self) -> bool:
        """Check index query performance"""
        try:
            # Test query performance for different operations
            start_time = time.time()
            
            # Test lineage queries
            for component_id in list(self.trace_index['component_paths'].keys())[:3]:
                self.query_component_lineage(component_id)
            
            # Test provenance queries
            for data_id in list(self.trace_index['data_lineage'].keys())[:3]:
                self.build_provenance_chain(data_id)
            
            query_time = time.time() - start_time
            
            # Performance should be under 100ms for basic queries
            return query_time < 0.1
            
        except Exception:
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        duration = (datetime.now() - self.start_time).total_seconds()
        report_file = f"trace_path_index_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Include trace index in report
        report_data = {
            'test_suite': 'Trace Path Index Test',
            'execution_time': duration,
            'results': self.test_results,
            'trace_index_summary': {
                'nodes_indexed': len(self.trace_index.get('node_lineage', {})),
                'data_items_indexed': len(self.trace_index.get('data_lineage', {})),
                'components_indexed': len(self.trace_index.get('component_paths', {}))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"🔍 Trace Path Index Test completed in {duration:.2f}s - Report: {report_file}")

def main():
    test_suite = TracePathIndexTest()
    results = test_suite.run_trace_tests()
    failed_tests = sum(1 for result in results.values() if result.get('status') == 'failed')
    sys.exit(failed_tests)

if __name__ == "__main__":
    main()
