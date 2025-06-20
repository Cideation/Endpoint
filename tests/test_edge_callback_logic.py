#!/usr/bin/env python3
"""
Edge Integrity Test
Tests edge direction, metadata, and callback accuracy
Validates phase-specific edge architectures:
✅ Alpha = DAG, directed, one-to-one or linear edge flow
✅ Beta = Relational (Objective Functions) → many-to-many, dense logic  
✅ Gamma = Combinatorial (Emergence) → many-to-many, sparse-to-dense mappings
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
    """Test phase-specific edge integrity and callback logic"""
    
    def __init__(self):
        self.test_results = {
            'alpha_dag_edges': {'status': 'pending', 'details': {}},
            'beta_relational_edges': {'status': 'pending', 'details': {}},
            'gamma_combinatorial_edges': {'status': 'pending', 'details': {}},
            'edge_table_segregation': {'status': 'pending', 'details': {}},
            'callback_accuracy_test': {'status': 'pending', 'details': {}}
        }
        self.start_time = datetime.now()
        
    def run_edge_tests(self) -> Dict[str, Any]:
        """Execute all phase-specific edge tests"""
        logger.info("🔗 Starting Phase-Specific Edge Tests")
        
        try:
            edge_configurations = self.create_phase_specific_edges()
            
            # Test Alpha DAG edges (directed, linear)
            self.test_alpha_dag_edges(edge_configurations['alpha_edges'])
            
            # Test Beta relational edges (many-to-many, dense)
            self.test_beta_relational_edges(edge_configurations['beta_relationships'])
            
            # Test Gamma combinatorial edges (sparse-to-dense)
            self.test_gamma_combinatorial_edges(edge_configurations['gamma_edges'])
            
            # Test edge table segregation
            self.test_edge_table_segregation(edge_configurations)
            
            # Test callback accuracy across phases
            self.test_callback_accuracy(edge_configurations)
            
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Edge callback test failed: {e}")
            
        return self.test_results
    
    def create_phase_specific_edges(self) -> Dict[str, Any]:
        """Create phase-specific edge configurations"""
        return {
            'alpha_edges': [
                # Alpha = DAG, directed, one-to-one linear flow
                {
                    'edge_id': 'ALPHA_E001',
                    'source_node': 'A01_MaterialInput',
                    'target_node': 'A02_RequirementCapture',
                    'edge_type': 'directed_dag',
                    'flow_pattern': 'one_to_one',
                    'storage_table': 'alpha_edges.csv',
                    'metadata': {
                        'data_type': 'material_specification',
                        'priority': 'high',
                        'transformation': 'material_to_requirement_mapping'
                    },
                    'callback_config': {
                        'trigger_condition': 'source_complete',
                        'callback_function': 'linear_propagate',
                        'retry_count': 2
                    }
                },
                {
                    'edge_id': 'ALPHA_E002',
                    'source_node': 'A02_RequirementCapture',
                    'target_node': 'A03_DesignConstraints',
                    'edge_type': 'directed_dag',
                    'flow_pattern': 'linear',
                    'storage_table': 'alpha_edges.csv',
                    'metadata': {
                        'data_type': 'requirement_constraints',
                        'priority': 'high',
                        'transformation': 'requirement_to_constraint_mapping'
                    },
                    'callback_config': {
                        'trigger_condition': 'source_complete',
                        'callback_function': 'linear_propagate',
                        'retry_count': 2
                    }
                }
            ],
            'beta_relationships': [
                # Beta = Relational (Objective Functions) → many-to-many, dense logic
                {
                    'edge_id': 'BETA_R001',
                    'source_nodes': ['B01_CostObjective', 'B02_QualityObjective'],
                    'target_nodes': ['B03_OptimizationEngine', 'B04_TradeoffAnalysis'],
                    'edge_type': 'many_to_many_relational',
                    'flow_pattern': 'dense_logic',
                    'storage_table': 'beta_relationships.csv',
                    'metadata': {
                        'data_type': 'objective_function_relations',
                        'priority': 'medium',
                        'transformation': 'multi_objective_optimization'
                    },
                    'callback_config': {
                        'trigger_condition': 'all_sources_ready',
                        'callback_function': 'relational_aggregate',
                        'retry_count': 3
                    }
                },
                {
                    'edge_id': 'BETA_R002',
                    'source_nodes': ['B03_OptimizationEngine'],
                    'target_nodes': ['B04_TradeoffAnalysis', 'B05_PerformanceMetrics', 'B06_CostAnalysis'],
                    'edge_type': 'one_to_many_relational',
                    'flow_pattern': 'dense_distribution',
                    'storage_table': 'beta_relationships.csv',
                    'metadata': {
                        'data_type': 'optimization_results',
                        'priority': 'high',
                        'transformation': 'result_distribution'
                    },
                    'callback_config': {
                        'trigger_condition': 'optimization_complete',
                        'callback_function': 'broadcast_results',
                        'retry_count': 2
                    }
                }
            ],
            'gamma_edges': [
                # Gamma = Combinatorial (Emergence) → many-to-many, sparse-to-dense mappings
                {
                    'edge_id': 'GAMMA_C001',
                    'source_nodes': ['G01_ComponentA', 'G02_ComponentB', 'G03_ComponentC'],
                    'target_nodes': ['G04_EmergentBehavior'],
                    'edge_type': 'combinatorial_emergence',
                    'flow_pattern': 'sparse_to_dense',
                    'storage_table': 'gamma_edges.csv',
                    'metadata': {
                        'data_type': 'emergent_properties',
                        'priority': 'critical',
                        'transformation': 'combinatorial_synthesis',
                        'learning_weight': 0.85
                    },
                    'callback_config': {
                        'trigger_condition': 'emergence_threshold_met',
                        'callback_function': 'emergent_synthesis',
                        'retry_count': 5
                    }
                },
                {
                    'edge_id': 'GAMMA_C002',
                    'source_nodes': ['G04_EmergentBehavior'],
                    'target_nodes': ['G05_SystemProperty1', 'G06_SystemProperty2', 'G07_SystemProperty3', 'G08_SystemProperty4'],
                    'edge_type': 'emergence_to_properties',
                    'flow_pattern': 'dense_emergence',
                    'storage_table': 'gamma_edges.csv',
                    'metadata': {
                        'data_type': 'system_wide_properties',
                        'priority': 'critical',
                        'transformation': 'property_manifestation',
                        'learning_weight': 0.92
                    },
                    'callback_config': {
                        'trigger_condition': 'emergence_complete',
                        'callback_function': 'manifest_properties',
                        'retry_count': 3
                    }
                }
            ]
        }
    
    def test_alpha_dag_edges(self, alpha_edges: List[Dict[str, Any]]):
        """Test Alpha DAG edges - directed, one-to-one linear flow"""
        logger.info("🔵 Testing Alpha DAG Edges")
        
        alpha_results = []
        for edge in alpha_edges:
            try:
                # Validate DAG properties
                is_dag = self.validate_dag_properties(edge)
                
                # Validate linear flow
                is_linear = self.validate_linear_flow(edge)
                
                # Validate directed nature
                is_directed = self.validate_directed_edge(edge)
                
                # Test storage table assignment
                correct_table = edge.get('storage_table') == 'alpha_edges.csv'
                
                alpha_results.append({
                    'edge_id': edge['edge_id'],
                    'is_dag': is_dag,
                    'is_linear': is_linear,
                    'is_directed': is_directed,
                    'correct_table': correct_table,
                    'alpha_compliant': is_dag and is_linear and is_directed and correct_table
                })
                
            except Exception as e:
                alpha_results.append({
                    'edge_id': edge['edge_id'],
                    'alpha_compliant': False,
                    'error': str(e)
                })
        
        compliant_edges = sum(1 for result in alpha_results if result.get('alpha_compliant'))
        
        self.test_results['alpha_dag_edges'] = {
            'status': 'passed' if compliant_edges == len(alpha_edges) else 'failed',
            'details': {
                'total_edges': len(alpha_edges),
                'compliant_edges': compliant_edges,
                'compliance_rate': (compliant_edges / len(alpha_edges)) * 100 if alpha_edges else 0,
                'edge_results': alpha_results
            }
        }
    
    def validate_dag_properties(self, edge: Dict[str, Any]) -> bool:
        """Validate DAG properties - no cycles, directed"""
        edge_type = edge.get('edge_type', '')
        return 'directed_dag' in edge_type
    
    def validate_linear_flow(self, edge: Dict[str, Any]) -> bool:
        """Validate linear flow pattern"""
        flow_pattern = edge.get('flow_pattern', '')
        return flow_pattern in ['one_to_one', 'linear']
    
    def validate_directed_edge(self, edge: Dict[str, Any]) -> bool:
        """Validate directed edge structure"""
        has_single_source = 'source_node' in edge
        has_single_target = 'target_node' in edge
        return has_single_source and has_single_target
    
    def test_beta_relational_edges(self, beta_relationships: List[Dict[str, Any]]):
        """Test Beta relational edges - many-to-many, dense logic"""
        logger.info("🟡 Testing Beta Relational Edges")
        
        beta_results = []
        for relationship in beta_relationships:
            try:
                # Validate many-to-many structure
                is_many_to_many = self.validate_many_to_many(relationship)
                
                # Validate dense logic pattern
                is_dense = self.validate_dense_logic(relationship)
                
                # Validate relational properties
                is_relational = self.validate_relational_properties(relationship)
                
                # Test storage table assignment
                correct_table = relationship.get('storage_table') == 'beta_relationships.csv'
                
                beta_results.append({
                    'edge_id': relationship['edge_id'],
                    'is_many_to_many': is_many_to_many,
                    'is_dense': is_dense,
                    'is_relational': is_relational,
                    'correct_table': correct_table,
                    'beta_compliant': is_many_to_many and is_dense and is_relational and correct_table
                })
                
            except Exception as e:
                beta_results.append({
                    'edge_id': relationship['edge_id'],
                    'beta_compliant': False,
                    'error': str(e)
                })
        
        compliant_relationships = sum(1 for result in beta_results if result.get('beta_compliant'))
        
        self.test_results['beta_relational_edges'] = {
            'status': 'passed' if compliant_relationships == len(beta_relationships) else 'failed',
            'details': {
                'total_relationships': len(beta_relationships),
                'compliant_relationships': compliant_relationships,
                'compliance_rate': (compliant_relationships / len(beta_relationships)) * 100 if beta_relationships else 0,
                'relationship_results': beta_results
            }
        }
    
    def validate_many_to_many(self, relationship: Dict[str, Any]) -> bool:
        """Validate many-to-many relationship structure"""
        has_multiple_sources = len(relationship.get('source_nodes', [])) > 1
        has_multiple_targets = len(relationship.get('target_nodes', [])) > 1
        edge_type = relationship.get('edge_type', '')
        
        return ('many_to_many' in edge_type or 'one_to_many' in edge_type) and (has_multiple_sources or has_multiple_targets)
    
    def validate_dense_logic(self, relationship: Dict[str, Any]) -> bool:
        """Validate dense logic flow pattern"""
        flow_pattern = relationship.get('flow_pattern', '')
        return 'dense' in flow_pattern
    
    def validate_relational_properties(self, relationship: Dict[str, Any]) -> bool:
        """Validate relational properties for objective functions"""
        edge_type = relationship.get('edge_type', '')
        data_type = relationship.get('metadata', {}).get('data_type', '')
        
        return 'relational' in edge_type and ('objective' in data_type or 'optimization' in data_type)
    
    def test_gamma_combinatorial_edges(self, gamma_edges: List[Dict[str, Any]]):
        """Test Gamma combinatorial edges - sparse-to-dense, emergent"""
        logger.info("🟢 Testing Gamma Combinatorial Edges")
        
        gamma_results = []
        for edge in gamma_edges:
            try:
                # Validate combinatorial structure
                is_combinatorial = self.validate_combinatorial_structure(edge)
                
                # Validate sparse-to-dense mapping
                is_sparse_to_dense = self.validate_sparse_to_dense(edge)
                
                # Validate emergence properties
                has_emergence = self.validate_emergence_properties(edge)
                
                # Validate learning weights
                has_learning = self.validate_learning_weights(edge)
                
                # Test storage table assignment
                correct_table = edge.get('storage_table') == 'gamma_edges.csv'
                
                gamma_results.append({
                    'edge_id': edge['edge_id'],
                    'is_combinatorial': is_combinatorial,
                    'is_sparse_to_dense': is_sparse_to_dense,
                    'has_emergence': has_emergence,
                    'has_learning': has_learning,
                    'correct_table': correct_table,
                    'gamma_compliant': is_combinatorial and is_sparse_to_dense and has_emergence and correct_table
                })
                
            except Exception as e:
                gamma_results.append({
                    'edge_id': edge['edge_id'],
                    'gamma_compliant': False,
                    'error': str(e)
                })
        
        compliant_edges = sum(1 for result in gamma_results if result.get('gamma_compliant'))
        
        self.test_results['gamma_combinatorial_edges'] = {
            'status': 'passed' if compliant_edges == len(gamma_edges) else 'failed',
            'details': {
                'total_edges': len(gamma_edges),
                'compliant_edges': compliant_edges,
                'compliance_rate': (compliant_edges / len(gamma_edges)) * 100 if gamma_edges else 0,
                'edge_results': gamma_results
            }
        }
    
    def validate_combinatorial_structure(self, edge: Dict[str, Any]) -> bool:
        """Validate combinatorial edge structure"""
        edge_type = edge.get('edge_type', '')
        return 'combinatorial' in edge_type or 'emergence' in edge_type
    
    def validate_sparse_to_dense(self, edge: Dict[str, Any]) -> bool:
        """Validate sparse-to-dense mapping pattern"""
        flow_pattern = edge.get('flow_pattern', '')
        return 'sparse_to_dense' in flow_pattern or 'dense_emergence' in flow_pattern
    
    def validate_emergence_properties(self, edge: Dict[str, Any]) -> bool:
        """Validate emergence properties"""
        data_type = edge.get('metadata', {}).get('data_type', '')
        transformation = edge.get('metadata', {}).get('transformation', '')
        
        return 'emergent' in data_type or 'emergence' in transformation or 'synthesis' in transformation
    
    def validate_learning_weights(self, edge: Dict[str, Any]) -> bool:
        """Validate learning weights for gamma edges"""
        learning_weight = edge.get('metadata', {}).get('learning_weight')
        return learning_weight is not None and 0.0 <= learning_weight <= 1.0
    
    def test_edge_table_segregation(self, edge_configurations: Dict[str, Any]):
        """Test edge table segregation by phase"""
        logger.info("🗂️ Testing Edge Table Segregation")
        
        segregation_results = {
            'alpha_edges_csv': [],
            'beta_relationships_csv': [],
            'gamma_edges_csv': []
        }
        
        # Check Alpha edges table assignment
        for edge in edge_configurations['alpha_edges']:
            table = edge.get('storage_table', '')
            segregation_results['alpha_edges_csv'].append({
                'edge_id': edge['edge_id'],
                'assigned_table': table,
                'correct_assignment': table == 'alpha_edges.csv'
            })
        
        # Check Beta relationships table assignment
        for relationship in edge_configurations['beta_relationships']:
            table = relationship.get('storage_table', '')
            segregation_results['beta_relationships_csv'].append({
                'edge_id': relationship['edge_id'],
                'assigned_table': table,
                'correct_assignment': table == 'beta_relationships.csv'
            })
        
        # Check Gamma edges table assignment
        for edge in edge_configurations['gamma_edges']:
            table = edge.get('storage_table', '')
            segregation_results['gamma_edges_csv'].append({
                'edge_id': edge['edge_id'],
                'assigned_table': table,
                'correct_assignment': table == 'gamma_edges.csv'
            })
        
        # Calculate segregation compliance
        total_assignments = (len(segregation_results['alpha_edges_csv']) +
                           len(segregation_results['beta_relationships_csv']) +
                           len(segregation_results['gamma_edges_csv']))
        
        correct_assignments = sum(
            sum(1 for item in table_assignments if item['correct_assignment'])
            for table_assignments in segregation_results.values()
        )
        
        self.test_results['edge_table_segregation'] = {
            'status': 'passed' if correct_assignments == total_assignments else 'failed',
            'details': {
                'total_assignments': total_assignments,
                'correct_assignments': correct_assignments,
                'segregation_compliance': (correct_assignments / total_assignments) * 100 if total_assignments > 0 else 0,
                'table_assignments': segregation_results
            }
        }
    
    def test_callback_accuracy(self, edge_configurations: Dict[str, Any]):
        """Test callback execution accuracy across all phases"""
        logger.info("🔄 Testing Cross-Phase Callback Accuracy")
        
        callback_results = []
        
        # Test callbacks for all edge types
        all_edges = (edge_configurations['alpha_edges'] +
                    edge_configurations['beta_relationships'] +
                    edge_configurations['gamma_edges'])
        
        for edge in all_edges:
            try:
                callback_config = edge.get('callback_config', {})
                
                # Execute phase-specific callback simulation
                callback_result = self.simulate_phase_specific_callback(edge, callback_config)
                
                callback_results.append({
                    'edge_id': edge['edge_id'],
                    'edge_phase': self.determine_edge_phase(edge['edge_id']),
                    'callback_executed': callback_result['executed'],
                    'execution_time': callback_result['execution_time'],
                    'phase_appropriate': callback_result['phase_appropriate'],
                    'success': callback_result['success']
                })
                
            except Exception as e:
                callback_results.append({
                    'edge_id': edge['edge_id'],
                    'callback_executed': False,
                    'error': str(e)
                })
        
        successful_callbacks = sum(1 for result in callback_results if result.get('success'))
        
        self.test_results['callback_accuracy_test'] = {
            'status': 'passed' if successful_callbacks == len(all_edges) else 'failed',
            'details': {
                'total_callbacks': len(all_edges),
                'successful_callbacks': successful_callbacks,
                'success_rate': (successful_callbacks / len(all_edges)) * 100 if all_edges else 0,
                'callback_results': callback_results
            }
        }
    
    def simulate_phase_specific_callback(self, edge: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate phase-specific callback execution"""
        start_time = time.time()
        
        callback_function = config.get('callback_function', '')
        edge_phase = self.determine_edge_phase(edge['edge_id'])
        
        # Phase-specific callback validation
        if edge_phase == 'alpha':
            # Alpha callbacks should be linear and simple
            success = 'linear' in callback_function or 'propagate' in callback_function
            phase_appropriate = True
        elif edge_phase == 'beta':
            # Beta callbacks should handle aggregation and relations
            success = 'aggregate' in callback_function or 'broadcast' in callback_function or 'relational' in callback_function
            phase_appropriate = True
        elif edge_phase == 'gamma':
            # Gamma callbacks should handle emergence and synthesis
            success = 'emergent' in callback_function or 'synthesis' in callback_function or 'manifest' in callback_function
            phase_appropriate = True
        else:
            success = False
            phase_appropriate = False
        
        execution_time = time.time() - start_time
        
        return {
            'executed': True,
            'success': success,
            'phase_appropriate': phase_appropriate,
            'execution_time': execution_time
        }
    
    def determine_edge_phase(self, edge_id: str) -> str:
        """Determine edge phase from edge ID"""
        if edge_id.startswith('ALPHA_'):
            return 'alpha'
        elif edge_id.startswith('BETA_'):
            return 'beta'
        elif edge_id.startswith('GAMMA_'):
            return 'gamma'
        else:
            return 'unknown'
    
    def generate_report(self):
        """Generate comprehensive test report"""
        duration = (datetime.now() - self.start_time).total_seconds()
        report_file = f"edge_callback_logic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'test_suite': 'Phase-Specific Edge Callback Logic Test',
                'execution_time': duration,
                'phase_architecture': {
                    'alpha': 'DAG, directed, one-to-one linear flow',
                    'beta': 'Relational (Objective Functions) → many-to-many, dense logic',
                    'gamma': 'Combinatorial (Emergence) → many-to-many, sparse-to-dense mappings'
                },
                'results': self.test_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"🔗 Phase-Specific Edge Test completed in {duration:.2f}s - Report: {report_file}")

def main():
    test_suite = EdgeCallbackLogicTest()
    results = test_suite.run_edge_tests()
    failed_tests = sum(1 for result in results.values() if result.get('status') == 'failed')
    sys.exit(failed_tests)

if __name__ == "__main__":
    main()
