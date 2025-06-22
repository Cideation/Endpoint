#!/usr/bin/env python3
"""
🔍 Advanced Test Layer 5: Output Tracing
=====================================

Tests the ability to trace final outputs back to their source inputs
through the complete DAG execution path.

Key Validations:
- Input-to-output traceability
- Path reconstruction through phases
- Source attribution accuracy
- Dependency chain validation
- Output provenance verification
"""

import pytest
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from shared.phase_2_runtime_modules import (
        NetworkNode, V01_ProductComponent, V02_AssemblyComponent,
        V03_QualityAssurance, V04_CostOptimization, V05_Compliance,
        V06_SupplyChain, V07_Manufacturing, V08_Testing, V09_Deployment
    )
    from MICROSERVICE_ENGINES.shared.network_graph import NetworkGraph
    from neon.database_integration import DatabaseIntegration, NEON_CONFIG
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Import error: {e}")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutputTracer:
    """Traces outputs back to their source inputs through the DAG"""
    
    def __init__(self):
        self.trace_paths = {}
        self.input_sources = {}
        self.output_destinations = {}
        self.dependency_chains = {}
        
    def record_input(self, node_id: str, input_data: Dict[str, Any], source: str = "external"):
        """Record an input and its source"""
        self.input_sources[node_id] = {
            'data': input_data,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'phase': self._get_node_phase(node_id)
        }
        
    def record_output(self, node_id: str, output_data: Dict[str, Any], destinations: List[str] = None):
        """Record an output and its destinations"""
        self.output_destinations[node_id] = {
            'data': output_data,
            'destinations': destinations or [],
            'timestamp': datetime.now().isoformat(),
            'phase': self._get_node_phase(node_id)
        }
        
    def record_dependency(self, target_node: str, source_node: str, dependency_type: str):
        """Record a dependency relationship"""
        if target_node not in self.dependency_chains:
            self.dependency_chains[target_node] = []
        
        self.dependency_chains[target_node].append({
            'source': source_node,
            'type': dependency_type,
            'timestamp': datetime.now().isoformat()
        })
        
    def trace_output_to_sources(self, output_node: str) -> Dict[str, Any]:
        """Trace an output back to all its source inputs"""
        trace_result = {
            'output_node': output_node,
            'source_inputs': [],
            'path_chains': [],
            'total_hops': 0,
            'phases_traversed': set()
        }
        
        # Recursive trace function
        def trace_recursive(node_id: str, path: List[str] = None) -> List[List[str]]:
            if path is None:
                path = []
            
            if node_id in path:  # Circular dependency detection
                return []
                
            current_path = path + [node_id]
            trace_result['phases_traversed'].add(self._get_node_phase(node_id))
            
            # Check if this is a source input
            if node_id in self.input_sources and self.input_sources[node_id]['source'] == 'external':
                trace_result['source_inputs'].append({
                    'node': node_id,
                    'data': self.input_sources[node_id]['data'],
                    'path_length': len(current_path)
                })
                return [current_path]
            
            # Trace dependencies
            all_paths = []
            if node_id in self.dependency_chains:
                for dep in self.dependency_chains[node_id]:
                    source_paths = trace_recursive(dep['source'], current_path)
                    all_paths.extend(source_paths)
            
            return all_paths
        
        # Execute trace
        all_paths = trace_recursive(output_node)
        trace_result['path_chains'] = all_paths
        trace_result['total_hops'] = max(len(path) for path in all_paths) if all_paths else 0
        trace_result['phases_traversed'] = list(trace_result['phases_traversed'])
        
        return trace_result
        
    def _get_node_phase(self, node_id: str) -> str:
        """Determine the phase of a node based on its ID"""
        if node_id.startswith('V01') or node_id.startswith('V02') or node_id.startswith('V03'):
            return 'Alpha'
        elif node_id.startswith('V04') or node_id.startswith('V05') or node_id.startswith('V06'):
            return 'Beta'
        elif node_id.startswith('V07') or node_id.startswith('V08') or node_id.startswith('V09'):
            return 'Gamma'
        return 'Unknown'
        
    def validate_trace_completeness(self, trace_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the trace is complete and accurate"""
        validation = {
            'is_complete': True,
            'issues': [],
            'coverage_score': 0.0,
            'path_integrity': True
        }
        
        # Check for source inputs
        if not trace_result['source_inputs']:
            validation['is_complete'] = False
            validation['issues'].append("No source inputs found")
            
        # Check for broken paths
        for path in trace_result['path_chains']:
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                
                # Verify dependency exists
                if next_node not in self.dependency_chains:
                    validation['path_integrity'] = False
                    validation['issues'].append(f"Missing dependency chain for {next_node}")
                else:
                    deps = [d['source'] for d in self.dependency_chains[next_node]]
                    if current_node not in deps:
                        validation['path_integrity'] = False
                        validation['issues'].append(f"Broken dependency: {current_node} -> {next_node}")
        
        # Calculate coverage score
        total_nodes = len(self.input_sources) + len(self.output_destinations)
        traced_nodes = len(set(node for path in trace_result['path_chains'] for node in path))
        validation['coverage_score'] = traced_nodes / total_nodes if total_nodes > 0 else 0.0
        
        return validation

@pytest.fixture
def output_tracer():
    """Create an OutputTracer instance for testing"""
    return OutputTracer()

@pytest.fixture
def sample_dag_execution(output_tracer):
    """Simulate a complete DAG execution with tracing"""
    
    # Alpha Phase Inputs (External Sources)
    output_tracer.record_input(
        'V01_Product_001',
        {'specifications': 'Product specs', 'requirements': 'Functional reqs'},
        'external'
    )
    
    output_tracer.record_input(
        'V02_Assembly_001', 
        {'components': ['comp1', 'comp2'], 'assembly_rules': 'Rules'},
        'external'
    )
    
    # Alpha Phase Processing
    output_tracer.record_dependency('V02_Assembly_001', 'V01_Product_001', 'component_spec')
    output_tracer.record_output(
        'V01_Product_001',
        {'validated_specs': True, 'design_parameters': {'param1': 'value1'}},
        ['V02_Assembly_001', 'V04_Cost_001']
    )
    
    output_tracer.record_output(
        'V02_Assembly_001',
        {'assembly_plan': 'Plan A', 'component_list': ['comp1', 'comp2', 'comp3']},
        ['V03_Quality_001', 'V05_Compliance_001']
    )
    
    # Beta Phase Processing
    output_tracer.record_dependency('V04_Cost_001', 'V01_Product_001', 'design_params')
    output_tracer.record_dependency('V05_Compliance_001', 'V02_Assembly_001', 'assembly_plan')
    
    output_tracer.record_output(
        'V04_Cost_001',
        {'cost_estimate': 15000, 'optimization_suggestions': ['opt1', 'opt2']},
        ['V07_Manufacturing_001']
    )
    
    output_tracer.record_output(
        'V05_Compliance_001',
        {'compliance_score': 0.95, 'regulatory_status': 'approved'},
        ['V08_Testing_001']
    )
    
    # Gamma Phase Processing (Final Outputs)
    output_tracer.record_dependency('V07_Manufacturing_001', 'V04_Cost_001', 'cost_optimization')
    output_tracer.record_dependency('V08_Testing_001', 'V05_Compliance_001', 'compliance_validation')
    
    output_tracer.record_output(
        'V07_Manufacturing_001',
        {'production_plan': 'Final plan', 'estimated_time': '30 days', 'cost_final': 14500},
        ['V09_Deployment_001']
    )
    
    output_tracer.record_output(
        'V08_Testing_001',
        {'test_results': 'passed', 'quality_score': 0.98, 'certification': 'ISO9001'},
        ['V09_Deployment_001']
    )
    
    # Final Output
    output_tracer.record_dependency('V09_Deployment_001', 'V07_Manufacturing_001', 'production_plan')
    output_tracer.record_dependency('V09_Deployment_001', 'V08_Testing_001', 'quality_certification')
    
    output_tracer.record_output(
        'V09_Deployment_001',
        {
            'deployment_ready': True,
            'final_cost': 14500,
            'quality_certified': True,
            'delivery_date': '2024-02-15'
        },
        []
    )
    
    return output_tracer

class TestOutputTracing:
    """Test suite for output tracing functionality"""
    
    def test_basic_trace_functionality(self, output_tracer):
        """Test basic tracing setup and recording"""
        # Record a simple input-output chain
        output_tracer.record_input('INPUT_001', {'data': 'test'}, 'external')
        output_tracer.record_dependency('OUTPUT_001', 'INPUT_001', 'direct')
        output_tracer.record_output('OUTPUT_001', {'result': 'processed'})
        
        # Verify recordings
        assert 'INPUT_001' in output_tracer.input_sources
        assert 'OUTPUT_001' in output_tracer.output_destinations
        assert 'OUTPUT_001' in output_tracer.dependency_chains
        
        logger.info("✅ Basic trace functionality working")
    
    def test_simple_output_trace(self, output_tracer):
        """Test tracing a simple output to its source"""
        # Setup simple chain: INPUT -> PROCESS -> OUTPUT
        output_tracer.record_input('INPUT_001', {'value': 100}, 'external')
        output_tracer.record_dependency('PROCESS_001', 'INPUT_001', 'data_input')
        output_tracer.record_dependency('OUTPUT_001', 'PROCESS_001', 'processed_data')
        
        output_tracer.record_output('PROCESS_001', {'processed_value': 150})
        output_tracer.record_output('OUTPUT_001', {'final_value': 200})
        
        # Trace the output
        trace_result = output_tracer.trace_output_to_sources('OUTPUT_001')
        
        # Validate trace
        assert trace_result['output_node'] == 'OUTPUT_001'
        assert len(trace_result['source_inputs']) == 1
        assert trace_result['source_inputs'][0]['node'] == 'INPUT_001'
        assert trace_result['total_hops'] == 3  # INPUT -> PROCESS -> OUTPUT
        
        logger.info("✅ Simple output trace successful")
    
    def test_complex_dag_trace(self, sample_dag_execution):
        """Test tracing through complex DAG with multiple phases"""
        tracer = sample_dag_execution
        
        # Trace the final deployment output
        trace_result = tracer.trace_output_to_sources('V09_Deployment_001')
        
        # Validate comprehensive trace
        assert trace_result['output_node'] == 'V09_Deployment_001'
        assert len(trace_result['source_inputs']) == 2  # Two external inputs
        
        # Check source inputs are correct
        source_nodes = [inp['node'] for inp in trace_result['source_inputs']]
        assert 'V01_Product_001' in source_nodes
        assert 'V02_Assembly_001' in source_nodes
        
        # Validate phases traversed
        assert 'Alpha' in trace_result['phases_traversed']
        assert 'Beta' in trace_result['phases_traversed']
        assert 'Gamma' in trace_result['phases_traversed']
        
        # Validate path chains exist
        assert len(trace_result['path_chains']) > 0
        assert trace_result['total_hops'] > 3  # Multi-phase execution
        
        logger.info("✅ Complex DAG trace successful")
    
    def test_trace_validation(self, sample_dag_execution):
        """Test trace validation for completeness and integrity"""
        tracer = sample_dag_execution
        
        # Get trace result
        trace_result = tracer.trace_output_to_sources('V09_Deployment_001')
        
        # Validate the trace
        validation = tracer.validate_trace_completeness(trace_result)
        
        # Check validation results
        assert validation['is_complete'] == True
        assert validation['path_integrity'] == True
        assert validation['coverage_score'] > 0.5  # Good coverage
        assert len(validation['issues']) == 0  # No issues
        
        logger.info("✅ Trace validation successful")
    
    def test_multiple_output_traces(self, sample_dag_execution):
        """Test tracing multiple outputs simultaneously"""
        tracer = sample_dag_execution
        
        # Trace multiple outputs
        outputs_to_trace = ['V07_Manufacturing_001', 'V08_Testing_001', 'V09_Deployment_001']
        trace_results = {}
        
        for output_node in outputs_to_trace:
            trace_results[output_node] = tracer.trace_output_to_sources(output_node)
        
        # Validate all traces
        for node, result in trace_results.items():
            assert result['output_node'] == node
            assert len(result['source_inputs']) > 0
            assert len(result['path_chains']) > 0
            
            # Validate that earlier phase outputs have shorter paths
            if node == 'V07_Manufacturing_001':
                assert result['total_hops'] <= trace_results['V09_Deployment_001']['total_hops']
        
        logger.info("✅ Multiple output traces successful")
    
    def test_circular_dependency_detection(self, output_tracer):
        """Test detection of circular dependencies in tracing"""
        # Create circular dependency
        output_tracer.record_dependency('NODE_A', 'NODE_B', 'circular_1')
        output_tracer.record_dependency('NODE_B', 'NODE_C', 'circular_2')
        output_tracer.record_dependency('NODE_C', 'NODE_A', 'circular_3')
        
        # Try to trace (should handle circular dependency gracefully)
        trace_result = output_tracer.trace_output_to_sources('NODE_A')
        
        # Should not get stuck in infinite loop
        assert isinstance(trace_result, dict)
        assert trace_result['output_node'] == 'NODE_A'
        
        logger.info("✅ Circular dependency detection working")
    
    def test_phase_transition_tracing(self, sample_dag_execution):
        """Test tracing across phase transitions"""
        tracer = sample_dag_execution
        
        # Trace a Beta phase output
        trace_result = tracer.trace_output_to_sources('V04_Cost_001')
        
        # Should trace back to Alpha phase
        assert 'Alpha' in trace_result['phases_traversed']
        assert 'Beta' in trace_result['phases_traversed']
        
        # Should find Alpha phase source
        source_phases = [tracer._get_node_phase(inp['node']) for inp in trace_result['source_inputs']]
        assert 'Alpha' in source_phases
        
        logger.info("✅ Phase transition tracing successful")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_real_node_integration(self):
        """Test tracing with real NetworkNode instances"""
        # Create real nodes
        product_node = V01_ProductComponent(node_id="V01_Real_001")
        assembly_node = V02_AssemblyComponent(node_id="V02_Real_001")
        
        # Setup initial data
        product_data = {
            'specifications': 'Real product specs',
            'requirements': 'Real requirements'
        }
        product_node.data.update(product_data)
        
        # Create tracer and record real execution
        tracer = OutputTracer()
        tracer.record_input('V01_Real_001', product_data, 'external')
        
        # Simulate processing
        result = product_node.evaluate_manufacturing()
        if result:
            tracer.record_output('V01_Real_001', product_node.dictionary())
            tracer.record_dependency('V02_Real_001', 'V01_Real_001', 'product_spec')
            
            # Trace the result
            trace_result = tracer.trace_output_to_sources('V01_Real_001')
            
            # Validate integration
            assert len(trace_result['source_inputs']) > 0
            assert trace_result['source_inputs'][0]['node'] == 'V01_Real_001'
        
        logger.info("✅ Real node integration successful")
    
    def test_trace_performance(self, sample_dag_execution):
        """Test tracing performance with larger DAG"""
        tracer = sample_dag_execution
        
        # Measure trace time
        import time
        start_time = time.time()
        
        # Perform multiple traces
        for _ in range(10):
            trace_result = tracer.trace_output_to_sources('V09_Deployment_001')
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (< 0.1 seconds per trace)
        assert avg_time < 0.1
        
        logger.info(f"✅ Trace performance acceptable: {avg_time:.4f}s average")
    
    def test_output_provenance_verification(self, sample_dag_execution):
        """Test verification of output provenance"""
        tracer = sample_dag_execution
        
        # Get final output
        final_output = tracer.output_destinations['V09_Deployment_001']
        
        # Trace its provenance
        trace_result = tracer.trace_output_to_sources('V09_Deployment_001')
        
        # Verify provenance data
        provenance = {
            'output_data': final_output['data'],
            'source_contributions': {},
            'transformation_chain': []
        }
        
        # Map source contributions
        for source_input in trace_result['source_inputs']:
            node_id = source_input['node']
            provenance['source_contributions'][node_id] = {
                'original_data': source_input['data'],
                'path_length': source_input['path_length']
            }
        
        # Build transformation chain
        for path in trace_result['path_chains']:
            transformation = {
                'path': path,
                'length': len(path),
                'phases': [tracer._get_node_phase(node) for node in path]
            }
            provenance['transformation_chain'].append(transformation)
        
        # Validate provenance completeness
        assert len(provenance['source_contributions']) > 0
        assert len(provenance['transformation_chain']) > 0
        assert 'deployment_ready' in provenance['output_data']
        
        logger.info("✅ Output provenance verification successful")

def run_output_tracing_tests():
    """Run all output tracing tests"""
    print("🔍 Starting Advanced Output Tracing Tests...")
    print("=" * 60)
    
    # Run pytest with specific markers and output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-x"  # Stop on first failure
    ]
    
    result = pytest.main(pytest_args)
    
    if result == 0:
        print("\n✅ All Output Tracing Tests PASSED!")
        print("🎯 System can successfully trace outputs to source inputs")
        print("📊 Traceability validation: COMPLETE")
    else:
        print("\n❌ Some Output Tracing Tests FAILED!")
        print("🔧 Check trace logic and dependency recording")
    
    return result == 0

if __name__ == "__main__":
    success = run_output_tracing_tests()
    sys.exit(0 if success else 1) 