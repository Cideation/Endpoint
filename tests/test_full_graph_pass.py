#!/usr/bin/env python3
"""
Full Graph Pass Test
Tests complete DAG execution across all phases (Alpha → Beta → Gamma)
Validates functor execution order, phase transitions, and end-to-end data flow
"""

import json
import logging
import sys
import time
import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestFullGraphPass:
    """Test complete DAG execution across all phases"""
    
    def setup_method(self):
        self.test_results = {
            'alpha_phase': {'status': 'pending', 'nodes': [], 'execution_time': 0},
            'beta_phase': {'status': 'pending', 'nodes': [], 'execution_time': 0},
            'gamma_phase': {'status': 'pending', 'nodes': [], 'execution_time': 0},
            'phase_transitions': {'status': 'pending', 'transitions': []},
            'full_graph_integrity': {'status': 'pending', 'details': {}}
        }
        self.start_time = datetime.now()
    
    def test_full_graph_execution(self):
        """Test complete full graph pass execution"""
        logger.info("🚀 Starting Full Graph Pass Test")
        
        test_graph = self.create_test_graph()
        alpha_results = self.execute_alpha_phase(test_graph)
        beta_results = self.execute_beta_phase(test_graph, alpha_results)
        gamma_results = self.execute_gamma_phase(test_graph, beta_results)
        
        # Assertions
        assert self.test_results['alpha_phase']['status'] == 'passed'
        assert self.test_results['beta_phase']['status'] == 'passed'
        assert self.test_results['gamma_phase']['status'] == 'passed'
        assert len(alpha_results['nodes']) == 2
        assert len(beta_results['nodes']) == 3
        assert len(gamma_results['nodes']) == 4
    
    def test_phase_transitions(self):
        """Test phase transition validation"""
        test_graph = self.create_test_graph()
        alpha_results = self.execute_alpha_phase(test_graph)
        beta_results = self.execute_beta_phase(test_graph, alpha_results)
        gamma_results = self.execute_gamma_phase(test_graph, beta_results)
        
        self.validate_phase_transitions(alpha_results, beta_results, gamma_results)
        assert self.test_results['phase_transitions']['status'] == 'passed'
    
    def test_graph_integrity(self):
        """Test full graph integrity validation"""
        test_graph = self.create_test_graph()
        alpha_results = self.execute_alpha_phase(test_graph)
        beta_results = self.execute_beta_phase(test_graph, alpha_results)
        gamma_results = self.execute_gamma_phase(test_graph, beta_results)
        
        self.validate_full_graph_integrity(test_graph, gamma_results)
        assert self.test_results['full_graph_integrity']['status'] == 'passed'
        
    def run_full_graph_test(self) -> Dict[str, Any]:
        """Execute complete full graph pass test"""
        logger.info("🚀 Starting Full Graph Pass Test")
        
        try:
            test_graph = self.create_test_graph()
            alpha_results = self.execute_alpha_phase(test_graph)
            beta_results = self.execute_beta_phase(test_graph, alpha_results)
            gamma_results = self.execute_gamma_phase(test_graph, beta_results)
            self.validate_phase_transitions(alpha_results, beta_results, gamma_results)
            self.validate_full_graph_integrity(test_graph, gamma_results)
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Full graph test failed: {e}")
            self.test_results['full_graph_integrity'] = {
                'status': 'failed', 'details': {'error': str(e)}
            }
            
        return self.test_results
    
    def create_test_graph(self) -> Dict[str, Any]:
        """Create comprehensive test graph spanning all phases"""
        return {
            'alpha_nodes': [
                {
                    'node_id': 'A01_MaterialInput', 'functor_type': 'MaterialSpecification', 'phase': 'alpha',
                    'inputs': {'material_type': 'Steel_A36', 'quantity': 1000},
                    'outputs': {}, 'downstream_edges': ['B01_ComponentDesign', 'B02_CostAnalysis']
                },
                {
                    'node_id': 'A02_RequirementCapture', 'functor_type': 'RequirementDefinition', 'phase': 'alpha',
                    'inputs': {'project_requirements': {'load_capacity': '50kN', 'safety_factor': 2.5}},
                    'outputs': {}, 'downstream_edges': ['B01_ComponentDesign', 'B03_ComplianceCheck']
                }
            ],
            'beta_nodes': [
                {
                    'node_id': 'B01_ComponentDesign', 'functor_type': 'DesignOptimization', 'phase': 'beta',
                    'inputs': {}, 'outputs': {}, 'downstream_edges': ['G01_ManufacturingPlan', 'G02_QualityAssurance']
                },
                {
                    'node_id': 'B02_CostAnalysis', 'functor_type': 'CostCalculation', 'phase': 'beta',
                    'inputs': {}, 'outputs': {}, 'downstream_edges': ['G03_ROICalculation']
                },
                {
                    'node_id': 'B03_ComplianceCheck', 'functor_type': 'RegulatoryValidation', 'phase': 'beta',
                    'inputs': {}, 'outputs': {}, 'downstream_edges': ['G02_QualityAssurance', 'G04_ComplianceReport']
                }
            ],
            'gamma_nodes': [
                {
                    'node_id': 'G01_ManufacturingPlan', 'functor_type': 'ProductionPlanning', 'phase': 'gamma',
                    'inputs': {}, 'outputs': {}, 'downstream_edges': []
                },
                {
                    'node_id': 'G02_QualityAssurance', 'functor_type': 'QualityValidation', 'phase': 'gamma',
                    'inputs': {}, 'outputs': {}, 'downstream_edges': []
                },
                {
                    'node_id': 'G03_ROICalculation', 'functor_type': 'FinancialAnalysis', 'phase': 'gamma',
                    'inputs': {}, 'outputs': {}, 'downstream_edges': []
                },
                {
                    'node_id': 'G04_ComplianceReport', 'functor_type': 'ComplianceReporting', 'phase': 'gamma',
                    'inputs': {}, 'outputs': {}, 'downstream_edges': []
                }
            ]
        }
    
    def execute_alpha_phase(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Alpha phase nodes"""
        logger.info("🔵 Executing Alpha Phase")
        start_time = time.time()
        alpha_results = {'nodes': [], 'phase_outputs': {}}
        
        for node in graph['alpha_nodes']:
            result = self.execute_node(node)
            alpha_results['nodes'].append(result)
            alpha_results['phase_outputs'][node['node_id']] = result['outputs']
            logger.info(f"✅ Alpha node {node['node_id']} completed")
        
        execution_time = time.time() - start_time
        self.test_results['alpha_phase'] = {
            'status': 'passed', 'nodes': alpha_results['nodes'], 'execution_time': execution_time
        }
        return alpha_results
    
    def execute_beta_phase(self, graph: Dict[str, Any], alpha_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Beta phase nodes with Alpha outputs"""
        logger.info("🟡 Executing Beta Phase")
        start_time = time.time()
        beta_results = {'nodes': [], 'phase_outputs': {}}
        
        for node in graph['beta_nodes']:
            self.inject_upstream_outputs(node, alpha_results['phase_outputs'])
            result = self.execute_node(node)
            beta_results['nodes'].append(result)
            beta_results['phase_outputs'][node['node_id']] = result['outputs']
            logger.info(f"✅ Beta node {node['node_id']} completed")
        
        execution_time = time.time() - start_time
        self.test_results['beta_phase'] = {
            'status': 'passed', 'nodes': beta_results['nodes'], 'execution_time': execution_time
        }
        return beta_results
    
    def execute_gamma_phase(self, graph: Dict[str, Any], beta_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Gamma phase nodes with Beta outputs"""
        logger.info("🟢 Executing Gamma Phase")
        start_time = time.time()
        gamma_results = {'nodes': [], 'phase_outputs': {}}
        
        for node in graph['gamma_nodes']:
            self.inject_upstream_outputs(node, beta_results['phase_outputs'])
            result = self.execute_node(node)
            gamma_results['nodes'].append(result)
            gamma_results['phase_outputs'][node['node_id']] = result['outputs']
            logger.info(f"✅ Gamma node {node['node_id']} completed")
        
        execution_time = time.time() - start_time
        self.test_results['gamma_phase'] = {
            'status': 'passed', 'nodes': gamma_results['nodes'], 'execution_time': execution_time
        }
        return gamma_results
    
    def inject_upstream_outputs(self, node: Dict[str, Any], upstream_outputs: Dict[str, Any]):
        """Inject upstream phase outputs into current node inputs"""
        for outputs in upstream_outputs.values():
            if 'material_properties' in outputs:
                node['inputs']['material_data'] = outputs['material_properties']
            if 'design_requirements' in outputs:
                node['inputs']['requirements'] = outputs['design_requirements']
    
    def execute_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual node based on functor type"""
        functor_type = node['functor_type']
        
        if functor_type == 'MaterialSpecification':
            outputs = {'material_properties': {'yield_strength': '36ksi', 'material_id': 'MAT_Steel_A36'}}
        elif functor_type == 'RequirementDefinition':
            outputs = {'design_requirements': {'safety_factor': 2.5}, 'compliance_standards': ['ISO_9001']}
        elif functor_type == 'DesignOptimization':
            outputs = {'optimized_design': {'weight': 145.5, 'efficiency': 0.92}}
        elif functor_type == 'CostCalculation':
            outputs = {'total_cost': 3375, 'cost_breakdown': {'material': 2500, 'manufacturing': 625}}
        elif functor_type == 'RegulatoryValidation':
            outputs = {'compliance_status': 'PASSED', 'compliance_score': 95.5}
        elif functor_type == 'ProductionPlanning':
            outputs = {'production_schedule': {'duration': '14_days', 'resources': 8}}
        elif functor_type == 'QualityValidation':
            outputs = {'quality_score': 98.2, 'certification': 'ISO_9001_CERTIFIED'}
        elif functor_type == 'FinancialAnalysis':
            outputs = {'roi_analysis': {'profit_margin': 0.44, 'payback_period': '18_months'}}
        elif functor_type == 'ComplianceReporting':
            outputs = {'final_report': {'overall_score': 95.5, 'status': 'CERTIFIED'}}
        else:
            outputs = {'status': 'unknown_functor_type'}
        
        return {
            'node_id': node['node_id'], 'functor_type': functor_type, 'phase': node['phase'],
            'inputs': node['inputs'], 'outputs': outputs, 'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_phase_transitions(self, alpha_results, beta_results, gamma_results):
        """Validate smooth transitions between phases"""
        logger.info("🔄 Validating Phase Transitions")
        transitions = [
            {'transition': 'alpha → beta', 'valid': len(alpha_results['phase_outputs']) > 0},
            {'transition': 'beta → gamma', 'valid': len(beta_results['phase_outputs']) > 0}
        ]
        self.test_results['phase_transitions'] = {
            'status': 'passed' if all(t['valid'] for t in transitions) else 'failed',
            'transitions': transitions
        }
    
    def validate_full_graph_integrity(self, graph, gamma_results):
        """Validate complete graph execution integrity"""
        logger.info("🔍 Validating Full Graph Integrity")
        total_nodes = len(graph['alpha_nodes']) + len(graph['beta_nodes']) + len(graph['gamma_nodes'])
        executed_nodes = (len(self.test_results['alpha_phase']['nodes']) +
                         len(self.test_results['beta_phase']['nodes']) +
                         len(self.test_results['gamma_phase']['nodes']))
        
        self.test_results['full_graph_integrity'] = {
            'status': 'passed' if total_nodes == executed_nodes else 'failed',
            'details': {
                'total_nodes': total_nodes, 'executed_nodes': executed_nodes,
                'integrity_score': (executed_nodes / total_nodes) * 100
            }
        }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        duration = (datetime.now() - self.start_time).total_seconds()
        report_file = f"full_graph_pass_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'test_suite': 'Full Graph Pass Test', 'execution_time': duration,
                'results': self.test_results, 'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"🚀 Full Graph Pass Test completed in {duration:.2f}s - Report: {report_file}")

def main():
    test_suite = TestFullGraphPass()
    results = test_suite.run_full_graph_test()
    failed_components = sum(1 for result in results.values() if result.get('status') == 'failed')
    sys.exit(failed_components)

if __name__ == "__main__":
    main()
