#!/usr/bin/env python3
"""
Agent Influence Test
Tests agent coefficients influence on node states
Validates agent decision-making and state modifications
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentImpactTest:
    """Test agent coefficients and their influence on node states"""
    
    def __init__(self):
        self.test_results = {
            'coefficient_influence': {'status': 'pending', 'details': {}},
            'agent_decision_making': {'status': 'pending', 'details': {}},
            'state_modification': {'status': 'pending', 'details': {}},
            'multi_agent_interaction': {'status': 'pending', 'details': {}}
        }
        self.start_time = datetime.now()
        
    def run_agent_tests(self) -> Dict[str, Any]:
        """Execute all agent influence tests"""
        logger.info("🤖 Starting Agent Impact Tests")
        
        try:
            # Create test agents and nodes
            test_agents = self.create_test_agents()
            test_nodes = self.create_test_nodes()
            
            # Test coefficient influence
            self.test_coefficient_influence(test_agents, test_nodes)
            
            # Test agent decision making
            self.test_agent_decision_making(test_agents, test_nodes)
            
            # Test state modification
            self.test_state_modification(test_agents, test_nodes)
            
            # Test multi-agent interaction
            self.test_multi_agent_interaction(test_agents, test_nodes)
            
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Agent impact test failed: {e}")
            
        return self.test_results
    
    def create_test_agents(self) -> List[Dict[str, Any]]:
        """Create test agent configurations"""
        return [
            {
                'agent_id': 'AGENT_COST_OPTIMIZER',
                'agent_type': 'CostOptimizationAgent',
                'coefficients': {
                    'cost_sensitivity': 0.8,
                    'quality_trade_off': 0.6,
                    'time_pressure': 0.4,
                    'risk_tolerance': 0.3
                },
                'influence_scope': ['material_selection', 'design_optimization', 'manufacturing_planning'],
                'decision_threshold': 0.7,
                'active': True
            },
            {
                'agent_id': 'AGENT_QUALITY_GUARDIAN',
                'agent_type': 'QualityAssuranceAgent',
                'coefficients': {
                    'quality_priority': 0.9,
                    'compliance_strictness': 0.85,
                    'safety_factor': 0.95,
                    'performance_requirement': 0.8
                },
                'influence_scope': ['quality_control', 'compliance_checking', 'safety_validation'],
                'decision_threshold': 0.85,
                'active': True
            },
            {
                'agent_id': 'AGENT_EFFICIENCY_MONITOR',
                'agent_type': 'EfficiencyMonitoringAgent',
                'coefficients': {
                    'time_efficiency': 0.7,
                    'resource_utilization': 0.75,
                    'process_optimization': 0.8,
                    'waste_reduction': 0.65
                },
                'influence_scope': ['process_optimization', 'resource_allocation', 'workflow_management'],
                'decision_threshold': 0.6,
                'active': True
            }
        ]
    
    def create_test_nodes(self) -> List[Dict[str, Any]]:
        """Create test node configurations"""
        return [
            {
                'node_id': 'N001_MaterialSelection',
                'functor_type': 'MaterialOptimization',
                'initial_state': {
                    'material_cost': 2500,
                    'quality_rating': 85,
                    'availability': 0.9,
                    'sustainability_score': 0.7
                },
                'agent_influences': [],
                'decision_factors': ['cost', 'quality', 'availability']
            },
            {
                'node_id': 'N002_DesignOptimization',
                'functor_type': 'DesignParameterization',
                'initial_state': {
                    'design_complexity': 0.8,
                    'performance_score': 92,
                    'manufacturing_feasibility': 0.85,
                    'cost_impact': 3200
                },
                'agent_influences': [],
                'decision_factors': ['performance', 'feasibility', 'cost']
            },
            {
                'node_id': 'N003_QualityControl',
                'functor_type': 'QualityValidation',
                'initial_state': {
                    'inspection_rigor': 0.8,
                    'compliance_level': 88,
                    'testing_coverage': 0.75,
                    'quality_cost': 450
                },
                'agent_influences': [],
                'decision_factors': ['compliance', 'coverage', 'cost']
            }
        ]
    
    def test_coefficient_influence(self, agents: List[Dict[str, Any]], nodes: List[Dict[str, Any]]):
        """Test how agent coefficients influence node states"""
        logger.info("📊 Testing Coefficient Influence")
        
        influence_results = []
        
        for agent in agents:
            for node in nodes:
                try:
                    # Check if agent has influence scope over node
                    if self.agent_influences_node(agent, node):
                        # Calculate influence
                        influence_result = self.calculate_agent_influence(agent, node)
                        
                        # Apply influence and measure state change
                        original_state = node['initial_state'].copy()
                        modified_state = self.apply_agent_influence(node, agent, influence_result)
                        
                        # Measure state change
                        state_change = self.measure_state_change(original_state, modified_state)
                        
                        influence_results.append({
                            'agent_id': agent['agent_id'],
                            'node_id': node['node_id'],
                            'influence_strength': influence_result['strength'],
                            'state_change_magnitude': state_change['magnitude'],
                            'affected_parameters': state_change['affected_parameters'],
                            'influence_valid': influence_result['strength'] > 0 and state_change['magnitude'] > 0
                        })
                        
                except Exception as e:
                    influence_results.append({
                        'agent_id': agent['agent_id'],
                        'node_id': node['node_id'],
                        'influence_valid': False,
                        'error': str(e)
                    })
        
        valid_influences = sum(1 for result in influence_results if result.get('influence_valid'))
        
        self.test_results['coefficient_influence'] = {
            'status': 'passed' if valid_influences > 0 else 'failed',
            'details': {
                'total_influences': len(influence_results),
                'valid_influences': valid_influences,
                'influence_results': influence_results
            }
        }
    
    def agent_influences_node(self, agent: Dict[str, Any], node: Dict[str, Any]) -> bool:
        """Check if agent has influence scope over node"""
        agent_scope = agent.get('influence_scope', [])
        node_type = node.get('functor_type', '').lower()
        
        # Check scope overlap
        for scope in agent_scope:
            if scope.lower() in node_type or any(factor in scope for factor in node.get('decision_factors', [])):
                return True
        
        return False
    
    def calculate_agent_influence(self, agent: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate agent influence on node"""
        coefficients = agent.get('coefficients', {})
        decision_threshold = agent.get('decision_threshold', 0.5)
        
        # Calculate weighted influence based on coefficients
        influence_factors = []
        
        for coeff_name, coeff_value in coefficients.items():
            if coeff_value > decision_threshold:
                influence_factors.append({
                    'factor': coeff_name,
                    'value': coeff_value,
                    'weight': coeff_value - decision_threshold
                })
        
        # Calculate overall influence strength
        if influence_factors:
            total_weight = sum(factor['weight'] for factor in influence_factors)
            influence_strength = min(1.0, total_weight)
        else:
            influence_strength = 0.0
        
        return {
            'strength': influence_strength,
            'factors': influence_factors,
            'threshold_met': len(influence_factors) > 0
        }
    
    def apply_agent_influence(self, node: Dict[str, Any], agent: Dict[str, Any], 
                            influence: Dict[str, Any]) -> Dict[str, Any]:
        """Apply agent influence to node state"""
        modified_state = node['initial_state'].copy()
        influence_strength = influence['strength']
        
        agent_type = agent['agent_type']
        
        # Apply influence based on agent type
        if agent_type == 'CostOptimizationAgent':
            if 'material_cost' in modified_state:
                # Cost optimization reduces cost
                reduction_factor = influence_strength * 0.15
                modified_state['material_cost'] *= (1 - reduction_factor)
            
            if 'cost_impact' in modified_state:
                reduction_factor = influence_strength * 0.12
                modified_state['cost_impact'] *= (1 - reduction_factor)
                
        elif agent_type == 'QualityAssuranceAgent':
            if 'quality_rating' in modified_state:
                # Quality agent improves quality
                improvement_factor = influence_strength * 0.1
                modified_state['quality_rating'] *= (1 + improvement_factor)
            
            if 'compliance_level' in modified_state:
                improvement_factor = influence_strength * 0.08
                modified_state['compliance_level'] *= (1 + improvement_factor)
                
        elif agent_type == 'EfficiencyMonitoringAgent':
            if 'manufacturing_feasibility' in modified_state:
                # Efficiency agent improves feasibility
                improvement_factor = influence_strength * 0.1
                modified_state['manufacturing_feasibility'] = min(1.0, 
                    modified_state['manufacturing_feasibility'] * (1 + improvement_factor))
            
            if 'resource_utilization' in modified_state:
                improvement_factor = influence_strength * 0.12
                modified_state['resource_utilization'] = min(1.0,
                    modified_state.get('resource_utilization', 0.8) * (1 + improvement_factor))
        
        return modified_state
    
    def measure_state_change(self, original_state: Dict[str, Any], 
                           modified_state: Dict[str, Any]) -> Dict[str, Any]:
        """Measure the magnitude of state change"""
        affected_parameters = []
        total_change = 0
        
        for key in original_state:
            if key in modified_state:
                original_val = original_state[key]
                modified_val = modified_state[key]
                
                if isinstance(original_val, (int, float)) and isinstance(modified_val, (int, float)):
                    if original_val != 0:
                        change_percent = abs((modified_val - original_val) / original_val)
                        if change_percent > 0.01:  # 1% threshold
                            affected_parameters.append({
                                'parameter': key,
                                'original': original_val,
                                'modified': modified_val,
                                'change_percent': change_percent
                            })
                            total_change += change_percent
        
        return {
            'magnitude': total_change,
            'affected_parameters': affected_parameters,
            'parameter_count': len(affected_parameters)
        }
    
    def test_agent_decision_making(self, agents: List[Dict[str, Any]], nodes: List[Dict[str, Any]]):
        """Test agent decision-making processes"""
        logger.info("🧠 Testing Agent Decision Making")
        
        decision_results = []
        
        for agent in agents:
            try:
                # Test decision scenarios
                scenarios = self.create_decision_scenarios()
                
                for scenario in scenarios:
                    decision = self.simulate_agent_decision(agent, scenario)
                    
                    decision_results.append({
                        'agent_id': agent['agent_id'],
                        'scenario': scenario['name'],
                        'decision_made': decision['decision_made'],
                        'decision_quality': decision['quality_score'],
                        'response_time': decision['response_time'],
                        'valid_decision': decision['decision_made'] and decision['quality_score'] > 0.6
                    })
                    
            except Exception as e:
                decision_results.append({
                    'agent_id': agent['agent_id'],
                    'valid_decision': False,
                    'error': str(e)
                })
        
        valid_decisions = sum(1 for result in decision_results if result.get('valid_decision'))
        
        self.test_results['agent_decision_making'] = {
            'status': 'passed' if valid_decisions > 0 else 'failed',
            'details': {
                'total_decisions': len(decision_results),
                'valid_decisions': valid_decisions,
                'decision_results': decision_results
            }
        }
    
    def create_decision_scenarios(self) -> List[Dict[str, Any]]:
        """Create decision scenarios for testing"""
        return [
            {
                'name': 'cost_vs_quality_tradeoff',
                'parameters': {
                    'cost_increase': 0.2,
                    'quality_improvement': 0.15,
                    'time_impact': 0.1
                }
            },
            {
                'name': 'compliance_requirement_change',
                'parameters': {
                    'new_compliance_level': 98,
                    'current_compliance': 88,
                    'implementation_cost': 800
                }
            },
            {
                'name': 'resource_constraint',
                'parameters': {
                    'available_resources': 0.7,
                    'required_resources': 1.0,
                    'alternative_approach_efficiency': 0.85
                }
            }
        ]
    
    def simulate_agent_decision(self, agent: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent decision-making process"""
        start_time = time.time()
        
        agent_type = agent['agent_type']
        coefficients = agent['coefficients']
        decision_threshold = agent['decision_threshold']
        
        # Calculate decision score based on agent type and scenario
        if scenario['name'] == 'cost_vs_quality_tradeoff':
            if agent_type == 'CostOptimizationAgent':
                decision_score = coefficients.get('cost_sensitivity', 0.5) * 0.8
            elif agent_type == 'QualityAssuranceAgent':
                decision_score = coefficients.get('quality_priority', 0.5) * 0.9
            else:
                decision_score = 0.6
                
        elif scenario['name'] == 'compliance_requirement_change':
            if agent_type == 'QualityAssuranceAgent':
                decision_score = coefficients.get('compliance_strictness', 0.5) * 0.95
            else:
                decision_score = 0.5
                
        elif scenario['name'] == 'resource_constraint':
            if agent_type == 'EfficiencyMonitoringAgent':
                decision_score = coefficients.get('resource_utilization', 0.5) * 0.85
            else:
                decision_score = 0.4
        else:
            decision_score = 0.5
        
        decision_made = decision_score > decision_threshold
        response_time = time.time() - start_time
        
        return {
            'decision_made': decision_made,
            'quality_score': decision_score,
            'response_time': response_time,
            'scenario_handled': True
        }
    
    def test_state_modification(self, agents: List[Dict[str, Any]], nodes: List[Dict[str, Any]]):
        """Test agent state modification capabilities"""
        logger.info("🔧 Testing State Modification")
        
        modification_results = []
        
        for node in nodes:
            try:
                original_state = node['initial_state'].copy()
                
                # Apply multiple agent influences
                current_state = original_state.copy()
                applied_agents = []
                
                for agent in agents:
                    if self.agent_influences_node(agent, node):
                        influence = self.calculate_agent_influence(agent, node)
                        if influence['strength'] > 0:
                            current_state = self.apply_agent_influence(
                                {'initial_state': current_state}, agent, influence
                            )['initial_state']
                            applied_agents.append(agent['agent_id'])
                
                # Measure cumulative modification
                cumulative_change = self.measure_state_change(original_state, current_state)
                
                modification_results.append({
                    'node_id': node['node_id'],
                    'applied_agents': applied_agents,
                    'cumulative_change': cumulative_change['magnitude'],
                    'parameters_modified': len(cumulative_change['affected_parameters']),
                    'modification_successful': cumulative_change['magnitude'] > 0 and len(applied_agents) > 0
                })
                
            except Exception as e:
                modification_results.append({
                    'node_id': node['node_id'],
                    'modification_successful': False,
                    'error': str(e)
                })
        
        successful_modifications = sum(1 for result in modification_results 
                                     if result.get('modification_successful'))
        
        self.test_results['state_modification'] = {
            'status': 'passed' if successful_modifications > 0 else 'failed',
            'details': {
                'total_nodes': len(modification_results),
                'successful_modifications': successful_modifications,
                'modification_results': modification_results
            }
        }
    
    def test_multi_agent_interaction(self, agents: List[Dict[str, Any]], nodes: List[Dict[str, Any]]):
        """Test multi-agent interaction and conflict resolution"""
        logger.info("🤝 Testing Multi-Agent Interaction")
        
        interaction_results = []
        
        # Test agent conflicts
        conflict_scenarios = [
            {
                'agents': ['AGENT_COST_OPTIMIZER', 'AGENT_QUALITY_GUARDIAN'],
                'conflict_type': 'cost_vs_quality',
                'node_target': 'N001_MaterialSelection'
            },
            {
                'agents': ['AGENT_QUALITY_GUARDIAN', 'AGENT_EFFICIENCY_MONITOR'],
                'conflict_type': 'quality_vs_efficiency',
                'node_target': 'N002_DesignOptimization'
            }
        ]
        
        for scenario in conflict_scenarios:
            try:
                # Get involved agents
                involved_agents = [agent for agent in agents 
                                 if agent['agent_id'] in scenario['agents']]
                
                # Get target node
                target_node = next((node for node in nodes 
                                  if node['node_id'] == scenario['node_target']), None)
                
                if target_node and len(involved_agents) >= 2:
                    # Simulate conflict resolution
                    resolution = self.simulate_conflict_resolution(involved_agents, target_node, scenario)
                    
                    interaction_results.append({
                        'scenario': scenario['conflict_type'],
                        'agents_involved': scenario['agents'],
                        'conflict_resolved': resolution['resolved'],
                        'resolution_method': resolution['method'],
                        'final_decision': resolution['decision'],
                        'interaction_successful': resolution['resolved']
                    })
                    
            except Exception as e:
                interaction_results.append({
                    'scenario': scenario['conflict_type'],
                    'interaction_successful': False,
                    'error': str(e)
                })
        
        successful_interactions = sum(1 for result in interaction_results 
                                    if result.get('interaction_successful'))
        
        self.test_results['multi_agent_interaction'] = {
            'status': 'passed' if successful_interactions > 0 else 'failed',
            'details': {
                'total_interactions': len(interaction_results),
                'successful_interactions': successful_interactions,
                'interaction_results': interaction_results
            }
        }
    
    def simulate_conflict_resolution(self, agents: List[Dict[str, Any]], 
                                   node: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate conflict resolution between agents"""
        
        # Calculate agent priorities
        agent_priorities = []
        for agent in agents:
            coefficients = agent['coefficients']
            
            if scenario['conflict_type'] == 'cost_vs_quality':
                if agent['agent_type'] == 'CostOptimizationAgent':
                    priority = coefficients.get('cost_sensitivity', 0.5)
                elif agent['agent_type'] == 'QualityAssuranceAgent':
                    priority = coefficients.get('quality_priority', 0.5)
                else:
                    priority = 0.5
            else:
                priority = 0.5
            
            agent_priorities.append({
                'agent_id': agent['agent_id'],
                'priority': priority
            })
        
        # Resolve conflict (highest priority wins)
        winning_agent = max(agent_priorities, key=lambda x: x['priority'])
        
        return {
            'resolved': True,
            'method': 'priority_based',
            'decision': f"Agent {winning_agent['agent_id']} decision applied",
            'winning_agent': winning_agent['agent_id']
        }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        duration = (datetime.now() - self.start_time).total_seconds()
        report_file = f"agent_impact_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'test_suite': 'Agent Impact Test',
                'execution_time': duration,
                'results': self.test_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"🤖 Agent Impact Test completed in {duration:.2f}s - Report: {report_file}")

def main():
    test_suite = AgentImpactTest()
    results = test_suite.run_agent_tests()
    failed_tests = sum(1 for result in results.values() if result.get('status') == 'failed')
    sys.exit(failed_tests)

if __name__ == "__main__":
    main()
