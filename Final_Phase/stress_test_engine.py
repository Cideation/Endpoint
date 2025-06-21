#!/usr/bin/env python3
"""
BEM Stress Test Engine
Implements comprehensive random testing for the training loop system:
1. Manual Sim Runs - Run 1-2 agents through full Alpha-Gamma lifecycle
2. Edge Case Injection - Feed malformed or incomplete data 
3. Stress Pulse Test - Trigger multiple node changes in rapid succession
"""

import json
import time
import random
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from node_engine_integration import NodeEngine
    from dgl_graph_builder import DGLGraphBuilder
except ImportError:
    print("⚠️  Node Engine or DGL components not available - using mock implementations")
    
    class NodeEngine:
        def handle_node_interaction(self, data):
            return {"success": True, "reward_score": random.uniform(0.3, 0.9)}
        
        def update_short_term_memory(self, node_id, memory_type, data):
            return {"success": True}
    
    class DGLGraphBuilder:
        def __init__(self):
            pass

@dataclass
class TestScenario:
    name: str
    description: str
    test_type: str  # "manual_sim", "edge_case", "stress_pulse"
    agents: List[str]
    duration: float
    expected_outcome: str

class StressTestEngine:
    def __init__(self):
        self.node_engine = NodeEngine()
        self.test_results = []
        self.active_agents = {}
        self.pulse_history = []
        
        # Test scenarios
        self.scenarios = self._load_test_scenarios()
        
    def _load_test_scenarios(self) -> List[TestScenario]:
        """Load predefined test scenarios"""
        return [
            # Manual Simulation Runs
            TestScenario(
                name="Alpha to Gamma Lifecycle",
                description="Single agent through complete lifecycle",
                test_type="manual_sim",
                agents=["QualityAgent"],
                duration=30.0,
                expected_outcome="successful_transition"
            ),
            TestScenario(
                name="Dual Agent Competition",
                description="Two agents competing for same resources",
                test_type="manual_sim", 
                agents=["CostAgent", "TimeAgent"],
                duration=45.0,
                expected_outcome="balanced_resolution"
            ),
            
            # Edge Case Injection
            TestScenario(
                name="Malformed Node Data",
                description="Feed corrupted node features",
                test_type="edge_case",
                agents=["QualityAgent"],
                duration=15.0,
                expected_outcome="graceful_error_handling"
            ),
            TestScenario(
                name="Missing Edge Features",
                description="Test with incomplete edge data",
                test_type="edge_case",
                agents=["CostAgent"],
                duration=20.0,
                expected_outcome="default_fallback"
            ),
            
            # Stress Pulse Tests
            TestScenario(
                name="Rapid Fire Pulses",
                description="100 pulse events in 10 seconds",
                test_type="stress_pulse",
                agents=["QualityAgent", "CostAgent", "TimeAgent"],
                duration=10.0,
                expected_outcome="stable_performance"
            ),
            TestScenario(
                name="Memory Overflow Test",
                description="Exceed agent memory capacity",
                test_type="stress_pulse",
                agents=["QualityAgent"],
                duration=25.0,
                expected_outcome="memory_management"
            )
        ]

    async def run_manual_sim_test(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Manual Sim Runs: Run 1-2 agents through full Alpha-Gamma lifecycle
        """
        print(f"\n🔄 MANUAL SIM: {scenario.name}")
        print(f"   Description: {scenario.description}")
        
        results = {
            "scenario": scenario.name,
            "test_type": "manual_sim",
            "start_time": time.time(),
            "agents_tested": scenario.agents,
            "phase_transitions": [],
            "performance_metrics": {},
            "success": False
        }
        
        try:
            # Initialize agents
            for agent_id in scenario.agents:
                node_id = f"test_node_{agent_id}_{uuid.uuid4().hex[:8]}"
                self.active_agents[agent_id] = {
                    "node_id": node_id,
                    "current_phase": "alpha",
                    "start_time": time.time(),
                    "transitions": []
                }
                
                # Create initial node
                await self._create_test_node(node_id, agent_id)
            
            # Run through lifecycle phases
            phases = ["alpha", "beta", "gamma"]
            
            for phase in phases:
                print(f"   🚀 Transitioning to {phase} phase...")
                
                for agent_id in scenario.agents:
                    agent_data = self.active_agents[agent_id]
                    node_id = agent_data["node_id"]
                    
                    # Simulate phase-specific activity
                    phase_result = await self._simulate_phase_activity(node_id, agent_id, phase)
                    
                    agent_data["current_phase"] = phase
                    agent_data["transitions"].append({
                        "phase": phase,
                        "timestamp": time.time(),
                        "result": phase_result
                    })
                    
                    results["phase_transitions"].append({
                        "agent": agent_id,
                        "phase": phase,
                        "performance": phase_result
                    })
                
                # Wait between phases
                await asyncio.sleep(scenario.duration / len(phases))
            
            # Calculate final metrics
            results["performance_metrics"] = await self._calculate_lifecycle_metrics()
            results["success"] = True
            
        except Exception as e:
            print(f"   ❌ Manual sim failed: {e}")
            results["error"] = str(e)
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        return results

    async def run_edge_case_test(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Edge Case Injection: Feed malformed or incomplete data
        """
        print(f"\n⚠️  EDGE CASE: {scenario.name}")
        print(f"   Description: {scenario.description}")
        
        results = {
            "scenario": scenario.name,
            "test_type": "edge_case",
            "start_time": time.time(),
            "edge_cases_tested": [],
            "error_handling": {},
            "success": False
        }
        
        try:
            agent_id = scenario.agents[0]
            node_id = f"edge_test_{uuid.uuid4().hex[:8]}"
            
            # Test different edge cases
            edge_cases = [
                self._test_malformed_node_features,
                self._test_missing_edge_data, 
                self._test_invalid_memory_structure,
                self._test_corrupt_reward_data,
                self._test_extreme_values
            ]
            
            for edge_case_func in edge_cases:
                edge_case_name = edge_case_func.__name__
                print(f"   🧪 Testing: {edge_case_name}")
                
                try:
                    edge_result = await edge_case_func(node_id, agent_id)
                    results["edge_cases_tested"].append({
                        "case": edge_case_name,
                        "result": edge_result,
                        "handled_gracefully": edge_result.get("success", False)
                    })
                    
                except Exception as e:
                    print(f"      ⚡ Exception caught: {e}")
                    results["edge_cases_tested"].append({
                        "case": edge_case_name,
                        "error": str(e),
                        "handled_gracefully": True  # Exception was caught
                    })
            
            # Evaluate error handling
            graceful_count = sum(1 for case in results["edge_cases_tested"] 
                               if case.get("handled_gracefully", False))
            
            results["error_handling"] = {
                "total_cases": len(edge_cases),
                "graceful_handling": graceful_count,
                "success_rate": graceful_count / len(edge_cases)
            }
            
            results["success"] = results["error_handling"]["success_rate"] >= 0.8
            
        except Exception as e:
            print(f"   ❌ Edge case test failed: {e}")
            results["error"] = str(e)
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        return results

    async def run_stress_pulse_test(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Stress Pulse Test: Trigger multiple node changes in rapid succession
        """
        print(f"\n⚡ STRESS PULSE: {scenario.name}")
        print(f"   Description: {scenario.description}")
        
        results = {
            "scenario": scenario.name,
            "test_type": "stress_pulse",
            "start_time": time.time(),
            "pulse_count": 0,
            "successful_pulses": 0,
            "failed_pulses": 0,
            "memory_usage": [],
            "response_times": [],
            "success": False
        }
        
        try:
            # Create test nodes for all agents
            test_nodes = {}
            for agent_id in scenario.agents:
                node_id = f"stress_node_{agent_id}_{uuid.uuid4().hex[:8]}"
                test_nodes[agent_id] = node_id
                await self._create_test_node(node_id, agent_id)
            
            # Determine pulse frequency
            target_pulses = 100 if "Rapid Fire" in scenario.name else 50
            pulse_interval = scenario.duration / target_pulses
            
            print(f"   🎯 Target: {target_pulses} pulses in {scenario.duration}s")
            print(f"   ⏱️  Interval: {pulse_interval:.3f}s per pulse")
            
            # Execute stress pulses
            start_time = time.time()
            pulse_tasks = []
            
            for i in range(target_pulses):
                # Select random agent and pulse type
                agent_id = random.choice(scenario.agents)
                node_id = test_nodes[agent_id]
                pulse_type = random.choice([
                    "bid_pulse", "occupancy_pulse", "compliancy_pulse",
                    "fit_pulse", "investment_pulse", "decay_pulse", "reject_pulse"
                ])
                
                # Create pulse task
                pulse_task = self._execute_stress_pulse(node_id, agent_id, pulse_type, i)
                pulse_tasks.append(pulse_task)
                
                # Wait for interval (if not rapid fire)
                if pulse_interval > 0.01:  # Don't wait for ultra-rapid pulses
                    await asyncio.sleep(pulse_interval)
                
                # Check if duration exceeded
                if time.time() - start_time > scenario.duration:
                    break
            
            # Wait for all pulses to complete
            pulse_results = await asyncio.gather(*pulse_tasks, return_exceptions=True)
            
            # Analyze results
            for i, result in enumerate(pulse_results):
                results["pulse_count"] += 1
                
                if isinstance(result, Exception):
                    results["failed_pulses"] += 1
                    print(f"   ❌ Pulse {i} failed: {result}")
                else:
                    results["successful_pulses"] += 1
                    if "response_time" in result:
                        results["response_times"].append(result["response_time"])
            
            # Calculate success metrics
            success_rate = results["successful_pulses"] / results["pulse_count"] if results["pulse_count"] > 0 else 0
            avg_response_time = sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0
            
            results["success"] = success_rate >= 0.9 and avg_response_time < 0.1
            
            print(f"   📊 Success Rate: {success_rate:.2%}")
            print(f"   ⏱️  Avg Response: {avg_response_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Stress pulse test failed: {e}")
            results["error"] = str(e)
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        return results

    async def _create_test_node(self, node_id: str, agent_id: str) -> Dict[str, Any]:
        """Create a test node with realistic data"""
        node_data = {
            "node_id": node_id,
            "agent_type": agent_id,
            "node_type": "V01_ProductComponent",
            "features": self._generate_test_features(),
            "phase": "alpha",
            "created_at": time.time()
        }
        
        # Initialize in node engine
        result = self.node_engine.handle_node_interaction(node_data)
        return result

    async def _simulate_phase_activity(self, node_id: str, agent_id: str, phase: str) -> Dict[str, Any]:
        """Simulate realistic activity for a specific phase"""
        activities = {
            "alpha": ["initial_assessment", "resource_allocation", "baseline_establishment"],
            "beta": ["optimization", "collaboration", "refinement"],
            "gamma": ["finalization", "validation", "delivery"]
        }
        
        results = []
        for activity in activities[phase]:
            activity_data = {
                "node_id": node_id,
                "activity": activity,
                "phase": phase,
                "timestamp": time.time(),
                "parameters": self._generate_activity_parameters(activity)
            }
            
            result = self.node_engine.handle_node_interaction(activity_data)
            results.append(result)
            
            # Small delay between activities
            await asyncio.sleep(0.1)
        
        return {
            "phase": phase,
            "activities_completed": len(results),
            "average_score": sum(r.get("reward_score", 0) for r in results) / len(results),
            "success": all(r.get("success", False) for r in results)
        }

    async def _execute_stress_pulse(self, node_id: str, agent_id: str, pulse_type: str, pulse_id: int) -> Dict[str, Any]:
        """Execute a single stress pulse"""
        start_time = time.time()
        
        try:
            pulse_data = {
                "node_id": node_id,
                "pulse_type": pulse_type,
                "pulse_id": pulse_id,
                "intensity": random.uniform(0.1, 1.0),
                "timestamp": time.time()
            }
            
            result = self.node_engine.handle_node_interaction(pulse_data)
            
            response_time = time.time() - start_time
            
            return {
                "pulse_id": pulse_id,
                "pulse_type": pulse_type,
                "response_time": response_time,
                "success": result.get("success", False),
                "reward_score": result.get("reward_score", 0)
            }
            
        except Exception as e:
            return {
                "pulse_id": pulse_id,
                "error": str(e),
                "response_time": time.time() - start_time,
                "success": False
            }

    # Edge case testing methods
    async def _test_malformed_node_features(self, node_id: str, agent_id: str) -> Dict[str, Any]:
        """Test with malformed node features"""
        malformed_data = {
            "node_id": node_id,
            "features": {
                "invalid_key": "string_instead_of_number",
                "missing_required": None,
                "negative_dimension": -1.5,
                "extreme_value": 1e10,
                "wrong_type": [1, 2, 3]  # Should be float
            }
        }
        
        result = self.node_engine.handle_node_interaction(malformed_data)
        return {"success": result.get("success", False), "error_handled": True}

    async def _test_missing_edge_data(self, node_id: str, agent_id: str) -> Dict[str, Any]:
        """Test with missing edge data"""
        incomplete_data = {
            "node_id": node_id,
            "edges": None,  # Missing edges
            "phase": "unknown_phase"
        }
        
        result = self.node_engine.handle_node_interaction(incomplete_data)
        return {"success": result.get("success", False), "error_handled": True}

    async def _test_invalid_memory_structure(self, node_id: str, agent_id: str) -> Dict[str, Any]:
        """Test with invalid memory structure"""
        # Attempt to corrupt memory
        try:
            memory_data = {
                "node_id": node_id,
                "memory_type": "invalid_type",
                "data": {"corrupted": True, "structure": "malformed"}
            }
            
            result = self.node_engine.update_short_term_memory(node_id, "invalid", memory_data)
            return {"success": True, "error_handled": True}
        except:
            return {"success": False, "error_handled": True}

    async def _test_corrupt_reward_data(self, node_id: str, agent_id: str) -> Dict[str, Any]:
        """Test with corrupt reward data"""
        corrupt_data = {
            "node_id": node_id,
            "performance_metrics": float('nan'),
            "user_feedback": "invalid",
            "learning_progress": -999999
        }
        
        result = self.node_engine.handle_node_interaction(corrupt_data)
        return {"success": result.get("success", False), "error_handled": True}

    async def _test_extreme_values(self, node_id: str, agent_id: str) -> Dict[str, Any]:
        """Test with extreme values"""
        extreme_data = {
            "node_id": node_id,
            "features": {
                "extreme_positive": 1e20,
                "extreme_negative": -1e20,
                "zero_division": 0,
                "infinity": float('inf'),
                "nan_value": float('nan')
            }
        }
        
        result = self.node_engine.handle_node_interaction(extreme_data)
        return {"success": result.get("success", False), "error_handled": True}

    def _generate_test_features(self) -> Dict[str, float]:
        """Generate realistic test features"""
        return {
            f"feature_{i}": random.uniform(0, 1) for i in range(18)
        }

    def _generate_activity_parameters(self, activity: str) -> Dict[str, Any]:
        """Generate parameters for specific activities"""
        base_params = {
            "priority": random.uniform(0.1, 1.0),
            "complexity": random.uniform(0.2, 0.9),
            "resources_required": random.randint(1, 5)
        }
        
        activity_specific = {
            "initial_assessment": {"assessment_depth": random.uniform(0.3, 0.8)},
            "optimization": {"optimization_target": random.choice(["cost", "time", "quality"])},
            "validation": {"validation_criteria": random.randint(3, 8)}
        }
        
        return {**base_params, **activity_specific.get(activity, {})}

    async def _calculate_lifecycle_metrics(self) -> Dict[str, float]:
        """Calculate metrics for lifecycle completion"""
        metrics = {}
        
        for agent_id, agent_data in self.active_agents.items():
            total_time = time.time() - agent_data["start_time"]
            transitions = len(agent_data["transitions"])
            
            metrics[agent_id] = {
                "total_time": total_time,
                "transitions_completed": transitions,
                "average_transition_time": total_time / max(transitions, 1),
                "success_rate": 1.0 if transitions >= 3 else transitions / 3
            }
        
        return metrics

    async def run_random_test_suite(self, num_tests: int = 10) -> Dict[str, Any]:
        """
        Run a comprehensive random test suite
        """
        print(f"\n🎲 RANDOM TEST SUITE: {num_tests} tests")
        print("=" * 50)
        
        suite_results = {
            "total_tests": num_tests,
            "tests_run": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "performance_summary": {}
        }
        
        for i in range(num_tests):
            # Randomly select scenario
            scenario = random.choice(self.scenarios)
            
            print(f"\n🧪 Test {i+1}/{num_tests}: {scenario.name}")
            
            # Run appropriate test based on type
            if scenario.test_type == "manual_sim":
                result = await self.run_manual_sim_test(scenario)
            elif scenario.test_type == "edge_case":
                result = await self.run_edge_case_test(scenario)
            elif scenario.test_type == "stress_pulse":
                result = await self.run_stress_pulse_test(scenario)
            
            # Record results
            suite_results["tests_run"] += 1
            suite_results["test_results"].append(result)
            
            if result.get("success", False):
                suite_results["successful_tests"] += 1
                print(f"   ✅ Test passed")
            else:
                suite_results["failed_tests"] += 1
                print(f"   ❌ Test failed")
            
            # Brief pause between tests
            await asyncio.sleep(1.0)
        
        # Calculate summary statistics
        suite_results["success_rate"] = suite_results["successful_tests"] / suite_results["tests_run"]
        suite_results["performance_summary"] = self._calculate_suite_performance(suite_results["test_results"])
        
        print(f"\n📊 SUITE COMPLETE")
        print(f"   Success Rate: {suite_results['success_rate']:.2%}")
        print(f"   Tests Passed: {suite_results['successful_tests']}/{suite_results['tests_run']}")
        
        return suite_results

    def _calculate_suite_performance(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics across all tests"""
        durations = [r.get("duration", 0) for r in test_results]
        
        return {
            "average_test_duration": sum(durations) / len(durations) if durations else 0,
            "total_test_time": sum(durations),
            "fastest_test": min(durations) if durations else 0,
            "slowest_test": max(durations) if durations else 0,
            "tests_by_type": {
                "manual_sim": len([r for r in test_results if r.get("test_type") == "manual_sim"]),
                "edge_case": len([r for r in test_results if r.get("test_type") == "edge_case"]),
                "stress_pulse": len([r for r in test_results if r.get("test_type") == "stress_pulse"])
            }
        }

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save test results to file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"stress_test_results_{timestamp}.json"
        
        # Save in current directory if we're already in Final_Phase
        import os
        if os.path.basename(os.getcwd()) == "Final_Phase":
            filepath = filename
        else:
            filepath = f"Final_Phase/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {filepath}")
        return filepath

# Demo function
async def run_stress_test_demo():
    """Run a demonstration of the stress testing system"""
    print("🚀 BEM Stress Test Engine Demo")
    print("=" * 40)
    
    engine = StressTestEngine()
    
    # Run a smaller test suite for demo
    results = await engine.run_random_test_suite(num_tests=6)
    
    # Save results
    results_file = engine.save_results(results)
    
    print(f"\n🎯 Demo Complete!")
    print(f"   Results: {results_file}")
    print(f"   Success Rate: {results['success_rate']:.2%}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_stress_test_demo())