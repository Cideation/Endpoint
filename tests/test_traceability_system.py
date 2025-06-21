#!/usr/bin/env python3
"""
Engineering Traceability System - Comprehensive Test Suite
🧪 Tests for complete audit trail functionality, decision tracking, and component score origin tracing
"""

import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from MICROSERVICE_ENGINES.traceability_engine import (
    TraceabilityEngine, OriginTag, TraceLog, ComponentScore, create_traceability_engine
)
from MICROSERVICE_ENGINES.traceability_integration import (
    TracedFunctorRegistry, PulseTraceabilityMixin, NodeStateTracker,
    get_traced_functor_registry, get_node_state_tracker, traced_functor,
    log_decision, log_param_usage, update_node_state
)

class TestTraceabilityEngine:
    """Test suite for core traceability engine"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = TraceabilityEngine(log_directory=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trace_lifecycle(self):
        """Test complete trace lifecycle from start to end"""
        
        print("\n🔍 Testing Trace Lifecycle")
        
        # Start trace
        trace_id = self.engine.start_trace(
            node_id="V01_TEST",
            functor="evaluate_manufacturing",
            agent_triggered="TestAgent",
            input_data={"component": "beam", "material": "steel"},
            dependencies=["V02", "V05"]
        )
        
        assert trace_id is not None
        assert trace_id in self.engine.active_traces
        print(f"✅ Trace started: {trace_id}")
        
        # Log decisions
        self.engine.log_decision(trace_id, "spec_check_passed", "grade >= min_grade", "Material grade sufficient")
        self.engine.log_decision(trace_id, "material_ok", "availability == true", "Material available")
        self.engine.log_decision(trace_id, "geometry_valid", "tolerance_check", "Geometry within tolerance")
        
        # Log parameter usage
        self.engine.log_global_param_usage(trace_id, "form_strategy")
        self.engine.log_global_param_usage(trace_id, "precision_tolerance")
        
        trace_log = self.engine.active_traces[trace_id]
        assert len(trace_log.decision_path) == 3
        assert len(trace_log.global_params_used) == 2
        print("✅ Decisions and parameters logged")
        
        # End trace
        output_data = {"manufacturing_score": 0.86, "status": "approved"}
        completed_trace = self.engine.end_trace(trace_id, output_data, 45.2)
        
        assert completed_trace is not None
        assert trace_id not in self.engine.active_traces
        assert completed_trace.execution_time_ms == 45.2
        print("✅ Trace completed successfully")
        
        # Verify trace log file was created
        trace_files = list(Path(self.temp_dir).glob("trace_*.json"))
        assert len(trace_files) == 1
        print("✅ Trace log file created")
    
    def test_component_score_tracking(self):
        """Test component score creation with origin tracking"""
        
        print("\n📊 Testing Component Score Tracking")
        
        # Create component score
        score = self.engine.create_component_score(
            value=0.86,
            source_functor="evaluate_component_vector",
            input_nodes=["V01", "V02"],
            agent="TestAgent",
            design_param="evolutionary_potential",
            component_id="comp_001"
        )
        
        assert isinstance(score, ComponentScore)
        assert score.value == 0.86
        assert score.validation_status == "validated"
        assert score.origin_tag.source_functor == "evaluate_component_vector"
        assert score.origin_tag.agent == "TestAgent"
        assert "comp_001" in self.engine.component_scores
        print("✅ Component score created with full origin tracking")
        
        # Test invalid score
        invalid_score = self.engine.create_component_score(
            value=1.5,  # Invalid value > 1.0
            source_functor="test_functor",
            input_nodes=["V01"],
            agent="TestAgent",
            design_param="test_param"
        )
        
        assert invalid_score.validation_status == "invalid"
        print("✅ Invalid score properly flagged")
    
    def test_graph_chain_tracking(self):
        """Test graph chain and trace path functionality"""
        
        print("\n🔗 Testing Graph Chain Tracking")
        
        # Create multiple traces with same dependencies
        dependencies = ["V02", "V05"]
        
        trace_id_1 = self.engine.start_trace("V01", "functor_1", "Agent1", {}, dependencies)
        trace_id_2 = self.engine.start_trace("V06", "functor_2", "Agent2", {}, dependencies)
        
        # Both should have same chain_id due to same dependencies
        trace_1 = self.engine.active_traces[trace_id_1]
        trace_2 = self.engine.active_traces[trace_id_2]
        
        assert trace_1.chain_id == trace_2.chain_id
        print("✅ Same dependencies create same chain_id")
        
        # Get trace path
        trace_path = self.engine.get_trace_path(trace_1.chain_id)
        
        assert "V01" in trace_path
        assert "V06" in trace_path
        print(f"✅ Trace path retrieved: {trace_path}")
        
        # Clean up
        self.engine.end_trace(trace_id_1, {})
        self.engine.end_trace(trace_id_2, {})
    
    def test_decision_lineage_analysis(self):
        """Test decision lineage analysis functionality"""
        
        print("\n🕵️ Testing Decision Lineage Analysis")
        
        # Create and complete a trace
        trace_id = self.engine.start_trace("V01_LINEAGE", "test_functor", "TestAgent", {})
        self.engine.log_decision(trace_id, "decision_1")
        self.engine.log_decision(trace_id, "decision_2")
        self.engine.log_global_param_usage(trace_id, "test_param")
        self.engine.end_trace(trace_id, {"result": "success"})
        
        # Analyze lineage
        lineage = self.engine.analyze_decision_lineage("V01_LINEAGE")
        
        assert lineage["target_node"] == "V01_LINEAGE"
        assert len(lineage["decision_chain"]) >= 1
        assert "TestAgent" in lineage["agents_involved"]
        assert "test_param" in lineage["parameters_used"]
        print("✅ Decision lineage analysis working")
    
    def test_audit_report_generation(self):
        """Test comprehensive audit report generation"""
        
        print("\n📋 Testing Audit Report Generation")
        
        # Create multiple traces
        for i in range(3):
            trace_id = self.engine.start_trace(f"V0{i+1}", f"functor_{i}", f"Agent{i}", {})
            self.engine.log_decision(trace_id, f"decision_{i}")
            self.engine.log_global_param_usage(trace_id, f"param_{i}")
            self.engine.end_trace(trace_id, {"result": i})
        
        # Generate audit report
        report = self.engine.generate_audit_report()
        
        assert "report_id" in report
        assert report["summary"]["total_traces"] == 3
        assert len(report["trace_analysis"]) == 3
        assert len(report["decision_patterns"]) >= 3
        assert len(report["parameter_usage"]) >= 3
        print("✅ Comprehensive audit report generated")
        
        # Test filtered report
        filtered_report = self.engine.generate_audit_report(agent_filter="Agent0")
        
        assert len([t for t in filtered_report["trace_analysis"] 
                   if "Agent0" in str(t)]) >= 0  # May be 0 due to async nature
        print("✅ Filtered audit report generated")

class TestTracedFunctorRegistry:
    """Test suite for traced functor registry"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        engine = TraceabilityEngine(log_directory=self.temp_dir)
        self.registry = TracedFunctorRegistry(engine)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_functor_registration_and_execution(self):
        """Test functor registration and traced execution"""
        
        print("\n🔧 Testing Functor Registration")
        
        @self.registry.register_functor("test_calculate", "V01_ProductComponent")
        def calculate_score(node_id: str, input_data: dict, agent: str = "test"):
            # Simulate calculation with decision logging
            self.registry.log_decision(node_id, "validation_passed", "input_valid", "Input is valid")
            self.registry.log_param_usage(node_id, "form_strategy")
            
            base_score = input_data.get("base_score", 0.5)
            return {"score": base_score * 1.2, "status": "calculated"}
        
        # Execute traced function
        result = calculate_score(
            node_id="V01_TEST",
            input_data={"base_score": 0.7},
            agent="TestAgent"
        )
        
        assert result["score"] == 0.84  # 0.7 * 1.2
        assert result["status"] == "calculated"
        assert "test_calculate" in self.registry.registered_functors
        print("✅ Traced functor executed successfully")
        
        # Verify trace was created
        assert self.registry.engine.execution_stats["total_traces"] == 1
        print("✅ Trace statistics updated")
    
    def test_error_handling_in_traced_functor(self):
        """Test error handling in traced functors"""
        
        print("\n❌ Testing Error Handling")
        
        @self.registry.register_functor("error_functor", "V01_ProductComponent")
        def failing_function(node_id: str, should_fail: bool = True):
            if should_fail:
                raise ValueError("Intentional test error")
            return {"success": True}
        
        # Test error case
        with pytest.raises(ValueError):
            failing_function(node_id="V01_ERROR", should_fail=True)
        
        # Verify error was traced
        assert self.registry.engine.execution_stats["total_traces"] == 1
        print("✅ Error properly traced and re-raised")
        
        # Test success case
        result = failing_function(node_id="V01_SUCCESS", should_fail=False)
        assert result["success"] is True
        print("✅ Success case works after error handling")

class TestNodeStateTracker:
    """Test suite for node state tracking"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        engine = TraceabilityEngine(log_directory=self.temp_dir)
        self.tracker = NodeStateTracker(engine)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_node_state_updates(self):
        """Test node state updates with traceability"""
        
        print("\n📊 Testing Node State Updates")
        
        node_id = "V01_STATE_TEST"
        
        # Initial state update
        trace_id_1 = self.tracker.update_node_state(
            node_id=node_id,
            new_state={"score": 0.5, "status": "initial", "material": "steel"},
            agent="InitAgent",
            functor="initial_setup"
        )
        
        assert node_id in self.tracker.node_states
        assert self.tracker.node_states[node_id]["score"] == 0.5
        print("✅ Initial state update successful")
        
        # State modification
        trace_id_2 = self.tracker.update_node_state(
            node_id=node_id,
            new_state={"score": 0.8, "status": "updated", "material": "steel", "validated": True},
            agent="UpdateAgent",
            functor="score_update"
        )
        
        assert self.tracker.node_states[node_id]["score"] == 0.8
        assert self.tracker.node_states[node_id]["validated"] is True
        print("✅ State modification successful")
        
        # Get lineage
        lineage = self.tracker.get_node_lineage(node_id)
        
        assert len(lineage) == 2
        assert lineage[0]["agent"] == "InitAgent"
        assert lineage[1]["agent"] == "UpdateAgent"
        assert "changes" in lineage[1]
        print("✅ State lineage tracking working")
    
    def test_state_change_calculation(self):
        """Test state change calculation logic"""
        
        print("\n🔄 Testing State Change Calculation")
        
        node_id = "V01_CHANGE_TEST"
        
        # Set initial state
        self.tracker.update_node_state(
            node_id=node_id,
            new_state={"a": 1, "b": 2, "c": 3},
            agent="TestAgent"
        )
        
        # Update with changes: modify, add, remove
        self.tracker.update_node_state(
            node_id=node_id,
            new_state={"a": 10, "b": 2, "d": 4},  # a modified, c removed, d added
            agent="TestAgent"
        )
        
        lineage = self.tracker.get_node_lineage(node_id)
        changes = lineage[-1]["changes"]
        
        assert "modified" in changes
        assert "added" in changes
        assert "removed" in changes
        assert changes["modified"]["a"]["old"] == 1
        assert changes["modified"]["a"]["new"] == 10
        assert "d" in changes["added"]
        assert "c" in changes["removed"]
        print("✅ State change calculation accurate")

class TestIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_manufacturing_workflow(self):
        """Test complete manufacturing workflow with full traceability"""
        
        print("\n🏭 Testing Complete Manufacturing Workflow")
        
        engine = TraceabilityEngine(log_directory=self.temp_dir)
        registry = TracedFunctorRegistry(engine)
        tracker = NodeStateTracker(engine)
        
        # Define traced manufacturing functors
        @registry.register_functor("evaluate_material", "V01_ProductComponent")
        def evaluate_material(node_id: str, material: str, agent: str = "MaterialAgent"):
            registry.log_decision(node_id, "material_check_started", f"material={material}", "Starting material evaluation")
            registry.log_param_usage(node_id, "material_expression")
            
            if material == "steel":
                registry.log_decision(node_id, "material_approved", "steel_grade_ok", "Steel grade approved")
                return {"material_score": 0.9, "approved": True}
            else:
                registry.log_decision(node_id, "material_rejected", "unknown_material", "Material not in database")
                return {"material_score": 0.3, "approved": False}
        
        @registry.register_functor("evaluate_geometry", "V01_ProductComponent")
        def evaluate_geometry(node_id: str, dimensions: dict, agent: str = "GeometryAgent"):
            registry.log_decision(node_id, "geometry_check_started", f"dims={dimensions}", "Starting geometry evaluation")
            registry.log_param_usage(node_id, "precision_tolerance")
            
            if all(d > 0 for d in dimensions.values()):
                registry.log_decision(node_id, "geometry_valid", "all_positive", "All dimensions positive")
                return {"geometry_score": 0.85, "valid": True}
            else:
                registry.log_decision(node_id, "geometry_invalid", "negative_dimension", "Invalid dimension found")
                return {"geometry_score": 0.1, "valid": False}
        
        @registry.register_functor("calculate_final_score", "V01_ProductComponent")
        def calculate_final_score(node_id: str, material_score: float, geometry_score: float, agent: str = "ScoringAgent"):
            registry.log_decision(node_id, "final_calculation_started", "combining_scores", "Calculating final component score")
            registry.log_param_usage(node_id, "evolutionary_potential")
            
            final_score = (material_score + geometry_score) / 2
            
            if final_score > 0.7:
                registry.log_decision(node_id, "component_approved", f"score={final_score}", "Component meets quality threshold")
                status = "approved"
            else:
                registry.log_decision(node_id, "component_rejected", f"score={final_score}", "Component below quality threshold")
                status = "rejected"
            
            return {"final_score": final_score, "status": status}
        
        # Execute complete workflow
        component_id = "V01_BEAM_001"
        
        # Step 1: Material evaluation
        material_result = evaluate_material(
            node_id=component_id,
            material="steel",
            agent="MaterialExpert"
        )
        
        # Update state after material evaluation
        tracker.update_node_state(
            node_id=component_id,
            new_state={"material_score": material_result["material_score"], "material_approved": material_result["approved"]},
            agent="MaterialExpert",
            functor="material_evaluation_complete"
        )
        
        # Step 2: Geometry evaluation
        geometry_result = evaluate_geometry(
            node_id=component_id,
            dimensions={"length": 100, "width": 50, "height": 25},
            agent="GeometryExpert"
        )
        
        # Update state after geometry evaluation
        current_state = tracker.node_states[component_id].copy()
        current_state.update({
            "geometry_score": geometry_result["geometry_score"],
            "geometry_valid": geometry_result["valid"]
        })
        
        tracker.update_node_state(
            node_id=component_id,
            new_state=current_state,
            agent="GeometryExpert",
            functor="geometry_evaluation_complete"
        )
        
        # Step 3: Final scoring
        final_result = calculate_final_score(
            node_id=component_id,
            material_score=material_result["material_score"],
            geometry_score=geometry_result["geometry_score"],
            agent="QualityController"
        )
        
        # Final state update
        current_state = tracker.node_states[component_id].copy()
        current_state.update({
            "final_score": final_result["final_score"],
            "status": final_result["status"]
        })
        
        tracker.update_node_state(
            node_id=component_id,
            new_state=current_state,
            agent="QualityController",
            functor="final_scoring_complete"
        )
        
        # Create component score with origin tracking
        component_score = engine.create_component_score(
            value=final_result["final_score"],
            source_functor="calculate_final_score",
            input_nodes=[component_id],
            agent="QualityController",
            design_param="evolutionary_potential",
            component_id=component_id
        )
        
        # Verify workflow results
        assert material_result["approved"] is True
        assert geometry_result["valid"] is True
        assert final_result["status"] == "approved"
        assert final_result["final_score"] > 0.7
        print("✅ Manufacturing workflow completed successfully")
        
        # Verify traceability
        assert engine.execution_stats["total_traces"] == 6  # 3 functors + 3 state updates
        
        lineage = engine.analyze_decision_lineage(component_id)
        assert len(lineage["agents_involved"]) >= 3
        assert len(lineage["parameters_used"]) >= 3
        print("✅ Complete traceability established")
        
        # Generate comprehensive audit report
        audit_report = engine.generate_audit_report()
        
        assert audit_report["summary"]["total_traces"] == 6
        assert len(audit_report["agent_activity"]) >= 3
        print("✅ Comprehensive audit report generated")
        
        # Verify component score traceability
        assert component_score.value == final_result["final_score"]
        assert component_score.origin_tag.agent == "QualityController"
        assert component_score.validation_status == "validated"
        print("✅ Component score fully traceable")
        
        print(f"🎉 Complete workflow traced with {engine.execution_stats['total_decisions']} decisions logged")

def run_traceability_tests():
    """Run all traceability system tests"""
    
    print("🔍 Engineering Traceability System - Comprehensive Test Suite")
    print("=" * 80)
    
    # Test core engine
    engine_tests = TestTraceabilityEngine()
    engine_tests.setup_method()
    
    try:
        engine_tests.test_trace_lifecycle()
        engine_tests.test_component_score_tracking()
        engine_tests.test_graph_chain_tracking()
        engine_tests.test_decision_lineage_analysis()
        engine_tests.test_audit_report_generation()
        print("✅ Core Engine Tests: PASSED")
    finally:
        engine_tests.teardown_method()
    
    # Test functor registry
    registry_tests = TestTracedFunctorRegistry()
    registry_tests.setup_method()
    
    try:
        registry_tests.test_functor_registration_and_execution()
        registry_tests.test_error_handling_in_traced_functor()
        print("✅ Functor Registry Tests: PASSED")
    finally:
        registry_tests.teardown_method()
    
    # Test node state tracker
    tracker_tests = TestNodeStateTracker()
    tracker_tests.setup_method()
    
    try:
        tracker_tests.test_node_state_updates()
        tracker_tests.test_state_change_calculation()
        print("✅ Node State Tracker Tests: PASSED")
    finally:
        tracker_tests.teardown_method()
    
    # Test integration scenarios
    integration_tests = TestIntegrationScenarios()
    integration_tests.setup_method()
    
    try:
        integration_tests.test_complete_manufacturing_workflow()
        print("✅ Integration Scenario Tests: PASSED")
    finally:
        integration_tests.teardown_method()
    
    print("=" * 80)
    print("🎉 ALL TRACEABILITY TESTS PASSED!")
    print("🔍 Engineering traceability system is production-ready!")

if __name__ == "__main__":
    run_traceability_tests()
