#!/usr/bin/env python3
"""
Test Graph Hints ABM System
🧠 Comprehensive testing for agent-based model with shared interpretation maps
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MICROSERVICE_ENGINES.graph_hints_system import (
    GraphHintsSystem, HintCategory, GraphHint, AgentAdaptation,
    create_graph_hints_system, register_system_hint
)

class TestGraphHintsSystem:
    """Test the core Graph Hints System functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.hints_system = GraphHintsSystem(hints_directory=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_hint_registration(self):
        """Test basic hint registration"""
        
        # Register type role hint
        hint = self.hints_system.register_hint(
            HintCategory.TYPE_ROLES,
            "V01_ProductComponent",
            {
                "default_tags": ["component", "product"],
                "expected_inputs": ["material", "dimensions"],
                "expected_outputs": ["score", "quality"],
                "rendering_priority": "high"
            },
            confidence=0.95,
            source="test"
        )
        
        assert hint.category == HintCategory.TYPE_ROLES
        assert hint.key == "V01_ProductComponent"
        assert hint.confidence == 0.95
        assert hint.source == "test"
        assert isinstance(hint.value, dict)
        
        # Verify hint is accessible
        retrieved_hint = self.hints_system.get_hint(
            HintCategory.TYPE_ROLES, 
            "V01_ProductComponent"
        )
        
        assert retrieved_hint is not None
        assert retrieved_hint["default_tags"] == ["component", "product"]
    
    def test_agent_adaptation_registration(self):
        """Test agent adaptation registration and updates"""
        
        # Register agent
        adaptation = self.hints_system.register_agent_adaptation(
            "TestAgent1",
            learning_rate=0.2,
            initial_bidding={"manufacturing": 1.5, "quality": 0.8}
        )
        
        assert adaptation.agent_id == "TestAgent1"
        assert adaptation.learning_rate == 0.2
        assert adaptation.bidding_pattern["manufacturing"] == 1.5
        assert adaptation.bidding_pattern["quality"] == 0.8
        
        # Test feedback update
        initial_feedback = adaptation.signal_feedback.get("manufacturing", 0.5)
        
        self.hints_system.update_agent_feedback(
            "TestAgent1", 
            "manufacturing", 
            0.9,
            context={"test": "positive_feedback"}
        )
        
        updated_adaptation = self.hints_system.agent_adaptations["TestAgent1"]
        new_feedback = updated_adaptation.signal_feedback["manufacturing"]
        
        # Should have moved toward positive feedback
        assert new_feedback > initial_feedback
        assert len(updated_adaptation.adaptation_history) == 1
        
        # Test bidding strength
        bidding_strength = self.hints_system.get_agent_bidding_strength(
            "TestAgent1", 
            "manufacturing"
        )
        
        assert bidding_strength > 0
    
    def test_signal_mapping(self):
        """Test signal interpretation mapping"""
        
        # Register signal mapping
        self.hints_system.register_hint(
            HintCategory.SIGNAL_MAP,
            "critical_performance",
            {
                "threshold": 0.95,
                "color_mapping": "#FF4444",
                "animation": "flash",
                "interpretation": "critical_state"
            }
        )
        
        # Retrieve signal mapping
        signal_data = self.hints_system.get_signal_mapping("critical_performance")
        
        assert signal_data["threshold"] == 0.95
        assert signal_data["color_mapping"] == "#FF4444"
        assert signal_data["animation"] == "flash"
        assert signal_data["interpretation"] == "critical_state"
        
        # Test default fallback
        unknown_signal = self.hints_system.get_signal_mapping("unknown_signal")
        assert unknown_signal["threshold"] == 0.5
        assert unknown_signal["color_mapping"] == "#808080"
    
    def test_phase_behavior_mapping(self):
        """Test phase behavior interpretation"""
        
        # Register phase behavior
        self.hints_system.register_hint(
            HintCategory.PHASE_MAP,
            "Alpha",
            {
                "execution_order": 1,
                "dependencies": [],
                "expected_functors": ["V01_ProductComponent", "V02_Material"],
                "convergence_criteria": {"min_nodes": 3, "stability_threshold": 0.8}
            }
        )
        
        # Retrieve phase behavior
        phase_data = self.hints_system.get_phase_behavior("Alpha")
        
        assert phase_data["execution_order"] == 1
        assert phase_data["expected_functors"] == ["V01_ProductComponent", "V02_Material"]
        assert phase_data["convergence_criteria"]["min_nodes"] == 3
    
    def test_visual_schema_guarantee(self):
        """Test guaranteed visual schema keys"""
        
        # Register visual schema
        self.hints_system.register_hint(
            HintCategory.VISUAL_SCHEMA,
            "node",
            {
                "required_keys": ["id", "type", "state", "position"],
                "optional_keys": ["color", "animation", "metadata", "label"],
                "rendering_hints": {
                    "min_size": 20,
                    "max_size": 100,
                    "default_color": "#4A90E2"
                }
            }
        )
        
        # Retrieve visual schema
        visual_data = self.hints_system.get_visual_schema("node")
        
        assert "id" in visual_data["required_keys"]
        assert "type" in visual_data["required_keys"]
        assert "state" in visual_data["required_keys"]
        assert "position" in visual_data["required_keys"]
        assert visual_data["rendering_hints"]["default_color"] == "#4A90E2"
    
    def test_emergence_rules(self):
        """Test emergence rule registration and evaluation"""
        
        # Register emergence rule
        self.hints_system.register_emergence_rule(
            "performance_cluster",
            conditions={
                "average_performance": {"operator": "greater_than", "value": 0.8},
                "node_count": {"operator": "greater_than", "value": 5},
                "stability": {"operator": "equals", "value": True}
            },
            actions={
                "highlight_cluster": True,
                "increase_priority": 1.5,
                "notify_operators": True
            },
            priority=0.9
        )
        
        # Test positive emergence detection
        positive_state = {
            "average_performance": 0.85,
            "node_count": 7,
            "stability": True
        }
        
        activated_rules = self.hints_system.check_emergence_conditions(positive_state)
        
        assert len(activated_rules) == 1
        assert activated_rules[0]["rule_name"] == "performance_cluster"
        
        # Test negative emergence detection
        negative_state = {
            "average_performance": 0.75,  # Below threshold
            "node_count": 7,
            "stability": True
        }
        
        activated_rules = self.hints_system.check_emergence_conditions(negative_state)
        
        assert len(activated_rules) == 0
    
    def test_interpretation_package_generation(self):
        """Test interpretation package generation for different systems"""
        
        # Setup various hints
        self.hints_system.register_hint(
            HintCategory.TYPE_ROLES, "TestType", {"role": "test"}
        )
        self.hints_system.register_hint(
            HintCategory.SIGNAL_MAP, "TestSignal", {"threshold": 0.7}
        )
        self.hints_system.register_agent_adaptation("TestAgent", 0.1)
        
        # Generate package for parser
        parser_package = self.hints_system.generate_interpretation_package("parser")
        
        assert parser_package["target_system"] == "parser"
        assert "type_roles" in parser_package["interpretation_maps"]
        assert "TestType" in parser_package["interpretation_maps"]["type_roles"]
        
        # Generate package for all systems
        all_package = self.hints_system.generate_interpretation_package("all")
        
        assert "type_roles" in all_package["interpretation_maps"]
        assert "signal_map" in all_package["interpretation_maps"]
        assert "TestAgent" in all_package["agent_adaptations"]
        assert all_package["metadata"]["total_hints"] >= 2
        assert all_package["metadata"]["total_agents"] >= 1
    
    def test_system_synchronization(self):
        """Test system state synchronization"""
        
        # Setup existing hints
        self.hints_system.register_hint(
            HintCategory.TYPE_ROLES,
            "ExistingType",
            {"existing": "data"}
        )
        
        # Mock system state with conflicts and new data
        mock_system_state = {
            "type_roles": {
                "ExistingType": {"existing": "modified_data"},  # Conflict
                "NewType": {"new": "data"}  # New hint suggestion
            },
            "signal_map": {
                "NewSignal": {"threshold": 0.6}  # New hint suggestion
            }
        }
        
        # Perform synchronization
        sync_results = self.hints_system.sync_with_system("test_system", mock_system_state)
        
        assert sync_results["system_name"] == "test_system"
        assert len(sync_results["conflicts_detected"]) >= 1
        assert len(sync_results["new_hints_suggested"]) >= 2
        
        # Check conflict details
        conflict = sync_results["conflicts_detected"][0]
        assert conflict["category"] == "type_roles"
        assert conflict["key"] == "ExistingType"
        assert conflict["hint_value"] == {"existing": "data"}
        assert conflict["system_value"] == {"existing": "modified_data"}
    
    def test_abm_configuration_export(self):
        """Test ABM configuration export"""
        
        # Setup comprehensive system
        self.hints_system.register_hint(
            HintCategory.TYPE_ROLES, "ComponentA", {"role": "primary"}
        )
        self.hints_system.register_hint(
            HintCategory.SIGNAL_MAP, "SignalX", {"threshold": 0.8}
        )
        self.hints_system.register_agent_adaptation("Agent1", 0.15)
        self.hints_system.register_emergence_rule(
            "test_rule", 
            {"condition": True}, 
            {"action": "test"}
        )
        
        # Export configuration
        abm_config = self.hints_system.export_abm_configuration()
        
        assert abm_config["abm_version"] == "1.0"
        assert "interpretation_maps" in abm_config
        assert "agent_adaptations" in abm_config
        assert "system_metadata" in abm_config
        assert "abm_characteristics" in abm_config
        
        # Check ABM characteristics
        characteristics = abm_config["abm_characteristics"]
        assert characteristics["data_first"] is True
        assert characteristics["render_aware"] is True
        assert characteristics["emergence_tuned"] is True
        assert characteristics["composable"] is True
        assert characteristics["trainable"] is True
        assert characteristics["interpretation_driven"] is True
        
        # Check system metadata
        metadata = abm_config["system_metadata"]
        assert metadata["total_hints"] >= 3
        assert metadata["active_agents"] >= 1
        assert metadata["coherence_score"] >= 0.0
        assert metadata["emergence_rules_count"] >= 1
    
    def test_agent_learning_progression(self):
        """Test agent learning over multiple feedback cycles"""
        
        # Register agent
        self.hints_system.register_agent_adaptation(
            "LearningAgent",
            learning_rate=0.3,  # High learning rate for faster testing
            initial_bidding={"task_a": 1.0}
        )
        
        # Simulate positive feedback cycle
        feedback_scores = [0.8, 0.85, 0.9, 0.95]
        
        for i, score in enumerate(feedback_scores):
            self.hints_system.update_agent_feedback(
                "LearningAgent",
                "task_a",
                score,
                context={"cycle": i + 1}
            )
        
        # Check learning progression
        adaptation = self.hints_system.agent_adaptations["LearningAgent"]
        
        assert len(adaptation.adaptation_history) == 4
        assert adaptation.signal_feedback["task_a"] > 0.8  # Should have learned
        
        # Check bidding pattern adjustment
        bidding_strength = self.hints_system.get_agent_bidding_strength(
            "LearningAgent", 
            "task_a"
        )
        
        assert bidding_strength > 1.0  # Should have increased due to positive feedback
    
    def test_system_coherence_score_calculation(self):
        """Test system coherence score calculation"""
        
        # Start with empty system
        initial_score = self.hints_system._calculate_coherence_score()
        assert initial_score == 0.0
        
        # Add high-confidence hints
        for i in range(10):
            self.hints_system.register_hint(
                HintCategory.TYPE_ROLES,
                f"HighConfidenceType{i}",
                {"confidence": "high"},
                confidence=0.9
            )
        
        # Add low-confidence hints
        for i in range(5):
            self.hints_system.register_hint(
                HintCategory.SIGNAL_MAP,
                f"LowConfidenceSignal{i}",
                {"confidence": "low"},
                confidence=0.3
            )
        
        # Calculate coherence score
        system_coherence_score = self.hints_system._calculate_coherence_score()
        
        assert system_coherence_score > 0.0
        assert system_coherence_score <= 1.0
        
        # Should be influenced by both coverage and consistency
        assert system_coherence_score > initial_score

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_graph_hints_system(self):
        """Test system creation convenience function"""
        
        # Test default creation
        system1 = create_graph_hints_system()
        assert isinstance(system1, GraphHintsSystem)
        
        # Test with custom directory
        with tempfile.TemporaryDirectory() as temp_dir:
            system2 = create_graph_hints_system(temp_dir)
            assert isinstance(system2, GraphHintsSystem)
            assert str(system2.hints_directory) == temp_dir
    
    def test_register_system_hint(self):
        """Test system hint registration convenience function"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            hints_system = create_graph_hints_system(temp_dir)
            
            # Register hint using convenience function
            hint = register_system_hint(
                "type_roles",
                "ConvenienceType",
                {"test": "data"},
                hints_system
            )
            
            assert hint.category == HintCategory.TYPE_ROLES
            assert hint.key == "ConvenienceType"
            assert hint.source == "system_registration"
            
            # Verify hint is accessible
            retrieved = hints_system.get_hint(
                HintCategory.TYPE_ROLES,
                "ConvenienceType"
            )
            assert retrieved["test"] == "data"

class TestABMIntegration:
    """Test ABM integration scenarios"""
    
    def setup_method(self):
        """Setup ABM test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.abm_system = GraphHintsSystem(hints_directory=self.temp_dir)
        
        # Setup realistic ABM scenario
        self._setup_manufacturing_abm()
    
    def teardown_method(self):
        """Cleanup ABM test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _setup_manufacturing_abm(self):
        """Setup manufacturing ABM scenario"""
        
        # Register component types
        component_types = {
            "V01_ProductComponent": {
                "default_tags": ["component", "product", "manufacturing"],
                "expected_inputs": ["material", "dimensions", "specifications"],
                "expected_outputs": ["score", "quality", "feasibility"],
                "rendering_priority": "high"
            },
            "V04_UserEconomicProfile": {
                "default_tags": ["user", "economic", "constraints"],
                "expected_inputs": ["budget", "preferences", "timeline"],
                "expected_outputs": ["profile", "constraints", "priorities"],
                "rendering_priority": "medium"
            },
            "V05_ManufacturingProcess": {
                "default_tags": ["process", "manufacturing", "workflow"],
                "expected_inputs": ["components", "resources", "schedule"],
                "expected_outputs": ["plan", "timeline", "cost"],
                "rendering_priority": "high"
            }
        }
        
        for comp_type, comp_data in component_types.items():
            self.abm_system.register_hint(HintCategory.TYPE_ROLES, comp_type, comp_data)
        
        # Register signal mappings
        signal_mappings = {
            "manufacturing_ready": {
                "threshold": 0.8,
                "color_mapping": "#00FF41",
                "animation": "pulse",
                "interpretation": "ready_for_production"
            },
            "quality_concern": {
                "threshold": 0.3,
                "color_mapping": "#FF8C00",
                "animation": "warning",
                "interpretation": "quality_review_needed"
            },
            "cost_optimal": {
                "threshold": 0.9,
                "color_mapping": "#32CD32",
                "animation": "glow",
                "interpretation": "cost_optimized"
            }
        }
        
        for signal_name, signal_data in signal_mappings.items():
            self.abm_system.register_hint(HintCategory.SIGNAL_MAP, signal_name, signal_data)
        
        # Register manufacturing agents
        agents = {
            "QualityAgent": {"learning_rate": 0.1, "bidding": {"quality": 1.5, "cost": 0.7}},
            "CostAgent": {"learning_rate": 0.15, "bidding": {"cost": 1.8, "quality": 0.6}},
            "TimeAgent": {"learning_rate": 0.12, "bidding": {"timeline": 1.6, "quality": 0.8}}
        }
        
        for agent_id, agent_config in agents.items():
            self.abm_system.register_agent_adaptation(
                agent_id,
                agent_config["learning_rate"],
                agent_config["bidding"]
            )
        
        # Register emergence rules
        self.abm_system.register_emergence_rule(
            "manufacturing_optimization",
            conditions={
                "quality_score": {"operator": "greater_than", "value": 0.8},
                "cost_efficiency": {"operator": "greater_than", "value": 0.7},
                "timeline_feasibility": {"operator": "greater_than", "value": 0.75}
            },
            actions={
                "trigger_production": True,
                "notify_stakeholders": True,
                "lock_configuration": True
            },
            priority=0.95
        )
    
    def test_manufacturing_abm_scenario(self):
        """Test complete manufacturing ABM scenario"""
        
        # Simulate manufacturing system state
        manufacturing_state = {
            "quality_score": 0.85,
            "cost_efficiency": 0.78,
            "timeline_feasibility": 0.80,
            "active_components": 12,
            "agent_conflicts": 0
        }
        
        # Check emergence conditions
        activated_rules = self.abm_system.check_emergence_conditions(manufacturing_state)
        
        assert len(activated_rules) == 1
        assert activated_rules[0]["rule_name"] == "manufacturing_optimization"
        
        # Simulate agent feedback from manufacturing results
        feedback_scenarios = [
            ("QualityAgent", "quality", 0.9),
            ("CostAgent", "cost", 0.7),
            ("TimeAgent", "timeline", 0.8)
        ]
        
        for agent_id, signal, feedback in feedback_scenarios:
            self.abm_system.update_agent_feedback(agent_id, signal, feedback)
        
        # Verify agent adaptations
        quality_agent = self.abm_system.agent_adaptations["QualityAgent"]
        assert quality_agent.signal_feedback["quality"] > 0.5
        
        cost_agent = self.abm_system.agent_adaptations["CostAgent"]
        assert cost_agent.signal_feedback["cost"] > 0.5
        
        # Generate interpretation package for manufacturing system
        manufacturing_package = self.abm_system.generate_interpretation_package("all")
        
        assert "V01_ProductComponent" in manufacturing_package["interpretation_maps"]["type_roles"]
        assert "manufacturing_ready" in manufacturing_package["interpretation_maps"]["signal_map"]
        assert "QualityAgent" in manufacturing_package["agent_adaptations"]
        assert manufacturing_package["metadata"]["system_coherence_score"] > 0.0
    
    def test_abm_export_and_reload(self):
        """Test ABM configuration export and reload"""
        
        # Export ABM configuration
        export_path = os.path.join(self.temp_dir, "abm_config.json")
        abm_config = self.abm_system.export_abm_configuration(export_path)
        
        # Verify export file exists
        assert os.path.exists(export_path)
        
        # Load exported configuration
        with open(export_path, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["abm_version"] == "1.0"
        assert "interpretation_maps" in loaded_config
        assert "agent_adaptations" in loaded_config
        
        # Verify specific content
        type_roles = loaded_config["interpretation_maps"]["type_roles"]
        assert "V01_ProductComponent" in type_roles
        assert "V04_UserEconomicProfile" in type_roles
        assert "V05_ManufacturingProcess" in type_roles
        
        agent_adaptations = loaded_config["agent_adaptations"]
        assert "QualityAgent" in agent_adaptations
        assert "CostAgent" in agent_adaptations
        assert "TimeAgent" in agent_adaptations

if __name__ == "__main__":
    print("🧠 Running Graph Hints ABM Tests...")
    
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
    
    print("\n🎉 Graph Hints ABM Testing Complete!")
