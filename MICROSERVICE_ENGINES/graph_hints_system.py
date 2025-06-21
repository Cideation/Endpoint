#!/usr/bin/env python3
"""
Graph Hints System v1.0
🧠 Shared interpretation maps for BEM Agent-Based Model (ABM)
Prevents divergence, maintains coherence, enables structured emergence

Features:
- Shared interpretation rules across all BEM components
- ABM-powered by interpretation maps, not hard-coded behavior
- Data-first, render-aware, emergence-tuned architecture
- Agent-level adaptation with learning and bidding patterns
- Zero-logic rendering with guaranteed visual keys
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HintCategory(Enum):
    """Categories of graph hints"""
    TYPE_ROLES = "type_roles"
    SIGNAL_MAP = "signal_map"
    PHASE_MAP = "phase_map"
    VISUAL_SCHEMA = "visual_schema"
    AGENT_BEHAVIOR = "agent_behavior"
    EMERGENCE_RULES = "emergence_rules"

@dataclass
class GraphHint:
    """Individual graph hint with metadata"""
    hint_id: str
    category: HintCategory
    key: str
    value: Any
    confidence: float
    source: str
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class AgentAdaptation:
    """Agent-level adaptation parameters"""
    agent_id: str
    learning_rate: float
    bidding_pattern: Dict[str, float]
    signal_feedback: Dict[str, float]
    adaptation_history: List[Dict[str, Any]]
    last_updated: str
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()

class GraphHintsSystem:
    """Central graph hints management system"""
    
    def __init__(self, hints_directory: str = "MICROSERVICE_ENGINES/graph_hints"):
        self.hints_directory = Path(hints_directory)
        self.hints_directory.mkdir(parents=True, exist_ok=True)
        
        # Core hint storage
        self.hints: Dict[HintCategory, Dict[str, GraphHint]] = {}
        self.agent_adaptations: Dict[str, AgentAdaptation] = {}
        
        # Interpretation maps
        self.interpretation_maps = {
            HintCategory.TYPE_ROLES: {},
            HintCategory.SIGNAL_MAP: {},
            HintCategory.PHASE_MAP: {},
            HintCategory.VISUAL_SCHEMA: {},
            HintCategory.AGENT_BEHAVIOR: {},
            HintCategory.EMERGENCE_RULES: {}
        }
        
        # Load existing hints
        self._load_all_hints()
        
        logger.info("🧠 Graph Hints System v1.0 initialized")
    
    def register_hint(self, category: HintCategory, key: str, value: Any,
                     confidence: float = 1.0, source: str = "system") -> GraphHint:
        """Register a new graph hint"""
        
        hint_id = f"hint_{uuid.uuid4().hex[:8]}"
        
        hint = GraphHint(
            hint_id=hint_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Store hint
        if category not in self.hints:
            self.hints[category] = {}
        
        self.hints[category][key] = hint
        
        # Update interpretation map
        self.interpretation_maps[category][key] = value
        
        # Persist hint
        self._save_hint(hint)
        
        logger.info(f"🧠 Registered hint: {category.value}.{key}")
        return hint
    
    def get_hint(self, category: HintCategory, key: str, default: Any = None) -> Any:
        """Get hint value with fallback"""
        
        if category in self.interpretation_maps and key in self.interpretation_maps[category]:
            return self.interpretation_maps[category][key]
        
        return default
    
    def get_type_role(self, node_type: str) -> Dict[str, Any]:
        """Get role definition for node type"""
        
        return self.get_hint(HintCategory.TYPE_ROLES, node_type, {
            "default_tags": ["component"],
            "expected_inputs": [],
            "expected_outputs": [],
            "rendering_priority": "normal"
        })
    
    def get_signal_mapping(self, signal_name: str) -> Dict[str, Any]:
        """Get signal interpretation mapping"""
        
        return self.get_hint(HintCategory.SIGNAL_MAP, signal_name, {
            "threshold": 0.5,
            "color_mapping": "#808080",
            "animation": "none",
            "interpretation": "unknown"
        })
    
    def get_phase_behavior(self, phase_name: str) -> Dict[str, Any]:
        """Get phase behavior mapping"""
        
        return self.get_hint(HintCategory.PHASE_MAP, phase_name, {
            "execution_order": 0,
            "dependencies": [],
            "expected_functors": [],
            "convergence_criteria": {}
        })
    
    def get_visual_schema(self, element_type: str) -> Dict[str, Any]:
        """Get guaranteed visual schema keys"""
        
        return self.get_hint(HintCategory.VISUAL_SCHEMA, element_type, {
            "required_keys": ["id", "type", "state"],
            "optional_keys": ["color", "animation", "metadata"],
            "rendering_hints": {}
        })
    
    def register_agent_adaptation(self, agent_id: str, learning_rate: float = 0.1,
                                 initial_bidding: Dict[str, float] = None) -> AgentAdaptation:
        """Register agent adaptation parameters"""
        
        if initial_bidding is None:
            initial_bidding = {"default": 1.0}
        
        adaptation = AgentAdaptation(
            agent_id=agent_id,
            learning_rate=learning_rate,
            bidding_pattern=initial_bidding,
            signal_feedback={},
            adaptation_history=[],
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        self.agent_adaptations[agent_id] = adaptation
        
        # Register as hint
        self.register_hint(
            HintCategory.AGENT_BEHAVIOR,
            agent_id,
            {
                "learning_rate": learning_rate,
                "bidding_pattern": initial_bidding,
                "adaptation_enabled": True
            },
            source="agent_registration"
        )
        
        logger.info(f"🤖 Registered agent adaptation: {agent_id}")
        return adaptation
    
    def update_agent_feedback(self, agent_id: str, signal_name: str, 
                            feedback_score: float, context: Dict[str, Any] = None) -> None:
        """Update agent signal feedback for learning"""
        
        if agent_id not in self.agent_adaptations:
            self.register_agent_adaptation(agent_id)
        
        adaptation = self.agent_adaptations[agent_id]
        
        # Update signal feedback with learning rate
        current_feedback = adaptation.signal_feedback.get(signal_name, 0.5)
        new_feedback = current_feedback + (adaptation.learning_rate * (feedback_score - current_feedback))
        adaptation.signal_feedback[signal_name] = new_feedback
        
        # Record adaptation event
        adaptation_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal_name,
            "feedback_score": feedback_score,
            "new_feedback": new_feedback,
            "context": context or {}
        }
        
        adaptation.adaptation_history.append(adaptation_event)
        adaptation.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Update bidding pattern based on feedback
        self._update_bidding_pattern(agent_id, signal_name, new_feedback)
        
        logger.debug(f"🔄 Updated agent {agent_id} feedback for {signal_name}: {new_feedback:.3f}")
    
    def get_agent_bidding_strength(self, agent_id: str, context: str = "default") -> float:
        """Get agent bidding strength for context"""
        
        if agent_id not in self.agent_adaptations:
            return 1.0
        
        adaptation = self.agent_adaptations[agent_id]
        return adaptation.bidding_pattern.get(context, 1.0)
    
    def register_emergence_rule(self, rule_name: str, conditions: Dict[str, Any],
                               actions: Dict[str, Any], priority: float = 1.0) -> None:
        """Register emergence detection and response rule"""
        
        emergence_rule = {
            "conditions": conditions,
            "actions": actions,
            "priority": priority,
            "activation_count": 0,
            "last_activated": None
        }
        
        self.register_hint(
            HintCategory.EMERGENCE_RULES,
            rule_name,
            emergence_rule,
            confidence=priority,
            source="emergence_system"
        )
        
        logger.info(f"🌟 Registered emergence rule: {rule_name}")
    
    def check_emergence_conditions(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for emergence rule activations"""
        
        activated_rules = []
        
        emergence_rules = self.interpretation_maps.get(HintCategory.EMERGENCE_RULES, {})
        
        for rule_name, rule_data in emergence_rules.items():
            if self._evaluate_emergence_conditions(rule_data["conditions"], system_state):
                activated_rules.append({
                    "rule_name": rule_name,
                    "rule_data": rule_data,
                    "activation_timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Update activation count
                rule_data["activation_count"] += 1
                rule_data["last_activated"] = datetime.now(timezone.utc).isoformat()
        
        return activated_rules
    
    def generate_interpretation_package(self, target_system: str) -> Dict[str, Any]:
        """Generate interpretation package for target system"""
        
        package = {
            "target_system": target_system,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "interpretation_maps": {},
            "agent_adaptations": {},
            "metadata": {
                "total_hints": sum(len(hints) for hints in self.hints.values()),
                "total_agents": len(self.agent_adaptations),
                "system_coherence_score": self._calculate_coherence_score()
            }
        }
        
        # Include relevant interpretation maps
        if target_system in ["parser", "all"]:
            package["interpretation_maps"]["type_roles"] = self.interpretation_maps[HintCategory.TYPE_ROLES]
        
        if target_system in ["sdfa", "all"]:
            package["interpretation_maps"]["signal_map"] = self.interpretation_maps[HintCategory.SIGNAL_MAP]
        
        if target_system in ["ne_engine", "all"]:
            package["interpretation_maps"]["phase_map"] = self.interpretation_maps[HintCategory.PHASE_MAP]
        
        if target_system in ["ui", "all"]:
            package["interpretation_maps"]["visual_schema"] = self.interpretation_maps[HintCategory.VISUAL_SCHEMA]
        
        if target_system in ["agents", "all"]:
            package["agent_adaptations"] = {
                agent_id: asdict(adaptation) 
                for agent_id, adaptation in self.agent_adaptations.items()
            }
        
        if target_system in ["emergence", "all"]:
            package["interpretation_maps"]["emergence_rules"] = self.interpretation_maps[HintCategory.EMERGENCE_RULES]
        
        return package
    
    def sync_with_system(self, system_name: str, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize hints with system state and return updates"""
        
        sync_results = {
            "system_name": system_name,
            "sync_timestamp": datetime.now(timezone.utc).isoformat(),
            "updates_applied": [],
            "conflicts_detected": [],
            "new_hints_suggested": []
        }
        
        # Check for conflicts between hints and system state
        for category, hints in self.interpretation_maps.items():
            system_section = system_state.get(category.value, {})
            
            for key, hint_value in hints.items():
                system_value = system_section.get(key)
                
                if system_value is not None and system_value != hint_value:
                    conflict = {
                        "category": category.value,
                        "key": key,
                        "hint_value": hint_value,
                        "system_value": system_value,
                        "resolution": "update_hint"  # or "update_system"
                    }
                    sync_results["conflicts_detected"].append(conflict)
        
        # Suggest new hints based on system state
        for category_name, system_data in system_state.items():
            if isinstance(system_data, dict):
                category = HintCategory(category_name) if category_name in [c.value for c in HintCategory] else None
                
                if category:
                    for key, value in system_data.items():
                        if key not in self.interpretation_maps.get(category, {}):
                            suggestion = {
                                "category": category.value,
                                "key": key,
                                "suggested_value": value,
                                "confidence": 0.8
                            }
                            sync_results["new_hints_suggested"].append(suggestion)
        
        return sync_results
    
    def export_abm_configuration(self, output_path: str = None) -> Dict[str, Any]:
        """Export complete ABM configuration"""
        
        abm_config = {
            "abm_version": "1.0",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "interpretation_maps": {
                category.value: hints for category, hints in self.interpretation_maps.items()
            },
            "agent_adaptations": {
                agent_id: asdict(adaptation) 
                for agent_id, adaptation in self.agent_adaptations.items()
            },
            "system_metadata": {
                "total_hints": sum(len(hints) for hints in self.hints.values()),
                "active_agents": len(self.agent_adaptations),
                "coherence_score": self._calculate_coherence_score(),
                "emergence_rules_count": len(self.interpretation_maps.get(HintCategory.EMERGENCE_RULES, {}))
            },
            "abm_characteristics": {
                "data_first": True,
                "render_aware": True,
                "emergence_tuned": True,
                "composable": True,
                "trainable": True,
                "interpretation_driven": True
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(abm_config, f, indent=2, default=str)
            logger.info(f"📦 ABM configuration exported to {output_path}")
        
        return abm_config
    
    def _load_all_hints(self) -> None:
        """Load all existing hints from storage"""
        
        try:
            # Load each hint category
            for category in HintCategory:
                category_file = self.hints_directory / f"{category.value}.json"
                
                if category_file.exists():
                    with open(category_file, 'r') as f:
                        category_data = json.load(f)
                    
                    self.interpretation_maps[category] = category_data.get("interpretation_map", {})
                    
                    # Reconstruct hints
                    if category not in self.hints:
                        self.hints[category] = {}
                    
                    for key, value in self.interpretation_maps[category].items():
                        hint = GraphHint(
                            hint_id=f"loaded_{uuid.uuid4().hex[:8]}",
                            category=category,
                            key=key,
                            value=value,
                            confidence=1.0,
                            source="loaded",
                            timestamp=datetime.now(timezone.utc).isoformat()
                        )
                        self.hints[category][key] = hint
            
            # Load agent adaptations
            agent_file = self.hints_directory / "agent_adaptations.json"
            if agent_file.exists():
                with open(agent_file, 'r') as f:
                    agent_data = json.load(f)
                
                for agent_id, adaptation_data in agent_data.items():
                    adaptation = AgentAdaptation(**adaptation_data)
                    self.agent_adaptations[agent_id] = adaptation
            
            logger.info("✅ Loaded existing graph hints")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load hints: {e}")
            self._initialize_default_hints()
    
    def _save_hint(self, hint: GraphHint) -> None:
        """Save individual hint to storage"""
        
        try:
            category_file = self.hints_directory / f"{hint.category.value}.json"
            
            # Load existing data
            if category_file.exists():
                with open(category_file, 'r') as f:
                    category_data = json.load(f)
            else:
                category_data = {"interpretation_map": {}, "hints_metadata": {}}
            
            # Update data
            category_data["interpretation_map"][hint.key] = hint.value
            category_data["hints_metadata"][hint.key] = {
                "hint_id": hint.hint_id,
                "confidence": hint.confidence,
                "source": hint.source,
                "timestamp": hint.timestamp
            }
            
            # Save updated data
            with open(category_file, 'w') as f:
                json.dump(category_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"❌ Failed to save hint: {e}")
    
    def _update_bidding_pattern(self, agent_id: str, signal_name: str, feedback: float) -> None:
        """Update agent bidding pattern based on feedback"""
        
        adaptation = self.agent_adaptations[agent_id]
        
        # Adjust bidding strength based on feedback
        current_bid = adaptation.bidding_pattern.get(signal_name, 1.0)
        
        # Increase bidding for positive feedback, decrease for negative
        if feedback > 0.7:
            new_bid = min(2.0, current_bid * 1.1)
        elif feedback < 0.3:
            new_bid = max(0.1, current_bid * 0.9)
        else:
            new_bid = current_bid
        
        adaptation.bidding_pattern[signal_name] = new_bid
    
    def _evaluate_emergence_conditions(self, conditions: Dict[str, Any], 
                                     system_state: Dict[str, Any]) -> bool:
        """Evaluate emergence rule conditions"""
        
        for condition_key, condition_value in conditions.items():
            system_value = system_state.get(condition_key)
            
            if isinstance(condition_value, dict):
                operator = condition_value.get("operator", "equals")
                threshold = condition_value.get("value")
                
                if operator == "greater_than" and (system_value is None or system_value <= threshold):
                    return False
                elif operator == "less_than" and (system_value is None or system_value >= threshold):
                    return False
                elif operator == "equals" and system_value != threshold:
                    return False
            else:
                if system_value != condition_value:
                    return False
        
        return True
    
    def _calculate_coherence_score(self) -> float:
        """Calculate system coherence score"""
        
        total_hints = sum(len(hints) for hints in self.hints.values())
        if total_hints == 0:
            return 0.0
        
        # Calculate based on hint coverage and consistency
        coverage_score = min(1.0, total_hints / 50)  # Assume 50 hints for full coverage
        
        # Calculate consistency (hints with high confidence)
        high_confidence_hints = 0
        for category_hints in self.hints.values():
            for hint in category_hints.values():
                if hint.confidence >= 0.8:
                    high_confidence_hints += 1
        
        consistency_score = high_confidence_hints / total_hints if total_hints > 0 else 0.0
        
        return (coverage_score + consistency_score) / 2
    
    def _initialize_default_hints(self) -> None:
        """Initialize default graph hints"""
        
        # Default type roles
        default_types = {
            "V01_ProductComponent": {
                "default_tags": ["component", "product"],
                "expected_inputs": ["material", "dimensions"],
                "expected_outputs": ["score", "quality"],
                "rendering_priority": "high"
            },
            "V04_UserEconomicProfile": {
                "default_tags": ["user", "economic"],
                "expected_inputs": ["budget", "preferences"],
                "expected_outputs": ["profile", "constraints"],
                "rendering_priority": "normal"
            }
        }
        
        for node_type, role_data in default_types.items():
            self.register_hint(HintCategory.TYPE_ROLES, node_type, role_data, source="default")
        
        # Default signal mappings
        default_signals = {
            "evolutionary_peak": {
                "threshold": 0.9,
                "color_mapping": "#00FF41",
                "animation": "pulse",
                "interpretation": "optimal_performance"
            },
            "neutral_state": {
                "threshold": 0.5,
                "color_mapping": "#FFD700",
                "animation": "none",
                "interpretation": "stable_operation"
            }
        }
        
        for signal_name, signal_data in default_signals.items():
            self.register_hint(HintCategory.SIGNAL_MAP, signal_name, signal_data, source="default")
        
        logger.info("🔧 Initialized default graph hints")

# Convenience functions
def create_graph_hints_system(hints_directory: str = None) -> GraphHintsSystem:
    """Create graph hints system instance"""
    return GraphHintsSystem(hints_directory) if hints_directory else GraphHintsSystem()

def register_system_hint(category: str, key: str, value: Any, 
                        hints_system: GraphHintsSystem = None) -> GraphHint:
    """Register a system hint"""
    if hints_system is None:
        hints_system = create_graph_hints_system()
    
    category_enum = HintCategory(category)
    return hints_system.register_hint(category_enum, key, value, source="system_registration")

if __name__ == "__main__":
    print("🧠 Graph Hints System v1.0 - ABM Demo")
    
    # Create hints system
    hints_system = create_graph_hints_system()
    
    # Demo 1: Type roles registration
    print("\n📋 Demo 1: Type Roles Registration")
    hints_system.register_hint(
        HintCategory.TYPE_ROLES,
        "V01_ProductComponent",
        {
            "default_tags": ["component", "product", "manufacturing"],
            "expected_inputs": ["material", "dimensions", "specifications"],
            "expected_outputs": ["score", "quality", "feasibility"],
            "rendering_priority": "high"
        }
    )
    
    # Demo 2: Agent adaptation
    print("\n🤖 Demo 2: Agent Adaptation")
    agent_adaptation = hints_system.register_agent_adaptation(
        "Agent1",
        learning_rate=0.15,
        initial_bidding={"manufacturing": 1.2, "quality": 0.8}
    )
    
    # Simulate feedback
    hints_system.update_agent_feedback("Agent1", "manufacturing", 0.85)
    hints_system.update_agent_feedback("Agent1", "quality", 0.45)
    
    # Demo 3: Emergence rules
    print("\n🌟 Demo 3: Emergence Rules")
    hints_system.register_emergence_rule(
        "high_performance_cluster",
        conditions={
            "average_performance": {"operator": "greater_than", "value": 0.8},
            "node_count": {"operator": "greater_than", "value": 5}
        },
        actions={
            "highlight_cluster": True,
            "increase_priority": 1.5,
            "notify_operators": True
        }
    )
    
    # Demo 4: System synchronization
    print("\n🔄 Demo 4: System Synchronization")
    mock_system_state = {
        "type_roles": {
            "V02_NewComponent": {
                "default_tags": ["component", "new"],
                "rendering_priority": "medium"
            }
        },
        "signal_map": {
            "critical_state": {
                "threshold": 0.1,
                "color_mapping": "#FF0000"
            }
        }
    }
    
    sync_results = hints_system.sync_with_system("test_system", mock_system_state)
    print(f"  Conflicts detected: {len(sync_results['conflicts_detected'])}")
    print(f"  New hints suggested: {len(sync_results['new_hints_suggested'])}")
    
    # Demo 5: ABM configuration export
    print("\n📦 Demo 5: ABM Configuration Export")
    abm_config = hints_system.export_abm_configuration()
    
    print(f"  Total hints: {abm_config['system_metadata']['total_hints']}")
    print(f"  Active agents: {abm_config['system_metadata']['active_agents']}")
    print(f"  Coherence score: {abm_config['system_metadata']['coherence_score']:.3f}")
    print(f"  ABM characteristics: {abm_config['abm_characteristics']}")
    
    print("\n🎉 Graph Hints System Demo Complete!")
    print("🧠 BEM transformed into ABM with shared interpretation maps!")
