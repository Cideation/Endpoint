#!/usr/bin/env python3
"""
Dashboard State Generator - Pure Data Pipes
Generates structured state from engine for dashboard consumption
All semantics sourced from graph_hints/ - no hardcoding
"""

import json
import os
import sys
from typing import Dict, List, Any
from datetime import datetime

# Add paths for system integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MICROSERVICE_ENGINES', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Final_Phase'))

try:
    from interaction_language_interpreter import InteractionLanguageInterpreter
    INTERPRETER_AVAILABLE = True
except ImportError:
    INTERPRETER_AVAILABLE = False

class DashboardStateGenerator:
    """
    Pure data pipe generator - no UI logic, only structured state
    All semantics sourced from graph_hints/ configuration
    """
    
    def __init__(self):
        self.graph_hints_path = os.path.join(os.path.dirname(__file__), '..', 'MICROSERVICE_ENGINES', 'graph_hints')
        self.interaction_language = self._load_interaction_language()
        self.signal_map = self._load_signal_map()
        self.visual_schema = self._load_visual_schema()
        self.phase_map = self._load_phase_map()
        
    def _load_interaction_language(self) -> Dict[str, Any]:
        """Load interaction language configuration"""
        try:
            with open(os.path.join(self.graph_hints_path, 'interaction_language.json'), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"design_signal": {}, "signal_intent": {}}
    
    def _load_signal_map(self) -> Dict[str, Any]:
        """Load signal mapping configuration"""
        signal_map_path = os.path.join(self.graph_hints_path, 'signal_map.json')
        if os.path.exists(signal_map_path):
            with open(signal_map_path, 'r') as f:
                return json.load(f)
        else:
            # Create default signal map
            default_map = {
                "score_ranges": {
                    "critical": {"min": 0.0, "max": 0.2, "label": "Critical", "color": "#F44336"},
                    "low": {"min": 0.2, "max": 0.4, "label": "Low", "color": "#FF9800"},
                    "medium": {"min": 0.4, "max": 0.7, "label": "Medium", "color": "#2196F3"},
                    "high": {"min": 0.7, "max": 0.9, "label": "High", "color": "#4CAF50"},
                    "optimal": {"min": 0.9, "max": 1.0, "label": "Optimal", "color": "#3F51B5"}
                }
            }
            self._save_signal_map(default_map)
            return default_map
    
    def _load_visual_schema(self) -> Dict[str, Any]:
        """Load visual schema for frontend expectations"""
        visual_schema_path = os.path.join(self.graph_hints_path, 'visual_schema.json')
        if os.path.exists(visual_schema_path):
            with open(visual_schema_path, 'r') as f:
                return json.load(f)
        else:
            # Create default visual schema
            default_schema = {
                "node_properties": ["id", "type", "design_signal", "intent", "urgency", "color", "opacity"],
                "agent_properties": ["id", "active_nodes", "score", "last_action", "status"],
                "phase_properties": ["name", "active", "behaviors", "node_count", "signal_distribution"],
                "update_frequency": 1000,
                "animation_duration": 500
            }
            self._save_visual_schema(default_schema)
            return default_schema
    
    def _load_phase_map(self) -> Dict[str, Any]:
        """Load phase mapping configuration"""
        phase_map_path = os.path.join(self.graph_hints_path, 'phase_map.json')
        if os.path.exists(phase_map_path):
            with open(phase_map_path, 'r') as f:
                return json.load(f)
        else:
            # Create default phase map
            default_map = {
                "alpha": {
                    "name": "Alpha - DAG Processing",
                    "color": "#3F51B5",
                    "node_types": ["V01_ProductComponent", "V06_MaterialSpec"],
                    "primary_signals": ["design_signal", "urgency_index"],
                    "behaviors": ["sequential_processing", "quality_evaluation"]
                },
                "beta": {
                    "name": "Beta - Relational Optimization", 
                    "color": "#FF9800",
                    "node_types": ["V02_EconomicProfile", "V03_AgentBehavior"],
                    "primary_signals": ["agent_feedback", "interaction_mode"],
                    "behaviors": ["optimization", "agent_learning"]
                },
                "gamma": {
                    "name": "Gamma - Emergent Synthesis",
                    "color": "#4CAF50", 
                    "node_types": ["V05_ComplianceCheck", "V04_Environment"],
                    "primary_signals": ["gradient_energy", "signal_intent"],
                    "behaviors": ["emergence_detection", "system_synthesis"]
                }
            }
            self._save_phase_map(default_map)
            return default_map
    
    def _save_signal_map(self, signal_map: Dict[str, Any]):
        """Save signal map to file"""
        with open(os.path.join(self.graph_hints_path, 'signal_map.json'), 'w') as f:
            json.dump(signal_map, f, indent=2)
    
    def _save_visual_schema(self, visual_schema: Dict[str, Any]):
        """Save visual schema to file"""
        with open(os.path.join(self.graph_hints_path, 'visual_schema.json'), 'w') as f:
            json.dump(visual_schema, f, indent=2)
    
    def _save_phase_map(self, phase_map: Dict[str, Any]):
        """Save phase map to file"""
        with open(os.path.join(self.graph_hints_path, 'phase_map.json'), 'w') as f:
            json.dump(phase_map, f, indent=2)
    
    def generate_dashboard_state(self, engine_nodes: List[Dict[str, Any]], 
                               engine_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate complete dashboard state from engine data
        Pure data transformation - no UI logic
        """
        
        # Process nodes through interaction language
        processed_nodes = []
        for node in engine_nodes:
            processed_node = self._process_node_for_dashboard(node)
            processed_nodes.append(processed_node)
        
        # Process agents
        processed_agents = []
        for agent in engine_agents:
            processed_agent = self._process_agent_for_dashboard(agent)
            processed_agents.append(processed_agent)
        
        # Generate phase state
        phase_state = self._generate_phase_state(processed_nodes)
        
        # Create complete dashboard state
        dashboard_state = {
            "timestamp": datetime.now().isoformat(),
            "nodes": processed_nodes,
            "agents": processed_agents, 
            "phases": phase_state,
            "system_summary": {
                "total_nodes": len(processed_nodes),
                "total_agents": len(processed_agents),
                "active_phases": len([p for p in phase_state.values() if p.get("active", False)]),
                "avg_urgency": self._calculate_avg_urgency(processed_nodes),
                "signal_distribution": self._calculate_signal_distribution(processed_nodes)
            },
            "metadata": {
                "visual_schema": self.visual_schema,
                "update_frequency": self.visual_schema.get("update_frequency", 1000)
            }
        }
        
        return dashboard_state
    
    def _process_node_for_dashboard(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Process single node for dashboard display"""
        
        # Extract core properties
        node_id = node.get("id", "unknown")
        node_type = node.get("type", "unknown")
        score = node.get("score", 0.5)
        
        # Get design signal from interaction language
        design_signal = self._get_design_signal_from_score(score)
        signal_config = self.interaction_language.get("design_signal", {}).get(design_signal, {})
        
        # Get intent and urgency
        intent = signal_config.get("intent", "wait")
        urgency = signal_config.get("urgency", 0.5)
        urgency_level = self._calculate_urgency_level(urgency)
        
        # Get color from signal map
        color = self._get_color_from_score(score)
        
        # Get phase
        phase = self._determine_node_phase(node_type)
        
        return {
            "id": node_id,
            "type": node_type,
            "design_signal": design_signal,
            "intent": intent,
            "urgency": urgency_level,
            "color": color,
            "opacity": self._calculate_opacity(urgency),
            "score": score,
            "phase": phase,
            "semantic_meaning": signal_config.get("semantic_meaning", "Unknown state"),
            "last_update": datetime.now().isoformat()
        }
    
    def _process_agent_for_dashboard(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Process single agent for dashboard display"""
        
        agent_id = agent.get("id", "unknown")
        active_nodes = agent.get("active_nodes", [])
        score = agent.get("score", 0.5)
        last_action = agent.get("last_action", "idle")
        
        return {
            "id": agent_id,
            "active_nodes": active_nodes,
            "node_count": len(active_nodes),
            "score": score,
            "last_action": last_action,
            "status": self._determine_agent_status(score, len(active_nodes)),
            "color": self._get_color_from_score(score),
            "last_update": datetime.now().isoformat()
        }
    
    def _generate_phase_state(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate phase state from processed nodes"""
        
        phase_state = {}
        
        for phase_name, phase_config in self.phase_map.items():
            # Count nodes in this phase
            phase_nodes = [n for n in nodes if n.get("phase") == phase_name]
            
            # Calculate signal distribution for this phase
            signal_dist = {}
            for node in phase_nodes:
                signal = node.get("design_signal", "neutral_state")
                signal_dist[signal] = signal_dist.get(signal, 0) + 1
            
            # Determine if phase is active
            is_active = len(phase_nodes) > 0 and any(n.get("urgency") in ["high", "immediate"] for n in phase_nodes)
            
            phase_state[phase_name] = {
                "name": phase_config.get("name", phase_name),
                "color": phase_config.get("color", "#9E9E9E"),
                "active": is_active,
                "node_count": len(phase_nodes),
                "behaviors": phase_config.get("behaviors", []),
                "signal_distribution": signal_dist,
                "avg_score": sum(n.get("score", 0) for n in phase_nodes) / max(1, len(phase_nodes)),
                "primary_signals": phase_config.get("primary_signals", [])
            }
        
        return phase_state
    
    def _get_design_signal_from_score(self, score: float) -> str:
        """Map score to design signal using interaction language"""
        if score < 0.1:
            return "critical_failure"
        elif score > 0.8:
            return "evolutionary_peak"
        elif score < 0.3:
            return "low_precision"
        else:
            return "neutral_state"
    
    def _get_color_from_score(self, score: float) -> str:
        """Get color from score using signal map"""
        for range_name, range_config in self.signal_map.get("score_ranges", {}).items():
            if range_config["min"] <= score <= range_config["max"]:
                return range_config["color"]
        return "#9E9E9E"  # Default gray
    
    def _calculate_urgency_level(self, urgency: float) -> str:
        """Calculate urgency level from urgency score"""
        if urgency >= 0.9:
            return "immediate"
        elif urgency >= 0.7:
            return "high"
        elif urgency >= 0.4:
            return "moderate"
        else:
            return "low"
    
    def _calculate_opacity(self, urgency: float) -> float:
        """Calculate opacity based on urgency"""
        return 0.3 + (urgency * 0.7)  # Range from 0.3 to 1.0
    
    def _determine_node_phase(self, node_type: str) -> str:
        """Determine phase from node type"""
        for phase_name, phase_config in self.phase_map.items():
            node_types = phase_config.get("node_types", [])
            if any(node_type.startswith(nt) for nt in node_types):
                return phase_name
        return "alpha"  # Default phase
    
    def _determine_agent_status(self, score: float, node_count: int) -> str:
        """Determine agent status"""
        if node_count == 0:
            return "idle"
        elif score > 0.7:
            return "active"
        elif score < 0.3:
            return "struggling"
        else:
            return "working"
    
    def _calculate_avg_urgency(self, nodes: List[Dict[str, Any]]) -> float:
        """Calculate average urgency across all nodes"""
        urgency_values = {"immediate": 1.0, "high": 0.8, "moderate": 0.5, "low": 0.2}
        total_urgency = sum(urgency_values.get(n.get("urgency", "low"), 0.2) for n in nodes)
        return total_urgency / max(1, len(nodes))
    
    def _calculate_signal_distribution(self, nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of design signals"""
        distribution = {}
        for node in nodes:
            signal = node.get("design_signal", "neutral_state")
            distribution[signal] = distribution.get(signal, 0) + 1
        return distribution
    
    def save_dashboard_state(self, dashboard_state: Dict[str, Any], output_path: str = "engine/dashboard_state.json"):
        """Save dashboard state to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dashboard_state, f, indent=2)

# Example usage for node_engine.py integration
def update_dashboard_from_engine(engine_nodes: List[Dict[str, Any]], 
                                engine_agents: List[Dict[str, Any]]):
    """
    Function to be called from node_engine.py every tick/update
    Pure data pipe - no UI logic
    """
    generator = DashboardStateGenerator()
    dashboard_state = generator.generate_dashboard_state(engine_nodes, engine_agents)
    generator.save_dashboard_state(dashboard_state)
    return dashboard_state

if __name__ == "__main__":
    # Test with sample data
    sample_nodes = [
        {"id": "V01_Product_001", "type": "V01_ProductComponent", "score": 0.8},
        {"id": "V02_Economic_001", "type": "V02_EconomicProfile", "score": 0.3},
        {"id": "V05_Compliance_001", "type": "V05_ComplianceCheck", "score": 0.95}
    ]
    
    sample_agents = [
        {"id": "agent_001", "active_nodes": ["V01_Product_001"], "score": 0.7, "last_action": "evaluate"},
        {"id": "agent_002", "active_nodes": ["V02_Economic_001", "V05_Compliance_001"], "score": 0.6, "last_action": "optimize"}
    ]
    
    dashboard_state = update_dashboard_from_engine(sample_nodes, sample_agents)
    print("✅ Dashboard state generated:", len(dashboard_state["nodes"]), "nodes") 