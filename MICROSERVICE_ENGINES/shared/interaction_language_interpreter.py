#!/usr/bin/env python3
"""
Semantic Interaction Language Interpreter
Converts raw system states into human-interpretable visual and behavioral outputs
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class InteractionState:
    """Represents the complete interaction state of a node/agent"""
    design_signal: str
    signal_intent: str
    agent_feedback: str
    interaction_mode: str
    gradient_energy: str
    urgency_index: str
    composite_expressions: List[str]
    timestamp: datetime
    
@dataclass
class VisualOutput:
    """Visual representation derived from interaction state"""
    color: str
    opacity: float
    glow_intensity: float
    pulse_pattern: str
    animation_duration: int
    visual_priority: str
    semantic_meaning: str

class InteractionLanguageInterpreter:
    """
    Core interpreter for the semantic interaction language system
    Bridges raw data and human perception without cosmetic UI logic
    """
    
    def __init__(self, language_config_path: str = None):
        self.language_config = {}
        self.interaction_history = []
        self.active_states = {}
        
        # Load interaction language configuration
        if language_config_path is None:
            language_config_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'graph_hints', 
                'interaction_language.json'
            )
        
        self.load_language_config(language_config_path)
        logger.info("🗣️ Interaction Language Interpreter initialized")
    
    def load_language_config(self, config_path: str):
        """Load the interaction language configuration"""
        try:
            with open(config_path, 'r') as f:
                self.language_config = json.load(f)
            logger.info(f"✅ Loaded interaction language config: {len(self.language_config)} categories")
        except FileNotFoundError:
            logger.warning(f"⚠️ Interaction language config not found: {config_path}")
            self.language_config = self._get_fallback_config()
        except Exception as e:
            logger.error(f"❌ Failed to load interaction language config: {e}")
            self.language_config = self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback configuration if file loading fails"""
        return {
            "design_signal": {
                "neutral_state": {"color": "#9E9E9E", "intent": "wait", "urgency": 0.5}
            },
            "signal_intent": {
                "wait": {"trigger": "idle", "pulse": "dim", "intensity": 0.3}
            }
        }
    
    def evaluate_design_interaction(self, score: float, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate design signal based on SDFA score and context
        Core function that maps quantitative scores to semantic meaning
        """
        design_signals = self.language_config.get("design_signal", {})
        
        # Find appropriate design signal based on score and urgency thresholds
        selected_signal = None
        for signal_name, signal_config in design_signals.items():
            urgency_threshold = signal_config.get("urgency", 0.5)
            
            # Different logic for different signal types
            if signal_name == "critical_failure" and score < 0.1:
                selected_signal = signal_name
                break
            elif signal_name == "evolutionary_peak" and score > 0.8:
                selected_signal = signal_name
                break
            elif signal_name == "learning_phase" and context and context.get("is_learning", False):
                selected_signal = signal_name
                break
            elif signal_name == "low_precision" and score < 0.3:
                selected_signal = signal_name
                break
            elif signal_name == "neutral_state" and 0.3 <= score <= 0.8:
                selected_signal = signal_name
                break
        
        # Default to neutral if no specific signal matches
        if not selected_signal:
            selected_signal = "neutral_state"
        
        signal_config = design_signals.get(selected_signal, {})
        
        return {
            "design_signal": selected_signal,
            "color": signal_config.get("color", "#9E9E9E"),
            "intent": signal_config.get("intent", "wait"),
            "urgency": signal_config.get("urgency", 0.5),
            "visual_cue": signal_config.get("visual_cue", "steady_pulse"),
            "semantic_meaning": signal_config.get("semantic_meaning", "Unknown state"),
            "score": score
        }
    
    def interpret_signal_intent(self, intent: str, node_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert signal intent to actionable triggers and visual outputs"""
        signal_intents = self.language_config.get("signal_intent", {})
        intent_config = signal_intents.get(intent, signal_intents.get("wait", {}))
        
        return {
            "trigger": intent_config.get("trigger", "idle"),
            "pulse": intent_config.get("pulse", "dim"),
            "action_type": intent_config.get("action_type", "monitor"),
            "propagation": intent_config.get("propagation", "local"),
            "duration": intent_config.get("duration", 5000),
            "intensity": intent_config.get("intensity", 0.3),
            "node_id": node_state.get("id", "unknown"),
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_urgency_index(self, change_rate: float, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate urgency based on rate of change and current state"""
        urgency_levels = self.language_config.get("urgency_index", {})
        
        # Determine urgency level based on change rate
        if change_rate >= 0.9:
            urgency_level = "immediate"
        elif change_rate >= 0.7:
            urgency_level = "high"
        elif change_rate >= 0.4:
            urgency_level = "moderate"
        else:
            urgency_level = "low"
        
        urgency_config = urgency_levels.get(urgency_level, urgency_levels.get("low", {}))
        
        return {
            "urgency_level": urgency_level,
            "flash": urgency_config.get("flash", "fade"),
            "rank": urgency_config.get("rank", 4),
            "change_rate": change_rate,
            "attention_demand": urgency_config.get("attention_demand", "minimal"),
            "visual_priority": urgency_config.get("visual_priority", "background")
        }
    
    def map_human_sense(self, sense_type: str, intensity: float) -> Dict[str, Any]:
        """Map human sensory experience to graph equivalent"""
        human_mapping = self.language_config.get("human_sense_mapping", {})
        sense_config = human_mapping.get(sense_type, {})
        
        graph_equivalent = sense_config.get("graph_equivalent", "neutral_state")
        threshold = sense_config.get("threshold", 0.5)
        
        # Determine if threshold is met
        threshold_met = intensity >= threshold
        
        return {
            "sense_type": sense_type,
            "graph_equivalent": graph_equivalent,
            "intensity": intensity,
            "threshold": threshold,
            "threshold_met": threshold_met,
            "response": sense_config.get("response", "no_action"),
            "visual_metaphor": sense_config.get("visual_metaphor", "neutral_display")
        }
    
    def generate_composite_expression(self, node_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate composite expressions from multiple signal types"""
        composite_expressions = self.language_config.get("composite_expressions", {})
        
        # Extract current signals from node state
        current_urgency = node_state.get("urgency_level", "low")
        current_intent = node_state.get("signal_intent", "wait")
        current_energy = node_state.get("gradient_energy", "low_intensity")
        current_mode = node_state.get("interaction_mode", "passive")
        
        # Check for matching composite expressions
        for expression_name, expression_config in composite_expressions.items():
            combines = expression_config.get("combines", [])
            
            # Simple matching logic - can be enhanced
            if any(signal in str(node_state.values()) for signal in combines):
                return {
                    "expression_name": expression_name,
                    "visual_result": expression_config.get("visual_result", "default_display"),
                    "semantic_meaning": expression_config.get("semantic_meaning", "Complex system state"),
                    "components": combines,
                    "node_state": node_state
                }
        
        return {
            "expression_name": "basic_state",
            "visual_result": "standard_node_display",
            "semantic_meaning": "Standard operational state",
            "components": [],
            "node_state": node_state
        }
    
    def create_interaction_state(self, node_data: Dict[str, Any]) -> InteractionState:
        """Create complete interaction state from node data"""
        
        # Evaluate design interaction
        score = node_data.get("score", 0.5)
        design_eval = self.evaluate_design_interaction(score, node_data)
        
        # Determine signal intent
        intent = design_eval["intent"]
        intent_result = self.interpret_signal_intent(intent, node_data)
        
        # Calculate urgency
        change_rate = node_data.get("change_rate", 0.1)
        urgency_result = self.calculate_urgency_index(change_rate, node_data)
        
        # Generate composite expression
        composite_result = self.generate_composite_expression(node_data)
        
        return InteractionState(
            design_signal=design_eval["design_signal"],
            signal_intent=intent_result["trigger"],
            agent_feedback=node_data.get("agent_feedback", "neutral_observation"),
            interaction_mode=node_data.get("interaction_mode", "passive"),
            gradient_energy=node_data.get("gradient_energy", "low_intensity"),
            urgency_index=urgency_result["urgency_level"],
            composite_expressions=[composite_result["expression_name"]],
            timestamp=datetime.now()
        )
    
    def create_visual_output(self, interaction_state: InteractionState) -> VisualOutput:
        """Convert interaction state to visual output"""
        
        # Get design signal configuration
        design_signals = self.language_config.get("design_signal", {})
        design_config = design_signals.get(interaction_state.design_signal, {})
        
        # Get interaction mode configuration
        interaction_modes = self.language_config.get("interaction_mode", {})
        mode_config = interaction_modes.get(interaction_state.interaction_mode, {})
        
        # Get gradient energy configuration
        gradient_energies = self.language_config.get("gradient_energy", {})
        energy_config = gradient_energies.get(interaction_state.gradient_energy, {})
        
        # Get urgency configuration
        urgency_levels = self.language_config.get("urgency_index", {})
        urgency_config = urgency_levels.get(interaction_state.urgency_index, {})
        
        return VisualOutput(
            color=design_config.get("color", "#9E9E9E"),
            opacity=mode_config.get("opacity", 0.5),
            glow_intensity=energy_config.get("energy_level", 0.3),
            pulse_pattern=urgency_config.get("flash", "fade"),
            animation_duration=2000,  # Default duration
            visual_priority=urgency_config.get("visual_priority", "normal"),
            semantic_meaning=design_config.get("semantic_meaning", "Unknown state")
        )
    
    def process_node_for_interaction(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing function - converts raw node data to interaction language output
        This is the primary interface for other systems
        """
        try:
            # Create interaction state
            interaction_state = self.create_interaction_state(node_data)
            
            # Generate visual output
            visual_output = self.create_visual_output(interaction_state)
            
            # Store in active states
            node_id = node_data.get("id", "unknown")
            self.active_states[node_id] = {
                "interaction_state": interaction_state,
                "visual_output": visual_output,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to history
            self.interaction_history.append({
                "node_id": node_id,
                "interaction_state": interaction_state,
                "visual_output": visual_output,
                "timestamp": datetime.now().isoformat()
            })
            
            # Return complete interaction package
            return {
                "node_id": node_id,
                "interaction_state": {
                    "design_signal": interaction_state.design_signal,
                    "signal_intent": interaction_state.signal_intent,
                    "agent_feedback": interaction_state.agent_feedback,
                    "interaction_mode": interaction_state.interaction_mode,
                    "gradient_energy": interaction_state.gradient_energy,
                    "urgency_index": interaction_state.urgency_index,
                    "composite_expressions": interaction_state.composite_expressions
                },
                "visual_output": {
                    "color": visual_output.color,
                    "opacity": visual_output.opacity,
                    "glow_intensity": visual_output.glow_intensity,
                    "pulse_pattern": visual_output.pulse_pattern,
                    "animation_duration": visual_output.animation_duration,
                    "visual_priority": visual_output.visual_priority,
                    "semantic_meaning": visual_output.semantic_meaning
                },
                "raw_data": node_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing node interaction: {e}")
            return self._get_fallback_interaction_output(node_data)
    
    def _get_fallback_interaction_output(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback output when processing fails"""
        return {
            "node_id": node_data.get("id", "unknown"),
            "interaction_state": {
                "design_signal": "neutral_state",
                "signal_intent": "wait",
                "agent_feedback": "neutral_observation",
                "interaction_mode": "passive",
                "gradient_energy": "low_intensity",
                "urgency_index": "low",
                "composite_expressions": ["basic_state"]
            },
            "visual_output": {
                "color": "#9E9E9E",
                "opacity": 0.5,
                "glow_intensity": 0.3,
                "pulse_pattern": "fade",
                "animation_duration": 2000,
                "visual_priority": "normal",
                "semantic_meaning": "Neutral system state"
            },
            "raw_data": node_data,
            "timestamp": datetime.now().isoformat(),
            "error": "Fallback interaction output"
        }
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of current interaction language state"""
        return {
            "active_nodes": len(self.active_states),
            "total_interactions": len(self.interaction_history),
            "language_categories": list(self.language_config.keys()),
            "recent_interactions": self.interaction_history[-5:] if self.interaction_history else [],
            "system_status": "operational"
        }

# Convenience functions for direct use
def evaluate_design_interaction(score: float, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Standalone function for design interaction evaluation"""
    interpreter = InteractionLanguageInterpreter()
    return interpreter.evaluate_design_interaction(score, context)

def process_node_interaction(node_data: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone function for complete node interaction processing"""
    interpreter = InteractionLanguageInterpreter()
    return interpreter.process_node_for_interaction(node_data)

if __name__ == "__main__":
    # Test the interaction language interpreter
    interpreter = InteractionLanguageInterpreter()
    
    # Test with sample node data
    sample_node = {
        "id": "V01_ProductComponent_001",
        "score": 0.7,
        "change_rate": 0.8,
        "is_learning": True,
        "interaction_mode": "active",
        "gradient_energy": "high_intensity"
    }
    
    result = interpreter.process_node_for_interaction(sample_node)
    print("🗣️ Interaction Language Test Result:")
    print(json.dumps(result, indent=2)) 