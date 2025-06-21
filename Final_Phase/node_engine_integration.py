#!/usr/bin/env python3
"""
Node Engine Integration with Semantic Interaction Language
Connects node processing with the interaction language interpreter
"""

import sys
import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add paths for system integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MICROSERVICE_ENGINES', 'shared'))

try:
    from interaction_language_interpreter import InteractionLanguageInterpreter, process_node_interaction
    INTERACTION_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ Interaction Language Interpreter not available")
    INTERACTION_AVAILABLE = False

logger = logging.getLogger(__name__)

class NodeEngineWithInteractionLanguage:
    """
    Enhanced node engine that processes nodes through the semantic interaction language
    Bridges computational node processing with human-interpretable outputs
    """
    
    def __init__(self):
        self.interaction_interpreter = InteractionLanguageInterpreter() if INTERACTION_AVAILABLE else None
        self.active_nodes = {}
        self.interaction_outputs = {}
        self.neighbor_activation_queue = []
        
        logger.info("🔧 Node Engine with Interaction Language initialized")
    
    def process_node_with_interaction(self, node_id: str, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a node through both computational logic and interaction language
        Returns enhanced node state with semantic interaction outputs
        """
        try:
            # Standard node processing
            processed_node = self._process_node_logic(node_id, node_data)
            
            # Apply interaction language interpretation
            if self.interaction_interpreter:
                interaction_result = self.interaction_interpreter.process_node_for_interaction(processed_node)
                
                # Merge computational and interaction results
                enhanced_node = {
                    **processed_node,
                    "interaction_language": interaction_result.get("interaction_state", {}),
                    "visual_output": interaction_result.get("visual_output", {}),
                    "semantic_meaning": interaction_result.get("visual_output", {}).get("semantic_meaning", "Unknown")
                }
                
                # Check for neighbor activation triggers
                if self._should_activate_neighbors(interaction_result):
                    self.neighbor_activation_queue.append({
                        "source_node": node_id,
                        "activation_type": interaction_result.get("interaction_state", {}).get("signal_intent", "wait"),
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Store interaction output
                self.interaction_outputs[node_id] = interaction_result
                
                return enhanced_node
            else:
                # Fallback without interaction language
                return {
                    **processed_node,
                    "interaction_language": {"design_signal": "neutral_state", "signal_intent": "wait"},
                    "visual_output": {"color": "#9E9E9E", "opacity": 0.5},
                    "semantic_meaning": "Standard node processing (no interaction language)"
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing node {node_id} with interaction language: {e}")
            return self._get_fallback_node_result(node_id, node_data)
    
    def _process_node_logic(self, node_id: str, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standard node processing logic"""
        
        # Simulate node computation
        input_value = node_data.get("input_value", 0.5)
        node_type = node_data.get("type", "unknown")
        
        # Different processing based on node type
        if node_type.startswith("V01_"):  # Product Component
            output_value = self._process_product_component(input_value, node_data)
        elif node_type.startswith("V02_"):  # Economic Profile
            output_value = self._process_economic_profile(input_value, node_data)
        elif node_type.startswith("V05_"):  # Compliance Check
            output_value = self._process_compliance_check(input_value, node_data)
        else:
            output_value = input_value * 0.8  # Default processing
        
        # Calculate node score and change rate
        previous_value = node_data.get("previous_value", input_value)
        change_rate = abs(output_value - previous_value) if previous_value != 0 else 0.1
        
        # Determine interaction mode based on processing results
        interaction_mode = self._determine_interaction_mode(output_value, change_rate, node_data)
        
        return {
            "id": node_id,
            "type": node_type,
            "input_value": input_value,
            "output_value": output_value,
            "score": output_value,
            "change_rate": change_rate,
            "interaction_mode": interaction_mode,
            "previous_value": previous_value,
            "processing_timestamp": datetime.now().isoformat(),
            "is_learning": change_rate > 0.5,  # High change rate indicates learning
            "gradient_energy": self._calculate_gradient_energy(change_rate),
            **node_data  # Include original data
        }
    
    def _process_product_component(self, input_value: float, node_data: Dict[str, Any]) -> float:
        """Process V01 Product Component nodes"""
        # Simulate product component evaluation
        quality_factor = node_data.get("quality_factor", 0.8)
        manufacturing_score = node_data.get("manufacturing_score", 0.7)
        
        # Combine factors
        output = (input_value * 0.4) + (quality_factor * 0.3) + (manufacturing_score * 0.3)
        return min(1.0, max(0.0, output))
    
    def _process_economic_profile(self, input_value: float, node_data: Dict[str, Any]) -> float:
        """Process V02 Economic Profile nodes"""
        # Simulate economic evaluation
        cost_efficiency = node_data.get("cost_efficiency", 0.6)
        market_demand = node_data.get("market_demand", 0.5)
        
        # Economic calculation
        output = (input_value * 0.5) + (cost_efficiency * 0.3) + (market_demand * 0.2)
        return min(1.0, max(0.0, output))
    
    def _process_compliance_check(self, input_value: float, node_data: Dict[str, Any]) -> float:
        """Process V05 Compliance Check nodes"""
        # Simulate compliance evaluation
        regulatory_score = node_data.get("regulatory_score", 0.9)
        safety_rating = node_data.get("safety_rating", 0.85)
        
        # Compliance is typically binary but we'll use weighted average
        output = (input_value * 0.2) + (regulatory_score * 0.4) + (safety_rating * 0.4)
        return min(1.0, max(0.0, output))
    
    def _determine_interaction_mode(self, output_value: float, change_rate: float, node_data: Dict[str, Any]) -> str:
        """Determine the interaction mode based on node state"""
        if change_rate > 0.7:
            return "relational"  # High change, needs interaction
        elif output_value > 0.8:
            return "active"  # High performance, actively participating
        elif output_value < 0.3:
            return "passive"  # Low performance, minimal interaction
        else:
            return "active"  # Default to active
    
    def _calculate_gradient_energy(self, change_rate: float) -> str:
        """Calculate gradient energy level based on change rate"""
        if change_rate > 0.8:
            return "high_intensity"
        elif change_rate > 0.5:
            return "medium_intensity"
        elif change_rate > 0.2:
            return "low_intensity"
        else:
            return "low_intensity"
    
    def _should_activate_neighbors(self, interaction_result: Dict[str, Any]) -> bool:
        """Check if neighbors should be activated based on interaction state"""
        signal_intent = interaction_result.get("interaction_state", {}).get("signal_intent", "wait")
        return signal_intent == "broadcast"
    
    def activate_neighbors(self, source_node_id: str) -> List[str]:
        """
        Activate neighboring nodes when broadcast signal is triggered
        This is the key function that implements: if node["intent"] == "broadcast": activate_neighbors(node_id)
        """
        activated_neighbors = []
        
        try:
            # Get neighbors (this would be implemented based on your graph structure)
            neighbor_ids = self._get_node_neighbors(source_node_id)
            
            for neighbor_id in neighbor_ids:
                if neighbor_id in self.active_nodes:
                    # Process neighbor with activation context
                    neighbor_data = self.active_nodes[neighbor_id].copy()
                    neighbor_data["activated_by"] = source_node_id
                    neighbor_data["activation_timestamp"] = datetime.now().isoformat()
                    
                    # Boost neighbor's input value due to activation
                    current_input = neighbor_data.get("input_value", 0.5)
                    neighbor_data["input_value"] = min(1.0, current_input + 0.1)
                    
                    # Reprocess the activated neighbor
                    enhanced_neighbor = self.process_node_with_interaction(neighbor_id, neighbor_data)
                    self.active_nodes[neighbor_id] = enhanced_neighbor
                    
                    activated_neighbors.append(neighbor_id)
                    
                    logger.info(f"🔗 Activated neighbor {neighbor_id} from {source_node_id}")
            
            if activated_neighbors:
                logger.info(f"📡 Broadcast from {source_node_id} activated {len(activated_neighbors)} neighbors")
            
        except Exception as e:
            logger.error(f"❌ Error activating neighbors for {source_node_id}: {e}")
        
        return activated_neighbors
    
    def _get_node_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring node IDs (implement based on your graph structure)"""
        # This is a simplified implementation - replace with actual graph traversal
        all_node_ids = list(self.active_nodes.keys())
        
        # Simple heuristic: nodes with similar prefixes are neighbors
        node_prefix = node_id.split('_')[0] if '_' in node_id else node_id[:3]
        neighbors = [nid for nid in all_node_ids if nid != node_id and nid.startswith(node_prefix)]
        
        # If no prefix matches, return random subset
        if not neighbors:
            neighbors = [nid for nid in all_node_ids if nid != node_id][:2]
        
        return neighbors
    
    def process_neighbor_activation_queue(self):
        """Process queued neighbor activations"""
        while self.neighbor_activation_queue:
            activation = self.neighbor_activation_queue.pop(0)
            source_node = activation["source_node"]
            
            activated_neighbors = self.activate_neighbors(source_node)
            
            if activated_neighbors:
                logger.info(f"🌊 Processed activation cascade from {source_node}: {activated_neighbors}")
    
    def render_pulse(self, node_id: str, pulse_type: str):
        """
        Render pulse visualization based on signal intent
        This implements: renderPulse(node.pulse) // from `signal_intent → pulse`
        """
        try:
            if node_id in self.interaction_outputs:
                interaction_state = self.interaction_outputs[node_id].get("interaction_state", {})
                visual_output = self.interaction_outputs[node_id].get("visual_output", {})
                
                pulse_data = {
                    "node_id": node_id,
                    "pulse_type": pulse_type,
                    "color": visual_output.get("color", "#9E9E9E"),
                    "intensity": visual_output.get("glow_intensity", 0.3),
                    "duration": visual_output.get("animation_duration", 2000),
                    "pattern": visual_output.get("pulse_pattern", "fade"),
                    "semantic_meaning": visual_output.get("semantic_meaning", "Unknown pulse"),
                    "timestamp": datetime.now().isoformat()
                }
                
                # This would be sent to ECM Gateway or frontend rendering system
                logger.info(f"🎨 Rendering pulse for {node_id}: {pulse_type} ({pulse_data['pattern']})")
                
                return pulse_data
            else:
                logger.warning(f"⚠️ No interaction output found for node {node_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error rendering pulse for {node_id}: {e}")
            return None
    
    def _get_fallback_node_result(self, node_id: str, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback result when processing fails"""
        return {
            "id": node_id,
            "type": node_data.get("type", "unknown"),
            "score": 0.5,
            "change_rate": 0.1,
            "interaction_mode": "passive",
            "interaction_language": {"design_signal": "neutral_state", "signal_intent": "wait"},
            "visual_output": {"color": "#9E9E9E", "opacity": 0.5},
            "semantic_meaning": "Fallback processing result",
            "error": "Processing failed, using fallback",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the node engine state"""
        return {
            "active_nodes": len(self.active_nodes),
            "interaction_outputs": len(self.interaction_outputs),
            "pending_activations": len(self.neighbor_activation_queue),
            "interaction_language_available": INTERACTION_AVAILABLE,
            "system_status": "operational"
        }

# Example usage function
def demo_node_engine_interaction():
    """Demonstrate the node engine with interaction language"""
    engine = NodeEngineWithInteractionLanguage()
    
    # Sample nodes
    sample_nodes = [
        {
            "id": "V01_ProductComponent_001",
            "type": "V01_ProductComponent",
            "input_value": 0.7,
            "quality_factor": 0.8,
            "manufacturing_score": 0.9
        },
        {
            "id": "V02_EconomicProfile_001", 
            "type": "V02_EconomicProfile",
            "input_value": 0.4,
            "cost_efficiency": 0.6,
            "market_demand": 0.8
        },
        {
            "id": "V05_ComplianceCheck_001",
            "type": "V05_ComplianceCheck", 
            "input_value": 0.9,
            "regulatory_score": 0.95,
            "safety_rating": 0.88
        }
    ]
    
    print("🔧 Node Engine with Interaction Language Demo")
    print("=" * 50)
    
    # Process each node
    for node_data in sample_nodes:
        print(f"\n🔄 Processing {node_data['id']}...")
        
        result = engine.process_node_with_interaction(node_data['id'], node_data)
        engine.active_nodes[node_data['id']] = result
        
        # Display results
        print(f"  Score: {result['score']:.3f}")
        print(f"  Design Signal: {result['interaction_language']['design_signal']}")
        print(f"  Signal Intent: {result['interaction_language']['signal_intent']}")
        print(f"  Semantic Meaning: {result['semantic_meaning']}")
        print(f"  Visual Color: {result['visual_output']['color']}")
        
        # Check for broadcast intent
        if result['interaction_language']['signal_intent'] == 'broadcast':
            print(f"  🔔 Broadcasting signal - will activate neighbors")
            
        # Render pulse
        pulse_data = engine.render_pulse(node_data['id'], result['visual_output']['pulse_pattern'])
        if pulse_data:
            print(f"  🎨 Pulse rendered: {pulse_data['pattern']} at {pulse_data['intensity']:.2f} intensity")
    
    # Process neighbor activations
    print(f"\n🌊 Processing neighbor activation queue...")
    engine.process_neighbor_activation_queue()
    
    # System summary
    summary = engine.get_system_summary()
    print(f"\n📊 System Summary:")
    print(f"  Active Nodes: {summary['active_nodes']}")
    print(f"  Interaction Outputs: {summary['interaction_outputs']}")
    print(f"  Interaction Language Available: {summary['interaction_language_available']}")

if __name__ == "__main__":
    demo_node_engine_interaction() 