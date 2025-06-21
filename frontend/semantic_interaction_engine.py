#!/usr/bin/env python3
"""
Semantic Interaction Engine - Frontend Integration
Bridges the interaction language interpreter with real-time GraphQL and WebSocket systems
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add paths for system integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MICROSERVICE_ENGINES', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Final_Phase'))

try:
    from interaction_language_interpreter import InteractionLanguageInterpreter, process_node_interaction
    INTERPRETER_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ Interaction Language Interpreter not available")
    INTERPRETER_AVAILABLE = False

try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter
    from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL
    STRAWBERRY_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ Strawberry GraphQL not available")
    STRAWBERRY_AVAILABLE = False

logger = logging.getLogger(__name__)

class SemanticInteractionEngine:
    """
    Frontend engine that processes semantic interactions and broadcasts them via GraphQL
    Connects raw system data to meaningful human-interpretable visual outputs
    """
    
    def __init__(self):
        self.interpreter = InteractionLanguageInterpreter() if INTERPRETER_AVAILABLE else None
        self.active_interactions = {}
        self.interaction_subscribers = set()
        self.pulse_queue = asyncio.Queue()
        self.running = False
        
        logger.info("🎨 Semantic Interaction Engine initialized")
    
    async def start_engine(self):
        """Start the semantic interaction processing engine"""
        self.running = True
        logger.info("🚀 Starting Semantic Interaction Engine...")
        
        # Start background tasks
        asyncio.create_task(self._process_interaction_queue())
        asyncio.create_task(self._broadcast_pulse_updates())
        
        logger.info("✅ Semantic Interaction Engine running")
    
    async def stop_engine(self):
        """Stop the semantic interaction engine"""
        self.running = False
        logger.info("🛑 Stopping Semantic Interaction Engine")
    
    async def process_node_interaction(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a node through the interaction language system
        Returns semantic visual output for frontend rendering
        """
        if not self.interpreter:
            return self._get_fallback_interaction(node_data)
        
        try:
            # Process through interaction language interpreter
            interaction_result = self.interpreter.process_node_for_interaction(node_data)
            
            # Extract key components for frontend
            visual_output = interaction_result.get("visual_output", {})
            interaction_state = interaction_result.get("interaction_state", {})
            
            # Create frontend-ready interaction package
            frontend_interaction = {
                "node_id": node_data.get("id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                
                # Visual properties for direct CSS/Canvas application
                "visual": {
                    "color": visual_output.get("color", "#9E9E9E"),
                    "opacity": visual_output.get("opacity", 0.5),
                    "glow_intensity": visual_output.get("glow_intensity", 0.3),
                    "pulse_pattern": visual_output.get("pulse_pattern", "fade"),
                    "animation_duration": visual_output.get("animation_duration", 2000),
                    "visual_priority": visual_output.get("visual_priority", "normal")
                },
                
                # Semantic meaning for human interpretation
                "semantic": {
                    "meaning": visual_output.get("semantic_meaning", "Unknown state"),
                    "design_signal": interaction_state.get("design_signal", "neutral_state"),
                    "signal_intent": interaction_state.get("signal_intent", "wait"),
                    "urgency_level": interaction_state.get("urgency_index", "low"),
                    "interaction_mode": interaction_state.get("interaction_mode", "passive")
                },
                
                # Action triggers for system behavior
                "actions": {
                    "trigger_type": self._map_intent_to_trigger(interaction_state.get("signal_intent", "wait")),
                    "propagation_type": self._get_propagation_type(interaction_state),
                    "should_activate_neighbors": interaction_state.get("signal_intent") == "broadcast",
                    "energy_level": interaction_state.get("gradient_energy", "low_intensity")
                },
                
                # Raw data for debugging/logging
                "raw_data": node_data,
                "interaction_state": interaction_state
            }
            
            # Store active interaction
            node_id = node_data.get("id", "unknown")
            self.active_interactions[node_id] = frontend_interaction
            
            # Queue for broadcasting
            await self.pulse_queue.put({
                "type": "interaction_update",
                "data": frontend_interaction
            })
            
            return frontend_interaction
            
        except Exception as e:
            logger.error(f"❌ Error processing node interaction: {e}")
            return self._get_fallback_interaction(node_data)
    
    def _map_intent_to_trigger(self, signal_intent: str) -> str:
        """Map signal intent to frontend trigger type"""
        intent_mapping = {
            "broadcast": "activate_neighbors",
            "idle": "maintain_state", 
            "fallback": "suppress_activity",
            "alert": "emergency_response",
            "morph": "adaptive_change"
        }
        return intent_mapping.get(signal_intent, "maintain_state")
    
    def _get_propagation_type(self, interaction_state: Dict[str, Any]) -> str:
        """Determine how interaction should propagate through the graph"""
        intent = interaction_state.get("signal_intent", "wait")
        mode = interaction_state.get("interaction_mode", "passive")
        
        if intent == "broadcast" and mode == "relational":
            return "network_cascade"
        elif intent == "alert":
            return "global_alert"
        elif mode == "active":
            return "local_activation"
        else:
            return "no_propagation"
    
    def _get_fallback_interaction(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback interaction when interpreter is unavailable"""
        return {
            "node_id": node_data.get("id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "visual": {
                "color": "#9E9E9E",
                "opacity": 0.5,
                "glow_intensity": 0.3,
                "pulse_pattern": "fade",
                "animation_duration": 2000,
                "visual_priority": "normal"
            },
            "semantic": {
                "meaning": "Neutral system state (fallback)",
                "design_signal": "neutral_state",
                "signal_intent": "wait",
                "urgency_level": "low",
                "interaction_mode": "passive"
            },
            "actions": {
                "trigger_type": "maintain_state",
                "propagation_type": "no_propagation",
                "should_activate_neighbors": False,
                "energy_level": "low_intensity"
            },
            "raw_data": node_data,
            "fallback": True
        }
    
    async def _process_interaction_queue(self):
        """Background task to process interaction updates"""
        while self.running:
            try:
                # Process queued interactions
                if not self.pulse_queue.empty():
                    pulse_data = await self.pulse_queue.get()
                    await self._handle_pulse_data(pulse_data)
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
            except Exception as e:
                logger.error(f"❌ Error in interaction queue processing: {e}")
                await asyncio.sleep(1)
    
    async def _handle_pulse_data(self, pulse_data: Dict[str, Any]):
        """Handle individual pulse data items"""
        try:
            pulse_type = pulse_data.get("type", "unknown")
            data = pulse_data.get("data", {})
            
            if pulse_type == "interaction_update":
                # Process interaction update
                node_id = data.get("node_id", "unknown")
                
                # Check if neighbors should be activated
                if data.get("actions", {}).get("should_activate_neighbors", False):
                    await self._activate_neighbors(node_id, data)
                
                # Log significant interactions
                urgency = data.get("semantic", {}).get("urgency_level", "low")
                if urgency in ["high", "immediate"]:
                    logger.info(f"🚨 High urgency interaction: {node_id} - {data.get('semantic', {}).get('meaning', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Error handling pulse data: {e}")
    
    async def _activate_neighbors(self, node_id: str, interaction_data: Dict[str, Any]):
        """Activate neighboring nodes based on interaction propagation"""
        try:
            propagation_type = interaction_data.get("actions", {}).get("propagation_type", "no_propagation")
            
            if propagation_type == "network_cascade":
                # Simulate network cascade activation
                logger.info(f"🌊 Network cascade triggered from {node_id}")
                
                # Queue cascade effects for neighboring nodes
                cascade_data = {
                    "type": "cascade_effect",
                    "source_node": node_id,
                    "propagation_type": propagation_type,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.pulse_queue.put(cascade_data)
            
            elif propagation_type == "global_alert":
                # Global alert propagation
                logger.info(f"🚨 Global alert triggered from {node_id}")
                
                # Broadcast global alert
                alert_data = {
                    "type": "global_alert",
                    "source_node": node_id,
                    "alert_level": interaction_data.get("semantic", {}).get("urgency_level", "low"),
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.pulse_queue.put(alert_data)
        
        except Exception as e:
            logger.error(f"❌ Error activating neighbors: {e}")
    
    async def _broadcast_pulse_updates(self):
        """Broadcast pulse updates to subscribed clients"""
        while self.running:
            try:
                # Broadcast current active interactions to subscribers
                if self.interaction_subscribers and self.active_interactions:
                    broadcast_data = {
                        "timestamp": datetime.now().isoformat(),
                        "active_interactions": len(self.active_interactions),
                        "recent_interactions": list(self.active_interactions.values())[-10:],  # Last 10
                        "system_status": "operational"
                    }
                    
                    # In a real implementation, this would broadcast via WebSocket/GraphQL subscriptions
                    logger.debug(f"📡 Broadcasting to {len(self.interaction_subscribers)} subscribers")
                
                await asyncio.sleep(1)  # Broadcast every second
                
            except Exception as e:
                logger.error(f"❌ Error in pulse broadcasting: {e}")
                await asyncio.sleep(5)
    
    def add_interaction_subscriber(self, subscriber_id: str):
        """Add a subscriber for interaction updates"""
        self.interaction_subscribers.add(subscriber_id)
        logger.info(f"➕ Added interaction subscriber: {subscriber_id}")
    
    def remove_interaction_subscriber(self, subscriber_id: str):
        """Remove a subscriber for interaction updates"""
        self.interaction_subscribers.discard(subscriber_id)
        logger.info(f"➖ Removed interaction subscriber: {subscriber_id}")
    
    def get_active_interactions(self) -> Dict[str, Any]:
        """Get current active interactions"""
        return {
            "active_count": len(self.active_interactions),
            "interactions": self.active_interactions,
            "subscribers": len(self.interaction_subscribers),
            "queue_size": self.pulse_queue.qsize(),
            "engine_running": self.running
        }
    
    async def simulate_interaction_demo(self):
        """Simulate interaction demo for testing"""
        logger.info("🎭 Starting interaction demo simulation...")
        
        demo_nodes = [
            {
                "id": "V01_ProductComponent_001",
                "score": 0.8,
                "change_rate": 0.7,
                "is_learning": True,
                "interaction_mode": "active"
            },
            {
                "id": "V02_UserEconomicProfile_001", 
                "score": 0.3,
                "change_rate": 0.9,
                "is_learning": False,
                "interaction_mode": "relational"
            },
            {
                "id": "V05_ComplianceCheck_001",
                "score": 0.95,
                "change_rate": 0.2,
                "is_learning": False,
                "interaction_mode": "passive"
            }
        ]
        
        for node_data in demo_nodes:
            interaction_result = await self.process_node_interaction(node_data)
            logger.info(f"🎨 Processed interaction for {node_data['id']}: {interaction_result['semantic']['meaning']}")
            await asyncio.sleep(1)
        
        logger.info("✅ Interaction demo simulation complete")

# GraphQL Integration (if Strawberry is available)
if STRAWBERRY_AVAILABLE:
    @strawberry.type
    class InteractionOutput:
        node_id: str
        color: str
        opacity: float
        glow_intensity: float
        pulse_pattern: str
        semantic_meaning: str
        urgency_level: str
        timestamp: str
    
    @strawberry.type
    class Query:
        @strawberry.field
        def get_active_interactions(self) -> List[InteractionOutput]:
            # This would integrate with the SemanticInteractionEngine
            return []
    
    @strawberry.type  
    class Subscription:
        @strawberry.subscription
        async def interaction_updates(self) -> InteractionOutput:
            # This would yield real-time interaction updates
            yield InteractionOutput(
                node_id="demo",
                color="#3F51B5",
                opacity=0.8,
                glow_intensity=0.7,
                pulse_pattern="bright",
                semantic_meaning="Demo interaction",
                urgency_level="moderate",
                timestamp=datetime.now().isoformat()
            )

# Main execution for testing
if __name__ == "__main__":
    async def main():
        engine = SemanticInteractionEngine()
        await engine.start_engine()
        
        # Run demo simulation
        await engine.simulate_interaction_demo()
        
        # Keep running for a bit
        await asyncio.sleep(10)
        
        await engine.stop_engine()
    
    asyncio.run(main()) 