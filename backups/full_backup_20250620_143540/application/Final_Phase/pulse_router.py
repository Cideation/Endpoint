#!/usr/bin/env python3
"""
Pulse Router - Computation Layer (Mutable)
Dispatches interaction signals to functors based on semantic pulse types
Part of ECM-Pulse separation: ECM relays, Pulse Router computes
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Complete 7-pulse system definitions
PULSE_DEFINITIONS = {
    "bid_pulse": {
        "description": "Competitive intent, pricing, offers",
        "color": "#FFC107",
        "direction": "downward",
        "target_node_type": "BiddingAgent",
        "visual_label": "Bid",
        "urgency": "high"
    },
    "occupancy_pulse": {
        "description": "Spatial request, usage claim",
        "color": "#2196F3", 
        "direction": "upward",
        "target_node_type": "OccupancyNode",
        "visual_label": "Occupancy",
        "urgency": "medium"
    },
    "compliancy_pulse": {
        "description": "Enforcement, constraints, violations",
        "color": "#1E3A8A",
        "direction": "lateral",
        "target_node_type": "ComplianceNode", 
        "visual_label": "Compliance",
        "urgency": "high"
    },
    "fit_pulse": {
        "description": "Geometric, contextual match",
        "color": "#4CAF50",
        "direction": "diagonal",
        "target_node_type": "FitNode",
        "visual_label": "Fit",
        "urgency": "medium"
    },
    "investment_pulse": {
        "description": "Capital signal, resource allocation",
        "color": "#FF9800",
        "direction": "downward", 
        "target_node_type": "InvestmentNode",
        "visual_label": "Investment",
        "urgency": "high"
    },
    "decay_pulse": {
        "description": "Decline, obsolescence, reset readiness",
        "color": "#9E9E9E",
        "direction": "downward",
        "target_node_type": "Any",
        "visual_label": "Decay", 
        "urgency": "low"
    },
    "reject_pulse": {
        "description": "Immediate denial of proposal/interaction",
        "color": "#F44336",
        "direction": "reflexive",
        "target_node_type": "Any",
        "visual_label": "Reject",
        "urgency": "immediate"
    }
}

class PulseRouter:
    """
    Semantic pulse router for BEM system
    Mutable computation layer that evolves with system logic
    """
    
    def __init__(self):
        self.pulse_history = []
        self.active_pulses = {}
        self.functor_registry = {}
        
    def detect_pulse_type(self, message: Dict[str, Any]) -> Optional[str]:
        """
        Semantic detection of pulse type from message content
        """
        pulse_type = message.get("type")
        
        # Validate against 7-pulse system
        if pulse_type in PULSE_DEFINITIONS:
            return pulse_type
            
        return None
    
    def route_pulse(self, pulse_type: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route pulse to appropriate functors based on type and direction
        """
        if pulse_type not in PULSE_DEFINITIONS:
            return {"status": "error", "message": "Unknown pulse type"}
        
        pulse_def = PULSE_DEFINITIONS[pulse_type]
        
        # Create routing decision
        routing_decision = {
            "pulse_type": pulse_type,
            "direction": pulse_def["direction"],
            "target_node_type": pulse_def["target_node_type"],
            "urgency": pulse_def["urgency"],
            "color": pulse_def["color"],
            "timestamp": datetime.now().isoformat(),
            "original_message": message
        }
        
        # Store pulse for tracking
        self.pulse_history.append(routing_decision)
        self.active_pulses[pulse_type] = routing_decision
        
        return routing_decision
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing entry point from ECM Gateway
        """
        try:
            # Detect pulse type
            pulse_type = self.detect_pulse_type(message)
            
            if not pulse_type:
                return {
                    "status": "no_pulse_detected",
                    "message": "Message did not trigger any pulse"
                }
            
            # Route pulse to functors
            routing_result = self.route_pulse(pulse_type, message)
            
            return {
                "status": "pulse_processed",
                "pulse_type": pulse_type,
                "routing": routing_result
            }
            
        except Exception as e:
            logger.error(f"Pulse processing error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

# Global pulse router instance
pulse_router = PulseRouter()

def process_ecm_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for messages from ECM Gateway
    """
    return pulse_router.process_message(message) 