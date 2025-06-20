"""
BEM System Interaction Flow Coordinator
Manages the complete flow: User → AA → ECM → DB → Visualization
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import websockets
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserActionType(Enum):
    """Types of user actions in the system"""
    NODE_CLICK = "node_click"
    EDGE_CLICK = "edge_click"
    NODE_DRAG = "node_drag"
    ZOOM = "zoom"
    PAN = "pan"
    PULSE_TRIGGER = "pulse_trigger"
    DATA_REQUEST = "data_request"

class UserClassification(Enum):
    """AA user classifications based on behavior"""
    EXPLORER = "explorer"  # Browsing and viewing
    ANALYST = "analyst"    # Deep diving into data
    OPERATOR = "operator"  # Making changes
    MONITOR = "monitor"    # Watching for events
    ADMIN = "admin"        # System management

@dataclass
class UserAction:
    """Represents a user action in the system"""
    user_id: str
    action_type: UserActionType
    target_id: Optional[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    source: str  # "cytoscape" or "unreal"

@dataclass
class PulseDecision:
    """ECM pulse routing decision"""
    pulse_type: str
    target_nodes: List[str]
    intensity: float
    metadata: Dict[str, Any]

class InteractionFlowCoordinator:
    """Coordinates the complete interaction flow between all system components"""
    
    def __init__(self, db_config: Dict[str, str], ecm_url: str = "ws://localhost:8765"):
        self.db_config = db_config
        self.ecm_url = ecm_url
        self.user_sessions = {}  # Track user sessions and classifications
        self.action_history = {}  # Store recent actions per user
        
    async def process_user_interaction(self, action: UserAction) -> Dict[str, Any]:
        """
        Main flow orchestrator - processes user interaction through all steps:
        1. Log action
        2. Classify user (AA)
        3. Decide pulses (ECM)
        4. Log to database
        5. Update visualizations
        """
        flow_result = {
            "action_id": f"{action.user_id}_{int(time.time())}",
            "steps": []
        }
        
        try:
            # Step 1: Log user action
            logger.info(f"📝 Step 1: Logging action from {action.source}")
            self._log_user_action(action)
            flow_result["steps"].append({
                "step": 1,
                "status": "completed",
                "description": "User action logged"
            })
            
            # Step 2: AA Classification
            logger.info(f"🤖 Step 2: AA classifying user {action.user_id}")
            classification = await self._classify_user(action)
            flow_result["classification"] = classification.value
            flow_result["steps"].append({
                "step": 2,
                "status": "completed",
                "description": f"User classified as: {classification.value}"
            })
            
            # Step 3: ECM Pulse Routing
            logger.info(f"⚡ Step 3: ECM deciding pulse routing")
            pulse_decision = await self._decide_pulse_routing(action, classification)
            flow_result["pulse"] = {
                "type": pulse_decision.pulse_type,
                "targets": pulse_decision.target_nodes
            }
            flow_result["steps"].append({
                "step": 3,
                "status": "completed",
                "description": f"Pulse routed: {pulse_decision.pulse_type}"
            })
            
            # Step 4: Database Logging
            logger.info(f"💾 Step 4: Logging to PostgreSQL")
            await self._log_to_database(action, classification, pulse_decision)
            flow_result["steps"].append({
                "step": 4,
                "status": "completed",
                "description": "State logged to database"
            })
            
            # Step 5: Update Cytoscape View
            logger.info(f"🔄 Step 5: Updating Cytoscape view")
            cytoscape_update = await self._update_cytoscape(action, pulse_decision)
            flow_result["cytoscape_update"] = cytoscape_update
            flow_result["steps"].append({
                "step": 5,
                "status": "completed",
                "description": "Cytoscape view updated"
            })
            
            # Step 6: Send to Unreal (if connected)
            logger.info(f"🎮 Step 6: Sending pulse effects to Unreal")
            unreal_update = await self._send_to_unreal(pulse_decision)
            flow_result["unreal_update"] = unreal_update
            flow_result["steps"].append({
                "step": 6,
                "status": "completed",
                "description": "Unreal visualization updated"
            })
            
        except Exception as e:
            logger.error(f"Error in interaction flow: {str(e)}")
            flow_result["error"] = str(e)
            
        return flow_result
    
    def _log_user_action(self, action: UserAction):
        """Log user action for analysis"""
        user_id = action.user_id
        if user_id not in self.action_history:
            self.action_history[user_id] = []
        
        # Keep last 100 actions per user
        self.action_history[user_id].append(action)
        if len(self.action_history[user_id]) > 100:
            self.action_history[user_id].pop(0)
    
    async def _classify_user(self, action: UserAction) -> UserClassification:
        """
        Automated Admin (AA) classifies user based on behavior patterns
        """
        user_id = action.user_id
        recent_actions = self.action_history.get(user_id, [])
        
        # Analyze recent action patterns
        action_counts = {}
        for past_action in recent_actions[-20:]:  # Last 20 actions
            action_type = past_action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Classification logic
        if action_counts.get('data_request', 0) > 5:
            return UserClassification.ANALYST
        elif action_counts.get('node_drag', 0) > 3:
            return UserClassification.OPERATOR
        elif action_counts.get('pulse_trigger', 0) > 2:
            return UserClassification.ADMIN
        elif len(recent_actions) < 5:
            return UserClassification.EXPLORER
        else:
            return UserClassification.MONITOR
    
    async def _decide_pulse_routing(self, action: UserAction, classification: UserClassification) -> PulseDecision:
        """
        ECM decides which pulse to trigger based on action and user classification
        """
        pulse_map = {
            UserActionType.NODE_CLICK: {
                UserClassification.EXPLORER: "bid_pulse",
                UserClassification.ANALYST: "fit_pulse",
                UserClassification.OPERATOR: "occupancy_pulse",
                UserClassification.MONITOR: "compliance_pulse",
                UserClassification.ADMIN: "investment_pulse"
            },
            UserActionType.EDGE_CLICK: {
                UserClassification.EXPLORER: "fit_pulse",
                UserClassification.ANALYST: "compliance_pulse",
                UserClassification.OPERATOR: "investment_pulse",
                UserClassification.MONITOR: "decay_pulse",
                UserClassification.ADMIN: "bid_pulse"
            },
            UserActionType.PULSE_TRIGGER: {
                UserClassification.ADMIN: "reject_pulse",  # Only admins can trigger reject
                UserClassification.OPERATOR: "investment_pulse"
            }
        }
        
        # Get pulse type based on action and classification
        pulse_type = pulse_map.get(action.action_type, {}).get(
            classification, "bid_pulse"  # Default pulse
        )
        
        # Determine target nodes
        target_nodes = []
        if action.target_id:
            target_nodes.append(action.target_id)
            # Add connected nodes based on pulse type
            if pulse_type in ["bid_pulse", "investment_pulse"]:
                # These pulses propagate to neighbors
                target_nodes.extend(self._get_connected_nodes(action.target_id))
        
        return PulseDecision(
            pulse_type=pulse_type,
            target_nodes=target_nodes[:10],  # Limit to 10 nodes
            intensity=0.8 if classification == UserClassification.ADMIN else 0.5,
            metadata={
                "user_classification": classification.value,
                "action_type": action.action_type.value
            }
        )
    
    def _get_connected_nodes(self, node_id: str) -> List[str]:
        """Get nodes connected to the given node (mock implementation)"""
        # In real implementation, this would query the graph database
        return [f"node_{i}" for i in range(3)]
    
    async def _log_to_database(self, action: UserAction, classification: UserClassification, pulse: PulseDecision):
        """Log complete interaction flow to PostgreSQL"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Log interaction
                    cur.execute("""
                        INSERT INTO interaction_logs 
                        (user_id, action_type, source, classification, 
                         pulse_type, target_nodes, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        action.user_id,
                        action.action_type.value,
                        action.source,
                        classification.value,
                        pulse.pulse_type,
                        json.dumps(pulse.target_nodes),
                        action.timestamp
                    ))
                    
                    # Update user session
                    cur.execute("""
                        INSERT INTO user_sessions (user_id, classification, last_action)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id) 
                        DO UPDATE SET 
                            classification = EXCLUDED.classification,
                            last_action = EXCLUDED.last_action
                    """, (
                        action.user_id,
                        classification.value,
                        action.timestamp
                    ))
                    
                conn.commit()
        except Exception as e:
            logger.error(f"Database logging error: {str(e)}")
    
    async def _update_cytoscape(self, action: UserAction, pulse: PulseDecision) -> Dict[str, Any]:
        """Update Cytoscape visualization with new state"""
        update_data = {
            "type": "state_update",
            "pulse": {
                "type": pulse.pulse_type,
                "targets": pulse.target_nodes,
                "intensity": pulse.intensity
            },
            "user_action": {
                "type": action.action_type.value,
                "target": action.target_id
            }
        }
        
        # In real implementation, this would send WebSocket message to Cytoscape clients
        return update_data
    
    async def _send_to_unreal(self, pulse: PulseDecision) -> Dict[str, Any]:
        """Send pulse effects to Unreal Engine via ECM"""
        try:
            async with websockets.connect(self.ecm_url) as ws:
                message = {
                    "type": "pulse_effect",
                    "pulse_type": pulse.pulse_type,
                    "targets": pulse.target_nodes,
                    "intensity": pulse.intensity,
                    "timestamp": datetime.now().isoformat()
                }
                
                await ws.send(json.dumps(message))
                response = await ws.recv()
                
                return json.loads(response)
        except Exception as e:
            logger.error(f"Error sending to Unreal: {str(e)}")
            return {"status": "error", "message": str(e)}

# Example usage
async def main():
    """Example of using the interaction flow coordinator"""
    coordinator = InteractionFlowCoordinator(
        db_config={
            "host": "localhost",
            "database": "bem_system",
            "user": "bem_user",
            "password": "password"
        }
    )
    
    # Simulate user interaction
    action = UserAction(
        user_id="user123",
        action_type=UserActionType.NODE_CLICK,
        target_id="node_42",
        metadata={"x": 100, "y": 200},
        timestamp=datetime.now(),
        source="cytoscape"
    )
    
    # Process through complete flow
    result = await coordinator.process_user_interaction(action)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 