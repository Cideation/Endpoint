#!/usr/bin/env python3
"""
Unreal Engine Integration Client for ECM Gateway

Connects Unreal Engine spatial events to the ECM Infrastructure Gateway.
This handles the actual Unreal integration for Phase 3.

Architecture:
Unreal Engine → Unreal Integration Client → ECM Gateway → Node Engine
              (spatial events)     (bridge)    (infrastructure)  (computation)
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import websockets
from websockets.client import WebSocketClientProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - UNREAL-CLIENT - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UNREAL_CLIENT")

class UnrealSpatialEvent:
    """
    Represents a spatial event from Unreal Engine
    """
    def __init__(self, event_type: str, spatial_data: Dict[str, Any], timestamp: Optional[datetime] = None):
        self.event_type = event_type
        self.spatial_data = spatial_data
        self.timestamp = timestamp or datetime.now()
        self.event_id = f"unreal_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    
    def to_ecm_message(self) -> Dict[str, Any]:
        """Convert to ECM-compatible message format"""
        return {
            'type': 'unreal_spatial_event',
            'event_id': self.event_id,
            'event_type': self.event_type,
            'spatial_data': self.spatial_data,
            'timestamp': self.timestamp.isoformat(),
            'source': 'unreal_engine'
        }

class UnrealECMClient:
    """
    Unreal Engine client for ECM Infrastructure Gateway
    
    Handles spatial event transmission from Unreal to ECM Gateway
    """
    
    def __init__(self, ecm_host: str = "localhost", ecm_port: int = 8765):
        self.ecm_host = ecm_host
        self.ecm_port = ecm_port
        self.ecm_uri = f"ws://{ecm_host}:{ecm_port}"
        
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connection_id: Optional[str] = None
        self.is_connected = False
        
        # Event handling
        self.event_handlers: Dict[str, Callable] = {}
        self.pending_events: List[UnrealSpatialEvent] = []
        
        logger.info(f"Unreal ECM Client initialized for {self.ecm_uri}")
    
    async def connect_to_ecm(self):
        """
        Connect to ECM Infrastructure Gateway
        """
        try:
            logger.info(f"Connecting to ECM Gateway at {self.ecm_uri}")
            self.websocket = await websockets.connect(self.ecm_uri)
            
            # Wait for ECM handshake
            handshake_message = await self.websocket.recv()
            handshake = json.loads(handshake_message)
            
            if handshake.get('type') == 'ecm_handshake':
                self.connection_id = handshake.get('connection_id')
                self.is_connected = True
                logger.info(f"Connected to ECM Gateway - Connection ID: {self.connection_id}")
                
                # Send any pending events
                await self._send_pending_events()
                
                return True
            else:
                logger.error(f"Unexpected handshake response: {handshake}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to ECM Gateway: {e}")
            self.is_connected = False
            return False
    
    async def send_spatial_event(self, event: UnrealSpatialEvent) -> bool:
        """
        Send spatial event to ECM Gateway
        """
        if not self.is_connected or not self.websocket:
            logger.warning("Not connected to ECM - queuing event")
            self.pending_events.append(event)
            return False
        
        try:
            message = event.to_ecm_message()
            await self.websocket.send(json.dumps(message))
            
            # Wait for ECM acknowledgment
            response = await self.websocket.recv()
            ack = json.loads(response)
            
            if ack.get('type') == 'ecm_bridge_ack':
                logger.info(f"Spatial event bridged: {event.event_id} -> {ack.get('ecm_message_id')}")
                return True
            else:
                logger.warning(f"Unexpected ECM response: {ack}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send spatial event: {e}")
            self.is_connected = False
            return False
    
    async def _send_pending_events(self):
        """Send any events that were queued while disconnected"""
        while self.pending_events:
            event = self.pending_events.pop(0)
            success = await self.send_spatial_event(event)
            if not success:
                # Re-queue if failed
                self.pending_events.insert(0, event)
                break
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register handler for specific spatial event types"""
        self.event_handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    async def listen_for_responses(self):
        """
        Listen for responses from ECM Gateway
        """
        if not self.is_connected or not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                response = json.loads(message)
                await self._handle_ecm_response(response)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("ECM connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error listening for ECM responses: {e}")
            self.is_connected = False
    
    async def _handle_ecm_response(self, response: Dict[str, Any]):
        """Handle responses from ECM Gateway"""
        response_type = response.get('type')
        
        if response_type == 'ecm_bridge_ack':
            # Standard acknowledgment - already handled in send_spatial_event
            pass
        elif response_type == 'ecm_error':
            logger.error(f"ECM Error: {response.get('message')}")
        else:
            logger.info(f"ECM Response: {response}")
    
    async def disconnect(self):
        """Disconnect from ECM Gateway"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from ECM Gateway")

# Unreal Engine Spatial Event Generators
class UnrealSpatialEventGenerator:
    """
    Simulates Unreal Engine spatial events for testing
    In production, this would be replaced by actual Unreal Engine integration
    """
    
    def __init__(self, client: UnrealECMClient):
        self.client = client
    
    async def simulate_player_movement(self, location: Dict[str, float], rotation: Dict[str, float]):
        """Simulate player movement in Unreal"""
        event = UnrealSpatialEvent(
            event_type='player_movement',
            spatial_data={
                'location': location,
                'rotation': rotation,
                'velocity': {'x': 0.5, 'y': 0.2, 'z': 0.0},
                'level': 'main_level'
            }
        )
        await self.client.send_spatial_event(event)
    
    async def simulate_object_interaction(self, object_id: str, interaction_type: str, location: Dict[str, float]):
        """Simulate object interaction in Unreal"""
        event = UnrealSpatialEvent(
            event_type='object_interaction',
            spatial_data={
                'object_id': object_id,
                'interaction_type': interaction_type,
                'location': location,
                'hit_normal': {'x': 0.0, 'y': 0.0, 'z': 1.0},
                'force': 150.0
            }
        )
        await self.client.send_spatial_event(event)
    
    async def simulate_spatial_query(self, query_type: str, parameters: Dict[str, Any]):
        """Simulate spatial query in Unreal"""
        event = UnrealSpatialEvent(
            event_type='spatial_query',
            spatial_data={
                'query_type': query_type,
                'parameters': parameters,
                'timestamp': datetime.now().isoformat()
            }
        )
        await self.client.send_spatial_event(event)
    
    async def simulate_construction_event(self, building_data: Dict[str, Any]):
        """Simulate construction/building event in Unreal"""
        event = UnrealSpatialEvent(
            event_type='construction_event',
            spatial_data={
                'action': building_data.get('action', 'place'),
                'component_type': building_data.get('component_type', 'wall'),
                'location': building_data.get('location', {'x': 0, 'y': 0, 'z': 0}),
                'dimensions': building_data.get('dimensions', {'length': 5.0, 'width': 0.2, 'height': 3.0}),
                'material': building_data.get('material', 'concrete'),
                'cost_estimate': building_data.get('cost_estimate', 1000.0)
            }
        )
        await self.client.send_spatial_event(event)

# Example usage and testing
async def main():
    """
    Example usage of Unreal ECM Client
    """
    logger.info("Starting Unreal Engine ECM Integration Test")
    
    # Create client
    client = UnrealECMClient()
    
    # Connect to ECM Gateway
    if not await client.connect_to_ecm():
        logger.error("Failed to connect to ECM Gateway")
        return
    
    # Create event generator for testing
    generator = UnrealSpatialEventGenerator(client)
    
    # Start listening for responses
    listen_task = asyncio.create_task(client.listen_for_responses())
    
    try:
        # Simulate various Unreal Engine events
        logger.info("Simulating Unreal Engine spatial events...")
        
        # Player movement
        await generator.simulate_player_movement(
            location={'x': 100.0, 'y': 200.0, 'z': 50.0},
            rotation={'pitch': 0.0, 'yaw': 90.0, 'roll': 0.0}
        )
        
        await asyncio.sleep(1)
        
        # Object interaction
        await generator.simulate_object_interaction(
            object_id='wall_001',
            interaction_type='select',
            location={'x': 150.0, 'y': 200.0, 'z': 50.0}
        )
        
        await asyncio.sleep(1)
        
        # Spatial query
        await generator.simulate_spatial_query(
            query_type='proximity_search',
            parameters={
                'center': {'x': 100.0, 'y': 200.0, 'z': 50.0},
                'radius': 500.0,
                'filter': 'structural_components'
            }
        )
        
        await asyncio.sleep(1)
        
        # Construction event
        await generator.simulate_construction_event({
            'action': 'place',
            'component_type': 'beam',
            'location': {'x': 200.0, 'y': 300.0, 'z': 60.0},
            'dimensions': {'length': 10.0, 'width': 0.4, 'height': 0.6},
            'material': 'steel',
            'cost_estimate': 2500.0
        })
        
        # Wait a bit to see responses
        await asyncio.sleep(3)
        
    except KeyboardInterrupt:
        logger.info("Stopping Unreal ECM integration...")
    finally:
        listen_task.cancel()
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 