#!/usr/bin/env python3
"""
Unreal Pulse Handler - 3D Visualization Bridge

Handles bidirectional communication between Unreal Engine and the pulse system.
Converts semantic pulses into 3D visual effects with directional flow animations.
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - UNREAL-PULSE - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unreal_pulse.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UNREAL_PULSE")

# Import pulse definitions
PULSE_DEFINITIONS = {
    "bid_pulse": {"color": "#FFC107", "direction": "downward", "visual_label": "Bid"},
    "occupancy_pulse": {"color": "#2196F3", "direction": "upward", "visual_label": "Occupancy"},
    "compliancy_pulse": {"color": "#1E3A8A", "direction": "cross-subtree", "visual_label": "Compliance"},
    "fit_pulse": {"color": "#4CAF50", "direction": "lateral", "visual_label": "Fit Check"},
    "investment_pulse": {"color": "#FF9800", "direction": "broadcast", "visual_label": "Investment"},
    "decay_pulse": {"color": "#9E9E9E", "direction": "downward", "visual_label": "Decay"},
    "reject_pulse": {"color": "#F44336", "direction": "reflexive", "visual_label": "Rejected"}
}

@dataclass
class UnrealPulseVisualization:
    """3D pulse visualization data for Unreal Engine"""
    pulse_type: str
    color: str
    direction: str
    position: Dict[str, float]
    intensity: float
    duration: float
    visual_effects: List[str]

class UnrealPulseHandler:
    """Bridges pulse system with Unreal Engine 3D visualization"""
    
    def __init__(self, ecm_port: int = 8765, unreal_port: int = 8767):
        self.ecm_port = ecm_port
        self.unreal_port = unreal_port
        self.unreal_clients = set()
        self.ecm_connection = None
        self.is_running = False
        self.pulse_queue = []
        
        logger.info(f"Unreal Pulse Handler initialized - Unreal port: {unreal_port}")
    
    async def start_handler(self):
        """Start the Unreal pulse handler"""
        logger.info("🎮 Starting Unreal Pulse Handler...")
        self.is_running = True
        
        # Start Unreal server and ECM connection
        tasks = [
            self._start_unreal_server(),
            self._connect_to_ecm(),
            self._process_pulse_visualizations()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Unreal Pulse Handler...")
            self.is_running = False
    
    async def _start_unreal_server(self):
        """Start WebSocket server for Unreal Engine"""
        async def handle_unreal_client(websocket, path):
            self.unreal_clients.add(websocket)
            client_addr = websocket.remote_address
            logger.info(f"🎮 Unreal Engine connected: {client_addr}")
            
            try:
                # Send handshake with pulse system info
                await websocket.send(json.dumps({
                    'type': 'unreal_handshake',
                    'supported_pulses': list(PULSE_DEFINITIONS.keys()),
                    'timestamp': datetime.now().isoformat()
                }))
                
                # Handle Unreal messages
                async for message in websocket:
                    await self._handle_unreal_message(websocket, json.loads(message))
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"🎮 Unreal Engine disconnected: {client_addr}")
            finally:
                self.unreal_clients.discard(websocket)
        
        server = await websockets.serve(handle_unreal_client, "localhost", self.unreal_port)
        logger.info(f"🎮 Unreal server running on ws://localhost:{self.unreal_port}")
        await server.wait_closed()
    
    async def _connect_to_ecm(self):
        """Connect to ECM Gateway for pulse data"""
        while self.is_running:
            try:
                uri = f"ws://localhost:{self.ecm_port}"
                async with websockets.connect(uri) as websocket:
                    self.ecm_connection = websocket
                    logger.info("✅ Connected to ECM Gateway")
                    
                    async for message in websocket:
                        await self._handle_ecm_pulse(json.loads(message))
                        
            except Exception as e:
                logger.warning(f"ECM connection error: {e}, retrying in 3s...")
                await asyncio.sleep(3)
    
    async def _handle_ecm_pulse(self, message: Dict[str, Any]):
        """Handle pulse data from ECM Gateway"""
        pulse_type = message.get('type', 'unknown')
        
        if pulse_type in PULSE_DEFINITIONS:
            # Create 3D visualization
            visualization = self._create_3d_visualization(pulse_type, message)
            self.pulse_queue.append(visualization)
            logger.info(f"📡 Received pulse: {pulse_type}")
    
    async def _handle_unreal_message(self, websocket, message: Dict[str, Any]):
        """Handle messages from Unreal Engine"""
        msg_type = message.get('type')
        
        if msg_type == 'spatial_event':
            # Forward spatial events to ECM
            if self.ecm_connection:
                await self.ecm_connection.send(json.dumps(message))
                logger.info(f"🎮 Forwarded spatial event to ECM")
        
        elif msg_type == 'pulse_request':
            # Unreal requesting specific pulse visualization
            pulse_type = message.get('pulse_type')
            if pulse_type in PULSE_DEFINITIONS:
                visualization = self._create_3d_visualization(pulse_type, message)
                await self._send_to_unreal(websocket, {
                    'type': 'pulse_visualization',
                    'visualization': visualization.__dict__
                })
    
    def _create_3d_visualization(self, pulse_type: str, message: Dict[str, Any]) -> UnrealPulseVisualization:
        """Create 3D visualization data for Unreal Engine"""
        pulse_def = PULSE_DEFINITIONS[pulse_type]
        
        return UnrealPulseVisualization(
            pulse_type=pulse_type,
            color=pulse_def['color'],
            direction=pulse_def['direction'],
            position=message.get('position', {'x': 0, 'y': 0, 'z': 100}),
            intensity=self._get_pulse_intensity(pulse_type),
            duration=self._get_pulse_duration(pulse_type),
            visual_effects=self._get_visual_effects(pulse_type, pulse_def['direction'])
        )
    
    def _get_pulse_intensity(self, pulse_type: str) -> float:
        """Get visual intensity for pulse type"""
        intensities = {
            'bid_pulse': 0.8,
            'occupancy_pulse': 0.6,
            'compliancy_pulse': 1.0,
            'fit_pulse': 0.4,
            'investment_pulse': 0.9,
            'decay_pulse': 0.2,
            'reject_pulse': 1.0
        }
        return intensities.get(pulse_type, 0.5)
    
    def _get_pulse_duration(self, pulse_type: str) -> float:
        """Get pulse duration in seconds"""
        durations = {
            'bid_pulse': 2.0,
            'occupancy_pulse': 3.0,
            'compliancy_pulse': 1.5,
            'fit_pulse': 1.0,
            'investment_pulse': 2.5,
            'decay_pulse': 4.0,
            'reject_pulse': 0.5
        }
        return durations.get(pulse_type, 2.0)
    
    def _get_visual_effects(self, pulse_type: str, direction: str) -> List[str]:
        """Get visual effects for pulse"""
        effects = ['particle_system', 'color_glow']
        
        # Direction-specific effects
        direction_effects = {
            'downward': ['cascade_particles', 'gravity_flow'],
            'upward': ['rising_stream', 'upward_spiral'],
            'cross-subtree': ['lateral_waves', 'horizontal_spread'],
            'lateral': ['side_ripples', 'peer_connections'],
            'broadcast': ['radial_explosion', 'expanding_rings'],
            'reflexive': ['instant_flash', 'rejection_burst']
        }
        
        effects.extend(direction_effects.get(direction, []))
        
        # Pulse-specific effects
        if pulse_type == 'bid_pulse':
            effects.append('attention_pulse')
        elif pulse_type == 'investment_pulse':
            effects.append('golden_sparkles')
        elif pulse_type == 'compliancy_pulse':
            effects.append('authority_aura')
        elif pulse_type == 'reject_pulse':
            effects.append('rejection_x_mark')
        
        return effects
    
    async def _process_pulse_visualizations(self):
        """Process queued pulse visualizations"""
        while self.is_running:
            if self.pulse_queue and self.unreal_clients:
                visualization = self.pulse_queue.pop(0)
                
                # Broadcast to all Unreal clients
                message = {
                    'type': 'pulse_visualization',
                    'visualization': {
                        'pulse_type': visualization.pulse_type,
                        'color': visualization.color,
                        'direction': visualization.direction,
                        'position': visualization.position,
                        'intensity': visualization.intensity,
                        'duration': visualization.duration,
                        'visual_effects': visualization.visual_effects
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                await self._broadcast_to_unreal(message)
                logger.info(f"🎨 Sent visualization: {visualization.pulse_type}")
            
            await asyncio.sleep(0.1)
    
    async def _send_to_unreal(self, websocket, message: Dict[str, Any]):
        """Send message to specific Unreal client"""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send to Unreal: {e}")
    
    async def _broadcast_to_unreal(self, message: Dict[str, Any]):
        """Broadcast to all Unreal clients"""
        disconnected = []
        for client in self.unreal_clients:
            try:
                await client.send(json.dumps(message))
            except Exception:
                disconnected.append(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            self.unreal_clients.discard(client)

async def main():
    """Main entry point"""
    handler = UnrealPulseHandler()
    await handler.start_handler()

if __name__ == "__main__":
    asyncio.run(main()) 