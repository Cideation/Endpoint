#!/usr/bin/env python3
"""
ECM Pulse System - Segregated Pulse Integration

A specialized WebSocket system for pulse integration that operates 
completely independently from the Node Engine. This handles rhythmic,
periodic, and event-driven pulses that coordinate across the system.

Architecture:
ECM Gateway ←→ ECM Pulse System (segregated) ←→ Pulse Clients
           (infrastructure)    (pulse coordination)    (various systems)

Key Features:
- Segregated from Node Engine and functor layers
- Handles pulse timing, coordination, and distribution
- Infrastructure-level pulse reliability
- Audit-safe pulse logging
- Agnostic to computational logic
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Set, List, Optional, Callable
import websockets
from websockets.server import WebSocketServerProtocol
from dataclasses import dataclass
from enum import Enum

# Configure pulse logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ECM-PULSE - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecm_pulse_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ECM_PULSE")

class PulseType(str, Enum):
    """Types of pulses in the system"""
    HEARTBEAT = "heartbeat"          # Regular system heartbeat
    SYNCHRONIZATION = "sync"         # Cross-system synchronization
    EVENT_TRIGGER = "event_trigger"  # Event-driven pulse
    COORDINATION = "coordination"    # Multi-system coordination
    TIMING = "timing"               # Precise timing pulse
    INFRASTRUCTURE = "infrastructure" # Infrastructure status pulse

@dataclass
class Pulse:
    """
    Represents a pulse in the system
    """
    pulse_id: str
    pulse_type: PulseType
    timestamp: datetime
    interval_ms: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None
    source: str = "unknown"
    target: Optional[str] = None
    ttl_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pulse_id': self.pulse_id,
            'pulse_type': self.pulse_type.value,
            'timestamp': self.timestamp.isoformat(),
            'interval_ms': self.interval_ms,
            'payload': self.payload or {},
            'source': self.source,
            'target': self.target,
            'ttl_seconds': self.ttl_seconds
        }
    
    def is_expired(self) -> bool:
        """Check if pulse has expired"""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)

class PulseClient:
    """
    Represents a client connected to the pulse system
    """
    def __init__(self, websocket: WebSocketServerProtocol, client_id: str, client_type: str):
        self.websocket = websocket
        self.client_id = client_id
        self.client_type = client_type
        self.connected_at = datetime.now()
        self.last_pulse = None
        self.pulse_count = 0
        self.subscribed_pulse_types: Set[PulseType] = set()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'client_id': self.client_id,
            'client_type': self.client_type,
            'connected_at': self.connected_at.isoformat(),
            'last_pulse': self.last_pulse.isoformat() if self.last_pulse else None,
            'pulse_count': self.pulse_count,
            'subscribed_types': [pt.value for pt in self.subscribed_pulse_types]
        }

class ECMPulseSystem:
    """
    ECM Pulse System - Segregated pulse coordination infrastructure
    
    Responsibilities:
    - Maintain pulse timing and coordination
    - Distribute pulses to subscribed clients
    - Audit all pulse traffic
    - Provide reliable pulse infrastructure
    
    Non-Responsibilities:
    - No computational logic or behavior
    - No functor processing or agent logic
    - No dynamic routing based on content
    - No state mutation or interpretation
    """
    
    def __init__(self, host: str = "localhost", port: int = 8766):
        self.host = host
        self.port = port
        
        # Pulse infrastructure state
        self.pulse_clients: Dict[str, PulseClient] = {}
        self.active_pulses: Dict[str, Pulse] = {}
        self.pulse_history: List[Pulse] = []
        
        # Pulse generators (infrastructure-level)
        self.pulse_generators: Dict[str, Callable] = {}
        self.pulse_tasks: Set[asyncio.Task] = set()
        
        # System metrics
        self.system_start_time = datetime.now()
        self.total_pulses_sent = 0
        self.total_clients_served = 0
        
        logger.info(f"ECM Pulse System initialized - {host}:{port}")
    
    async def handle_pulse_connection(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle pulse client connections
        """
        client_id = str(uuid.uuid4())
        
        try:
            # Send pulse handshake
            handshake = {
                'type': 'pulse_handshake',
                'client_id': client_id,
                'pulse_system_version': '1.0.0',
                'available_pulse_types': [pt.value for pt in PulseType],
                'timestamp': datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(handshake))
            
            # Wait for client identification
            client_info_message = await websocket.recv()
            client_info = json.loads(client_info_message)
            
            client_type = client_info.get('client_type', 'unknown')
            pulse_client = PulseClient(websocket, client_id, client_type)
            
            # Register client
            self.pulse_clients[client_id] = pulse_client
            self.total_clients_served += 1
            
            logger.info(f"PULSE-CONNECT: {client_id} ({client_type}) from {websocket.remote_address}")
            
            # Handle client messages
            async for message in websocket:
                await self._handle_pulse_message(pulse_client, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"PULSE-DISCONNECT: {client_id} - Connection closed")
        except Exception as e:
            logger.error(f"PULSE-ERROR: {client_id} - {str(e)}")
        finally:
            # Cleanup
            if client_id in self.pulse_clients:
                del self.pulse_clients[client_id]
            logger.info(f"PULSE-CLEANUP: {client_id} - Client cleanup complete")
    
    async def _handle_pulse_message(self, client: PulseClient, raw_message: str):
        """
        Handle messages from pulse clients
        """
        try:
            message = json.loads(raw_message)
            message_type = message.get('type')
            
            # Audit pulse message
            audit_entry = {
                'client_id': client.client_id,
                'message_type': message_type,
                'timestamp': datetime.now().isoformat(),
                'payload_size': len(raw_message)
            }
            logger.info(f"PULSE-MESSAGE: {json.dumps(audit_entry)}")
            
            if message_type == 'subscribe_pulse':
                await self._handle_pulse_subscription(client, message)
            elif message_type == 'unsubscribe_pulse':
                await self._handle_pulse_unsubscription(client, message)
            elif message_type == 'send_pulse':
                await self._handle_pulse_send(client, message)
            elif message_type == 'pulse_status':
                await self._handle_pulse_status_request(client)
            else:
                # Unknown message type
                response = {
                    'type': 'pulse_error',
                    'error': 'unknown_message_type',
                    'message_type': message_type,
                    'timestamp': datetime.now().isoformat()
                }
                await client.websocket.send(json.dumps(response))
                
        except json.JSONDecodeError as e:
            error_response = {
                'type': 'pulse_error',
                'error': 'invalid_json',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            await client.websocket.send(json.dumps(error_response))
        except Exception as e:
            logger.error(f"PULSE-HANDLE-ERROR: {client.client_id} - {str(e)}")
    
    async def _handle_pulse_subscription(self, client: PulseClient, message: Dict[str, Any]):
        """Handle pulse subscription requests"""
        pulse_types = message.get('pulse_types', [])
        
        for pulse_type_str in pulse_types:
            try:
                pulse_type = PulseType(pulse_type_str)
                client.subscribed_pulse_types.add(pulse_type)
                logger.info(f"PULSE-SUBSCRIBE: {client.client_id} subscribed to {pulse_type.value}")
            except ValueError:
                logger.warning(f"PULSE-SUBSCRIBE-ERROR: Invalid pulse type {pulse_type_str}")
        
        # Send subscription confirmation
        response = {
            'type': 'pulse_subscription_ack',
            'subscribed_types': [pt.value for pt in client.subscribed_pulse_types],
            'timestamp': datetime.now().isoformat()
        }
        await client.websocket.send(json.dumps(response))
    
    async def _handle_pulse_unsubscription(self, client: PulseClient, message: Dict[str, Any]):
        """Handle pulse unsubscription requests"""
        pulse_types = message.get('pulse_types', [])
        
        for pulse_type_str in pulse_types:
            try:
                pulse_type = PulseType(pulse_type_str)
                client.subscribed_pulse_types.discard(pulse_type)
                logger.info(f"PULSE-UNSUBSCRIBE: {client.client_id} unsubscribed from {pulse_type.value}")
            except ValueError:
                logger.warning(f"PULSE-UNSUBSCRIBE-ERROR: Invalid pulse type {pulse_type_str}")
        
        # Send unsubscription confirmation
        response = {
            'type': 'pulse_unsubscription_ack',
            'subscribed_types': [pt.value for pt in client.subscribed_pulse_types],
            'timestamp': datetime.now().isoformat()
        }
        await client.websocket.send(json.dumps(response))
    
    async def _handle_pulse_send(self, client: PulseClient, message: Dict[str, Any]):
        """Handle pulse send requests from clients"""
        try:
            pulse_type = PulseType(message.get('pulse_type'))
            pulse = Pulse(
                pulse_id=str(uuid.uuid4()),
                pulse_type=pulse_type,
                timestamp=datetime.now(),
                interval_ms=message.get('interval_ms'),
                payload=message.get('payload'),
                source=client.client_id,
                target=message.get('target'),
                ttl_seconds=message.get('ttl_seconds', 60)
            )
            
            # Distribute pulse
            await self._distribute_pulse(pulse)
            
            # Send acknowledgment
            response = {
                'type': 'pulse_send_ack',
                'pulse_id': pulse.pulse_id,
                'timestamp': datetime.now().isoformat()
            }
            await client.websocket.send(json.dumps(response))
            
        except ValueError as e:
            error_response = {
                'type': 'pulse_error',
                'error': 'invalid_pulse_type',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            await client.websocket.send(json.dumps(error_response))
    
    async def _handle_pulse_status_request(self, client: PulseClient):
        """Handle pulse status requests"""
        status = {
            'type': 'pulse_status_response',
            'system_uptime': str(datetime.now() - self.system_start_time),
            'total_clients': len(self.pulse_clients),
            'total_pulses_sent': self.total_pulses_sent,
            'active_pulse_types': list(set(pt.value for c in self.pulse_clients.values() for pt in c.subscribed_pulse_types)),
            'client_info': client.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        await client.websocket.send(json.dumps(status))
    
    async def _distribute_pulse(self, pulse: Pulse):
        """
        Distribute pulse to subscribed clients
        """
        self.active_pulses[pulse.pulse_id] = pulse
        self.pulse_history.append(pulse)
        self.total_pulses_sent += 1
        
        # Keep pulse history manageable
        if len(self.pulse_history) > 1000:
            self.pulse_history = self.pulse_history[-1000:]
        
        # Distribute to subscribed clients
        clients_notified = 0
        for client in self.pulse_clients.values():
            if pulse.pulse_type in client.subscribed_pulse_types:
                try:
                    pulse_message = {
                        'type': 'pulse',
                        **pulse.to_dict()
                    }
                    await client.websocket.send(json.dumps(pulse_message))
                    client.pulse_count += 1
                    client.last_pulse = datetime.now()
                    clients_notified += 1
                except Exception as e:
                    logger.error(f"PULSE-SEND-ERROR: {client.client_id} - {str(e)}")
        
        # Audit pulse distribution
        logger.info(f"PULSE-DISTRIBUTED: {pulse.pulse_id} ({pulse.pulse_type.value}) to {clients_notified} clients")
    
    def start_infrastructure_pulses(self):
        """
        Start infrastructure-level pulse generators
        """
        # Heartbeat pulse
        async def heartbeat_generator():
            while True:
                pulse = Pulse(
                    pulse_id=str(uuid.uuid4()),
                    pulse_type=PulseType.HEARTBEAT,
                    timestamp=datetime.now(),
                    interval_ms=5000,
                    source="ecm_pulse_system",
                    payload={'system_status': 'operational'}
                )
                await self._distribute_pulse(pulse)
                await asyncio.sleep(5)  # 5-second heartbeat
        
        # Infrastructure status pulse
        async def infrastructure_status_generator():
            while True:
                pulse = Pulse(
                    pulse_id=str(uuid.uuid4()),
                    pulse_type=PulseType.INFRASTRUCTURE,
                    timestamp=datetime.now(),
                    interval_ms=30000,
                    source="ecm_pulse_system",
                    payload={
                        'uptime': str(datetime.now() - self.system_start_time),
                        'active_clients': len(self.pulse_clients),
                        'total_pulses': self.total_pulses_sent
                    }
                )
                await self._distribute_pulse(pulse)
                await asyncio.sleep(30)  # 30-second infrastructure status
        
        # Start pulse generators
        heartbeat_task = asyncio.create_task(heartbeat_generator())
        status_task = asyncio.create_task(infrastructure_status_generator())
        
        self.pulse_tasks.add(heartbeat_task)
        self.pulse_tasks.add(status_task)
        
        logger.info("Infrastructure pulse generators started")
    
    async def cleanup_expired_pulses(self):
        """
        Cleanup expired pulses
        """
        while True:
            expired_pulses = [
                pulse_id for pulse_id, pulse in self.active_pulses.items()
                if pulse.is_expired()
            ]
            
            for pulse_id in expired_pulses:
                del self.active_pulses[pulse_id]
            
            if expired_pulses:
                logger.info(f"PULSE-CLEANUP: Removed {len(expired_pulses)} expired pulses")
            
            await asyncio.sleep(60)  # Cleanup every minute
    
    async def start_pulse_system(self):
        """
        Start ECM Pulse System
        """
        logger.info(f"ECM Pulse System starting on ws://{self.host}:{self.port}")
        logger.info("Pulse System Features:")
        logger.info("  ✅ Segregated from Node Engine")
        logger.info("  ✅ Infrastructure-level pulse coordination")
        logger.info("  ✅ Audit-safe pulse logging")
        logger.info("  ✅ Reliable pulse distribution")
        logger.info("  ✅ Multiple pulse type support")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_pulse_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        # Start infrastructure pulse generators
        self.start_infrastructure_pulses()
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self.cleanup_expired_pulses())
        self.pulse_tasks.add(cleanup_task)
        
        logger.info(f"ECM Pulse System operational on ws://{self.host}:{self.port}")
        return server

# Main pulse system startup
async def main():
    """
    ECM Pulse System startup
    """
    pulse_system = ECMPulseSystem()
    
    # Start pulse system
    server = await pulse_system.start_pulse_system()
    
    try:
        # Keep pulse system running
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("ECM Pulse System shutting down...")
        
        # Cancel pulse tasks
        for task in pulse_system.pulse_tasks:
            task.cancel()
        
        server.close()
        await server.wait_closed()
        logger.info("ECM Pulse System stopped")

if __name__ == "__main__":
    asyncio.run(main()) 