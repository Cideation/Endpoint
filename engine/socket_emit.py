#!/usr/bin/env python3
"""
Socket Emitter for Real-time Dashboard State Streaming
Streams structured state updates to dashboard via SocketIO
"""

import socketio
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logger = logging.getLogger(__name__)

class DashboardSocketEmitter:
    """
    Real-time state emitter for dashboard
    Streams pure data - no UI logic
    """
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self.sio = socketio.AsyncClient()
        self.connected = False
        self.emit_queue = asyncio.Queue()
        self.running = False
        
        # Setup event handlers
        self.sio.on('connect', self._on_connect)
        self.sio.on('disconnect', self._on_disconnect)
        
    async def _on_connect(self):
        """Handle connection to dashboard server"""
        self.connected = True
        logger.info(f"🔗 Connected to dashboard server: {self.server_url}")
    
    async def _on_disconnect(self):
        """Handle disconnection from dashboard server"""
        self.connected = False
        logger.info("🔌 Disconnected from dashboard server")
    
    async def connect(self):
        """Connect to dashboard server"""
        try:
            await self.sio.connect(self.server_url)
            self.running = True
            
            # Start emit queue processor
            asyncio.create_task(self._process_emit_queue())
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to dashboard server: {e}")
    
    async def disconnect(self):
        """Disconnect from dashboard server"""
        self.running = False
        if self.connected:
            await self.sio.disconnect()
    
    async def emit_agent_state_update(self, node_state_dict: Dict[str, Any]):
        """
        Emit agent state update - core function for node_engine.py
        Usage: sio.emit("agent_state_update", node_state_dict)
        """
        await self._queue_emit("agent_state_update", node_state_dict)
    
    async def emit_pulse_render(self, pulse_data: Dict[str, Any]):
        """
        Emit pulse rendering data
        Usage: socket.on("pulse_render", (data) => { renderPulse(data) })
        """
        await self._queue_emit("pulse_render", {
            "type": "pulse_render",
            "data": pulse_data,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _queue_emit(self, event_name: str, data: Dict[str, Any]):
        """Queue emit for processing"""
        await self.emit_queue.put({
            "event": event_name,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _process_emit_queue(self):
        """Process queued emits"""
        while self.running:
            try:
                if not self.emit_queue.empty():
                    emit_item = await self.emit_queue.get()
                    
                    if self.connected:
                        await self.sio.emit(emit_item["event"], emit_item["data"])
                        logger.debug(f"📡 Emitted {emit_item['event']}")
                    else:
                        logger.warning(f"⚠️ Not connected - queuing {emit_item['event']}")
                
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                logger.error(f"❌ Error processing emit queue: {e}")
                await asyncio.sleep(1)

# Global emitter instance
_dashboard_emitter: Optional[DashboardSocketEmitter] = None

async def get_dashboard_emitter(server_url: str = "http://localhost:5000") -> DashboardSocketEmitter:
    """Get or create global dashboard emitter"""
    global _dashboard_emitter
    
    if _dashboard_emitter is None:
        _dashboard_emitter = DashboardSocketEmitter(server_url)
        await _dashboard_emitter.connect()
    
    return _dashboard_emitter

# Convenience functions for node_engine.py integration
async def emit_node_state_update(node_state_dict: Dict[str, Any], server_url: str = "http://localhost:5000"):
    """
    Convenience function for node_engine.py
    Usage: await emit_node_state_update(node_state_dict)
    """
    try:
        emitter = await get_dashboard_emitter(server_url)
        await emitter.emit_agent_state_update(node_state_dict)
    except Exception as e:
        logger.error(f"❌ Failed to emit node state update: {e}")

async def emit_pulse_to_dashboard(pulse_data: Dict[str, Any], server_url: str = "http://localhost:5000"):
    """
    Convenience function for pulse rendering
    Usage: await emit_pulse_to_dashboard(pulse_data)
    """
    try:
        emitter = await get_dashboard_emitter(server_url)
        await emitter.emit_pulse_render(pulse_data)
    except Exception as e:
        logger.error(f"❌ Failed to emit pulse data: {e}")

# Synchronous wrappers for non-async contexts
def emit_node_state_sync(node_state_dict: Dict[str, Any], server_url: str = "http://localhost:5000"):
    """Synchronous wrapper for node state updates"""
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(emit_node_state_update(node_state_dict, server_url))
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(emit_node_state_update(node_state_dict, server_url))

def emit_pulse_sync(pulse_data: Dict[str, Any], server_url: str = "http://localhost:5000"):
    """Synchronous wrapper for pulse rendering"""
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(emit_pulse_to_dashboard(pulse_data, server_url))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(emit_pulse_to_dashboard(pulse_data, server_url))

if __name__ == "__main__":
    # Test the socket emitter
    async def test_emitter():
        emitter = DashboardSocketEmitter()
        await emitter.connect()
        
        # Test emit
        test_data = {
            "node_id": "V01_Product_001",
            "design_signal": "evolutionary_peak",
            "intent": "promote",
            "urgency": "high",
            "color": "#3F51B5"
        }
        
        await emitter.emit_agent_state_update(test_data)
        print("✅ Test emit completed")
        
        await emitter.disconnect()
    
    asyncio.run(test_emitter())

# Missing: Production-grade rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Missing: Comprehensive health checks
@app.route('/health/detailed')
def detailed_health():
    return {
        "database": check_db_connection(),
        "redis": check_redis_connection(),
        "websocket": check_websocket_health(),
        "disk_space": check_disk_space(),
        "memory_usage": get_memory_usage()
    }
