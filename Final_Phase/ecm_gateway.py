#!/usr/bin/env python3
"""
ECM (Event Coordination Module) - Infrastructure Gateway

💡 FUNDAMENTAL PRINCIPLE: "Infrastructure is immutable; computation is emergent."

🎯 CRITICAL ARCHITECTURE: ECM-Pulse Separation
✅ ECM = Immutable infrastructure (pure relay, fixed behavior, stable foundation)
✅ Pulse = Emergent computation (interprets, triggers, decides, adapts)

Pure computational infrastructure for spatial computation bridging.
- Fixed, reliable, persistent gateway
- No functor logic, dynamic routing, or emergent behavior
- Stateless execution with persistent connection
- Bridge only - transmits and receives without interpretation
- Agnostic to agent types, functors, or AC processing
- Audit-safe with full message logging and traceability

Architecture Position:
Unreal Engine ←→ ECM Gateway ←→ Node Engine / Pulse Handler
              (infrastructure)    (computation)
              
ECM Role: Log, validate structure, timestamp, relay downstream
Pulse Role: Interpret content, trigger functors, update state
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Set, Optional, Dict, Any

# Configure logging with more detailed format
logging.basicConfig(
    filename="ecm_log.txt", 
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode='a'
)
logger = logging.getLogger(__name__)

# Global state for connected clients
connected_clients: Set[websockets.WebSocketServerProtocol] = set()

def safe_json_loads(raw_msg: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON with error handling"""
    try:
        return json.loads(raw_msg)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected JSON parsing error: {e}")
        return None

def safe_json_dumps(data: Dict[str, Any]) -> str:
    """Safely serialize JSON with error handling"""
    try:
        return json.dumps(data)
    except TypeError as e:
        logger.error(f"JSON encode error: {e}")
        return json.dumps({"error": "Serialization failed", "type": "json_error"})
    except Exception as e:
        logger.error(f"Unexpected JSON serialization error: {e}")
        return json.dumps({"error": "Unknown serialization error", "type": "json_error"})

async def safe_send(websocket: websockets.WebSocketServerProtocol, message: Dict[str, Any]) -> bool:
    """Safely send message through WebSocket with error handling"""
    try:
        if websocket.closed:
            logger.warning("Attempted to send to closed WebSocket")
            return False
            
        json_message = safe_json_dumps(message)
        await websocket.send(json_message)
        return True
        
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed during send")
        return False
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket exception during send: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during send: {e}")
        return False

async def ecm_handler(websocket: websockets.WebSocketServerProtocol, path: str):
    """Enhanced WebSocket handler with comprehensive error boundaries"""
    
    # Add client to connected set
    connected_clients.add(websocket)
    client_address = websocket.remote_address
    logger.info(f"Client connected: {client_address}")
    
    try:
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "connected",
            "message": "ECM Gateway ready"
        }
        
        if not await safe_send(websocket, welcome_msg):
            logger.error("Failed to send welcome message")
            return
            
        # Main message processing loop
        async for raw_msg in websocket:
            timestamp = datetime.utcnow().isoformat()
            
            try:
                # Parse incoming message
                message = safe_json_loads(raw_msg)
                
                if message is None:
                    # Invalid JSON received
                    error_response = {
                        "timestamp": timestamp,
                        "status": "error",
                        "error": "Invalid JSON format",
                        "type": "json_parse_error"
                    }
                    await safe_send(websocket, error_response)
                    continue
                
                # Log valid inbound message
                logger.info(f"[RECEIVED] {timestamp} - {message}")
                
                # Validate message structure
                message_type = message.get("type", "unknown")
                payload = message.get("payload", {})
                
                # ECM DOES NOT EVALUATE LOGIC — only relays
                outbound = {
                    "timestamp": timestamp,
                    "status": "received",
                    "echo": payload,
                    "type": message_type,
                    "original_message_id": message.get("id"),
                    "processing_status": "relayed"
                }
                
                # Send response
                success = await safe_send(websocket, outbound)
                if success:
                    logger.info(f"[SENT] {timestamp} - {outbound}")
                else:
                    logger.error(f"[FAILED_SEND] {timestamp} - Failed to send response")
                
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client {client_address} disconnected during message processing")
                break
            except Exception as e:
                # Handle any other exceptions during message processing
                logger.error(f"Message processing error: {e}")
                
                error_response = {
                    "timestamp": timestamp,
                    "status": "error",
                    "error": str(e),
                    "type": "processing_error"
                }
                
                # Attempt to send error response
                await safe_send(websocket, error_response)
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_address} connection closed")
    except Exception as e:
        logger.error(f"Handler error for client {client_address}: {e}")
    finally:
        # Clean up client connection
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected: {client_address}")

async def health_check_endpoint():
    """Simple health check for monitoring"""
    while True:
        try:
            logger.info(f"ECM Health: {len(connected_clients)} connected clients")
            await asyncio.sleep(30)  # Health check every 30 seconds
        except Exception as e:
            logger.error(f"Health check error: {e}")
            await asyncio.sleep(30)

def launch_ecm(host: str = "0.0.0.0", port: int = 8765):
    """Launch ECM WebSocket server with enhanced error handling"""
    print(f"📡 ECM WebSocket launching on ws://{host}:{port}")
    logger.info(f"ECM Gateway starting on {host}:{port}")
    
    # Configure WebSocket server with timeouts and limits
    return websockets.serve(
        ecm_handler,
        host,
        port,
        ping_interval=20,
        ping_timeout=10,
        close_timeout=10,
        max_size=1024*1024,  # 1MB max message size
        max_queue=32  # Max queued messages
    )

async def main():
    """Main entry point with health monitoring"""
    print("🚀 Starting ECM Gateway with enhanced error handling...")
    
    # Start WebSocket server
    server = launch_ecm()
    
    # Start health check task
    health_task = asyncio.create_task(health_check_endpoint())
    
    print("✅ ECM Gateway ready for connections")
    print("📊 Health monitoring active")
    
    # Run server and health check concurrently
    await asyncio.gather(server, health_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 ECM Gateway shutdown requested")
        logger.info("ECM Gateway shutdown by user")
    except Exception as e:
        print(f"❌ ECM Gateway startup error: {e}")
        logger.error(f"Startup error: {e}") 