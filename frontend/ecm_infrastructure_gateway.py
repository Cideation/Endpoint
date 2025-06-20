#!/usr/bin/env python3
"""
ECM (Event Coordination Module) - Infrastructure Gateway

🎯 CRITICAL ARCHITECTURE: ECM-Pulse Separation
✅ ECM = Pure relay (no pulse execution, no decisions)
✅ Pulse = Post-ECM delivery (interprets, triggers, decides)

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

# Configure logging
logging.basicConfig(filename="ecm_log.txt", level=logging.INFO, format="%(asctime)s %(message)s")

connected_clients = set()

async def ecm_handler(websocket, path):
    connected_clients.add(websocket)
    logging.info(f"Client connected: {websocket.remote_address}")

    try:
        async for raw_msg in websocket:
            timestamp = datetime.utcnow().isoformat()

            try:
                message = json.loads(raw_msg)
                # Log all inbound messages
                logging.info(f"[RECEIVED] {timestamp} - {message}")

                # ECM DOES NOT EVALUATE LOGIC — only relays
                outbound = {
                    "timestamp": timestamp,
                    "status": "received",
                    "echo": message.get("payload", {}),
                    "type": message.get("type", "unknown")
                }

                await websocket.send(json.dumps(outbound))
                logging.info(f"[SENT] {timestamp} - {outbound}")

            except Exception as e:
                error_msg = {"error": str(e), "type": "error"}
                await websocket.send(json.dumps(error_msg))
                logging.error(f"[ERROR] {timestamp} - {str(e)}")
    finally:
        connected_clients.remove(websocket)
        logging.info(f"Client disconnected: {websocket.remote_address}")

def launch_ecm(host="0.0.0.0", port=8765):
    print(f"📡 ECM WebSocket launched on ws://{host}:{port}")
    return websockets.serve(ecm_handler, host, port)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    server = launch_ecm()
    loop.run_until_complete(server)
    loop.run_forever() 