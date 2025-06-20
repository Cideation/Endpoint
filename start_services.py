#!/usr/bin/env python3
"""
Simple Service Launcher for Phase 2 Testing
Starts essential services for 100% integration test coverage
"""

import subprocess
import time
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_simple_api_server():
    """Start a simple API server for testing"""
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI()
    
    @app.get("/api/status")
    def get_status():
        return {"status": "running", "service": "dual_ac_api", "test_mode": True}
    
    @app.post("/cosmetic_ac")
    def cosmetic_ac_endpoint(data: dict):
        return {"success": True, "message": "Cosmetic AC processed", "data": data}
    
    @app.post("/unreal_ac") 
    def unreal_ac_endpoint(data: dict):
        return {"success": True, "message": "Unreal AC processed", "data": data}
    
    logger.info("🎮 Starting Simple Dual AC API on port 8002")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")

def start_simple_websocket_server():
    """Start a simple WebSocket server for ECM testing"""
    import asyncio
    import websockets
    import json
    
    async def ecm_handler(websocket, path):
        """Simple ECM message handler"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = {
                        "status": "received",
                        "echo": data.get("payload", {}),
                        "timestamp": data.get("timestamp"),
                        "type": data.get("type", "unknown")
                    }
                    await websocket.send(json.dumps(response))
                except:
                    await websocket.send('{"status": "error", "message": "Invalid JSON"}')
        except:
            pass
    
    logger.info("🌐 Starting Simple ECM Gateway on port 8765")
    start_server = websockets.serve(ecm_handler, "localhost", 8765)
    return start_server

def start_simple_pulse_server():
    """Start a simple pulse WebSocket server"""
    import asyncio
    import websockets
    import json
    
    async def pulse_handler(websocket, path):
        """Simple pulse message handler"""
        try:
            async for message in websocket:
                response = {"status": "pulse_received", "timestamp": "2024-01-01T00:00:00"}
                await websocket.send(json.dumps(response))
        except:
            pass
    
    logger.info("🔄 Starting Simple Pulse System on port 8766")
    start_server = websockets.serve(pulse_handler, "localhost", 8766)
    return start_server

async def start_websocket_servers():
    """Start both WebSocket servers"""
    ecm_server = start_simple_websocket_server()
    pulse_server = start_simple_pulse_server()
    
    await asyncio.gather(
        ecm_server,
        pulse_server
    )
    
    # Keep servers running
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    import multiprocessing
    
    if len(sys.argv) > 1:
        service = sys.argv[1]
        
        if service == "api":
            start_simple_api_server()
        elif service == "websockets":
            import asyncio
            asyncio.run(start_websocket_servers())
    else:
        logger.info("🚀 Starting all services for Phase 2 testing...")
        
        # Start API server in separate process
        api_process = multiprocessing.Process(target=start_simple_api_server)
        api_process.start()
        
        # Start WebSocket servers in separate process  
        def run_websockets():
            import asyncio
            asyncio.run(start_websocket_servers())
        
        ws_process = multiprocessing.Process(target=run_websockets)
        ws_process.start()
        
        logger.info("✅ All services started!")
        logger.info("🎮 Dual AC API: http://localhost:8002")
        logger.info("🌐 ECM Gateway: ws://localhost:8765") 
        logger.info("🔄 Pulse System: ws://localhost:8766")
        logger.info("Press Ctrl+C to stop all services")
        
        try:
            api_process.join()
            ws_process.join()
        except KeyboardInterrupt:
            logger.info("🛑 Stopping services...")
            api_process.terminate()
            ws_process.terminate()
            api_process.join()
            ws_process.join()
            logger.info("✅ All services stopped") 