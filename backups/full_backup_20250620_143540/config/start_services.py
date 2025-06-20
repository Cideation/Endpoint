#!/usr/bin/env python3
"""
Simple Service Launcher for Phase 2 Testing
Starts essential services for 100% integration test coverage
"""

import subprocess
import time
import sys
import logging
import websockets.exceptions
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_simple_api_server():
    """Start a simple API server for testing"""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    import uvicorn
    
    # Rate limiting setup
    limiter = Limiter(key_func=get_remote_address)
    
    app = FastAPI(
        title="BEM System Dual AC API",
        description="Production-ready API with security features",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080", "https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    @app.get("/api/status")
    @limiter.limit("30/minute")  # Rate limit: 30 requests per minute
    def get_status(request: Request):
        return {
            "status": "running", 
            "service": "dual_ac_api", 
            "test_mode": True,
            "timestamp": time.time(),
            "security": "enabled"
        }
    
    @app.post("/cosmetic_ac")
    @limiter.limit("60/minute")  # Rate limit: 60 requests per minute
    def cosmetic_ac_endpoint(request: Request, data: dict):
        return {
            "success": True, 
            "message": "Cosmetic AC processed", 
            "data": data,
            "timestamp": time.time()
        }
    
    @app.post("/unreal_ac") 
    @limiter.limit("60/minute")  # Rate limit: 60 requests per minute
    def unreal_ac_endpoint(request: Request, data: dict):
        return {
            "success": True, 
            "message": "Unreal AC processed", 
            "data": data,
            "timestamp": time.time()
        }
    
    @app.get("/health")
    def health_check():
        """Health check endpoint for monitoring"""
        return {"status": "healthy", "timestamp": time.time()}
    
    logger.info("🎮 Starting Enhanced Dual AC API with Security on port 8002")
    logger.info("🔒 Security features: CORS, Rate Limiting, Health Checks")
    
    # SSL/TLS configuration for production
    ssl_config = {
        "ssl_keyfile": "key.pem",
        "ssl_certfile": "cert.pem"
    } if os.path.exists("cert.pem") and os.path.exists("key.pem") else {}
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002, 
        log_level="warning",
        **ssl_config
    )

def start_simple_websocket_server():
    """Start a simple WebSocket server for ECM testing"""
    import asyncio
    import websockets
    import json
    
    async def ecm_handler(websocket):
        """Simple ECM message handler - Fixed signature"""
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
                except json.JSONDecodeError:
                    await websocket.send('{"status": "error", "message": "Invalid JSON"}')
                except Exception as e:
                    print(f"ECM Message processing error: {e}")
                    await websocket.send('{"status": "error", "message": "Processing error"}')
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"ECM Handler error: {e}")
    
    logger.info("🌐 Starting Simple ECM Gateway on port 8765")
    start_server = websockets.serve(ecm_handler, "localhost", 8765)
    return start_server

def start_simple_pulse_server():
    """Start a simple pulse WebSocket server"""
    import asyncio
    import websockets
    import json
    
    async def pulse_handler(websocket):
        """Simple pulse message handler - Fixed signature"""
        try:
            async for message in websocket:
                try:
                    # Parse message for better pulse handling
                    data = json.loads(message) if message.strip() else {}
                    response = {"status": "pulse_received", "timestamp": "2024-01-01T00:00:00"}
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    response = {"status": "pulse_error", "message": "Invalid pulse format"}
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    print(f"Pulse processing error: {e}")
                    response = {"status": "pulse_error", "message": "Processing error"}
                    await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"Pulse Handler error: {e}")
    
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