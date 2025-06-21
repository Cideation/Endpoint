#!/usr/bin/env python3
"""
SocketIO Dashboard Server
Receives real-time data from engine and broadcasts to dashboard clients
Complements GraphQL API with event-based streaming
"""

import socketio
import logging
from typing import Dict, Set, Any
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger(__name__)

class DashboardSocketServer:
    def __init__(self):
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=True
        )
        
        self.connected_clients: Set[str] = set()
        self.setup_event_handlers()
        
    def setup_event_handlers(self):
        @self.sio.event
        async def connect(sid, environ):
            self.connected_clients.add(sid)
            logger.info(f"🔗 Dashboard client connected: {sid}")
            
            await self.sio.emit('connection_status', {
                'status': 'connected',
                'client_id': sid,
                'timestamp': datetime.now().isoformat(),
                'message': 'Dashboard streaming ready'
            }, room=sid)
            
        @self.sio.event
        async def disconnect(sid):
            self.connected_clients.discard(sid)
            logger.info(f"🔌 Dashboard client disconnected: {sid}")
        
        @self.sio.event
        async def ping(sid, data):
            await self.sio.emit('pong', {
                'timestamp': datetime.now().isoformat(),
                'client_id': sid
            }, room=sid)
    
    async def broadcast_agent_state_update(self, data: Dict[str, Any]):
        if self.connected_clients:
            await self.sio.emit('agent_state_update', data)
            logger.debug(f"📡 Broadcasted to {len(self.connected_clients)} clients")
    
    async def broadcast_pulse_render(self, data: Dict[str, Any]):
        if self.connected_clients:
            await self.sio.emit('pulse_render', data)
            logger.debug(f"🎨 Broadcasted pulse render to {len(self.connected_clients)} clients")
    
    def get_client_count(self) -> int:
        return len(self.connected_clients)

dashboard_server = DashboardSocketServer()

def create_dashboard_server_app():
    app = FastAPI(
        title="BEM Dashboard Server",
        description="Real-time streaming server for dashboard console",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    socketio_app = socketio.ASGIApp(dashboard_server.sio, app)
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "connected_clients": dashboard_server.get_client_count(),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/")
    async def root():
        return {
            "message": "BEM Dashboard Server",
            "socketio_endpoint": "/socket.io/",
            "connected_clients": dashboard_server.get_client_count()
        }
    
    @app.post("/test/agent_state")
    async def test_agent_state(data: Dict[str, Any]):
        await dashboard_server.broadcast_agent_state_update(data)
        return {"status": "broadcasted", "clients": dashboard_server.get_client_count()}
    
    return socketio_app

if __name__ == "__main__":
    print("🎛️ Starting BEM Dashboard Server...")
    print("📡 SocketIO endpoint: http://localhost:5000/socket.io/")
    
    app = create_dashboard_server_app()
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
