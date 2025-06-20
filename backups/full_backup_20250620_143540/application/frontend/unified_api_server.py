"""
Unified API Server for 1-Way Compute Architecture
Routes structured (Cosmetic UI) and spatial (3D Environment) inputs through central Node Engine
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Add the parent directory to Python path to import neon modules
sys.path.append(str(Path(__file__).parent.parent))

from neon.frontend_service import FrontendAPIService
from neon.api_models import ComponentSearchRequest, DashboardStatistics

app = FastAPI(
    title="Neon Unified Interface API",
    description="1-Way Compute Architecture: All inputs → Node Engine → Controlled emergence",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
frontend_service = FrontendAPIService()

# =====================================================
# NODE ENGINE CORE CLASSES
# =====================================================

class InputSource(str, Enum):
    STRUCTURED = "structured"  # Cosmetic UI forms
    SPATIAL = "spatial"        # 3D environment interactions

class NodeEngineEvent:
    """Central event structure for all inputs"""
    def __init__(self, 
                 event_id: str,
                 source: InputSource,
                 event_type: str,
                 data: Dict[str, Any],
                 timestamp: datetime = None):
        self.event_id = event_id
        self.source = source
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.agent_id = self._determine_agent(event_type)
        
    def _determine_agent(self, event_type: str) -> int:
        """Route event types to appropriate agents"""
        agent_mapping = {
            'agent_signup': 1,          # Agent 1 (Product)
            'user_interaction': 2,      # Agent 2 (User)
            'spatial_query': 3,         # Agent 3 (Spatial)
            'material_update': 4,       # Agent 4 (Material)
            'structural_analysis': 5,   # Agent 5 (Structural)
            'mep_calculation': 6,       # Agent 6 (MEP)
            'cost_estimation': 7,       # Agent 7 (Cost)
            'time_analysis': 8,         # Agent 8 (Time)
            'quality_check': 9,         # Agent 9 (Quality)
            'system_integration': 10    # Agent 10 (Integration)
        }
        return agent_mapping.get(event_type, 1)

class NodeEngine:
    """Central compute engine that processes all inputs and generates controlled emergence"""
    
    def __init__(self):
        self.active_agents = {}
        self.processing_queue = []
        self.event_history = []
        self.emergence_patterns = {}
        self.subscribers = []  # WebSocket connections for real-time updates
        
    async def process_event(self, event: NodeEngineEvent) -> Dict[str, Any]:
        """Central processing method - all inputs flow through here"""
        
        # Add to processing queue
        self.processing_queue.append({
            'event_id': event.event_id,
            'type': event.event_type,
            'progress': 0,
            'started_at': datetime.now()
        })
        
        # Activate appropriate agent
        await self._activate_agent(event.agent_id)
        
        # Process in engine (simulated complex computation)
        result = await self._compute_emergence(event)
        
        # Update processing queue
        await self._update_processing_queue(event.event_id, 100)
        
        # Store in history
        self.event_history.append({
            'event': event,
            'result': result,
            'processed_at': datetime.now()
        })
        
        # Broadcast to subscribers
        await self._broadcast_emergence(event, result)
        
        return result
    
    async def _activate_agent(self, agent_id: int):
        """Activate agent in the system"""
        self.active_agents[agent_id] = {
            'id': agent_id,
            'activated_at': datetime.now(),
            'active': True
        }
        
        # Auto-deactivate after processing
        asyncio.create_task(self._deactivate_agent_after_delay(agent_id, 5))
    
    async def _deactivate_agent_after_delay(self, agent_id: int, delay: int):
        """Deactivate agent after delay"""
        await asyncio.sleep(delay)
        if agent_id in self.active_agents:
            self.active_agents[agent_id]['active'] = False
    
    async def _compute_emergence(self, event: NodeEngineEvent) -> Dict[str, Any]:
        """Simulate complex computation that generates controlled emergence"""
        
        # Simulate processing delay based on complexity
        complexity_map = {
            'agent_signup': 0.5,
            'spatial_query': 1.0,
            'structural_analysis': 2.0,
            'system_integration': 3.0
        }
        delay = complexity_map.get(event.event_type, 1.0)
        await asyncio.sleep(delay)
        
        # Generate emergence patterns based on input type and data
        emergence = self._generate_emergence_pattern(event)
        
        return {
            'event_id': event.event_id,
            'emergence_type': emergence['type'],
            'environment_changes': emergence['changes'],
            'agent_responses': emergence['agents'],
            'data_updates': emergence['data'],
            'computed_at': datetime.now().isoformat()
        }
    
    def _generate_emergence_pattern(self, event: NodeEngineEvent) -> Dict[str, Any]:
        """Generate controlled emergence patterns"""
        
        patterns = {
            'agent_signup': {
                'type': 'agent_activation',
                'changes': ['new_agent_node', 'capability_expansion'],
                'agents': [event.agent_id],
                'data': {'new_agent': event.data}
            },
            'spatial_query': {
                'type': 'spatial_analysis',
                'changes': ['component_highlight', 'spatial_overlay'],
                'agents': [3, 5],  # Spatial + Structural
                'data': {'selected_component': event.data}
            },
            'component_selection': {
                'type': 'component_focus',
                'changes': ['detail_panel', 'property_display'],
                'agents': [4, 9],  # Material + Quality
                'data': {'component_analysis': event.data}
            }
        }
        
        return patterns.get(event.event_type, {
            'type': 'generic_processing',
            'changes': ['state_update'],
            'agents': [event.agent_id],
            'data': event.data
        })
    
    async def _update_processing_queue(self, event_id: str, progress: int):
        """Update processing queue progress"""
        for item in self.processing_queue:
            if item['event_id'] == event_id:
                item['progress'] = progress
                if progress >= 100:
                    asyncio.create_task(self._remove_completed_item(event_id, 3))
                break
    
    async def _remove_completed_item(self, event_id: str, delay: int):
        """Remove completed item from queue after delay"""
        await asyncio.sleep(delay)
        self.processing_queue = [item for item in self.processing_queue if item['event_id'] != event_id]
    
    async def _broadcast_emergence(self, event: NodeEngineEvent, result: Dict[str, Any]):
        """Broadcast emergence to all subscribers"""
        broadcast_data = {
            'type': 'emergence',
            'event_id': event.event_id,
            'source': event.source.value,
            'emergence': result,
            'timestamp': datetime.now().isoformat()
        }
        
        for websocket in self.subscribers[:]:
            try:
                await websocket.send_text(json.dumps(broadcast_data))
            except:
                self.subscribers.remove(websocket)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'active_agents': list(self.active_agents.values()),
            'processing_queue': self.processing_queue,
            'recent_events': self.event_history[-10:],
            'metrics': {
                'cpu_usage': 45 + len(self.processing_queue) * 10,
                'active_nodes': len([a for a in self.active_agents.values() if a.get('active', False)]),
                'events_per_second': len(self.event_history) / max(1, 60) if self.event_history else 0
            }
        }

# Global Node Engine instance
node_engine = NodeEngine()

# =====================================================
# STARTUP/SHUTDOWN
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services"""
    await frontend_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    await frontend_service.close()

# =====================================================
# STATIC FILES
# =====================================================

@app.get("/")
async def serve_unified_interface():
    """Serve the unified interface"""
    return FileResponse("frontend/unified_interface.html")

@app.get("/legacy")
async def serve_legacy_interface():
    """Serve the original interface"""
    return FileResponse("frontend/index.html")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# =====================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# =====================================================

@app.websocket("/ws/engine")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time engine updates"""
    await websocket.accept()
    node_engine.subscribers.append(websocket)
    
    try:
        await websocket.send_text(json.dumps({
            'type': 'engine_status',
            'data': node_engine.get_engine_status()
        }))
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        if websocket in node_engine.subscribers:
            node_engine.subscribers.remove(websocket)

# =====================================================
# UNIFIED INPUT ENDPOINTS
# =====================================================

@app.post("/api/v2/engine/structured-input")
async def submit_structured_input(
    event_type: str = Form(...),
    data: str = Form(...)
):
    """Submit structured input from Cosmetic UI to Node Engine"""
    try:
        event_data = json.loads(data)
        
        event = NodeEngineEvent(
            event_id=f"struct_{datetime.now().timestamp()}",
            source=InputSource.STRUCTURED,
            event_type=event_type,
            data=event_data
        )
        
        result = await node_engine.process_event(event)
        
        return {
            'success': True,
            'event_id': event.event_id,
            'emergence': result,
            'message': f'Structured input processed through Node Engine'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/engine/spatial-input")
async def submit_spatial_input(
    action: str = Form(...),
    component_id: str = Form(None),
    position: str = Form(None),
    data: str = Form("{}")
):
    """Submit spatial input from 3D Environment to Node Engine"""
    try:
        event_data = json.loads(data)
        event_data.update({
            'action': action,
            'component_id': component_id,
            'position': json.loads(position) if position else None
        })
        
        # Convert spatial action to structured event type
        event_type = _convert_spatial_to_event_type(action)
        
        event = NodeEngineEvent(
            event_id=f"spatial_{datetime.now().timestamp()}",
            source=InputSource.SPATIAL,
            event_type=event_type,
            data=event_data
        )
        
        result = await node_engine.process_event(event)
        
        return {
            'success': True,
            'event_id': event.event_id,
            'emergence': result,
            'message': f'Spatial input converted and processed through Node Engine'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _convert_spatial_to_event_type(spatial_action: str) -> str:
    """Convert spatial actions to structured event types"""
    conversion_map = {
        'select_component': 'component_selection',
        'touch_zone': 'user_interaction',
        'move_component': 'spatial_modification',
        'measure_distance': 'spatial_measurement',
        'analyze_component': 'structural_analysis',
        'component_selected': 'spatial_query'
    }
    return conversion_map.get(spatial_action, 'generic_spatial_event')

# =====================================================
# ENGINE STATUS ENDPOINTS
# =====================================================

@app.get("/api/v2/engine/status")
async def get_engine_status():
    """Get current Node Engine status"""
    return {
        'success': True,
        'data': node_engine.get_engine_status()
    }

@app.get("/api/v1/dashboard/statistics")
async def get_dashboard_statistics():
    """Legacy dashboard statistics endpoint"""
    return {
        'success': True,
        'data': {
            'totalComponents': 5,
            'totalFiles': 3,
            'spatialComponents': 4,
            'materialComponents': 3,
            'componentsByType': {'structural': 3, 'mep': 2}
        }
    }

@app.get("/api/v1/components")
async def get_components():
    """Legacy components endpoint"""
    mock_components = [
        {
            'component_id': '1',
            'component_name': 'Steel Beam H400',
            'component_type': 'structural',
            'description': 'Primary load-bearing beam',
            'has_spatial_data': True,
            'has_materials': True,
            'material_count': 2,
            'created_at': datetime.now().isoformat()
        },
        {
            'component_id': '2',
            'component_name': 'Concrete Column C30',
            'component_type': 'structural',
            'description': 'Reinforced concrete column',
            'has_spatial_data': True,
            'has_materials': True,
            'material_count': 1,
            'created_at': datetime.now().isoformat()
        },
        {
            'component_id': '3',
            'component_name': 'HVAC Duct 600mm',
            'component_type': 'mep',
            'description': 'Main HVAC distribution duct',
            'has_spatial_data': True,
            'has_materials': False,
            'material_count': 0,
            'created_at': datetime.now().isoformat()
        }
    ]
    
    return {
        'success': True,
        'data': mock_components,
        'total_count': len(mock_components),
        'page': 1,
        'page_size': 10,
        'total_pages': 1
    }

@app.get("/api/v2/health")
async def health_check():
    """Enhanced health check"""
    return {
        'status': 'healthy',
        'service': 'Neon Unified Interface API',
        'version': '2.0.0',
        'architecture': '1-way compute',
        'node_engine': 'active',
        'active_agents': len(node_engine.active_agents),
        'processing_queue_size': len(node_engine.processing_queue)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 