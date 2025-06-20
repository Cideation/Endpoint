#!/usr/bin/env python3
"""
Enhanced Dual Agent Coefficient API Server

Implements the dual AC system:
- Cosmetic AC Group: Structured inputs from UI (sliders, forms, dropdowns)
- Unreal AC Group: Spatial actions from 3D environment
- Both feed into same unified Node Engine
- 1-way compute with 2-way interactive feel

Architecture:
UI/Spatial → AC Groups → Node Engine → Controlled Emergence → Environment
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# DUAL AGENT COEFFICIENT MODELS
# =====================================================

class ACType(str, Enum):
    """Agent Coefficient source types"""
    COSMETIC = "cosmetic"  # From UI sliders, forms, dropdowns
    UNREAL = "unreal"      # From spatial actions, 3D interactions

class AgentClass(str, Enum):
    """The 10 Agent Classes"""
    PRODUCT = "product"
    USER = "user"
    SPATIAL = "spatial"
    MATERIAL = "material"
    STRUCTURAL = "structural"
    MEP = "mep"
    COST = "cost"
    TIME = "time"
    QUALITY = "quality"
    INTEGRATION = "integration"

@dataclass
class AgentCoefficient:
    """Individual Agent Coefficient"""
    key: str
    value: Union[float, int, str]
    ac_type: ACType
    source: str
    timestamp: datetime
    agent_class: Optional[AgentClass] = None
    metadata: Dict[str, Any] = None

    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class CosmeticACGroup:
    """Cosmetic Agent Coefficients from UI"""
    budget: float = 50.0
    investment: float = 75.0
    quality: float = 60.0
    component_priority: str = "structural"
    timeline: float = 80.0
    agent_name: str = ""
    agent_type: Optional[AgentClass] = None
    
    def get_coefficients(self) -> Dict[str, AgentCoefficient]:
        """Convert to AgentCoefficient objects"""
        coefficients = {}
        for key, value in asdict(self).items():
            if key not in ['agent_name', 'agent_type'] and value is not None:
                coefficients[key] = AgentCoefficient(
                    key=key,
                    value=value,
                    ac_type=ACType.COSMETIC,
                    source="cosmetic_ui",
                    timestamp=datetime.now(),
                    agent_class=self.agent_type
                )
        return coefficients

@dataclass
class UnrealACGroup:
    """Unreal Agent Coefficients from spatial actions"""
    location_zone: Optional[int] = None
    wall_size: Optional[float] = None
    selection_precision: Optional[float] = None
    room_area: Optional[float] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    placement_accuracy: Optional[float] = None
    interaction_count: int = 0
    touch_pressure: Optional[float] = None
    zone_id: Optional[int] = None
    geometry_size: Optional[float] = None
    measurement_precision: Optional[float] = None
    complexity_score: Optional[int] = None
    component_volume: Optional[float] = None
    surface_area: Optional[float] = None
    complexity_rating: Optional[int] = None
    selection_frequency: Optional[int] = None
    focus_duration: Optional[int] = None
    interaction_type: Optional[str] = None
    
    def get_coefficients(self) -> Dict[str, AgentCoefficient]:
        """Convert to AgentCoefficient objects"""
        coefficients = {}
        for key, value in asdict(self).items():
            if value is not None:
                coefficients[key] = AgentCoefficient(
                    key=key,
                    value=value,
                    ac_type=ACType.UNREAL,
                    source="spatial_environment",
                    timestamp=datetime.now(),
                    agent_class=AgentClass.SPATIAL
                )
        return coefficients

# =====================================================
# NODE ENGINE WITH DUAL AC PROCESSING
# =====================================================

@dataclass
class NodeEngineEvent:
    """Unified event structure for both AC types"""
    id: str
    timestamp: datetime
    ac_type: ACType
    event_type: str
    source: str
    coefficients: Dict[str, AgentCoefficient]
    agent_class: Optional[AgentClass] = None
    metadata: Dict[str, Any] = None
    processed: bool = False
    result: Optional[str] = None

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'ac_type': self.ac_type.value,
            'event_type': self.event_type,
            'source': self.source,
            'coefficients': {k: v.to_dict() for k, v in self.coefficients.items()},
            'agent_class': self.agent_class.value if self.agent_class else None,
            'metadata': self.metadata or {},
            'processed': self.processed,
            'result': self.result
        }

class NodeEngine:
    """
    Unified Node Engine processing both Cosmetic and Unreal AC groups
    
    Architecture:
    - Receives AC from both sources
    - Processes through unified computation
    - Applies controlled emergence to environments
    - Maintains 1-way compute flow with 2-way interactive feel
    """
    
    def __init__(self):
        self.active_cosmetic_ac: Dict[str, AgentCoefficient] = {}
        self.active_unreal_ac: Dict[str, AgentCoefficient] = {}
        self.event_queue: List[NodeEngineEvent] = []
        self.processed_events: List[NodeEngineEvent] = []
        self.active_agents: Dict[AgentClass, dict] = {}
        self.metrics = {
            'cpu_usage': 45.0,
            'ac_load': 0,
            'events_per_second': 0,
            'total_coefficients': 0,
            'cosmetic_ac_count': 0,
            'unreal_ac_count': 0
        }
        self.websocket_connections: List[WebSocket] = []
        
        # Initialize agent classes
        self._initialize_agents()
        
        # Start processing loop
        asyncio.create_task(self._processing_loop())
    
    def _initialize_agents(self):
        """Initialize the 10 agent classes"""
        for agent_class in AgentClass:
            self.active_agents[agent_class] = {
                'id': agent_class.value,
                'role': agent_class.value.title(),
                'active': agent_class in [AgentClass.PRODUCT, AgentClass.STRUCTURAL],
                'coefficient_count': 0,
                'last_activity': None
            }
    
    async def process_cosmetic_ac(self, ac_group: CosmeticACGroup, event_type: str = "cosmetic_update") -> NodeEngineEvent:
        """Process Cosmetic Agent Coefficients"""
        coefficients = ac_group.get_coefficients()
        
        # Update active cosmetic AC
        self.active_cosmetic_ac.update(coefficients)
        
        # Create unified event
        event = NodeEngineEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            ac_type=ACType.COSMETIC,
            event_type=event_type,
            source="cosmetic_ui",
            coefficients=coefficients,
            agent_class=ac_group.agent_type,
            metadata={'agent_name': ac_group.agent_name}
        )
        
        # Add to processing queue
        self.event_queue.append(event)
        
        # Update metrics
        self._update_metrics()
        
        # Broadcast to websockets
        await self._broadcast_ac_update()
        
        return event
    
    async def process_unreal_ac(self, ac_group: UnrealACGroup, event_type: str, metadata: Dict[str, Any] = None) -> NodeEngineEvent:
        """Process Unreal Agent Coefficients from spatial actions"""
        coefficients = ac_group.get_coefficients()
        
        # Update active unreal AC
        self.active_unreal_ac.update(coefficients)
        
        # Create unified event
        event = NodeEngineEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            ac_type=ACType.UNREAL,
            event_type=event_type,
            source="spatial_environment",
            coefficients=coefficients,
            agent_class=AgentClass.SPATIAL,
            metadata=metadata or {}
        )
        
        # Add to processing queue
        self.event_queue.append(event)
        
        # Activate spatial agent
        self.active_agents[AgentClass.SPATIAL]['active'] = True
        self.active_agents[AgentClass.SPATIAL]['last_activity'] = datetime.now()
        
        # Update metrics
        self._update_metrics()
        
        # Broadcast to websockets
        await self._broadcast_ac_update()
        
        return event
    
    async def _processing_loop(self):
        """Main processing loop for unified AC processing"""
        while True:
            if self.event_queue:
                event = self.event_queue.pop(0)
                
                # Simulate processing time based on AC complexity
                processing_time = 0.5 + len(event.coefficients) * 0.1
                await asyncio.sleep(processing_time)
                
                # Process event through unified engine
                result = await self._execute_unified_processing(event)
                event.result = result
                event.processed = True
                
                # Apply controlled emergence
                await self._apply_emergence(event)
                
                # Store processed event
                self.processed_events.append(event)
                
                # Keep only recent events (last 100)
                if len(self.processed_events) > 100:
                    self.processed_events = self.processed_events[-100:]
                
                # Update agent coefficient counts
                self._update_agent_coefficients()
                
                # Broadcast processing completion
                await self._broadcast_event_processed(event)
            
            await asyncio.sleep(0.1)
    
    async def _execute_unified_processing(self, event: NodeEngineEvent) -> str:
        """Execute unified processing for both AC types"""
        emergence_patterns = {
            ACType.COSMETIC: {
                'agent_signup': 'Agent activated with cosmetic preferences',
                'bulk_coefficients': 'System parameters optimized',
                'cosmetic_update': 'UI preferences integrated',
                'budget_update': 'Cost calculations adjusted',
                'quality_update': 'Quality standards updated',
                'investment_update': 'Investment profile updated',
                'timeline_update': 'Schedule optimization applied'
            },
            ACType.UNREAL: {
                'select_wall': 'Wall analysis initiated → geometry patterns detected',
                'place_room': 'Room configuration updated → spatial optimization applied',
                'touch_zone': 'Zone interaction recorded → behavioral patterns learned',
                'measure_geometry': 'Geometric analysis complete → precision metrics updated',
                'select_component': 'Component selection processed → relationship mapping updated',
                'component_geometry': 'Component properties analyzed → material optimization suggested',
                'component_interaction': 'Interaction patterns learned → usage frequency optimized'
            }
        }
        
        pattern = emergence_patterns.get(event.ac_type, {}).get(event.event_type)
        if pattern:
            return pattern
        
        # Default emergence based on AC type
        if event.ac_type == ACType.COSMETIC:
            return f"Cosmetic AC processed → {len(event.coefficients)} parameters updated"
        else:
            return f"Spatial AC processed → {len(event.coefficients)} environmental factors learned"
    
    async def _apply_emergence(self, event: NodeEngineEvent):
        """Apply controlled emergence based on processed AC"""
        # Update relevant agents based on AC type and coefficients
        for coeff_key, coefficient in event.coefficients.items():
            # Map coefficients to relevant agents
            relevant_agents = self._map_coefficient_to_agents(coeff_key, coefficient)
            
            for agent_class in relevant_agents:
                if agent_class in self.active_agents:
                    self.active_agents[agent_class]['active'] = True
                    self.active_agents[agent_class]['last_activity'] = datetime.now()
    
    def _map_coefficient_to_agents(self, coeff_key: str, coefficient: AgentCoefficient) -> List[AgentClass]:
        """Map coefficients to relevant agent classes"""
        mapping = {
            # Cosmetic AC mappings
            'budget': [AgentClass.COST, AgentClass.PRODUCT],
            'investment': [AgentClass.COST, AgentClass.TIME],
            'quality': [AgentClass.QUALITY, AgentClass.MATERIAL],
            'component_priority': [AgentClass.STRUCTURAL, AgentClass.MEP],
            'timeline': [AgentClass.TIME, AgentClass.INTEGRATION],
            
            # Unreal AC mappings
            'location_zone': [AgentClass.SPATIAL, AgentClass.STRUCTURAL],
            'wall_size': [AgentClass.STRUCTURAL, AgentClass.MATERIAL],
            'room_area': [AgentClass.SPATIAL, AgentClass.MEP],
            'interaction_count': [AgentClass.USER, AgentClass.INTEGRATION],
            'geometry_size': [AgentClass.STRUCTURAL, AgentClass.MATERIAL],
            'component_volume': [AgentClass.MATERIAL, AgentClass.COST]
        }
        
        return mapping.get(coeff_key, [AgentClass.INTEGRATION])
    
    def _update_agent_coefficients(self):
        """Update agent coefficient counts"""
        for agent_class, agent_data in self.active_agents.items():
            # Count relevant coefficients for each agent
            cosmetic_count = sum(1 for coeff in self.active_cosmetic_ac.values() 
                               if agent_class in self._map_coefficient_to_agents(coeff.key, coeff))
            unreal_count = sum(1 for coeff in self.active_unreal_ac.values() 
                             if agent_class in self._map_coefficient_to_agents(coeff.key, coeff))
            
            agent_data['coefficient_count'] = cosmetic_count + unreal_count
    
    def _update_metrics(self):
        """Update engine metrics"""
        total_cosmetic = len(self.active_cosmetic_ac)
        total_unreal = len(self.active_unreal_ac)
        total_ac = total_cosmetic + total_unreal
        
        self.metrics.update({
            'ac_load': total_ac,
            'total_coefficients': total_ac,
            'cosmetic_ac_count': total_cosmetic,
            'unreal_ac_count': total_unreal,
            'cpu_usage': min(90.0, 45.0 + total_ac * 2.5),
            'events_per_second': len(self.event_queue) + 3
        })
    
    async def _broadcast_ac_update(self):
        """Broadcast AC updates to all connected websockets"""
        if self.websocket_connections:
            message = {
                'type': 'ac_update',
                'data': {
                    'cosmetic_ac': {k: v.to_dict() for k, v in self.active_cosmetic_ac.items()},
                    'unreal_ac': {k: v.to_dict() for k, v in self.active_unreal_ac.items()},
                    'metrics': self.metrics,
                    'active_agents': self.active_agents
                }
            }
            
            # Send to all connections
            disconnected = []
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_text(json.dumps(message, default=str))
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                self.websocket_connections.remove(ws)
    
    async def _broadcast_event_processed(self, event: NodeEngineEvent):
        """Broadcast when an event is processed"""
        if self.websocket_connections:
            message = {
                'type': 'event_processed',
                'data': event.to_dict()
            }
            
            # Send to all connections
            disconnected = []
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_text(json.dumps(message, default=str))
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                self.websocket_connections.remove(ws)
    
    def get_status(self) -> dict:
        """Get current engine status"""
        return {
            'metrics': self.metrics,
            'active_agents': list(self.active_agents.values()),
            'queue_length': len(self.event_queue),
            'processed_events': len(self.processed_events),
            'active_cosmetic_ac': len(self.active_cosmetic_ac),
            'active_unreal_ac': len(self.active_unreal_ac),
            'recent_events': [e.to_dict() for e in self.processed_events[-5:]]
        }

# =====================================================
# API MODELS
# =====================================================

class CosmeticACRequest(BaseModel):
    agent_name: str = ""
    agent_type: Optional[str] = None
    budget: float = Field(50.0, ge=0, le=100)
    investment: float = Field(75.0, ge=0, le=100)
    quality: float = Field(60.0, ge=0, le=100)
    component_priority: str = "structural"
    timeline: float = Field(80.0, ge=0, le=100)

class SpatialACRequest(BaseModel):
    action: str
    coefficients: Dict[str, Union[float, int, str]]
    metadata: Optional[Dict[str, Any]] = None

class AgentSignupRequest(BaseModel):
    agent_name: str
    agent_type: str
    initial_coefficients: Optional[Dict[str, Union[float, int, str]]] = None

# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(title="Dual Agent Coefficient API", version="1.0.0")

# Mobile responsive design integrated throughout the system

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Node Engine
engine = NodeEngine()

# Mount static files
app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/")
async def get_interface():
    """Serve the enhanced dual AC interface"""
    try:
        with open("frontend/enhanced_unified_interface.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Enhanced Dual AC Interface</h1>
                <p>Interface file not found. Please ensure enhanced_unified_interface.html exists in the frontend directory.</p>
            </body>
        </html>
        """)

@app.get("/api/status")
async def get_engine_status():
    """Get current Node Engine status"""
    return engine.get_status()

@app.post("/cosmetic_ac")
@app.post("/api/cosmetic-ac")
async def process_cosmetic_ac(request: CosmeticACRequest):
    """Process Cosmetic Agent Coefficients from UI"""
    try:
        # Convert to CosmeticACGroup
        ac_group = CosmeticACGroup(
            budget=request.budget,
            investment=request.investment,
            quality=request.quality,
            component_priority=request.component_priority,
            timeline=request.timeline,
            agent_name=request.agent_name,
            agent_type=AgentClass(request.agent_type) if request.agent_type else None
        )
        
        # Process through engine
        event = await engine.process_cosmetic_ac(ac_group, "cosmetic_update")
        
        return {
            "success": True,
            "event_id": event.id,
            "message": "Cosmetic AC processed successfully",
            "coefficients_processed": len(event.coefficients),
            "result": event.result
        }
    
    except Exception as e:
        logger.error(f"Error processing cosmetic AC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unreal_ac")
@app.post("/api/unreal-ac")
async def process_unreal_ac(request: SpatialACRequest):
    """Process Unreal Agent Coefficients from spatial actions"""
    try:
        # Create UnrealACGroup from coefficients
        ac_group = UnrealACGroup()
        
        # Update with provided coefficients
        for key, value in request.coefficients.items():
            if hasattr(ac_group, key):
                setattr(ac_group, key, value)
        
        # Process through engine
        event = await engine.process_unreal_ac(ac_group, request.action, request.metadata)
        
        return {
            "success": True,
            "event_id": event.id,
            "message": "Unreal AC processed successfully",
            "action": request.action,
            "coefficients_processed": len(event.coefficients),
            "result": event.result
        }
    
    except Exception as e:
        logger.error(f"Error processing unreal AC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent-signup")
async def agent_signup(request: AgentSignupRequest):
    """Register new agent with initial coefficients"""
    try:
        # Create cosmetic AC group for agent signup
        ac_group = CosmeticACGroup(
            agent_name=request.agent_name,
            agent_type=AgentClass(request.agent_type)
        )
        
        # Add initial coefficients if provided
        if request.initial_coefficients:
            for key, value in request.initial_coefficients.items():
                if hasattr(ac_group, key):
                    setattr(ac_group, key, value)
        
        # Process through engine
        event = await engine.process_cosmetic_ac(ac_group, "agent_signup")
        
        return {
            "success": True,
            "event_id": event.id,
            "agent_name": request.agent_name,
            "agent_type": request.agent_type,
            "message": f"Agent {request.agent_name} registered successfully",
            "coefficients_initialized": len(event.coefficients)
        }
    
    except Exception as e:
        logger.error(f"Error in agent signup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/coefficients")
async def get_active_coefficients():
    """Get all active agent coefficients"""
    return {
        "cosmetic_ac": {k: v.to_dict() for k, v in engine.active_cosmetic_ac.items()},
        "unreal_ac": {k: v.to_dict() for k, v in engine.active_unreal_ac.items()},
        "total_coefficients": len(engine.active_cosmetic_ac) + len(engine.active_unreal_ac),
        "metrics": engine.metrics
    }

@app.get("/api/events")
async def get_processed_events(limit: int = 20):
    """Get recent processed events"""
    recent_events = engine.processed_events[-limit:]
    return {
        "events": [event.to_dict() for event in recent_events],
        "total_processed": len(engine.processed_events),
        "queue_length": len(engine.event_queue)
    }

# =====================================================
# WEBSOCKET ENDPOINT
# =====================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    engine.websocket_connections.append(websocket)
    
    try:
        # Send initial status
        initial_status = {
            'type': 'initial_status',
            'data': engine.get_status()
        }
        await websocket.send_text(json.dumps(initial_status, default=str))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        if websocket in engine.websocket_connections:
            engine.websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in engine.websocket_connections:
            engine.websocket_connections.remove(websocket)

# =====================================================
# STARTUP SIMULATION
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Initialize with some sample data"""
    logger.info("Starting Dual Agent Coefficient System...")
    
    # Simulate some initial cosmetic AC
    initial_cosmetic = CosmeticACGroup(
        budget=65.0,
        investment=80.0,
        quality=70.0,
        component_priority="structural",
        timeline=75.0,
        agent_name="System Default",
        agent_type=AgentClass.PRODUCT
    )
    
    await engine.process_cosmetic_ac(initial_cosmetic, "system_initialization")
    
    # Simulate some initial spatial interaction
    initial_unreal = UnrealACGroup(
        interaction_count=1,
        zone_id=1,
        touch_pressure=50.0
    )
    
    await engine.process_unreal_ac(initial_unreal, "system_initialization", 
                                  {"source": "startup_simulation"})
    
    logger.info("Dual AC System initialized with sample coefficients")

if __name__ == "__main__":
    print("🎯 Starting Enhanced Dual Agent Coefficient System")
    print("📊 Cosmetic AC Group: UI sliders, forms, dropdowns")
    print("🎮 Unreal AC Group: Spatial actions, 3D interactions")
    print("🧠 Node Engine: Unified processing of both AC types")
    print("🌟 Architecture: 1-way compute, 2-way interactive feel")
    print("🚀 Access: http://localhost:8002")
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info") 