#!/usr/bin/env python3
"""
BEM System GraphQL Server
Provides dynamic schema for Cytoscape Agent Console with real-time subscriptions
"""

import strawberry
import asyncio
import json
import os
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, WebSocket, BackgroundTasks
from strawberry.fastapi import GraphQLRouter
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncpg
import redis
from datetime import datetime
from enum import Enum

# Import existing BEM utilities
import sys
sys.path.append('.')
sys.path.append('..')

# Data affinity utilities
def load_affinity_configuration():
    """Load functor affinity configuration from JSON files"""
    try:
        affinity_config = {}
        
        # Load functor types with affinity
        with open("MICROSERVICE_ENGINES/functor_types_with_affinity.json", "r") as f:
            affinity_config["types"] = json.load(f)
        
        # Load functor data affinity
        with open("MICROSERVICE_ENGINES/functor_data_affinity.json", "r") as f:
            affinity_config["data"] = json.load(f)
        
        return affinity_config
    except Exception as e:
        print(f"Warning: Could not load affinity configuration: {e}")
        return {
            "types": {
                "stateless_spatial": {"data_affinity": "Works with geometric or spatial data"},
                "local_calcs": {"data_affinity": "Performs numeric evaluations"},
                "aggregator": {"data_affinity": "Consolidates multiple results"}
            },
            "data": []
        }

from neon.database_integration import DatabaseIntegration
from neon.config import NEON_CONFIG
from neon.container_client import ContainerClient, ContainerType

# Database configuration - Use Neon instead of local PostgreSQL
DB_CONFIG = NEON_CONFIG

# Redis for real-time subscriptions
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Initialize Neon Database Integration
db_integration = DatabaseIntegration()

# GraphQL Enums
@strawberry.enum
class NodeType(Enum):
    STRUCTURAL = "structural"
    COST = "cost"
    ENERGY = "energy"
    MEP = "mep"
    FABRICATION = "fabrication"
    TIME = "time"

@strawberry.enum
class PhaseType(Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    CROSS_PHASE = "cross_phase"

@strawberry.enum
class PulseType(Enum):
    BID_PULSE = "bid_pulse"
    OCCUPANCY_PULSE = "occupancy_pulse"
    COMPLIANCY_PULSE = "compliancy_pulse"
    FIT_PULSE = "fit_pulse"
    INVESTMENT_PULSE = "investment_pulse"
    DECAY_PULSE = "decay_pulse"
    REJECT_PULSE = "reject_pulse"

# GraphQL Types for Cytoscape
@strawberry.type
class Position:
    x: float
    y: float

@strawberry.type
class Coefficients:
    structural: Optional[float] = None
    cost: Optional[float] = None
    energy: Optional[float] = None
    mep: Optional[float] = None
    fabrication: Optional[float] = None
    time: Optional[float] = None

@strawberry.type
class NodeProperties:
    material: Optional[str] = None
    dimensions: Optional[Dict[str, float]] = None
    specifications: Optional[Dict[str, Any]] = None

@strawberry.type
class Node:
    id: str
    type: NodeType
    phase: PhaseType
    primary_functor: NodeType
    secondary_functor: Optional[NodeType] = None
    coefficients: Coefficients
    position: Optional[Position] = None
    properties: Optional[NodeProperties] = None
    status: Optional[str] = None

@strawberry.type
class Edge:
    id: str
    source: str
    target: str
    relationship_type: str
    weight: Optional[float] = None
    properties: Optional[Dict[str, Any]] = None

@strawberry.type
class Graph:
    nodes: List[Node]
    edges: List[Edge]
    metadata: Optional[Dict[str, Any]] = None

@strawberry.type
class PulseEvent:
    id: str
    pulse_type: PulseType
    source_node: str
    target_node: Optional[str] = None
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    status: str

@strawberry.type
class DataAffinityResult:
    affinity_type: str
    calculation_results: Dict[str, Any]
    execution_status: str
    formulas_executed: int
    component_id: str

@strawberry.type
class AffinityAnalysis:
    structural_affinity: Optional[DataAffinityResult] = None
    cost_affinity: Optional[DataAffinityResult] = None
    energy_affinity: Optional[DataAffinityResult] = None
    mep_affinity: Optional[DataAffinityResult] = None
    spatial_affinity: Optional[DataAffinityResult] = None
    time_affinity: Optional[DataAffinityResult] = None

@strawberry.type
class ComponentAnalysis:
    structural_score: Optional[float] = None
    cost_estimate: Optional[float] = None
    energy_efficiency: Optional[float] = None
    manufacturing_feasibility: Optional[float] = None

@strawberry.type
class FunctorAffinityType:
    name: str
    data_affinity: str
    included_functors: List[str]

@strawberry.type
class AffinityConfiguration:
    functor_types: List[FunctorAffinityType]
    available_affinity_types: List[str]

# Input Types
@strawberry.input
class PositionInput:
    x: float
    y: float

@strawberry.input
class ProjectContextInput:
    agent_project_tag: Optional[str] = None
    project_id: Optional[str] = None
    phase: Optional[str] = None
    agent_id: Optional[str] = None

@strawberry.input
class NodeFilter:
    phases: Optional[List[PhaseType]] = None
    functor_types: Optional[List[NodeType]] = None
    node_ids: Optional[List[str]] = None
    status: Optional[str] = None
    project_context: Optional[ProjectContextInput] = None

@strawberry.input
class AffinityRequest:
    affinity_types: List[str]
    component_id: str
    execution_mode: Optional[str] = "symbolic_reasoning"
    parameters: Optional[Dict[str, Any]] = None

@strawberry.input
class NodeUpdateInput:
    id: str
    position: Optional[PositionInput] = None
    properties: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

# Database functions
async def get_async_db_connection():
    """Get async database connection via Neon DatabaseIntegration"""
    await db_integration.initialize()
    return db_integration

def get_sync_db_connection():
    """Get synchronous database connection via Neon DatabaseIntegration"""
    return db_integration

async def load_graph_from_db(filter_params: Optional[NodeFilter] = None, project_context: Optional[ProjectContextInput] = None) -> Graph:
    """Load graph data from database with optional filtering"""
    db_conn = await get_async_db_connection()
    
    try:
        # For now, let's use component data from Neon since we don't have graph_nodes table yet
        # Get components with relations from Neon database
        components = await db_conn.get_components_with_relations(limit=100)
        
        # Convert components to graph nodes
        nodes = []
        edges = []
        
        for comp in components:
            # Create node from component data
            node = Node(
                id=str(comp.get('component_id', '')),
                type=NodeType.STRUCTURAL,  # Default to structural
                phase=PhaseType.PHASE_2,  # Default to phase 2
                primary_functor=NodeType.STRUCTURAL,
                secondary_functor=None,
                coefficients=Coefficients(
                    structural=1.0,
                    cost=comp.get('area_m2', 0.0) if comp.get('area_m2') else 0.0,
                    energy=0.0,
                    mep=0.0,
                    fabrication=comp.get('volume_cm3', 0.0) if comp.get('volume_cm3') else 0.0,
                    time=0.0
                ),
                position=Position(
                    x=comp.get('centroid_x', 0.0) if comp.get('centroid_x') else 0.0,
                    y=comp.get('centroid_y', 0.0) if comp.get('centroid_y') else 0.0
                ),
                properties=NodeProperties(
                    material=comp.get('material_name', ''),
                    dimensions={
                        'length_mm': comp.get('length_mm', 0.0) if comp.get('length_mm') else 0.0,
                        'width_mm': comp.get('width_mm', 0.0) if comp.get('width_mm') else 0.0,
                        'height_mm': comp.get('height_mm', 0.0) if comp.get('height_mm') else 0.0
                    },
                    specifications={
                        'component_type': comp.get('component_type', ''),
                        'geometry_type': comp.get('geometry_type', ''),
                        'surface_area_m2': comp.get('surface_area_m2', 0.0) if comp.get('surface_area_m2') else 0.0
                    }
                ),
                status='active'
            )
            nodes.append(node)
        
        return Graph(nodes=nodes, edges=edges, metadata={'source': 'neon_components', 'count': len(nodes)})
        
    except Exception as e:
        print(f"Error loading graph from database: {e}")
        # Return empty graph if error
        return Graph(nodes=[], edges=[], metadata={'error': str(e)})

async def publish_pulse_event(pulse_event: PulseEvent):
    """Publish pulse event to Redis for real-time subscriptions"""
    event_data = {
        "id": pulse_event.id,
        "pulse_type": pulse_event.pulse_type.value,
        "source_node": pulse_event.source_node,
        "target_node": pulse_event.target_node,
        "timestamp": pulse_event.timestamp.isoformat(),
        "data": pulse_event.data,
        "status": pulse_event.status
    }
    
    redis_client.publish("pulse_events", json.dumps(event_data))

# GraphQL Queries
@strawberry.type
class Query:
    
    @strawberry.field
    async def graph(self, filter: Optional[NodeFilter] = None, project_context: Optional[ProjectContextInput] = None) -> Graph:
        """Get the entire graph with optional filtering and project context"""
        # Merge project context from direct parameter or filter
        if project_context and filter and filter.project_context:
            # Use the one from filter if both provided
            effective_context = filter.project_context
        elif project_context:
            effective_context = project_context
        elif filter and filter.project_context:
            effective_context = filter.project_context
        else:
            effective_context = None
            
        return await load_graph_from_db(filter, effective_context)
    
    @strawberry.field
    async def node(self, id: str, project_context: Optional[ProjectContextInput] = None) -> Optional[Node]:
        """Get a specific node by ID with project context validation"""
        graph = await load_graph_from_db(NodeFilter(node_ids=[id]), project_context)
        return graph.nodes[0] if graph.nodes else None
    
    @strawberry.field
    async def pulse_history(self, limit: int = 100) -> List[PulseEvent]:
        """Get recent pulse events"""
        try:
            db_conn = await get_async_db_connection()
            
            # Since we don't have pulse_events table in Neon yet, return mock data
            # TODO: Implement pulse_events table in Neon schema
            mock_events = []
            
            # Create some mock pulse events for demonstration
            from datetime import datetime
            import uuid
            
            for i in range(min(limit, 5)):  # Return up to 5 mock events
                event = PulseEvent(
                    id=str(uuid.uuid4()),
                    pulse_type=PulseType.BID_PULSE,
                    source_node=f"node_{i+1}",
                    target_node=f"node_{i+2}" if i < 4 else None,
                    timestamp=datetime.now(),
                    data={"mock": True, "index": i},
                    status="active"
                )
                mock_events.append(event)
            
            return mock_events
            
        except Exception as e:
            print(f"Error loading pulse history: {e}")
            return []
    
    @strawberry.field
    async def affinity_configuration(self) -> AffinityConfiguration:
        """Get available data affinity types and functor configuration"""
        config = load_affinity_configuration()
        
        functor_types = []
        for type_name, type_data in config["types"].items():
            functor_type = FunctorAffinityType(
                name=type_name,
                data_affinity=type_data.get("data_affinity", ""),
                included_functors=type_data.get("included_functors", [])
            )
            functor_types.append(functor_type)
        
        return AffinityConfiguration(
            functor_types=functor_types,
            available_affinity_types=["structural", "cost", "energy", "mep", "spatial", "time"]
        )
    
    @strawberry.field
    async def analyze_component(self, node_id: str) -> ComponentAnalysis:
        """Run real-time component analysis via microservices"""
        try:
            # Get node data
            node = await self.node(node_id)
            if not node:
                return ComponentAnalysis()
            
            # Call microservices for analysis
            container_client = ContainerClient()
            
            # Prepare component data
            component_data = {
                "components": [{
                    "id": node.id,
                    "type": node.type.value,
                    "primary_functor": node.primary_functor.value,
                    "properties": node.properties.__dict__ if node.properties else {}
                }]
            }
            
            # Get structural analysis
            structural_result = container_client.call_container(
                ContainerType.DAG_ALPHA, 
                component_data
            )
            
            # Get cost analysis
            cost_result = container_client.call_container(
                ContainerType.SFDE_ENGINE,
                {**component_data, "affinity_types": ["cost"]}
            )
            
            # Extract scores
            structural_score = None
            if structural_result.get("status") == "success":
                structural_score = structural_result.get("results", {}).get("manufacturing_score", 0.0)
            
            cost_estimate = None
            if cost_result.get("status") == "success":
                cost_data = cost_result.get("results", {}).get("calculations", {}).get("cost", {})
                cost_estimate = cost_data.get("total_cost", 0.0)
            
            return ComponentAnalysis(
                structural_score=structural_score,
                cost_estimate=cost_estimate,
                energy_efficiency=0.85,  # Placeholder - integrate when available
                manufacturing_feasibility=0.90  # Placeholder
            )
            
        except Exception as e:
            print(f"Component analysis error: {e}")
            return ComponentAnalysis()

# GraphQL Mutations
@strawberry.type
class Mutation:
    
    @strawberry.mutation
    async def update_node_position(self, id: str, position: PositionInput) -> str:
        """Update node position in Cytoscape"""
        try:
            db_conn = await get_async_db_connection()
            
            # Since we don't have graph_nodes table yet, store position update in memory/Redis
            # TODO: Implement graph_nodes table for proper position storage
            
            # For now, just publish the update event for real-time UI updates
            redis_client.publish("graph_updates", json.dumps({
                "type": "node_position_update",
                "node_id": id,
                "position": {"x": position.x, "y": position.y},
                "timestamp": datetime.now().isoformat()
            }))
            
            return f"Updated position for node {id}"
            
        except Exception as e:
            print(f"Error updating node position: {e}")
            return f"Failed to update position for node {id}: {str(e)}"
    
    @strawberry.mutation
    async def execute_data_affinity(self, request: AffinityRequest) -> AffinityAnalysis:
        """Execute data affinity calculations using SFDE microservice"""
        try:
            # Get component data
            node = await Query().node(request.component_id)
            if not node:
                raise Exception(f"Component {request.component_id} not found")
            
            # Prepare component data for SFDE processing
            component_data = {
                "id": node.id,
                "type": node.type.value,
                "primary_functor": node.primary_functor.value,
                "properties": node.properties.__dict__ if node.properties else {},
                "coefficients": node.coefficients.__dict__ if node.coefficients else {}
            }
            
            # Build SFDE requests for each affinity type
            sfde_requests = []
            for affinity_type in request.affinity_types:
                sfde_request = {
                    "component_id": request.component_id,
                    "affinity_type": affinity_type,
                    "input_parameters": request.parameters or {},
                    "formula_context": {
                        "component_type": node.type.value,
                        "has_spatial_data": bool(node.properties and node.properties.specifications),
                        "has_geometry": bool(node.properties and node.properties.dimensions),
                        "has_dimensions": bool(node.properties and node.properties.dimensions),
                        "materials_count": 1 if node.properties else 0
                    }
                }
                sfde_requests.append(sfde_request)
            
            # Call SFDE microservice
            container_client = ContainerClient()
            sfde_result = container_client.call_container(
                ContainerType.SFDE_ENGINE,
                {
                    "sfde_requests": sfde_requests,
                    "affinity_types": request.affinity_types,
                    "execution_mode": request.execution_mode or "symbolic_reasoning"
                }
            )
            
            # Process results into affinity analysis
            analysis = AffinityAnalysis()
            
            if sfde_result.get("status") == "success":
                results = sfde_result.get("results", {})
                calculations = results.get("calculations", {})
                
                # Map each affinity type to result
                for affinity_type in request.affinity_types:
                    if affinity_type in calculations:
                        affinity_result = DataAffinityResult(
                            affinity_type=affinity_type,
                            calculation_results=calculations[affinity_type],
                            execution_status="completed",
                            formulas_executed=results.get("formulas_executed", 0),
                            component_id=request.component_id
                        )
                        
                        # Set the appropriate affinity field
                        if affinity_type == "structural":
                            analysis.structural_affinity = affinity_result
                        elif affinity_type == "cost":
                            analysis.cost_affinity = affinity_result
                        elif affinity_type == "energy":
                            analysis.energy_affinity = affinity_result
                        elif affinity_type == "mep":
                            analysis.mep_affinity = affinity_result
                        elif affinity_type == "spatial":
                            analysis.spatial_affinity = affinity_result
                        elif affinity_type == "time":
                            analysis.time_affinity = affinity_result
            
            return analysis
            
        except Exception as e:
            # Return empty analysis with error info
            print(f"Data affinity execution failed: {e}")
            return AffinityAnalysis()
    
    @strawberry.mutation
    async def trigger_pulse(self, pulse_type: PulseType, source_node: str, target_node: Optional[str] = None) -> PulseEvent:
        """Trigger pulse in the system"""
        pulse_event = PulseEvent(
            id=f"pulse_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            pulse_type=pulse_type,
            source_node=source_node,
            target_node=target_node,
            timestamp=datetime.now(),
            data={"manual_trigger": True},
            status="active"
        )
        
        try:
            # For now, just publish the pulse event to Redis since we don't have pulse_events table yet
            # TODO: Implement pulse_events table in Neon schema for persistence
            
            # Publish real-time event
            await publish_pulse_event(pulse_event)
            
            return pulse_event
            
        except Exception as e:
            print(f"Error triggering pulse: {e}")
            # Return the pulse event anyway for UI feedback
            return pulse_event

# GraphQL Subscriptions for Real-time Updates
@strawberry.type
class Subscription:
    
    @strawberry.subscription
    async def pulse_events(self) -> AsyncGenerator[PulseEvent, None]:
        """Subscribe to real-time pulse events"""
        pubsub = redis_client.pubsub()
        pubsub.subscribe("pulse_events")
        
        try:
            for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield PulseEvent(
                        id=data["id"],
                        pulse_type=PulseType(data["pulse_type"]),
                        source_node=data["source_node"],
                        target_node=data["target_node"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        data=data["data"],
                        status=data["status"]
                    )
        finally:
            pubsub.close()
    
    @strawberry.subscription
    async def graph_updates(self) -> AsyncGenerator[str, None]:
        """Subscribe to graph structure updates"""
        pubsub = redis_client.pubsub()
        pubsub.subscribe("graph_updates")
        
        try:
            for message in pubsub.listen():
                if message["type"] == "message":
                    yield message["data"].decode()
        finally:
            pubsub.close()

# Initialize GraphQL Schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)

# FastAPI Integration
app = FastAPI(title="BEM GraphQL API", description="GraphQL API for BEM System Cytoscape Visualization")

# GraphQL endpoint
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "bem-graphql-server",
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint for enhanced real-time features
@app.websocket("/ws/cytoscape")
async def cytoscape_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # Subscribe to Redis events
    pubsub = redis_client.pubsub()
    pubsub.subscribe("pulse_events", "graph_updates")
    
    try:
        while True:
            message = pubsub.get_message(timeout=1)
            if message and message["type"] == "message":
                await websocket.send_text(message["data"].decode())
            
            # Keep connection alive
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        pubsub.close()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 