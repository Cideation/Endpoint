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

# Data affinity utilities
def load_affinity_configuration():
    """Load functor affinity configuration from JSON files"""
    try:
        affinity_config = {}
        
        # Load functor types with affinity
        with open("../MICROSERVICE_ENGINES/functor_types_with_affinity.json", "r") as f:
            affinity_config["types"] = json.load(f)
        
        # Load functor data affinity
        with open("../MICROSERVICE_ENGINES/functor_data_affinity.json", "r") as f:
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

from neon.database_integration import DatabaseManager
from neon.container_client import ContainerClient, ContainerType

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "bem_production"),
    "user": os.getenv("DB_USER", "bem_user"),
    "password": os.getenv("DB_PASSWORD", "your_password")
}

# Redis for real-time subscriptions
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

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
    """Get async database connection"""
    return await asyncpg.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )

def get_sync_db_connection():
    """Get synchronous database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

async def load_graph_from_db(filter_params: Optional[NodeFilter] = None, project_context: Optional[ProjectContextInput] = None) -> Graph:
    """Load graph data from database with optional filtering"""
    conn = await get_async_db_connection()
    
    try:
        # Build WHERE clause based on filters
        where_conditions = []
        params = []
        
        if filter_params:
            if filter_params.phases:
                phases_str = [phase.value for phase in filter_params.phases]
                where_conditions.append(f"phase = ANY(${len(params) + 1})")
                params.append(phases_str)
            
            if filter_params.functor_types:
                functors_str = [ft.value for ft in filter_params.functor_types]
                where_conditions.append(f"primary_functor = ANY(${len(params) + 1})")
                params.append(functors_str)
            
            if filter_params.node_ids:
                where_conditions.append(f"id = ANY(${len(params) + 1})")
                params.append(filter_params.node_ids)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Load nodes
        nodes_query = f"""
            SELECT id, type, phase, primary_functor, secondary_functor,
                   coefficients, position, properties, status
            FROM graph_nodes {where_clause}
        """
        
        nodes_rows = await conn.fetch(nodes_query, *params)
        
        # Load edges
        edges_query = """
            SELECT id, source_node, target_node, relationship_type, weight, properties
            FROM graph_edges
            WHERE source_node = ANY($1) OR target_node = ANY($1)
        """
        
        node_ids = [row['id'] for row in nodes_rows]
        edges_rows = await conn.fetch(edges_query, node_ids) if node_ids else []
        
        # Convert to GraphQL types
        nodes = []
        for row in nodes_rows:
            coefficients_data = row['coefficients'] or {}
            position_data = row['position']
            
            node = Node(
                id=row['id'],
                type=NodeType(row['type']),
                phase=PhaseType(row['phase']),
                primary_functor=NodeType(row['primary_functor']),
                secondary_functor=NodeType(row['secondary_functor']) if row['secondary_functor'] else None,
                coefficients=Coefficients(**coefficients_data),
                position=Position(**position_data) if position_data else None,
                properties=NodeProperties(**row['properties']) if row['properties'] else None,
                status=row['status']
            )
            nodes.append(node)
        
        edges = []
        for row in edges_rows:
            edge = Edge(
                id=row['id'],
                source=row['source_node'],
                target=row['target_node'],
                relationship_type=row['relationship_type'],
                weight=row['weight'],
                properties=row['properties']
            )
            edges.append(edge)
        
        return Graph(nodes=nodes, edges=edges)
        
    finally:
        await conn.close()

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
        conn = await get_async_db_connection()
        
        try:
            query = """
                SELECT id, pulse_type, source_node, target_node, timestamp, data, status
                FROM pulse_events
                ORDER BY timestamp DESC
                LIMIT $1
            """
            
            rows = await conn.fetch(query, limit)
            
            events = []
            for row in rows:
                event = PulseEvent(
                    id=row['id'],
                    pulse_type=PulseType(row['pulse_type']),
                    source_node=row['source_node'],
                    target_node=row['target_node'],
                    timestamp=row['timestamp'],
                    data=row['data'],
                    status=row['status']
                )
                events.append(event)
            
            return events
            
        finally:
            await conn.close()
    
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
        conn = await get_async_db_connection()
        
        try:
            await conn.execute(
                "UPDATE graph_nodes SET position = $1 WHERE id = $2",
                json.dumps({"x": position.x, "y": position.y}),
                id
            )
            
            # Publish update event
            redis_client.publish("graph_updates", json.dumps({
                "type": "node_position_update",
                "node_id": id,
                "position": {"x": position.x, "y": position.y},
                "timestamp": datetime.now().isoformat()
            }))
            
            return f"Updated position for node {id}"
            
        finally:
            await conn.close()
    
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
        
        # Save to database
        conn = await get_async_db_connection()
        try:
            await conn.execute("""
                INSERT INTO pulse_events (id, pulse_type, source_node, target_node, timestamp, data, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
            pulse_event.id, pulse_event.pulse_type.value, pulse_event.source_node,
            pulse_event.target_node, pulse_event.timestamp, 
            json.dumps(pulse_event.data), pulse_event.status)
        finally:
            await conn.close()
        
        # Publish real-time event
        await publish_pulse_event(pulse_event)
        
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