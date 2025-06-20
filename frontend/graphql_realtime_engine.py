#!/usr/bin/env python3
"""
Real-Time UX GraphQL Engine - No Cosmetic Delays
⚡ Immediate graph updates triggered by backend state changes via GraphQL events
Stack: FastAPI + Strawberry GraphQL + WebSocket subscriptions + Cytoscape.js
"""

import asyncio
import json
import logging
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator, List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# REAL-TIME STATE MANAGEMENT
# ============================================================================

class RealTimeStateManager:
    """Manages real-time graph state and publishes immediate updates"""
    
    def __init__(self):
        self.subscribers: Dict[str, WebSocket] = {}
        self.graph_state = {
            'nodes': {},
            'edges': {},
            'last_update': datetime.now().isoformat(),
            'version': 1
        }
        self.update_queue = asyncio.Queue()
        
    async def subscribe(self, client_id: str, websocket: WebSocket):
        """Subscribe client to real-time updates"""
        self.subscribers[client_id] = websocket
        logger.info(f"⚡ Client {client_id} subscribed to real-time updates")
        
        # Send current state immediately
        await self.send_to_client(client_id, {
            'type': 'INITIAL_STATE',
            'payload': self.graph_state
        })
    
    async def unsubscribe(self, client_id: str):
        """Unsubscribe client from updates"""
        if client_id in self.subscribers:
            del self.subscribers[client_id]
            logger.info(f"❌ Client {client_id} unsubscribed")
    
    async def publish_node_update(self, node_data: Dict[str, Any]):
        """Publish immediate node state change"""
        update_id = str(uuid.uuid4())
        
        # Update internal state
        self.graph_state['nodes'][node_data['node_id']] = node_data
        self.graph_state['last_update'] = datetime.now().isoformat()
        self.graph_state['version'] += 1
        
        # Broadcast to all subscribers immediately
        update_message = {
            'type': 'NODE_UPDATE',
            'update_id': update_id,
            'timestamp': datetime.now().isoformat(),
            'payload': {
                'node': node_data,
                'graph_version': self.graph_state['version']
            }
        }
        
        await self.broadcast_update(update_message)
        logger.info(f"⚡ Published node update: {node_data['node_id']}")
    
    async def publish_edge_update(self, edge_data: Dict[str, Any]):
        """Publish immediate edge state change"""
        update_id = str(uuid.uuid4())
        
        # Update internal state
        edge_key = f"{edge_data['source_node']}->{edge_data['target_node']}"
        self.graph_state['edges'][edge_key] = edge_data
        self.graph_state['last_update'] = datetime.now().isoformat()
        self.graph_state['version'] += 1
        
        # Broadcast to all subscribers immediately
        update_message = {
            'type': 'EDGE_UPDATE',
            'update_id': update_id,
            'timestamp': datetime.now().isoformat(),
            'payload': {
                'edge': edge_data,
                'graph_version': self.graph_state['version']
            }
        }
        
        await self.broadcast_update(update_message)
        logger.info(f"⚡ Published edge update: {edge_key}")
    
    async def publish_functor_execution(self, functor_result: Dict[str, Any]):
        """Publish immediate functor execution result"""
        update_id = str(uuid.uuid4())
        
        # Update affected nodes
        for node_id, node_changes in functor_result.get('affected_nodes', {}).items():
            if node_id in self.graph_state['nodes']:
                self.graph_state['nodes'][node_id].update(node_changes)
        
        self.graph_state['last_update'] = datetime.now().isoformat()
        self.graph_state['version'] += 1
        
        # Broadcast functor execution result
        update_message = {
            'type': 'FUNCTOR_EXECUTION',
            'update_id': update_id,
            'timestamp': datetime.now().isoformat(),
            'payload': {
                'functor_result': functor_result,
                'graph_version': self.graph_state['version']
            }
        }
        
        await self.broadcast_update(update_message)
        logger.info(f"⚡ Published functor execution: {functor_result.get('functor_type', 'unknown')}")
    
    async def broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected clients immediately"""
        disconnected_clients = []
        
        for client_id, websocket in self.subscribers.items():
            try:
                await websocket.send_text(json.dumps(message))
            except WebSocketDisconnect:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.unsubscribe(client_id)
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.subscribers:
            try:
                await self.subscribers[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                await self.unsubscribe(client_id)

# Global state manager instance
state_manager = RealTimeStateManager()

# ============================================================================
# GRAPHQL SCHEMA DEFINITIONS
# ============================================================================

@strawberry.type
class Node:
    """GraphQL Node type for real-time updates"""
    node_id: str
    functor_type: str
    phase: str
    status: str
    inputs: str  # JSON string
    outputs: str  # JSON string
    position_x: float
    position_y: float
    last_update: str
    version: int

@strawberry.type
class Edge:
    """GraphQL Edge type for real-time updates"""
    edge_id: str
    source_node: str
    target_node: str
    edge_type: str
    weight: float
    metadata: str  # JSON string
    last_update: str
    version: int

@strawberry.type
class FunctorExecution:
    """GraphQL Functor execution result"""
    execution_id: str
    functor_type: str
    node_id: str
    execution_time_ms: float
    success: bool
    inputs: str  # JSON string
    outputs: str  # JSON string
    affected_nodes: str  # JSON string
    timestamp: str

@strawberry.type
class GraphUpdate:
    """Real-time graph update event"""
    update_id: str
    update_type: str
    timestamp: str
    node: Optional[Node] = None
    edge: Optional[Edge] = None
    functor_execution: Optional[FunctorExecution] = None
    graph_version: int

# ============================================================================
# GRAPHQL RESOLVERS
# ============================================================================

@strawberry.type
class Query:
    """GraphQL Query resolvers"""
    
    @strawberry.field
    async def current_graph_state(self) -> str:
        """Get current complete graph state"""
        return json.dumps(state_manager.graph_state)
    
    @strawberry.field
    async def graph_version(self) -> int:
        """Get current graph version number"""
        return state_manager.graph_state['version']
    
    @strawberry.field
    async def nodes(self) -> List[Node]:
        """Get all current nodes"""
        nodes = []
        for node_data in state_manager.graph_state['nodes'].values():
            nodes.append(Node(
                node_id=node_data['node_id'],
                functor_type=node_data.get('functor_type', 'unknown'),
                phase=node_data.get('phase', 'alpha'),
                status=node_data.get('status', 'idle'),
                inputs=json.dumps(node_data.get('inputs', {})),
                outputs=json.dumps(node_data.get('outputs', {})),
                position_x=node_data.get('position_x', 0.0),
                position_y=node_data.get('position_y', 0.0),
                last_update=node_data.get('last_update', datetime.now().isoformat()),
                version=node_data.get('version', 1)
            ))
        return nodes
    
    @strawberry.field
    async def edges(self) -> List[Edge]:
        """Get all current edges"""
        edges = []
        for edge_data in state_manager.graph_state['edges'].values():
            edges.append(Edge(
                edge_id=edge_data.get('edge_id', 'unknown'),
                source_node=edge_data['source_node'],
                target_node=edge_data['target_node'],
                edge_type=edge_data.get('edge_type', 'default'),
                weight=edge_data.get('weight', 1.0),
                metadata=json.dumps(edge_data.get('metadata', {})),
                last_update=edge_data.get('last_update', datetime.now().isoformat()),
                version=edge_data.get('version', 1)
            ))
        return edges

@strawberry.type
class Mutation:
    """GraphQL Mutation resolvers"""
    
    @strawberry.mutation
    async def execute_functor(self, node_id: str, functor_type: str, inputs: str) -> FunctorExecution:
        """Execute functor and trigger real-time updates"""
        execution_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Parse inputs
            input_data = json.loads(inputs)
            
            # Simulate functor execution (replace with actual execution)
            result = await simulate_functor_execution(node_id, functor_type, input_data)
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Create execution result
            execution_result = {
                'execution_id': execution_id,
                'functor_type': functor_type,
                'node_id': node_id,
                'execution_time_ms': execution_time,
                'success': result['success'],
                'inputs': inputs,
                'outputs': json.dumps(result['outputs']),
                'affected_nodes': json.dumps(result['affected_nodes']),
                'timestamp': datetime.now().isoformat()
            }
            
            # Publish real-time update immediately
            await state_manager.publish_functor_execution(execution_result)
            
            return FunctorExecution(**execution_result)
            
        except Exception as e:
            logger.error(f"Functor execution failed: {e}")
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            error_result = FunctorExecution(
                execution_id=execution_id,
                functor_type=functor_type,
                node_id=node_id,
                execution_time_ms=execution_time,
                success=False,
                inputs=inputs,
                outputs=json.dumps({'error': str(e)}),
                affected_nodes='{}',
                timestamp=datetime.now().isoformat()
            )
            
            return error_result
    
    @strawberry.mutation
    async def update_node_position(self, node_id: str, x: float, y: float) -> bool:
        """Update node position and trigger real-time update"""
        if node_id in state_manager.graph_state['nodes']:
            node_data = state_manager.graph_state['nodes'][node_id].copy()
            node_data['position_x'] = x
            node_data['position_y'] = y
            node_data['last_update'] = datetime.now().isoformat()
            
            await state_manager.publish_node_update(node_data)
            return True
        return False

@strawberry.type
class Subscription:
    """GraphQL Subscription resolvers for real-time updates"""
    
    @strawberry.subscription
    async def graph_updates(self) -> AsyncGenerator[GraphUpdate, None]:
        """Subscribe to real-time graph updates"""
        client_id = str(uuid.uuid4())
        
        # Create a queue for this subscription
        update_queue = asyncio.Queue()
        
        # Store subscription
        subscription_queues[client_id] = update_queue
        
        try:
            while True:
                # Wait for updates
                update_data = await update_queue.get()
                
                # Convert to GraphQL type
                graph_update = GraphUpdate(
                    update_id=update_data['update_id'],
                    update_type=update_data['type'],
                    timestamp=update_data['timestamp'],
                    graph_version=update_data['payload']['graph_version']
                )
                
                # Add specific data based on update type
                if update_data['type'] == 'NODE_UPDATE':
                    node_data = update_data['payload']['node']
                    graph_update.node = Node(
                        node_id=node_data['node_id'],
                        functor_type=node_data.get('functor_type', 'unknown'),
                        phase=node_data.get('phase', 'alpha'),
                        status=node_data.get('status', 'idle'),
                        inputs=json.dumps(node_data.get('inputs', {})),
                        outputs=json.dumps(node_data.get('outputs', {})),
                        position_x=node_data.get('position_x', 0.0),
                        position_y=node_data.get('position_y', 0.0),
                        last_update=node_data.get('last_update', datetime.now().isoformat()),
                        version=node_data.get('version', 1)
                    )
                elif update_data['type'] == 'EDGE_UPDATE':
                    edge_data = update_data['payload']['edge']
                    graph_update.edge = Edge(
                        edge_id=edge_data.get('edge_id', 'unknown'),
                        source_node=edge_data['source_node'],
                        target_node=edge_data['target_node'],
                        edge_type=edge_data.get('edge_type', 'default'),
                        weight=edge_data.get('weight', 1.0),
                        metadata=json.dumps(edge_data.get('metadata', {})),
                        last_update=edge_data.get('last_update', datetime.now().isoformat()),
                        version=edge_data.get('version', 1)
                    )
                elif update_data['type'] == 'FUNCTOR_EXECUTION':
                    functor_data = update_data['payload']['functor_result']
                    graph_update.functor_execution = FunctorExecution(**functor_data)
                
                yield graph_update
                
        except asyncio.CancelledError:
            # Clean up subscription
            if client_id in subscription_queues:
                del subscription_queues[client_id]
            raise

# Global subscription queues
subscription_queues: Dict[str, asyncio.Queue] = {}

# ============================================================================
# BACKEND FUNCTOR SIMULATION
# ============================================================================

async def simulate_functor_execution(node_id: str, functor_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate functor execution with realistic processing"""
    
    # Simulate processing time (remove in production)
    await asyncio.sleep(0.1)
    
    # Generate outputs based on functor type
    if functor_type == 'MaterialSpecification':
        outputs = {
            'material_properties': {
                'yield_strength': inputs.get('strength_requirement', 36000),
                'density': 7850,
                'cost_per_kg': 2.5
            },
            'specification_complete': True
        }
        affected_nodes = {
            node_id: {
                'status': 'completed',
                'outputs': outputs,
                'last_update': datetime.now().isoformat()
            }
        }
    
    elif functor_type == 'DesignOptimization':
        outputs = {
            'optimized_parameters': {
                'weight': inputs.get('target_weight', 150) * 0.95,
                'cost': inputs.get('budget', 1000) * 0.88,
                'efficiency': 0.92
            },
            'optimization_complete': True
        }
        affected_nodes = {
            node_id: {
                'status': 'optimized',
                'outputs': outputs,
                'last_update': datetime.now().isoformat()
            }
        }
    
    elif functor_type == 'QualityValidation':
        outputs = {
            'quality_score': 94.5,
            'validation_passed': True,
            'compliance_standards': ['ISO_9001', 'AISC_360']
        }
        affected_nodes = {
            node_id: {
                'status': 'validated',
                'outputs': outputs,
                'last_update': datetime.now().isoformat()
            }
        }
    
    else:
        outputs = {'result': 'processed', 'timestamp': datetime.now().isoformat()}
        affected_nodes = {
            node_id: {
                'status': 'processed',
                'outputs': outputs,
                'last_update': datetime.now().isoformat()
            }
        }
    
    return {
        'success': True,
        'outputs': outputs,
        'affected_nodes': affected_nodes
    }

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

# Create GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)

# Create FastAPI app
app = FastAPI(
    title="BEM Real-Time GraphQL Engine",
    description="⚡ Immediate graph updates without cosmetic delays",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GraphQL router
graphql_app = GraphQLRouter(
    schema,
    subscription_protocols=[GRAPHQL_TRANSPORT_WS_PROTOCOL]
)

app.include_router(graphql_app, prefix="/graphql")

# ============================================================================
# WEBSOCKET ENDPOINT FOR DIRECT REAL-TIME UPDATES
# ============================================================================

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """Direct WebSocket endpoint for real-time updates"""
    await websocket.accept()
    client_id = str(uuid.uuid4())
    
    try:
        await state_manager.subscribe(client_id, websocket)
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client messages (ping, requests, etc.)
                if message.get('type') == 'ping':
                    await websocket.send_text(json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {client_id}")
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {e}")
                break
                
    finally:
        await state_manager.unsubscribe(client_id)

# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Real-Time GraphQL Engine",
        "connected_clients": len(state_manager.subscribers),
        "graph_version": state_manager.graph_state['version'],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get real-time statistics"""
    return {
        "connected_clients": len(state_manager.subscribers),
        "total_nodes": len(state_manager.graph_state['nodes']),
        "total_edges": len(state_manager.graph_state['edges']),
        "graph_version": state_manager.graph_state['version'],
        "last_update": state_manager.graph_state['last_update'],
        "active_subscriptions": len(subscription_queues)
    }

# ============================================================================
# DEVELOPMENT ENDPOINTS FOR TESTING
# ============================================================================

@app.post("/dev/trigger_node_update")
async def trigger_node_update(node_data: dict):
    """Development endpoint to trigger node updates"""
    await state_manager.publish_node_update(node_data)
    return {"status": "update_published", "node_id": node_data.get('node_id')}

@app.post("/dev/trigger_edge_update")
async def trigger_edge_update(edge_data: dict):
    """Development endpoint to trigger edge updates"""
    await state_manager.publish_edge_update(edge_data)
    return {"status": "update_published", "edge": f"{edge_data.get('source_node')}->{edge_data.get('target_node')}"}

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Real-Time GraphQL Engine")
    logger.info("⚡ No cosmetic delays - immediate backend state synchronization")
    logger.info("🔗 GraphQL Playground: http://localhost:8004/graphql")
    logger.info("🌐 WebSocket endpoint: ws://localhost:8004/ws/realtime")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info",
        access_log=True
    ) 