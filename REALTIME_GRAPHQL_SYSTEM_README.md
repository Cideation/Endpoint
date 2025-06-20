# ⚡ Real-Time GraphQL System - No Cosmetic Delays

**Immediate graph updates directly triggered by backend state changes via GraphQL events — no fake animations, no client trickery.**

## 🏗️ Architecture Overview

| Layer                     | Purpose                                 | Stack                           |
| ------------------------- | --------------------------------------- | ------------------------------- |
| **GraphQL Subscriptions** | Receive live backend node/edge updates  | Apollo Client / GraphQL WS      |
| **Cytoscape.js**          | Visualize updated graph in real time    | React + Cytoscape.js            |
| **Backend Engine**        | Executes functors and publishes updates | Python, FastAPI, Strawberry GQL |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_realtime.txt
```

### 2. Launch Complete System
```bash
python start_realtime_system.py
```

### 3. Access Interfaces
- **Real-time Interface**: http://localhost:8005/realtime_graph_interface.html
- **GraphQL Playground**: http://localhost:8004/graphql
- **WebSocket Endpoint**: ws://localhost:8004/ws/realtime
- **Health Check**: http://localhost:8004/health

## 📦 System Components

### 1. GraphQL Real-Time Engine (`graphql_realtime_engine.py`)
- **FastAPI + Strawberry GraphQL** server
- **WebSocket subscriptions** for immediate updates
- **Real-time state management** without delays
- **Functor execution** with immediate result broadcasting

**Key Features:**
- ✅ **Zero animation delays** - immediate state reflection
- ✅ **WebSocket real-time** - instant backend-to-frontend sync
- ✅ **GraphQL subscriptions** - structured real-time data
- ✅ **Functor execution** - trigger graph computations
- ✅ **Multi-client support** - broadcast to all connected clients

### 2. Cytoscape Real-Time Client (`cytoscape_realtime_client.js`)
- **Cytoscape.js integration** with real-time updates
- **Apollo Client** for GraphQL subscriptions
- **WebSocket client** for direct real-time messaging
- **Phase-based styling** (Alpha, Beta, Gamma)

**Key Features:**
- ✅ **Immediate visual updates** - no cosmetic animations
- ✅ **Real-time node/edge updates** - direct from backend
- ✅ **Interactive graph** - drag nodes, execute functors
- ✅ **Phase visualization** - Alpha (blue), Beta (orange), Gamma (green)
- ✅ **Status indicators** - executing, completed, validated states

### 3. Real-Time Web Interface (`realtime_graph_interface.html`)
- **Complete web application** with sidebar controls
- **Connection status monitoring** 
- **Real-time statistics display**
- **Functor execution controls**
- **Activity logging** with timestamps

## 🔧 Configuration

### Environment Variables
```bash
# GraphQL Engine
GRAPHQL_PORT=8004
WEBSOCKET_PORT=8004

# Frontend Server
FRONTEND_PORT=8005

# Debug Options
ENABLE_DEBUG_LOG=true
```

### Service Configuration
```python
SERVICES = {
    'graphql_engine': {
        'script': 'frontend/graphql_realtime_engine.py',
        'port': 8004,
        'name': 'Real-Time GraphQL Engine'
    },
    'web_server': {
        'command': ['python', '-m', 'http.server', '8005'],
        'port': 8005,
        'name': 'Frontend Web Server'
    }
}
```

## 📡 GraphQL Schema

### Queries
```graphql
type Query {
  currentGraphState: String
  graphVersion: Int
  nodes: [Node!]!
  edges: [Edge!]!
}
```

### Mutations
```graphql
type Mutation {
  executeFunctor(nodeId: String!, functorType: String!, inputs: String!): FunctorExecution!
  updateNodePosition(nodeId: String!, x: Float!, y: Float!): Boolean!
}
```

### Subscriptions
```graphql
type Subscription {
  graphUpdates: GraphUpdate!
}
```

### Real-Time Types
```graphql
type Node {
  nodeId: String!
  functorType: String!
  phase: String!
  status: String!
  inputs: String!    # JSON
  outputs: String!   # JSON
  positionX: Float!
  positionY: Float!
  lastUpdate: String!
  version: Int!
}

type Edge {
  edgeId: String!
  sourceNode: String!
  targetNode: String!
  edgeType: String!
  weight: Float!
  metadata: String!  # JSON
  lastUpdate: String!
  version: Int!
}

type GraphUpdate {
  updateId: String!
  updateType: String!
  timestamp: String!
  node: Node
  edge: Edge
  functorExecution: FunctorExecution
  graphVersion: Int!
}
```

## 🔌 WebSocket Protocol

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8004/ws/realtime');
```

### Message Types
```javascript
// Ping/Pong for connection health
{
  "type": "ping",
  "timestamp": "2024-01-20T10:30:00Z"
}

// Real-time updates
{
  "type": "NODE_UPDATE",
  "update_id": "uuid-here",
  "timestamp": "2024-01-20T10:30:00Z",
  "payload": {
    "node": { /* node data */ },
    "graph_version": 123
  }
}

{
  "type": "FUNCTOR_EXECUTION",
  "update_id": "uuid-here", 
  "timestamp": "2024-01-20T10:30:00Z",
  "payload": {
    "functor_result": { /* execution result */ },
    "graph_version": 124
  }
}
```

## ⚙️ Functor Types

### Available Functors
1. **MaterialSpecification** - Define material properties
2. **DesignOptimization** - Optimize design parameters
3. **QualityValidation** - Validate quality standards
4. **CostAnalysis** - Analyze manufacturing costs
5. **ComplianceCheck** - Check regulatory compliance

### Execution Flow
```javascript
// Execute functor via client
const result = await realTimeClient.executeFunctor(
  'V01_ProductComponent',
  'MaterialSpecification', 
  {
    material_type: 'Steel_A36',
    strength_requirement: 36000
  }
);

// Immediate broadcast to all clients
// No delays - real backend state change
```

## 🎨 Visual Styling

### Phase-Based Node Colors
- **Alpha Phase** (DAG): Blue `#3498db` - Linear workflow
- **Beta Phase** (Relational): Orange `#f39c12` - Objective functions
- **Gamma Phase** (Combinatorial): Green `#27ae60` - Emergent properties

### Status Indicators
- **Idle**: Default border
- **Executing**: Red border `#e74c3c` with thick outline
- **Completed**: Green border `#27ae60` with medium outline
- **Validated**: Special completion indicator

### Edge Types
- **Alpha Edges**: Blue `#3498db` - Direct DAG flow
- **Beta Relationships**: Orange `#f39c12` - Objective connections
- **Gamma Edges**: Green `#27ae60` - Emergent dependencies
- **Cross-Phase**: Purple `#9b59b6` dashed - Phase transitions

## 🛠️ Development Tools

### Launch Options
```bash
# Complete system
python start_realtime_system.py

# GraphQL engine only
python start_realtime_system.py --graphql-only

# Test running system
python start_realtime_system.py --test
```

### Development Endpoints
```bash
# Trigger test node update
POST /dev/trigger_node_update
{
  "node_id": "TEST_123",
  "functor_type": "TestNode",
  "phase": "alpha",
  "status": "active"
}

# Trigger test edge update
POST /dev/trigger_edge_update
{
  "source_node": "V01",
  "target_node": "V02", 
  "edge_type": "test_edge",
  "weight": 2.5
}
```

### Client API
```javascript
// Initialize real-time client
const client = new CytoscapeRealTimeClient({
  containerId: 'graph-container',
  onUpdate: (updateType, data, version) => {
    console.log(`Update: ${updateType}, Version: ${version}`);
  }
});

// Execute functor
await client.executeFunctor('node_id', 'FunctorType', inputs);

// Update node position
await client.updateNodePosition('node_id', x, y);

// Get graph version
const version = client.getGraphVersion();

// Check connection status
const connected = client.isRealTimeConnected();
```

## 📊 Monitoring & Statistics

### Real-Time Stats
- **Connected Clients**: Number of active WebSocket connections
- **Graph Version**: Current version number (increments on each update)
- **Total Nodes**: Current node count in graph
- **Total Edges**: Current edge count in graph
- **Last Update**: Timestamp of most recent change

### Health Check Response
```json
{
  "status": "healthy",
  "service": "Real-Time GraphQL Engine",
  "connected_clients": 3,
  "graph_version": 142,
  "timestamp": "2024-01-20T10:30:00Z"
}
```

## 🔥 Performance Features

### Zero Delay Updates
- **No animations** - immediate visual state changes
- **Direct DOM updates** - bypass animation queues
- **WebSocket messaging** - fastest possible transport
- **Efficient diffing** - only update changed elements

### Scalability
- **Multi-client broadcasting** - efficient fan-out
- **Version tracking** - prevent update conflicts
- **Connection monitoring** - automatic reconnection
- **Resource cleanup** - memory leak prevention

## 🐛 Troubleshooting

### Common Issues

**WebSocket Connection Failed**
```bash
# Check if GraphQL engine is running
curl http://localhost:8004/health

# Check WebSocket endpoint
wscat -c ws://localhost:8004/ws/realtime
```

**GraphQL Queries Failing**
```bash
# Test GraphQL endpoint
curl -X POST http://localhost:8004/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ graphVersion }"}'
```

**Frontend Not Loading**
```bash
# Check frontend server
curl http://localhost:8005/realtime_graph_interface.html

# Start frontend manually
cd frontend && python -m http.server 8005
```

### Debug Logging
Enable detailed logging in client:
```javascript
const client = new CytoscapeRealTimeClient({
  enableDebugLog: true  // Enables console logging
});
```

## 📈 Production Deployment

### Docker Configuration
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_realtime.txt .
RUN pip install -r requirements_realtime.txt

COPY frontend/ ./frontend/
COPY start_realtime_system.py .

EXPOSE 8004 8005

CMD ["python", "start_realtime_system.py"]
```

### Environment Variables
```bash
GRAPHQL_HOST=0.0.0.0
GRAPHQL_PORT=8004
FRONTEND_PORT=8005
LOG_LEVEL=INFO
ENABLE_CORS=true
```

## 🔗 Integration with BEM System

### Connection Points
1. **ECM Gateway** → GraphQL subscriptions for pulse updates
2. **Functor Router** → GraphQL mutations for functor execution
3. **FSM Runtime** → WebSocket messages for state changes
4. **Agent State** → Real-time coefficient updates

### Data Flow
```
Backend State Change → GraphQL Subscription → WebSocket Broadcast → Cytoscape Update
     (Immediate)           (Immediate)           (Immediate)         (Immediate)
```

**No delays, no animations, no cosmetic trickery - just pure backend state synchronization.**

---

## 📝 Summary

This real-time GraphQL system provides **immediate, authentic graph updates** without any cosmetic delays. Every visual change in Cytoscape.js directly reflects a real backend state change, transmitted via GraphQL subscriptions and WebSocket connections for maximum responsiveness.

**Key Achievement**: ⚡ **Zero-delay real-time graph visualization** with authentic backend state synchronization. 