# BEM GraphQL Server - Frontend Integration

## 🎯 Overview

The BEM GraphQL Server provides a modern API for the Cytoscape Agent Console with **data affinity** integration, real-time pulse visualization, and component analysis capabilities.

## 🔧 Data Affinity Focus

This GraphQL implementation is centered around the BEM system's **data affinity** concept:

### **Data Affinity Types:**
- **`stateless_spatial`** - Works with geometric/spatial data; pure transformations based on topology
- **`local_calcs`** - Performs numeric evaluations on specific node inputs; no global aggregation  
- **`aggregator`** - Consolidates multiple results into decisions and routing logic

### **Affinity Categories:**
- **`structural`** - Load calculations, material properties, safety factors
- **`cost`** - Cost estimation, material costs, labor calculations
- **`energy`** - Energy efficiency, consumption, HVAC loads
- **`mep`** - Mechanical, electrical, plumbing system calculations
- **`spatial`** - Geometric transformations, positioning, topology
- **`time`** - Scheduling, sequencing, time-based calculations

## 🚀 Quick Start

### 1. Start the GraphQL Server
```bash
cd frontend
python start_graphql_server.py
```

### 2. Access GraphQL Playground
Visit `http://localhost:8000/graphql` for interactive testing

### 3. Run Tests
```bash
python test_graphql_demo.py
```

## 📋 Key API Examples

### Get Affinity Configuration
```graphql
query GetAffinityConfig {
    affinityConfiguration {
        functorTypes {
            name
            dataAffinity
            includedFunctors
        }
        availableAffinityTypes
    }
}
```

### Execute Data Affinity Analysis
```graphql
mutation ExecuteDataAffinity {
    executeDataAffinity(
        request: {
            componentId: "structural_beam_001"
            affinityTypes: ["structural", "cost", "energy"]
            executionMode: "symbolic_reasoning"
            parameters: {
                volume_cm3: 1500.0
                area_m2: 12.5
                length_mm: 3000.0
            }
        }
    ) {
        structuralAffinity {
            affinityType
            executionStatus
            formulasExecuted
            calculationResults
        }
        costAffinity {
            affinityType
            calculationResults
        }
        energyAffinity {
            affinityType
            calculationResults
        }
    }
}
```

### Load Graph for Cytoscape
```graphql
query LoadCytoscapeGraph {
    graph {
        nodes {
            id type phase primaryFunctor
            position { x y }
            coefficients { structural cost energy }
        }
        edges { id source target relationshipType weight }
    }
}
```

### Real-time Pulse Events
```graphql
subscription LivePulseEvents {
    pulseEvents {
        id pulseType sourceNode targetNode timestamp status
    }
}
```

## 🏗️ Integration with BEM Architecture

### **SFDE Microservice Integration**
- Calls existing `SFDE_ENGINE` container for calculations
- Uses `neon/container_client.py` for communication
- Processes results through established affinity pipelines

### **Data Affinity Files Used:**
- `MICROSERVICE_ENGINES/functor_types_with_affinity.json`
- `MICROSERVICE_ENGINES/functor_data_affinity.json`
- `MICROSERVICE_ENGINES/sfde_runner_net.py`

### **Real-time Architecture:**
- GraphQL subscriptions via Redis pub/sub
- WebSocket endpoint for enhanced real-time features
- Pulse system integration with 7-pulse architecture

## 🎨 Cytoscape Frontend Integration

### **JavaScript Integration Example:**
```javascript
// GraphQL client for Cytoscape
const client = new ApolloClient({
    uri: 'http://localhost:8000/graphql',
    wsUri: 'ws://localhost:8000/graphql'
});

// Load affinity configuration
const { data } = await client.query({
    query: AFFINITY_CONFIGURATION_QUERY
});

// Execute data affinity on node selection
const executeAffinity = async (nodeId, affinityTypes) => {
    const result = await client.mutate({
        mutation: EXECUTE_DATA_AFFINITY_MUTATION,
        variables: {
            request: {
                componentId: nodeId,
                affinityTypes: affinityTypes,
                executionMode: "symbolic_reasoning"
            }
        }
    });
    
    // Update Cytoscape visualization with results
    updateNodeVisualization(nodeId, result.data.executeDataAffinity);
};

// Real-time pulse subscription
client.subscribe({
    query: PULSE_EVENTS_SUBSCRIPTION
}).subscribe({
    next: (pulseEvent) => {
        visualizePulseEvent(pulseEvent.data.pulseEvents);
    }
});
```

## 🧪 Testing & Development

### **Test Suite Coverage:**
- ✅ Health check and server status
- ✅ Schema introspection validation
- ✅ Affinity configuration queries
- ✅ Data affinity execution with SFDE
- ✅ Graph data for Cytoscape
- ✅ Real-time pulse events
- ✅ WebSocket connectivity
- ✅ Node position updates

### **Development Server:**
```bash
# Start with hot reload
python start_graphql_server.py

# Start with testing
python start_graphql_server.py --test

# Start with Docker
python start_graphql_server.py --docker
```

## 🐳 Docker Deployment

```bash
docker-compose -f docker-compose.graphql.yml up --build
```

The Docker setup includes:
- PostgreSQL with BEM schema
- Redis for real-time subscriptions  
- GraphQL server with hot reload
- Health checks and networking

## 📊 Data Flow

1. **Cytoscape UI** → GraphQL query for graph data
2. **Node Selection** → Execute data affinity mutation
3. **SFDE Microservice** → Process affinity calculations
4. **GraphQL Response** → Structured affinity results
5. **Real-time Updates** → Pulse events via subscription
6. **Visualization** → Updated Cytoscape display

## 🔮 Advanced Features

### **Schema Introspection Benefits:**
- Dynamic UI generation from GraphQL schema
- Automatic field discovery for Cytoscape properties
- Type-safe development with GraphQL clients
- Built-in documentation via GraphQL playground

### **Data Affinity Advantages:**
- Precise calculation targeting by affinity type
- SFDE formula execution based on component context
- Microservice integration with existing container orchestration
- Real-time results for interactive design workflows

## 🔧 Configuration

### **Environment Variables:**
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bem_production
DB_USER=bem_user
DB_PASSWORD=bem_secure_pass_2024
REDIS_URL=redis://localhost:6379
```

### **Affinity Types Available:**
- `structural` - NSCP 2015 LRFD, beam calculations
- `cost` - Material costs, labor estimation
- `energy` - HVAC loads, lighting calculations
- `mep` - Electrical, plumbing, mechanical systems
- `spatial` - Geometric transformations, positioning
- `time` - Project scheduling, sequencing

## 🎯 Next Steps

1. **Enhanced Cytoscape Integration** - Full agent console integration
2. **Batch Affinity Processing** - Multiple components simultaneously  
3. **Caching Layer** - Redis-based calculation result caching
4. **Advanced Filtering** - Complex graph queries with multiple criteria
5. **Performance Monitoring** - GraphQL query performance analytics

The GraphQL server provides a clean, modern interface to the BEM system's data affinity capabilities while maintaining full compatibility with the existing microservice architecture and pulse system. 