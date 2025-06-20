# Cytoscape + GraphQL Integration Guide

## Overview
The integrated BEM Agent Console combines Cytoscape.js graph visualization with GraphQL API for real-time data affinity analysis. This creates a powerful interface for interacting with building information models through semantic graph operations.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cytoscape     │    │   GraphQL API    │    │   SFDE Engine   │
│   Frontend      │◄──►│   Server         │◄──►│   + Others      │
│                 │    │                  │    │                 │
│ • Graph Viz     │    │ • Schema Intro   │    │ • Calculations  │
│ • Node Select   │    │ • Data Affinity  │    │ • Formulas      │
│ • UI Controls   │    │ • Real-time      │    │ • Results       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   ECM Gateway       │
                    │   WebSocket         │
                    │   Infrastructure    │
                    └─────────────────────┘
```

## Quick Start

### 1. Launch Complete System
```bash
cd frontend/
python start_complete_system.py
```

This automatically:
- Starts ECM Gateway (WebSocket server)
- Starts GraphQL API server
- Launches microservices (if Docker available)
- Opens integrated interface in browser

### 2. Alternative Manual Launch
```bash
# Terminal 1: ECM Gateway
cd Final_Phase/
python ecm_gateway.py

# Terminal 2: GraphQL Server  
cd frontend/
python graphql_server.py

# Terminal 3: Open interface
open agent_console_graphql.html
```

## Using the Interface

### Graph Visualization
- **Node Selection**: Click any node to view details and enable data affinity analysis
- **Graph Navigation**: Use mouse to pan/zoom the Cytoscape graph
- **Node Types**: Different colors represent structural, cost, energy, MEP, spatial, time
- **Real-time Updates**: Graph reflects live data from GraphQL subscriptions

### Data Affinity Analysis

#### Step 1: Select Node
1. Click on any node in the graph
2. Node details appear in the right panel
3. Node highlights with green border when selected

#### Step 2: Choose Affinity Types
Select one or more data affinity types:
- **Structural**: Load calculations, safety factors, structural analysis
- **Cost**: Material costs, labor estimation, budget analysis  
- **Energy**: HVAC loads, efficiency calculations, energy modeling
- **MEP**: Mechanical, electrical, plumbing system analysis
- **Spatial**: Geometric transformations, spatial relationships
- **Time**: Scheduling, sequencing, project timeline analysis

#### Step 3: Execute Analysis
1. Click "Execute Data Affinity Analysis" button
2. System sends GraphQL mutation to backend
3. SFDE microservice processes calculations
4. Results display in Analysis Results section

### Connection Status
Monitor system health via connection indicators:
- **GraphQL Server**: Green = API available, Red = offline
- **WebSocket**: Green = real-time connected, Red = disconnected

## GraphQL API Reference

### Key Queries
```graphql
# Load graph data for Cytoscape
query LoadCytoscapeGraph($filter: NodeFilter) {
  graph(filter: $filter) {
    nodes {
      id
      type
      phase
      primaryFunctor
      position { x y }
      coefficients {
        structural
        cost
        energy
        mep
        fabrication
        time
      }
      status
    }
    edges {
      id
      source
      target
      relationshipType
      weight
    }
  }
}

# Get data affinity configuration
query GetAffinityConfiguration {
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

### Key Mutations
```graphql
# Execute data affinity analysis
mutation ExecuteDataAffinity($request: AffinityRequest!) {
  executeDataAffinity(request: $request) {
    structuralAffinity {
      affinityType
      executionStatus
      formulasExecuted
      componentId
      calculationResults
    }
    costAffinity {
      affinityType
      executionStatus
      formulasExecuted
      componentId
      calculationResults
    }
    # ... other affinity types
  }
}
```

### Real-time Subscriptions
```graphql
# Subscribe to live pulse events
subscription LivePulseEvents {
  pulseEvents {
    id
    pulseType
    sourceNode
    targetNode
    timestamp
    status
    data
  }
}

# Subscribe to graph updates
subscription GraphUpdates {
  graphUpdates {
    nodeId
    updateType
    newData
    timestamp
  }
}
```

## Data Affinity System

### Affinity Types
The system supports six core data affinity types based on BEM domain expertise:

1. **Structural Affinity**
   - Load path analysis
   - Safety factor calculations
   - Material strength verification
   - Deflection analysis

2. **Cost Affinity**
   - Material cost estimation
   - Labor calculation
   - Equipment costs
   - Total project budgeting

3. **Energy Affinity**
   - HVAC load calculations
   - Lighting power density
   - Energy efficiency metrics
   - Thermal performance

4. **MEP Affinity**
   - Mechanical system sizing
   - Electrical load analysis  
   - Plumbing flow calculations
   - System integration

5. **Spatial Affinity**
   - Geometric transformations
   - Volume/area calculations
   - Spatial relationship analysis
   - Coordinate system mapping

6. **Time Affinity**
   - Construction sequencing
   - Project scheduling
   - Duration estimation
   - Critical path analysis

### Execution Modes
- **Symbolic Reasoning**: Uses mathematical formulas and engineering principles
- **Lookup Tables**: References pre-calculated values and standards
- **Hybrid**: Combines symbolic and lookup approaches

## Integration Points

### Frontend → GraphQL
- Cytoscape events trigger GraphQL queries/mutations
- Real-time subscriptions update graph visualization
- User selections drive data affinity requests

### GraphQL → Microservices
- SFDE Engine processes structural/cost/energy calculations
- DAG Engine handles graph relationships and paths
- Functor Types Engine manages component compatibility

### WebSocket → Real-time Updates
- ECM Gateway provides persistent connection infrastructure
- Pulse events propagate through WebSocket to frontend
- Graph state changes trigger live updates

## Troubleshooting

### Common Issues

**GraphQL server not connecting:**
- Check if port 8000 is available
- Verify Python dependencies installed
- Check server logs for startup errors

**ECM Gateway offline:**
- Ensure port 8765 is available
- Check Final_Phase/ecm_log.txt for errors
- Verify WebSocket client compatibility

**No data affinity results:**
- Check microservice connectivity (Docker containers)
- Verify SFDE engine is responding
- Review GraphQL server logs for backend errors

**Graph not loading:**
- Check browser console for JavaScript errors
- Verify GraphQL API is returning valid data
- Test GraphQL queries directly at http://localhost:8000/

### Debug Mode
Enable detailed logging by modifying GraphQL server:
```python
# In graphql_server.py
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Affinity Parameters
Modify calculation parameters via GraphQL variables:
```javascript
const parameters = {
    volume_cm3: 2000.0,    // Component volume
    area_m2: 15.0,         // Surface area
    length_mm: 3000.0,     // Primary dimension
    material_type: "steel", // Material specification
    load_factor: 1.6       // Safety factor
};
```

### Graph Layout Customization
Modify Cytoscape layout in `agent_console_graphql.html`:
```javascript
cy.layout({
    name: 'cose',           // Force-directed layout
    idealEdgeLength: 100,   // Edge length
    nodeOverlap: 20,        // Node spacing
    animate: true          // Smooth transitions
}).run();
```

### Real-time Event Handling
Customize WebSocket event processing:
```javascript
function handleRealtimeUpdate(data) {
    if (data.type === 'node_update') {
        updateNodeInGraph(data);
    } else if (data.type === 'calculation_complete') {
        refreshAffinityResults(data);
    }
    // Add custom event types here
}
```

## Performance Optimization

### Large Graphs
- Implement node filtering in GraphQL queries
- Use Cytoscape compound nodes for grouping
- Enable viewport culling for rendering optimization

### Real-time Updates
- Throttle WebSocket message processing
- Batch graph updates for better performance
- Use delta updates instead of full graph refresh

### Memory Management
- Clean up WebSocket connections on page unload
- Implement GraphQL query result caching
- Monitor browser memory usage with dev tools

## Security Considerations

- GraphQL endpoint exposed on localhost only
- No authentication implemented (development environment)
- WebSocket connections accept all origins
- Consider adding rate limiting for production

For production deployment, implement:
- JWT authentication
- CORS restrictions
- Rate limiting
- Input validation
- HTTPS/WSS encryption 