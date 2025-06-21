# Unreal Engine Blueprint Integration Guide

## Architecture Overview

**CRITICAL PRINCIPLE**: Pulse logic and state management remain in the backend to prevent data disconnects.

```
Backend (Authoritative):
├── ECM Gateway - Static infrastructure pipe for pulse routing
├── Node Engine - Pulse state management and logic  
├── Pulse Router - Semantic pulse processing
└── Functor Router - Business logic execution

Unreal Engine (Visual Only):
├── Pulse Visualization - Renders visual effects from backend commands
├── Spatial Events - Sends user interactions to backend
└── Component Display - Shows building objects with no business logic
```

## File Structure

```
UnrealBlueprints/
├── README_BLUEPRINT_INTEGRATION.md          # This guide
├── PulseSystem/
│   ├── BP_PulseAnimator.h                   # Visual-only pulse animation
│   ├── Materials/
│   │   ├── M_PulseMaterial_Master.uasset    # Master material for all pulses
│   │   ├── M_BidPulse.uasset               # Amber bid pulse material
│   │   ├── M_OccupancyPulse.uasset         # Blue occupancy pulse material
│   │   ├── M_CompliancyPulse.uasset        # Indigo compliance pulse material
│   │   ├── M_FitPulse.uasset               # Green fit pulse material
│   │   ├── M_InvestmentPulse.uasset        # Orange investment pulse material
│   │   ├── M_DecayPulse.uasset             # Gray decay pulse material
│   │   └── M_RejectPulse.uasset            # Red rejection pulse material
│   └── ParticleSystems/
│       ├── PS_PulsingAmber.uasset          # Bid pulse particles
│       ├── PS_FlowingBlue.uasset           # Occupancy pulse particles
│       ├── PS_SteadyIndigo.uasset          # Compliance pulse particles
│       ├── PS_ConfirmingGreen.uasset       # Fit pulse particles
│       ├── PS_GoldenSparkle.uasset         # Investment pulse particles
│       ├── PS_FadingGray.uasset            # Decay pulse particles
│       └── PS_RejectionFlash.uasset        # Rejection pulse particles
├── SpatialObjects/
│   ├── BP_ComponentBase.h                   # Base class for building components (visual only)
│   ├── BP_Wall.uasset                      # Wall component blueprint
│   ├── BP_Beam.uasset                      # Beam component blueprint
│   ├── BP_Column.uasset                    # Column component blueprint
│   └── Materials/
│       ├── M_ComponentMaterial_Master.uasset # Master material for components
│       ├── M_Concrete.uasset               # Concrete material
│       ├── M_Steel.uasset                  # Steel material
│       └── M_Wood.uasset                   # Wood material
├── NetworkIntegration/
│   ├── BP_WebSocketManager.h               # WebSocket communication (network only)
│   ├── BP_ECMConnector.uasset              # ECM Gateway connection blueprint
│   ├── BP_SpatialEventSender.uasset       # Spatial event transmission blueprint
│   └── BP_PulseReceiver.uasset            # Pulse visualization receiver blueprint
├── LevelBlueprints/
│   ├── BP_BEMLevel.uasset                  # Main level controller blueprint
│   ├── BP_ConstructionSite.uasset         # Construction environment blueprint
│   └── BP_ComponentSpawner.uasset         # Dynamic component spawning blueprint
└── PluginConfiguration/
    ├── BEMSystem.uplugin                   # Plugin definition file
    ├── Config/
    │   ├── DefaultEngine.ini               # Engine configuration
    │   └── DefaultGame.ini                 # Game configuration
    └── Source/
        ├── BEMSystem.Build.cs              # Build configuration
        └── Private/
            └── BEMSystemModule.cpp         # Module initialization
```

## Core Principles

### 1. Backend Authority
- **All pulse logic** remains in ECM Gateway and Node Engine
- **All business rules** stay in the backend systems
- **All state management** handled by backend
- Unreal Engine is purely a **display client**

### 2. Visual-Only Components
- `BP_PulseAnimator` - Only renders visual effects, no pulse logic
- `BP_ComponentBase` - Only handles visual representation, no business logic
- `BP_WebSocketManager` - Only network communication, no data processing

### 3. Data Flow
```
User Interaction (Unreal) 
    ↓ (Spatial Event)
ECM Gateway 
    ↓ (Route to Node Engine)
Node Engine (Process Logic)
    ↓ (Generate Pulse Command)
Pulse Router (Semantic Processing)
    ↓ (Visual Command)
Unreal Engine (Render Visual Effect)
```

## Installation Instructions

### 1. Plugin Installation
1. Copy the entire `UnrealBlueprints` folder to your Unreal project's `Plugins/` directory
2. Restart Unreal Engine
3. Enable the "BEM System" plugin in the Plugins window
4. Regenerate project files if using C++

### 2. Dependencies Setup
Required plugins (enable in Plugins window):
- **WebSocketNetworking** - For ECM Gateway communication
- **JsonBlueprintUtilities** - For JSON message parsing

### 3. Backend Connection Setup
1. Ensure backend services are running:
   - ECM Gateway (port 8765)
   - Pulse Handler (port 8767)
   - Node Engine
2. Configure WebSocket URLs in BP_BEMLevel:
   - **ECM Gateway URL**: `ws://localhost:8765`
   - **Pulse Handler URL**: `ws://localhost:8767`

## Blueprint Node Reference

### Visual-Only Nodes
These nodes handle ONLY visual representation:

#### Pulse Visualization Nodes
- **Start Pulse Animation** - Renders pulse visual based on backend command
- **Stop Pulse Animation** - Stops visual effect when backend commands
- **Set Pulse Intensity** - Adjusts visual intensity (0.0 to 1.0)
- **Apply Directional Flow** - Sets visual direction (downward, upward, etc.)

#### Spatial Object Nodes
- **Set Component Material** - Changes visual material only
- **Set Component Dimensions** - Updates visual size only
- **Set Highlight State** - Visual selection highlight only
- **Update Visual From Backend** - Syncs visual with backend state

#### Network Communication Nodes
- **Connect to ECM Gateway** - Establishes WebSocket connection only
- **Send Spatial Event** - Transmits user interaction to backend only
- **Request Pulse Visualization** - Requests visual effect from backend only

### Event Handling Nodes
These nodes trigger when backend sends commands:

- **On Pulse Received** - Event when backend sends pulse visualization command
- **On Component Selected** - Event when user selects building component (sends to backend)
- **On WebSocket Connected** - Event when ECM connection established
- **On Backend State Changed** - Event when backend updates component state

## Message Flow Examples

### Spatial Event to Backend
```json
{
  "type": "spatial_event",
  "event_type": "component_selected",
  "position": {"x": 100.0, "y": 200.0, "z": 50.0},
  "component_id": "wall_001",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Pulse Visualization from Backend
```json
{
  "type": "pulse_visualization_command",
  "pulse_type": "bid_pulse",
  "color": "#FFC107",
  "position": {"x": 150.0, "y": 250.0, "z": 75.0},
  "intensity": 0.8,
  "duration": 2.0,
  "direction": "downward",
  "visual_effects": ["particle_system", "attention_pulse"]
}
```

### Component State Update from Backend
```json
{
  "type": "component_state_update",
  "component_id": "wall_001",
  "visual_properties": {
    "material": "concrete",
    "dimensions": {"x": 500, "y": 20, "z": 300},
    "highlight": false,
    "cost_estimate": 1500.0
  }
}
```

## Visual Scripting Examples

### Example 1: Receive Pulse from Backend
```
Event On Pulse Received
├── Parse JSON (Pulse Data)
├── Get Pulse Type
├── Get Position
├── Get Intensity
├── Get Duration
└── Start Pulse Animation
    ├── Pulse Type: From Backend
    ├── Position: From Backend
    ├── Intensity: From Backend
    └── Duration: From Backend
```

### Example 2: Send Spatial Event to Backend
```
Event On Component Clicked
├── Get Component ID
├── Get Component Position
├── Create Spatial Event JSON
│   ├── Event Type: "component_selected"
│   ├── Component ID: From Component
│   └── Position: From Component
└── Send to ECM Gateway
```

### Example 3: Update Visual from Backend
```
Event On Backend State Changed
├── Parse Component Data
├── Get Component by ID
├── Update Component Material
├── Update Component Dimensions
└── Update Component Highlight
```

## Performance Optimization

### Network Optimization
- **Batch Events** - Group multiple spatial events
- **Event Throttling** - Limit event frequency to prevent spam
- **Connection Pooling** - Reuse WebSocket connections

### Visual Optimization
- **Object Pooling** - Reuse pulse visual effects
- **LOD System** - Reduce detail based on distance
- **Culling** - Hide effects outside camera view

## Debugging and Testing

### Debug Console Commands
- `BEM.ShowConnections 1` - Display WebSocket connection status
- `BEM.LogNetworkMessages 1` - Log all network communication
- `BEM.ShowPulseCommands 1` - Display pulse commands from backend
- `BEM.TestBackendConnection` - Test ECM Gateway connection

### Testing Workflow
1. Start backend services (ECM Gateway, Node Engine, Pulse Router)
2. Launch Unreal Engine with BEM plugin
3. Test WebSocket connection to backend
4. Send test spatial event
5. Verify pulse visualization received from backend
6. Confirm no business logic in Unreal components

## Integration Checklist

### Backend Integration
- [ ] ECM Gateway running on port 8765
- [ ] Pulse Handler running on port 8767
- [ ] Node Engine processing spatial events
- [ ] Pulse Router generating visual commands

### Unreal Integration  
- [ ] BEM System plugin installed and enabled
- [ ] WebSocket connection to ECM Gateway working
- [ ] Spatial events sending to backend
- [ ] Pulse visualizations receiving from backend
- [ ] No business logic in Unreal components
- [ ] All state management in backend

## Troubleshooting

### Connection Issues
1. **WebSocket Connection Failed**
   - Verify backend services are running
   - Check firewall settings
   - Confirm URL configuration

2. **Pulses Not Received**
   - Check backend pulse generation
   - Verify WebSocket message format
   - Confirm JSON parsing in Unreal

3. **Spatial Events Not Sent**
   - Check WebSocket connection status
   - Verify event JSON format
   - Confirm ECM Gateway receives events

### Architecture Violations
- **Business Logic in Unreal** - Move all logic to backend
- **State Management in Unreal** - Use backend as single source of truth
- **Direct Database Access** - Route all data through ECM Gateway

This architecture ensures data consistency, prevents disconnects, and maintains clear separation between backend logic and frontend visualization. 