# BP_BEMLevel Visual Script Template

## Visual Blueprint Logic (No Business Logic)

This shows the proper visual scripting approach where Unreal Engine handles ONLY visualization and user interaction events, with all logic remaining in the backend.

### Main Level Blueprint Events

#### Event BeginPlay
```
Event BeginPlay
├── Get WebSocket Manager Component
├── Connect to ECM Gateway
│   └── URL: "ws://localhost:8765"
├── Connect to Pulse Handler  
│   └── URL: "ws://localhost:8767"
└── Print String ("🎮 BEM Level Started - Visual Client Only")
```

#### Event On Pulse Received (From Backend)
```
Event On Pulse Received
├── Parse JSON (Pulse Command from Backend)
├── Get Pulse Type (String)
├── Get Position (Vector)
├── Get Intensity (Float)
├── Get Duration (Float)
├── Get Direction (String)
├── Find Pulse Animator at Location
└── Start Pulse Animation (Visual Only)
    ├── Pulse Type: From Backend
    ├── Position: From Backend  
    ├── Intensity: From Backend
    ├── Duration: From Backend
    └── Direction: From Backend
```

#### Event On Component Clicked (User Interaction)
```
Event On Component Clicked
├── Get Component ID
├── Get Component Position
├── Get Component Type
├── Create Spatial Event JSON
│   ├── Type: "spatial_event"
│   ├── Event Type: "component_selected"
│   ├── Component ID: From Component
│   ├── Position: From Component
│   └── Timestamp: Current Time
├── Send to ECM Gateway (Backend Processes Logic)
└── Set Component Highlight (Visual Feedback Only)
```

### Key Principles Demonstrated

1. **No Business Logic**: All logic decisions come from backend
2. **Visual Only**: Unreal only renders what backend commands
3. **Event Driven**: Responds to backend commands and user interactions
4. **State Sync**: Visual state always syncs with backend state

### What NOT to Include in Visual Scripts

❌ **Business Rules**: No cost calculations, material validations, etc.
❌ **State Management**: No storing of component properties or relationships
❌ **Decision Logic**: No if/then business decisions
❌ **Data Processing**: No transforming or analyzing data

✅ **Visual Rendering**: Particle effects, materials, animations
✅ **User Interaction**: Click events, movement, selection
✅ **Network Communication**: Send events, receive commands
✅ **UI Updates**: Status displays, notifications, feedback
