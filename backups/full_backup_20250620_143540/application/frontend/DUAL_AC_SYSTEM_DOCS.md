# Dual Agent Coefficient System

## Architecture Overview

The Dual Agent Coefficient (AC) System implements a sophisticated **1-way compute architecture** that provides a **2-way interactive experience** through two distinct input groups feeding into a unified Node Engine.

### AC Groups

#### 1. Cosmetic AC Group
**Source**: Structured UI inputs (sliders, forms, dropdowns)  
**Purpose**: Intentional user preferences and parameters  
**Examples**:
- Budget levels (0-100%)
- Quality preferences (0-100%)
- Timeline urgency (0-100%)
- Component type priorities
- Investment levels
- Agent registration forms

#### 2. Unreal AC Group  
**Source**: Spatial actions in 3D environment  
**Purpose**: Behavioral triggers and spatial interactions  
**Examples**:
- Wall/component selections → location zones, geometry sizes
- Room placement → areas, positions, accuracy metrics
- Zone interactions → interaction counts, pressure levels
- Geometry measurements → sizes, precision, complexity scores
- Component analysis → volume, surface area, complexity ratings

### Unified Node Engine

Both AC groups feed into the same central Node Engine that:
- Processes coefficients through unified computation
- Maps coefficients to relevant agent classes (1-10)
- Applies controlled emergence to both environments
- Maintains real-time metrics and processing queues
- Routes events through single computational pipeline

### Agent Classes (1-10)

1. **Product Agent** - Product specifications and features
2. **User Agent** - User interactions and preferences  
3. **Spatial Agent** - 3D environment and positioning
4. **Material Agent** - Material properties and selection
5. **Structural Agent** - Structural engineering constraints
6. **MEP Agent** - Mechanical, electrical, plumbing systems
7. **Cost Agent** - Budget optimization and cost analysis
8. **Time Agent** - Timeline management and scheduling
9. **Quality Agent** - Quality standards and assurance
10. **Integration Agent** - System integration and coordination

### Data Flow Architecture

```
Cosmetic UI (sliders, forms) → Cosmetic AC Group
                                     ↓
3D Spatial Environment → Unreal AC Group → Unified Node Engine → Controlled Emergence
                                     ↓                              ↓
Both environments receive emergence feedback ←─────────────────────┘
```

**Key Insight**: Even though users interact with both structured forms and spatial 3D environment, **all inputs flow through the same computational engine**, creating the appearance of bidirectional interaction while maintaining unidirectional compute flow.

### Core Features

#### 1-Way Compute Architecture
- **Single computational pipeline** for all inputs
- **Unified event processing** regardless of source
- **Controlled emergence** applied consistently
- **Clean separation** between input and computation

#### 2-Way Interactive Feel
- **Cosmetic UI** feels responsive with real-time feedback
- **Spatial environment** provides immediate visual responses
- **Cross-environment synchronization** via WebSocket updates
- **Agent activation** creates dynamic system behavior

#### Dual AC Processing
- **Cosmetic coefficients** from structured interactions
- **Unreal coefficients** from spatial actions
- **Real-time coefficient tracking** and visualization
- **Agent mapping** based on coefficient types

### Technical Implementation

#### Frontend Technologies
- **Alpine.js** - Reactive data binding and state management
- **Three.js** - 3D spatial environment rendering
- **Tailwind CSS** - Modern UI styling and responsiveness
- **WebSocket** - Real-time bidirectional communication

#### Backend Architecture
- **FastAPI** - High-performance API server
- **Python** - Core business logic and AC processing
- **WebSocket Server** - Real-time event broadcasting
- **Pydantic** - Data validation and serialization

#### Database Integration
- **Neon PostgreSQL** - Component and project data
- **Real-time queries** - Live component information
- **Schema integration** - Consistent data structures

### AC Processing Pipeline

#### Cosmetic AC Flow
1. **User Input** → Slider/form interaction
2. **Coefficient Generation** → Structured data creation
3. **Engine Routing** → Unified processing pipeline
4. **Agent Activation** → Relevant agents receive coefficients
5. **Emergence Application** → Environmental feedback

#### Unreal AC Flow
1. **Spatial Action** → 3D environment interaction
2. **Coefficient Generation** → Behavioral data creation
3. **Engine Routing** → Same unified processing pipeline
4. **Agent Activation** → Spatial and related agents activated
5. **Emergence Application** → Both UI and 3D feedback

### Real-Time Features

#### Live Coefficient Tracking
- **Active Cosmetic AC** - Current structured parameters
- **Active Unreal AC** - Current spatial interactions
- **Cross-reference display** - Coefficients mapped to agents

#### Processing Queue
- **Event prioritization** - Critical coefficients first
- **Progress tracking** - Visual processing indicators
- **Type identification** - Cosmetic vs Unreal AC events

#### Engine Metrics
- **CPU utilization** - Based on AC processing load
- **Coefficient count** - Total active coefficients
- **Events per second** - Real-time processing rate
- **Agent activity** - Live agent status updates

### Usage Patterns

#### Structured Workflow (Cosmetic AC)
1. Register new agent with specific role
2. Set budget, quality, timeline preferences
3. Choose component type priorities
4. Send bulk coefficients to engine
5. Monitor agent activation and processing

#### Spatial Workflow (Unreal AC)
1. Interact with 3D components directly
2. Perform spatial actions (select, place, measure)
3. Generate location-based coefficients
4. Trigger automatic spatial agent activation
5. Observe visual emergence in environment

#### Hybrid Workflow (Both AC Types)
1. Set initial preferences via Cosmetic UI
2. Refine through spatial interactions
3. Monitor unified processing in Node Engine
4. Observe emergence in both environments
5. Iterate through combined AC generation

### Advanced Features

#### Coefficient Mapping
- **Automatic agent assignment** based on coefficient types
- **Multi-agent activation** for complex coefficients
- **Coefficient inheritance** between related agents
- **Priority-based processing** for conflicting coefficients

#### Emergence Patterns
- **Environmental feedback** - Visual changes in 3D space
- **UI updates** - Dynamic interface modifications
- **Agent communication** - Cross-agent coefficient sharing
- **System optimization** - Automatic parameter tuning

#### WebSocket Events
- **Real-time AC updates** - Live coefficient changes
- **Processing notifications** - Event completion status
- **Agent state changes** - Activation/deactivation events
- **Emergence broadcasts** - System-wide state updates

### Development Workflow

#### Local Setup
```bash
# Start the dual AC system
python start_dual_ac.py

# Access interface
open http://localhost:8002

# Monitor real-time updates
# WebSocket: ws://localhost:8002/ws
```

#### Testing Scenarios
1. **Cosmetic AC Testing**
   - Register multiple agents
   - Adjust sliders while monitoring coefficients
   - Send bulk updates and observe processing

2. **Unreal AC Testing**
   - Click 3D components
   - Perform spatial actions
   - Monitor coefficient generation

3. **Integration Testing**
   - Use both AC types simultaneously
   - Verify unified processing
   - Confirm cross-environment emergence

### Future Enhancements

#### Advanced AC Types
- **Temporal coefficients** - Time-based interactions
- **Collaborative coefficients** - Multi-user interactions
- **AI-generated coefficients** - Machine learning integration

#### Enhanced Emergence
- **Predictive modeling** - Anticipatory system responses
- **Learning algorithms** - Adaptive coefficient processing
- **Optimization engines** - Automatic system tuning

#### Extended Integrations
- **External APIs** - Third-party coefficient sources
- **IoT sensors** - Physical environment data
- **VR/AR interfaces** - Immersive spatial interactions

This architecture demonstrates how sophisticated interactive systems can maintain clean computational principles while providing rich, responsive user experiences through multiple input modalities.
