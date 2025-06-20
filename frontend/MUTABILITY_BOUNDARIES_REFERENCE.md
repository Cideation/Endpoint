# Complete Architecture File Structure Reference

## 📊 **Complete Architecture File Structure** [**[[memory:3352720096928090367]]**]

> **"Infrastructure is immutable; computation is emergent."**

| File/Component               | Role                           | Mutable?                  | Notes                                           |
| ---------------------------- | ------------------------------ | ------------------------- | ----------------------------------------------- |
| **ecm_gateway.py**           | WebSocket handler (fixed infra)| ❌ Immutable               | Audit-safe, stateless, fixed interface          |
| **pulse_router.py**          | Dispatches interaction signals | ✅ Mutable                 | Routes triggers to functors, evolves with logic |
| **fsm_runtime.py**           | Manages node FSM states        | ✅ Mutable (optional)      | Optional structured state management            |
| **agent_state.json**         | Runtime memory/cache           | ✅ Mutable                 | Holds runtime data, coefficients, cache        |
| **Unreal Engine**            | Environment visualization      | ❌ Stateless (visual only) | Visualizes graph state via pulses              |

## 🔒 **Immutable Infrastructure Layer**

### **ecm_gateway.py - WebSocket Handler**
- **Purpose:** Fixed infrastructure layer for message transport
- **Behavior:** Never changes, pure relay functionality
- **Responsibilities:**
  - Persistent WebSocket connections
  - Message logging and timestamping
  - Basic structure validation
  - Audit-safe message relay
- **Anti-patterns:** No interpretation, routing, or computation logic

### **Unreal Engine - Environment**
- **Purpose:** Environment that visualizes graph state via pulses
- **Behavior:** Stateless visual reflection only
- **Responsibilities:**
  - Display system state visually
  - Capture spatial interactions
  - Send events to ECM Gateway
  - Reflect computed results
- **Anti-patterns:** No computational logic or state management

## ✅ **Mutable Computation Layer**

### **pulse_router.py - Interaction Signal Dispatcher**
- **Purpose:** Dispatches interaction signals to functors
- **Behavior:** Adaptive routing logic that can evolve
- **Responsibilities:**
  - Interpret messages from ECM (post-delivery)
  - Route interaction signals to appropriate handlers
  - Manage event type handling
  - Interface with Node Engine operations
  - Evolve routing patterns as system needs change
- **Mutability:** Can register/unregister handlers, adapt routing logic

### **fsm_runtime.py - Node State Management (Optional)**
- **Purpose:** Manages node FSM states for structured behavior
- **Behavior:** Optional component for explicit state management
- **Responsibilities:**
  - Define state transition rules for graph nodes
  - Handle state-based behavior and transitions
  - Provide structured state management beyond functors
  - Integrate with pulse routing results
- **Mutability:** State transitions, rules, and handlers can evolve

### **agent_state.json - Runtime Memory/Cache**
- **Purpose:** Holds runtime memory/cache if needed
- **Behavior:** Persistent storage for runtime data
- **Responsibilities:**
  - Store agent coefficients (cosmetic AC, unreal AC)
  - Cache recent computations and interactions
  - Track session data and activity
  - Provide runtime configuration settings
- **Mutability:** All runtime data can be updated dynamically

## 🌊 **Complete Data Flow**

```
Input Sources (Unreal/UI)
         ↓
ecm_gateway.py (immutable relay)
         ↓
pulse_router.py (mutable interpretation & routing)
         ↓
fsm_runtime.py (optional state management)
         ↓
Node Engine (computation & functors)
         ↓
agent_state.json (cache & persistence)
         ↓
Environment (visual reflection)
```

## 🎯 **Integration Points**

### **ECM → Pulse Router**
```python
# ECM delivers message, Pulse Router interprets and routes
routing_result = await pulse_router.route_message(ecm_message)
```

### **Pulse Router → FSM Runtime**
```python
# Pulse routing results trigger FSM state transitions
await fsm_runtime.handle_pulse_routing_result(routing_result)
```

### **FSM Runtime → Agent State**
```python
# FSM state changes persist to agent state
state_snapshot = fsm_runtime.export_state_snapshot()
```

## 🔧 **Development Guidelines**

### **Modifying Infrastructure (Immutable)**
- ❌ Never add computation logic to `ecm_gateway.py`
- ❌ Never add state management to Unreal visualization
- ✅ Infrastructure changes require careful consideration
- ✅ Maintain audit trails and logging

### **Evolving Computation (Mutable)**
- ✅ Freely adapt routing logic in `pulse_router.py`
- ✅ Add new interaction types and handlers
- ✅ Evolve FSM transition rules as needed
- ✅ Update agent coefficients dynamically
- ✅ Expand cache and memory structures

### **Component Integration**
- Connect components through well-defined interfaces
- Maintain clear separation between immutable and mutable layers
- Use async patterns for scalable message processing
- Implement proper error handling and logging

This architecture ensures **robust infrastructure stability** with **unlimited computational evolution** potential! 🚀 