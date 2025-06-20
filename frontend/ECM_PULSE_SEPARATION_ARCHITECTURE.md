# ECM-Pulse Separation Architecture

## 🎯 **Critical Architectural Understanding**

**✅ Pulse is NOT inside ECM — but triggered AFTER ECM delivers**

ECM and Pulse are completely separate layers with distinct responsibilities in the message flow pipeline.

## 🧩 **Stage 1: Input Stage (ECM Layer)**

### **ECM Infrastructure Gateway (Port 8765)**
```
Unreal/UI → WebSocket → ECM receives → ECM processes → ECM passes downstream
```

**ECM Responsibilities:**
- ✅ **Pure relay** - No interpretation or computation
- ✅ **Persistent** - Maintain stable WebSocket connections  
- ✅ **Never executes a pulse** - Zero pulse logic
- ✅ **No decisions** - No routing or behavioral logic
- 📝 **Log & validate structure** - Basic message integrity only
- 🕐 **Timestamp** - Infrastructure-level timestamping
- 📋 **Audit trail** - Full message logging for compliance

**ECM Does NOT:**
- ❌ Interpret message content
- ❌ Trigger any functors
- ❌ Make routing decisions
- ❌ Execute pulse logic
- ❌ Modify system state
- ❌ Compute responses

## 🔁 **Stage 2: Dispatch Stage (Pulse Layer)**

### **Node Engine / Pulse Handler (Post-ECM)**
```
ECM delivers → Node Engine receives → Pulse Handler interprets → Actions triggered
```

**Pulse Layer Responsibilities:**
- ✅ **Decides graph edge/node impact** - Interprets spatial/UI messages
- ✅ **Triggers functor/state updates** - Executes computational logic
- ✅ **Records system impact** - Updates graph state
- ✅ **Visual feedback to environment** - Responses back to UI/Unreal

**Message Type Handling:**
```javascript
// After ECM delivery, Pulse Handler checks:
if (message.type === "pulse_trigger") {
    // Trigger specific graph node
}
if (message.type === "event_signal") {
    // Update system state
}
if (message.type === "interaction") {
    // Process spatial interaction
}
```

## 🌐 **Complete Message Flow**

```
1. Input Source (Unreal/UI)
   ↓
2. ECM Infrastructure Gateway 
   ├── Log message
   ├── Validate structure
   ├── Timestamp
   └── Relay downstream
   ↓
3. Node Engine / Pulse Handler
   ├── Interpret content
   ├── Decide impact
   ├── Trigger functors
   ├── Update state
   └── Generate feedback
   ↓
4. Response back to source
```

## 💓 **ECM Pulse System (Segregated)**

**Separate Infrastructure (Port 8766):**
- Independent of ECM Gateway
- Handles rhythmic/coordination pulses
- Not part of message interpretation flow
- Pure infrastructure heartbeat system

## 🔒 **Separation Benefits**

### **Infrastructure Stability:**
- ECM can be deployed/scaled independently
- No computational load on infrastructure layer
- Audit-safe message relay
- Container-ready deployment

### **Computational Flexibility:**
- Pulse layer can evolve without affecting ECM
- Complex functor logic isolated from infrastructure
- Easy to test message flow vs. computational logic
- Clear debugging boundaries

## 🎯 **Implementation Verification**

**ECM Gateway (ecm_infrastructure_gateway.py):**
```python
# ✅ CORRECT: Pure relay
outbound = {
    "timestamp": timestamp,
    "status": "received",
    "echo": message.get("payload", {}),
    "type": message.get("type", "unknown")
}

# ❌ NEVER in ECM:
# if message.type == "pulse_trigger":
#     execute_functor()  # This belongs in Pulse Layer!
```

**Node Engine / Pulse Handler:**
```python
# ✅ CORRECT: Post-ECM interpretation
def handle_ecm_delivery(message):
    if message.type == "pulse_trigger":
        trigger_graph_node(message.target)
    elif message.type == "interaction":
        update_spatial_state(message.data)
```

## 📋 **Architectural Compliance Checklist**

- ✅ ECM never executes pulses
- ✅ ECM never makes decisions  
- ✅ ECM only validates basic structure
- ✅ Pulse layer handles all interpretation
- ✅ Clear separation of infrastructure vs. computation
- ✅ Audit trail maintained at infrastructure level
- ✅ Computational complexity isolated to appropriate layer

This separation ensures **robust, scalable, audit-safe architecture** where infrastructure remains stable while computational logic can evolve independently. 