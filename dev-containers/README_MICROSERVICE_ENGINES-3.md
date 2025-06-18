# MICROSERVICE ENGINES

A containerized microservice architecture for executing system functors and managing emergence logic across a multi-phase SoS model using structured dictionaries, callback routing, and a required runtime NetworkX graph environment.

---

## 🧱 Containers Overview

### 1. `ne-dag-alpha`
Executes all **Alpha-phase DAG functors**:
- Deterministic, node-directed logic
- Examples: `evaluate_manufacturing`, `convert_to_geometry`, `evaluate_site_fit`
- Uses node dictionaries and the runtime graph

### 2. `ne-functor-types`
Handles **stateless functor types**:
- Includes: spatial logic, aggregation, and local calculations
- Still builds and uses the full graph internally (e.g. spatial reach, zone context)

### 3. `ne-callback-engine`
Executes all **emergence-qualified callbacks** across phases:
- **Beta-phase** (relational logic)
- **Gamma-phase** (combinatorial logic)
- Fully integrates system-wide Pulse logic
- Uses NetworkX to route and interpret combinatorial dynamics

---

## 🧠 Shared Runtime Environment

All containers load the following **shared environmental data** at runtime:

| Type                        | Source Path        | Purpose                                                  |
|-----------------------------|--------------------|----------------------------------------------------------|
| **Node Dictionaries**       | `/inputs/*.json`   | Per-node data (phase, agent, geometry, coefficients)     |
| **Agent Coefficients**      | `/shared/agent_coefficients.json` | System weights per agent            |
| **Global Variables**        | `/shared/global_variables.json`   | Limits, constants, simulation bounds |
| **Enriched Global Dictionary** | `/shared/enriched_global_dictionary.json` | Computed output snapshot       |
| **Callback Registry**       | `/shared/callback_registry.json` | Controls valid callback transitions |
| **Pulse Graph Directives**  | `/pulse/pulse_graph_directives.json` | Emergence routing for callback phase |
| **Phase Summary**           | `/outputs/phase_summary.json`     | Tracks DAG and callback stage completions |

---

## 🔁 NetworkX Graph Integration

All containers **construct the full graph** at runtime using `/shared/network_graph.py`, which loads:
- All nodes from `/inputs/`
- All relationships (edges) from `callback_registry.json` or embedded fields
- Optionally weighted edges from Pulse

The graph is used in:
- DAG traversal
- Callback routing
- Aggregation, centrality scoring, pathfinding
- Emergence validation

---

## 📂 Directory Structure

```
MICROSERVICE ENGINES/
├── docker-compose.yml
├── inputs/                   # Node dictionaries
├── outputs/                  # Functor and phase results
├── shared/                   # Coefficients, globals, graph builder
│   └── network_graph.py      # REQUIRED graph loader for all containers
├── pulse/                    # Emergence directives (used by callbacks)
├── ne-dag-alpha/             # Alpha DAG functors
├── ne-functor-types/         # Stateless logic
└── ne-callback-engine/       # ABG callback execution
```

---

## 🔧 Run Commands

To build and run all containers:
```bash
docker compose up --build
```

To run a specific container:
```bash
docker compose run --rm ne-callback-engine
```

---