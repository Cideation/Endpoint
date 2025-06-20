# Phase 2 Microservice Architecture

## Overview
This phase implements a deterministic, modular microservice architecture for CAD parsing and graph-based computation. All logic is deterministic and structured, with no LLM/AI except for a future, isolated SFDE module.

## Containers (5 Total)

1. **ne-dag-alpha**
   - DAG logic and orchestration
2. **ne-functor-types**
   - Functor type logic and computation
3. **ne-callback-engine**
   - Callback/event-driven computation
4. **sfde**
   - System Functor Data Engine (deterministic for now; LLM/AI in future)
5. **ne-graph-runtime-engine**
   - Dedicated NetworkX graph engine for runtime graph building, execution, and future DGL integration

## Shared Code
- All shared logic is in the `shared/` package, mounted into each container.
- Each microservice imports shared modules using `from shared import ...`.

## Docker & Compose
- Each service has its own Dockerfile.
- `docker-compose.yml` orchestrates all services, mounting shared code and data as needed.

## How to Run
1. Build and start all services:
   ```bash
   docker compose build --no-cache
   docker compose up --abort-on-container-exit
   ```
2. Debug output in each service will confirm shared imports and graph logic are working.

## Key Principles
- Deterministic, testable, and maintainable
- All shared code is centralized
- Containers are isolated but share code/data via Docker volumes

---

For more details on each container, see their respective folders and documentation.

# MICROSERVICE ENGINES (Phase 2 Runtime) – Final Revised with JSON-Driven Logic

This version reflects the full container architecture **with JSON-level integration insights** including: formula sourcing, coefficient logic, callback modes, and runtime graph interplay — ensuring a structured execution model across all functor types and node phases.

---

## ⚙️ Microservice Container Overview

| Container Name        | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `ne-dag-alpha`        | Executes Alpha-phase deterministic functors (DAG-only).                 |
| `ne-functor-types`    | Handles stateless logic: spatial, aggregation, and local calculations.   |
| `ne-callback-engine`  | Executes Beta- (relational) & Gamma- (combinatorial) callbacks.          |
| `sfde-engine`         | Discovers and injects scientific formulas using symbolic reasoning (SymPy).|
| `api-gateway`         | Exposes HTTP APIs for querying microservice functions and SFDE.         |

---

## 📦 File Dependencies – Updated with JSON Schema Insights

| File Name                            | Description                                                          |
|-------------------------------------|----------------------------------------------------------------------|
| `formula_dictionary.json`           | Source of `evaluatable_code` + unit/scientific domain per functor   |
| `functor_registry.json`             | Defines functor types, data affinity, phase bindings                |
| `global_variables.json`             | Constants (gravity, density, max budget, etc.)                      |
| `unified_functor_variable_edges.json` | Core source of all edge construction based on variable pass-through |
| `agent_coefficients_by_node.json`   | ACs now labeled **by Node ID**, not just by Agent Class             |
| `callback_registry.json`            | Maps `callback_type` (DAG, RELATIONAL, COMBINATORIAL) by phase      |
| `functor_deep_edges_fixed.json`     | Formula-seeking mappings between functors using shared data fields  |
| `formula_dictionary_runtime.json`   | Stores dynamic output of `sfde-engine`, injects real formulas       |

---

## 🧠 Functor Type Map (Cross-Phase)

Defined in: `functor_registry.json`

| Functor Type       | Data Affinity     | Used In Phase(s) | Sample Computations                  |
|--------------------|-------------------|------------------|---------------------------------------|
| Stateless Spatial  | Spatial Topology  | Alpha–Gamma      | Zone mapping, distance thresholds     |
| Local Calcs        | Deterministic     | Alpha            | Manufacturing cost, profile fit      |
| Aggregator         | Relational        | Beta–Gamma       | ROI, market matching, design priority |
| Local Calcs        | Deterministic     | Alpha            | Manufacturing score, cost/unit        |
| Aggregator         | Relational        | Beta, Gamma      | ROI aggregation, user-market fit      |

---

## 🔁 Graph Construction

All containers use `network_graph.py` to build a **full runtime NetworkX graph**:
- Loads nodes from `/inputs/*.json`
- Loads edges from `callback_registry.json` OR `unified_functor_variable_edges.json`
- All nodes labeled with `callback_type`, `functor_type`, `agent_coefficients`

---



## ✅ Node Edge Labeling Convention

Edges now follow updated JSON-based types:

| Key              | Value Description                  |
|------------------|------------------------------------|
| `callback_type`  | DAG, RELATIONAL, COMBINATORIAL     |
| `functor_type`   | Stateless Spatial, Aggregator, etc |
| `edge_type`      | derived from `unified_functor_variable_edges.json` |
| `phase`          | Alpha, Beta, Gamma                 |

---

## 📦 Directory Layout

```
/docker-compose.yml
/inputs/                   -> Node dictionaries
/shared/                   -> Configuration & variable files
/pulse/                    -> Emergence directives for callbacks
/ne-dag-alpha/             -> Alpha-phase DAG functors
/ne-functor-types/         -> Stateless functor logic
/ne-callback-engine/       -> Beta/Gamma callback execution
/sfde-engine/              -> Scientific formula discovery
/api-gateway/              -> HTTP API interface to microservices
```
/inputs/                   -> Node dictionaries
/shared/                   -> All config + variable files
/pulse/                    -> Emergence graph overlays
/ne-dag-alpha/             -> Phase A logic
/ne-callback-engine/       -> Phases B + G routing
/sfde-engine/              -> Formula discovery module
/docker-compose.yml
```

---

## ✅ Summary

- JSON architecture fully governs edge logic and formula behavior
- SFDE gives fallback AI + symbolic reasoning for unpopulated functors
- ACs are now node-specific and injected dynamically
- All containers support combined dictionary-driven logic

This README absorbs all recent file updates and is safe to use as the base reference for Phase 2 container orchestration.

