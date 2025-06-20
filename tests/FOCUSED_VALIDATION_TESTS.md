# 🎯 Focused Validation Tests

This document describes the 5 focused validation tests designed to improve code quality and system reliability.

## Overview

The Focused Validation Test Suite implements comprehensive testing across 5 critical areas:

1. **🔹 Smoke Test (Containers)** - Container orchestration validation
2. **🔹 Node Test** - Individual node functionality validation  
3. **🔹 Mini DAG Test** - Callback chain validation
4. **🔹 Persistence Test** - Database storage/retrieval validation
5. **🔹 Routing Test** - Inter-node communication validation

## Quick Start

### Run All Tests
```bash
python run_focused_tests.py
```

### Run Individual Test File
```bash
python test_focused_validation.py
```

## Test Details

### 🔹1. Smoke Test (Containers)

**Purpose**: Validate that all Docker containers start correctly and core services are operational.

**Actions**:
- Run `docker-compose up -d`
- Check all containers are in "running" state
- Hit `/healthz` and `/status` endpoints
- Scan logs for fatal errors (FATAL, CRITICAL, ERROR)

**Success Criteria**:
- At least one container running
- No fatal log patterns detected
- Health endpoints responding

### 🔹2. Node Test

**Purpose**: Validate individual node functionality with manufacturing evaluation.

**Actions**:
- Create test node: `V01_ProductComponent_TEST`
- Simulate `evaluate_manufacturing` process
- Validate `.dictionary()` output structure
- Assert expected keys are present

**Success Criteria**:
- Valid dictionary structure returned
- Required keys present: `node_id`, `functor_type`, `inputs`, `outputs`, `state`, `timestamp`
- No crashes during evaluation
- Manufacturing calculations completed

### 🔹3. Mini DAG Test Callbacks

**Purpose**: Validate functor execution in sequence with downstream value resolution.

**DAG Structure**: `V01_Component → V02_Assembly → V05_Quality`

**Success Criteria**:
- All nodes execute successfully
- Data flows correctly between nodes
- Final node produces valid outputs
- No broken callback chains

### 🔹4. Persistence Test

**Purpose**: Validate PostgreSQL storage and retrieval of node data.

**Actions**:
- Create test node with structured data
- Save node to PostgreSQL database
- Reload node from database
- Compare original vs reloaded dictionaries

**Success Criteria**:
- Node saves successfully to database
- Node reloads with identical data
- Dictionary comparison passes
- No data corruption or loss

### 🔹5. Routing Test

**Purpose**: Validate `route_to_node` functionality and inter-node communication.

**Actions**:
- Create source node (V03_Router_Source) with routing data
- Create target node (V04_Target) to receive data
- Trigger `route_to_node` from source to target
- Verify target receives inputs and executes

**Success Criteria**:
- Routing mechanism successfully transfers data
- Target node receives expected inputs
- Target node executes and produces outputs
- Communication chain completes

## Dependencies

Required Python packages:
- `docker` - Container management
- `psycopg2-binary` - PostgreSQL connectivity
- `requests` - HTTP endpoint testing

Install with:
```bash
pip install docker psycopg2-binary requests
```

## Test Reports

Each test run generates a timestamped JSON report: `focused_test_report_YYYYMMDD_HHMMSS.json`

## Performance Benchmarks

Expected test execution times:
- **Smoke Test**: 15-30 seconds (container startup)
- **Node Test**: 1-2 seconds (computation)
- **DAG Test**: 2-3 seconds (sequential execution)
- **Persistence Test**: 3-5 seconds (database I/O)
- **Routing Test**: 1-2 seconds (data transfer)

**Total Suite**: ~25-45 seconds
