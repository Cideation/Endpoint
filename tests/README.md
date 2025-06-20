# 🧪 BEM System Test Suite

This directory contains all testing infrastructure for the Building Environment Management (BEM) system.

## 📁 Test Organization

### 🎯 Focused Validation Tests
- **`test_focused_validation.py`** - 5 core validation tests (containers, nodes, DAG, persistence, routing)
- **`run_focused_tests.py`** - Simple test runner for focused validation
- **`FOCUSED_VALIDATION_TESTS.md`** - Detailed documentation

### 🧠 Advanced Graph Tests
- **`test_full_graph_pass.py`** - Complete DAG execution across all phases (Alpha → Beta → Gamma)
- **`test_edge_callback_logic.py`** - Edge direction, metadata, and callback accuracy validation
- **`test_emergent_values.py`** - Cross-phase output convergence (ROI, compliance score, etc.)
- **`test_agent_impact.py`** - Agent coefficients influence on node states
- **`test_trace_path_index.py`** - Final output trace to source inputs (component ID path)

### 🚀 System Integration Tests
- **`test_complete_system_integration.py`** - End-to-end system testing
- **`quick_system_test.py`** - Rapid system validation
- **`run_system_tests.py`** - System test orchestrator
- **`setup_test_environment.py`** - Test environment configuration

### 🔒 Security & Performance Tests
- **`test_deployment_security.py`** - Security validation for deployments
- **`test_performance_load.py`** - Load testing and performance validation

### 📋 Component Tests
- **`test_agent_console.py`** - Agent console functionality
- **`test_behavior_driven_ac.py`** - Behavior-driven acceptance criteria
- **`test_bem_system.py`** - Core BEM system components

### 🏗️ Phase-Specific Tests
- **`test_phase1_implementation.py`** - Phase 1 implementation validation
- **`test_phase2_integration.py`** - Phase 2 integration testing
- **`test_phase2_container_orchestration.py`** - Container orchestration tests
- **`test_phase3_production.py`** - Phase 3 production readiness
- **`test_phase3_quick_retest.py`** - Quick Phase 3 validation

### 🐳 Docker Test Configuration
- **`Dockerfile.test`** - Test environment container
- **`docker-compose.test.yml`** - Isolated testing environment
- **`docker-compose.deploy-test.yml`** - Deployment testing configuration

## 🚀 Quick Start

### Run All Focused Tests
```bash
cd tests
python run_focused_tests.py
```

### Run Specific Test Categories
```bash
# Focused validation (recommended)
python test_focused_validation.py

# Complete system integration
python test_complete_system_integration.py

# Quick system validation
python quick_system_test.py
```

## 📊 Test Reports

All tests generate detailed reports:
- **JSON reports**: `*_test_report_YYYYMMDD_HHMMSS.json`
- **Log files**: Detailed execution logs
- **Performance metrics**: Timing and resource usage

## 🛠️ Dependencies

Required packages (see `../requirements.txt`):
- `pytest` - Testing framework
- `docker` - Container management
- `psycopg2-binary` - Database connectivity
- `requests` - HTTP testing

## 📈 Performance Expectations

| Test Category | Duration | Coverage |
|---------------|----------|----------|
| Focused Validation | 25-45s | Core functionality |
| System Integration | 2-5min | End-to-end flows |
| Performance Load | 5-15min | Stress testing |
| Security Testing | 1-3min | Security validation |

# Test Suite Documentation

This directory contains comprehensive test suites for validating the Endpoint-1 system functionality, organized into focused validation tests and advanced graph testing layers.

## Test Architecture Overview

The test suite is designed around a **three-phase architecture** with distinct edge flow patterns:

### Phase-Specific Edge Architecture
- **✅ Alpha Phase**: DAG (Directed Acyclic Graph) - directed, one-to-one or linear edge flow
- **✅ Beta Phase**: Relational (Objective Functions) - many-to-many, dense logic patterns  
- **✅ Gamma Phase**: Combinatorial (Emergence) - many-to-many, sparse-to-dense mappings

### Edge Table Segregation
| Phase | Edge Type | Storage Table | Description |
|-------|-----------|---------------|-------------|
| Alpha | Directed DAG | `alpha_edges.csv` | Static logic flow between nodes |
| Beta | Many-to-Many | `beta_relationships.csv` | Objective Function relations |
| Gamma | Combinatorial Net | `gamma_edges.csv` | Learning-based, emergent dependencies |

## Test Categories

### 1. Focused Validation Tests (Basic Functionality)
Located in root test files, these provide essential system validation:

- **`test_focused_validation.py`** - 5 core validation scenarios
- **`run_focused_tests.py`** - Simple test runner
- **Expected execution time**: 25-45 seconds

#### Test Scenarios:
1. **Smoke Test (Containers)** - Docker services health check
2. **Node Test** - V01_ProductComponent manufacturing evaluation
3. **Mini DAG Test** - V01 → V02 → V05 functor sequence
4. **Persistence Test** - PostgreSQL round-trip validation
5. **Routing Test** - Inter-node routing simulation

### 2. Advanced Graph Testing (Production-Level Validation)
Located in individual test files, these provide comprehensive graph behavior testing:

#### Core Advanced Tests:
- **`test_full_graph_pass.py`** - Complete DAG execution across Alpha → Beta → Gamma phases
- **`test_edge_callback_logic.py`** - Phase-specific edge validation and callback accuracy
- **`test_emergent_values.py`** - Cross-phase convergence and sparse-to-dense emergence
- **`test_agent_impact.py`** - Agent coefficient influence and multi-agent conflict resolution
- **`test_trace_path_index.py`** - Complete traceability and data provenance validation

#### Test Runner:
- **`test_runner_advanced.py`** - Executes all 5 advanced tests with comprehensive reporting

### 3. Phase-Specific Test Details

#### Alpha Phase Testing (DAG Foundation)
- **Edge Type**: Directed, linear flow
- **Data Flow**: Material specifications → Requirements → Design constraints
- **Validation**: DAG properties, linear propagation, single-source/single-target edges
- **Storage**: `alpha_edges.csv`

#### Beta Phase Testing (Objective Functions)
- **Edge Type**: Many-to-many relational
- **Data Flow**: Cost/Quality/Performance objectives → Optimization → Tradeoff analysis
- **Validation**: Dense logic patterns, multi-objective convergence, ROI calculations
- **Storage**: `beta_relationships.csv`

#### Gamma Phase Testing (Emergent Synthesis)
- **Edge Type**: Combinatorial emergence
- **Data Flow**: Sparse inputs → Dense emergent properties → System behaviors
- **Validation**: Learning weights, emergence thresholds, sparse-to-dense mappings
- **Storage**: `gamma_edges.csv`

## Test Execution

### Quick Start
```bash
# Run focused validation tests
python run_focused_tests.py

# Run advanced graph tests
python test_runner_advanced.py

# Run individual test
python test_full_graph_pass.py
```

### Expected Results
- **Focused Tests**: 5/5 scenarios passing, ~30 seconds execution
- **Advanced Tests**: 5/5 graph layers passing, ~60-90 seconds execution
- **JSON Reports**: Timestamped detailed results for each test suite

## Test Infrastructure

### Dependencies
- **Docker**: Container health checking
- **PostgreSQL**: Database persistence testing
- **WebSocket**: Real-time communication testing
- **JSON**: Configuration and result reporting

### Configuration
- **Health Endpoints**: `/healthz`, `/status` on ports 8000, 8001
- **Database**: Environment variable support for connection strings
- **Logging**: Structured logging with timestamps and levels
- **Error Handling**: Graceful failure with detailed error reporting

## Reporting and Analysis

### Report Generation
Each test suite generates timestamped JSON reports containing:
- **Execution metrics**: Duration, success rates, performance benchmarks
- **Phase analysis**: Alpha/Beta/Gamma transition validation
- **Edge validation**: Direction, metadata, callback accuracy
- **Emergence tracking**: Cross-phase synthesis and learning adaptation
- **Traceability**: Complete audit trails and data provenance

### Report Locations
- **Focused Tests**: `focused_validation_report_YYYYMMDD_HHMMSS.json`
- **Advanced Tests**: `[test_name]_report_YYYYMMDD_HHMMSS.json`
- **Combined Results**: Available through test runners

## Integration with CI/CD

The test suite is designed for continuous integration:
- **Exit Codes**: Non-zero exit codes for failed tests
- **Parallel Execution**: Tests can run concurrently
- **Environment Agnostic**: Docker-based infrastructure testing
- **Automated Reporting**: JSON output for CI/CD pipeline integration

## Troubleshooting

### Common Issues
1. **Docker Connection**: Ensure Docker daemon is running
2. **PostgreSQL Access**: Check database connection environment variables
3. **Port Conflicts**: Verify ports 8000, 8001 are available
4. **Memory Usage**: Advanced tests may require 2GB+ available memory

### Debug Mode
Enable detailed logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
python test_full_graph_pass.py
```

## Development Guidelines

### Adding New Tests
1. Follow phase-specific architecture (Alpha/Beta/Gamma)
2. Include edge table segregation validation
3. Generate timestamped JSON reports
4. Add comprehensive error handling
5. Update this README with new test descriptions

### Test Naming Convention
- **Focused Tests**: `test_focused_[functionality].py`
- **Advanced Tests**: `test_[graph_layer]_[validation_type].py`
- **Utility Scripts**: `run_[test_category]_tests.py`

This test suite provides comprehensive validation of the Endpoint-1 system's graph-based architecture, ensuring robust functionality across all phases of the Alpha → Beta → Gamma pipeline with proper edge segregation and emergent behavior validation.
