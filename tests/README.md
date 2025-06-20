# 🧪 BEM System Test Suite

This directory contains all testing infrastructure for the Building Environment Management (BEM) system.

## 📁 Test Organization

### 🎯 Focused Validation Tests
- **`test_focused_validation.py`** - 5 core validation tests (containers, nodes, DAG, persistence, routing)
- **`run_focused_tests.py`** - Simple test runner for focused validation
- **`FOCUSED_VALIDATION_TESTS.md`** - Detailed documentation

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
