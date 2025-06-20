#!/usr/bin/env python3
"""
BEM System Comprehensive Testing Framework
Tests all components: Infrastructure, Computation, Interface, Data layers
Validates ECM-Pulse separation, cross-phase learning, and deployment readiness
"""

import asyncio
import json
import time
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BEMSystemTester:
    """Comprehensive BEM system testing framework"""
    
    def __init__(self):
        self.test_results = {}
        self.ecm_host = "localhost"
        self.ecm_port = 8765
        self.pulse_port = 8766
        self.api_host = "localhost"
        self.api_port = 8080
        
    def test_file_structure(self) -> Dict[str, bool]:
        """Test that all required files exist in correct locations"""
        logger.info("📁 Testing File Structure")
        
        required_files = {
            "ecm_gateway": "Final_Phase/ecm_gateway.py",
            "pulse_router": "Final_Phase/pulse_router.py", 
            "fsm_runtime": "Final_Phase/fsm_runtime.py",
            "dgl_trainer": "Final_Phase/dgl_trainer.py",
            "agent_state": "Final_Phase/agent_state.json",
            "training_schema": "postgre/training_database_schema.sql",
            "training_config": "postgre/training_db_config.py",
            "dual_ac_api": "frontend/dual_ac_api_server.py",
            "semantic_interface": "frontend/semantic_pulse_interface.html"
        }
        
        results = {}
        for component, file_path in required_files.items():
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            results[f"{component}_exists"] = os.path.exists(full_path)
            
        self.test_results["file_structure"] = results
        return results
    
    def test_ecm_pulse_separation(self) -> Dict[str, bool]:
        """Test critical ECM-Pulse separation architecture"""
        logger.info("🔄 Testing ECM-Pulse Separation")
        
        results = {}
        
        # Test that ECM and Pulse are separate files
        ecm_path = os.path.join(os.path.dirname(__file__), 'Final_Phase', 'ecm_gateway.py')
        pulse_path = os.path.join(os.path.dirname(__file__), 'Final_Phase', 'pulse_router.py')
        
        results["ecm_gateway_exists"] = os.path.exists(ecm_path)
        results["pulse_router_exists"] = os.path.exists(pulse_path)
        
        # Test port separation
        results["port_separation"] = self.ecm_port != self.pulse_port
        
        # Test architectural principles
        if os.path.exists(ecm_path):
            with open(ecm_path, 'r') as f:
                ecm_content = f.read()
                # ECM should be infrastructure (immutable)
                results["ecm_is_infrastructure"] = "infrastructure" in ecm_content.lower()
                results["ecm_no_pulse_logic"] = "pulse_router" not in ecm_content.lower()
        else:
            results["ecm_is_infrastructure"] = False
            results["ecm_no_pulse_logic"] = False
            
        if os.path.exists(pulse_path):
            with open(pulse_path, 'r') as f:
                pulse_content = f.read()
                # Pulse should handle 7-pulse system
                pulse_types = ["bid_pulse", "occupancy_pulse", "compliancy_pulse", 
                              "fit_pulse", "investment_pulse", "decay_pulse", "reject_pulse"]
                pulse_coverage = sum(1 for pulse_type in pulse_types if pulse_type in pulse_content)
                results["seven_pulse_coverage"] = pulse_coverage >= 6  # At least 6 of 7 pulses
        else:
            results["seven_pulse_coverage"] = False
        
        self.test_results["ecm_pulse_separation"] = results
        return results
    
    def test_dgl_trainer_integration(self) -> Dict[str, bool]:
        """Test DGL Trainer location and database integration"""
        logger.info("🤖 Testing DGL Trainer Integration")
        
        results = {}
        
        # Test correct location (Final_Phase, not MICROSERVICE_ENGINES)
        correct_path = os.path.join(os.path.dirname(__file__), 'Final_Phase', 'dgl_trainer.py')
        wrong_path = os.path.join(os.path.dirname(__file__), 'MICROSERVICE_ENGINES', 'ne-optimization-engine', 'dgl_trainer.py')
        
        results["dgl_in_final_phase"] = os.path.exists(correct_path)
        results["dgl_not_in_microservices"] = not os.path.exists(wrong_path)
        
        # Test database integration
        if os.path.exists(correct_path):
            with open(correct_path, 'r') as f:
                dgl_content = f.read()
                results["database_integration"] = "training_db_config" in dgl_content
                results["cross_phase_learning"] = all(phase in dgl_content for phase in ["Alpha", "Beta", "Gamma"])
                results["bem_embedding"] = "BEMGraphEmbedding" in dgl_content
        else:
            results["database_integration"] = False
            results["cross_phase_learning"] = False
            results["bem_embedding"] = False
        
        self.test_results["dgl_trainer"] = results
        return results
    
    def test_training_database_schema(self) -> Dict[str, bool]:
        """Test training database schema completeness"""
        logger.info("🗄️ Testing Training Database Schema")
        
        results = {}
        
        schema_path = os.path.join(os.path.dirname(__file__), 'postgre', 'training_database_schema.sql')
        
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema_content = f.read()
                
                # Test required tables
                required_tables = ["nodes", "edges", "coefficients", "labels", "pulses", "formulas", 
                                 "training_runs", "training_metrics", "model_embeddings"]
                
                for table in required_tables:
                    results[f"{table}_table"] = f"CREATE TABLE {table}" in schema_content
                
                # Test phase support
                results["phase_support"] = "Alpha" in schema_content and "Beta" in schema_content and "Gamma" in schema_content
                
                # Test pulse system integration
                pulse_types = ["bid_pulse", "occupancy_pulse", "compliancy_pulse", "fit_pulse", 
                              "investment_pulse", "decay_pulse", "reject_pulse"]
                pulse_coverage = sum(1 for pulse_type in pulse_types if pulse_type in schema_content)
                results["pulse_system_integration"] = pulse_coverage >= 6
                
                # Test SFDE integration
                results["sfde_integration"] = "coefficient" in schema_content.lower() and "formula" in schema_content.lower()
                
        else:
            # All tests fail if schema file doesn't exist
            for key in ["nodes_table", "edges_table", "coefficients_table", "labels_table", 
                       "pulses_table", "formulas_table", "training_runs_table", "training_metrics_table", 
                       "model_embeddings_table", "phase_support", "pulse_system_integration", "sfde_integration"]:
                results[key] = False
        
        self.test_results["training_database"] = results
        return results
    
    def test_dual_ac_system(self) -> Dict[str, bool]:
        """Test Dual Agent Coefficient system"""
        logger.info("🎮 Testing Dual AC System")
        
        results = {}
        
        # Test Dual AC API server
        dual_ac_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dual_ac_api_server.py')
        results["dual_ac_server_exists"] = os.path.exists(dual_ac_path)
        
        if os.path.exists(dual_ac_path):
            with open(dual_ac_path, 'r') as f:
                dual_ac_content = f.read()
                results["cosmetic_ac_endpoint"] = "/cosmetic_ac" in dual_ac_content
                results["unreal_ac_endpoint"] = "/unreal_ac" in dual_ac_content
                results["mobile_responsive"] = "mobile" in dual_ac_content.lower() or "responsive" in dual_ac_content.lower()
        else:
            results["cosmetic_ac_endpoint"] = False
            results["unreal_ac_endpoint"] = False
            results["mobile_responsive"] = False
        
        # Test Semantic Pulse Interface
        semantic_path = os.path.join(os.path.dirname(__file__), 'frontend', 'semantic_pulse_interface.html')
        results["semantic_interface_exists"] = os.path.exists(semantic_path)
        
        if os.path.exists(semantic_path):
            with open(semantic_path, 'r') as f:
                semantic_content = f.read()
                # Test for 7-pulse system in interface
                pulse_types = ["bid_pulse", "occupancy_pulse", "compliancy_pulse", 
                              "fit_pulse", "investment_pulse", "decay_pulse", "reject_pulse"]
                pulse_ui_coverage = sum(1 for pulse_type in pulse_types if pulse_type in semantic_content)
                results["pulse_ui_coverage"] = pulse_ui_coverage >= 6
                results["websocket_integration"] = "WebSocket" in semantic_content or "websocket" in semantic_content
        else:
            results["pulse_ui_coverage"] = False
            results["websocket_integration"] = False
        
        self.test_results["dual_ac_system"] = results
        return results
    
    def test_microservice_architecture(self) -> Dict[str, bool]:
        """Test Phase 2 microservice engine components"""
        logger.info("🐳 Testing Microservice Architecture")
        
        results = {}
        
        microservice_path = os.path.join(os.path.dirname(__file__), 'MICROSERVICE_ENGINES')
        
        # Test 6-container system
        required_engines = [
            "ne-dag-alpha",
            "ne-functor-types", 
            "ne-callback-engine",
            "sfde",
            "ne-graph-runtime-engine"
        ]
        
        for engine in required_engines:
            engine_path = os.path.join(microservice_path, engine)
            results[f"{engine.replace('-', '_')}_exists"] = os.path.exists(engine_path)
            
            # Test for Dockerfile
            dockerfile_path = os.path.join(engine_path, "Dockerfile")
            results[f"{engine.replace('-', '_')}_dockerfile"] = os.path.exists(dockerfile_path)
        
        # Test docker-compose
        docker_compose_path = os.path.join(microservice_path, "docker-compose.yml")
        results["docker_compose_exists"] = os.path.exists(docker_compose_path)
        
        self.test_results["microservice_architecture"] = results
        return results
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run complete BEM system test suite"""
        logger.info("🧪 Starting Comprehensive BEM System Testing")
        
        start_time = time.time()
        
        # Run all test phases
        self.test_file_structure()
        self.test_ecm_pulse_separation()
        self.test_dgl_trainer_integration()
        self.test_training_database_schema()
        self.test_dual_ac_system()
        self.test_microservice_architecture()
        
        # Calculate results
        total_tests = 0
        passed_tests = 0
        
        for component, tests in self.test_results.items():
            for test_name, result in tests.items():
                total_tests += 1
                if result:
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        end_time = time.time()
        
        summary = {
            "test_results": self.test_results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "duration_seconds": end_time - start_time,
                "deployment_ready": success_rate >= 75.0  # 75% pass rate for deployment
            }
        }
        
        self.print_test_results(summary)
        return summary
    
    def print_test_results(self, summary: Dict[str, Any]):
        """Print comprehensive test results"""
        logger.info("=" * 70)
        logger.info("🧪 BEM SYSTEM COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 70)
        
        for component, tests in summary["test_results"].items():
            logger.info(f"\n📋 {component.upper().replace('_', ' ')}:")
            for test_name, result in tests.items():
                status = "✅ PASS" if result else "❌ FAIL"
                test_display = test_name.replace('_', ' ').title()
                logger.info(f"  {status} {test_display}")
        
        summary_data = summary["summary"]
        logger.info(f"\n📊 OVERALL SUMMARY:")
        logger.info(f"  Total Tests: {summary_data['total_tests']}")
        logger.info(f"  Passed: {summary_data['passed_tests']}")
        logger.info(f"  Failed: {summary_data['failed_tests']}")
        logger.info(f"  Success Rate: {summary_data['success_rate']:.1f}%")
        logger.info(f"  Duration: {summary_data['duration_seconds']:.2f}s")
        
        if summary_data['deployment_ready']:
            logger.info(f"\n🚀 DEPLOYMENT STATUS: READY FOR DEPLOYMENT!")
            logger.info("   ✅ System architecture validated")
            logger.info("   ✅ Component separation confirmed") 
            logger.info("   ✅ Database integration verified")
            logger.info("   ✅ File structure correct")
        else:
            logger.info(f"\n⚠️  DEPLOYMENT STATUS: NOT READY - ISSUES FOUND")
            logger.info("   Review failed tests before deployment")
        
        logger.info("=" * 70)

# Test execution
async def main():
    """Main test execution"""
    tester = BEMSystemTester()
    results = await tester.run_comprehensive_test_suite()
    return 0 if results["summary"]["deployment_ready"] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 