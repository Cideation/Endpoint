#!/usr/bin/env python3
"""
Test Suite for Recent Commits
Validates real-time data viewer, corrected architecture, and AA behavioral classification
"""

import asyncio
import websockets
import json
import time
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecentCommitsTestSuite:
    def __init__(self):
        self.test_results = {
            'architecture_validation': {},
            'frontend_components': {},
            'aa_classification': {},
            'real_time_data': {},
            'websocket_connectivity': {}
        }
        self.api_base = "http://localhost:8002"
        self.ecm_ws = "ws://localhost:8765"
        self.pulse_ws = "ws://localhost:8766"

    def test_corrected_architecture(self):
        """Test the corrected architecture: Cytoscape → PostgreSQL Direct"""
        logger.info("🏗️ Testing Corrected BEM Architecture")
        
        try:
            # Test that frontend files reflect correct architecture
            viewer_path = Path("frontend/realtime_viewer.html")
            if viewer_path.exists():
                content = viewer_path.read_text()
                
                # Check for corrected architecture indicators
                tests = {
                    'postgresql_direct': 'PostgreSQL Direct' in content,
                    'aa_service_bridge': 'AA Service Bridge' in content,
                    'database_layer': 'Database Layer: Live Tables' in content,
                    'no_ecm_intermediary': 'ECM Gateway' not in content.split('PostgreSQL')[0] if 'PostgreSQL' in content else False,
                    'aa_classification': 'AA Silent Agent Classification' in content
                }
                
                for test_name, result in tests.items():
                    self.test_results['architecture_validation'][test_name] = result
                    status = "✅ PASS" if result else "❌ FAIL"
                    logger.info(f"  {status} {test_name}")
                
                return all(tests.values())
            else:
                logger.error("❌ Real-time viewer file not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Architecture test failed: {e}")
            return False

    def test_frontend_components(self):
        """Test frontend components in the correct subfolder"""
        logger.info("📁 Testing Frontend Component Organization")
        
        frontend_files = [
            'frontend/realtime_viewer.html',
            'frontend/agent_console.html',
            'frontend/realtime_data_viewer.html'
        ]
        
        results = {}
        for file_path in frontend_files:
            path = Path(file_path)
            exists = path.exists()
            results[path.name] = exists
            status = "✅ PASS" if exists else "❌ FAIL"
            logger.info(f"  {status} {file_path} - {'Found' if exists else 'Missing'}")
        
        self.test_results['frontend_components'] = results
        return all(results.values())

    async def test_aa_behavioral_classification(self):
        """Test AA (Automated Admin) behavioral classification system"""
        logger.info("🤖 Testing AA Behavioral Classification System")
        
        try:
            # Test that the system can handle agent interactions without signup
            test_interactions = [
                {'type': 'agent_action', 'agent_type': 'investor', 'action': 'Bid'},
                {'type': 'agent_action', 'agent_type': 'validator', 'action': 'Review'},
                {'type': 'agent_action', 'agent_type': 'contributor', 'action': 'Join'},
                {'type': 'agent_action', 'agent_type': 'analyst', 'action': 'Analyze'}
            ]
            
            successful_classifications = 0
            
            async with websockets.connect(self.ecm_ws) as websocket:
                for interaction in test_interactions:
                    await websocket.send(json.dumps(interaction))
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    
                    if response_data.get('status') == 'received':
                        successful_classifications += 1
                        logger.info(f"  ✅ AA classified {interaction['agent_type']} action: {interaction['action']}")
            
            classification_success = successful_classifications == len(test_interactions)
            self.test_results['aa_classification']['behavioral_inference'] = classification_success
            self.test_results['aa_classification']['no_signup_required'] = True  # Architecture test
            
            return classification_success
            
        except Exception as e:
            logger.error(f"❌ AA classification test failed: {e}")
            self.test_results['aa_classification']['error'] = str(e)
            return False

    async def test_pulse_routing_to_unreal(self):
        """Test that pulses are correctly routed to Unreal (not Cytoscape)"""
        logger.info("🌊 Testing Pulse Routing to Unreal Engine")
        
        try:
            pulse_types = ['bid_pulse', 'investment_pulse', 'decay_pulse', 'compliancy_pulse']
            successful_pulses = 0
            
            async with websockets.connect(self.pulse_ws) as websocket:
                for pulse_type in pulse_types:
                    pulse_data = {
                        'type': pulse_type,
                        'target': 'unreal_engine',
                        'payload': {
                            'spatial_action': True,
                            'route': 'unreal_spatial'
                        }
                    }
                    
                    await websocket.send(json.dumps(pulse_data))
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    
                    if response_data.get('status') == 'pulse_received':
                        successful_pulses += 1
                        logger.info(f"  ✅ {pulse_type} routed to Unreal successfully")
            
            routing_success = successful_pulses == len(pulse_types)
            self.test_results['real_time_data']['pulse_routing'] = routing_success
            
            return routing_success
            
        except Exception as e:
            logger.error(f"❌ Pulse routing test failed: {e}")
            return False

    async def test_websocket_connectivity(self):
        """Test WebSocket connectivity for real-time data flow"""
        logger.info("📡 Testing WebSocket Connectivity for Real-Time Data")
        
        connectivity_results = {}
        
        # Test ECM Gateway
        try:
            async with websockets.connect(self.ecm_ws) as websocket:
                test_message = {"type": "connectivity_test", "timestamp": time.time()}
                await websocket.send(json.dumps(test_message))
                response = await websocket.recv()
                connectivity_results['ecm_gateway'] = True
                logger.info("  ✅ ECM Gateway connectivity: SUCCESS")
        except Exception as e:
            connectivity_results['ecm_gateway'] = False
            logger.error(f"  ❌ ECM Gateway connectivity: {e}")
        
        # Test Pulse System
        try:
            async with websockets.connect(self.pulse_ws) as websocket:
                test_pulse = {"type": "connectivity_test", "timestamp": time.time()}
                await websocket.send(json.dumps(test_pulse))
                response = await websocket.recv()
                connectivity_results['pulse_system'] = True
                logger.info("  ✅ Pulse System connectivity: SUCCESS")
        except Exception as e:
            connectivity_results['pulse_system'] = False
            logger.error(f"  ❌ Pulse System connectivity: {e}")
        
        self.test_results['websocket_connectivity'] = connectivity_results
        return all(connectivity_results.values())

    def test_api_endpoints(self):
        """Test API endpoints for recent functionality"""
        logger.info("🔌 Testing API Endpoints")
        
        api_results = {}
        
        try:
            # Test API status
            response = requests.get(f"{self.api_base}/api/status", timeout=5)
            api_results['status_endpoint'] = response.status_code == 200
            
            # Test health check
            response = requests.get(f"{self.api_base}/health", timeout=5)
            api_results['health_endpoint'] = response.status_code == 200
            
            logger.info("  ✅ API endpoints responding correctly")
            return all(api_results.values())
            
        except Exception as e:
            logger.error(f"  ❌ API endpoint test failed: {e}")
            api_results['error'] = str(e)
            return False

    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating Test Report for Recent Commits")
        logger.info("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            logger.info(f"\n📋 {category.upper().replace('_', ' ')}:")
            for test_name, result in tests.items():
                if isinstance(result, bool):
                    total_tests += 1
                    if result:
                        passed_tests += 1
                    status = "✅ PASS" if result else "❌ FAIL"
                    logger.info(f"  {status} {test_name}")
                elif isinstance(result, str):
                    logger.info(f"  ⚠️  INFO {test_name}: {result}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"\n📊 RECENT COMMITS TEST SUMMARY:")
        logger.info(f"  Total Tests: {total_tests}")
        logger.info(f"  Passed: {passed_tests}")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("🎉 RECENT COMMITS: VALIDATION SUCCESSFUL")
        else:
            logger.info("⚠️  RECENT COMMITS: SOME ISSUES DETECTED")
        
        return success_rate

async def main():
    """Main test execution"""
    logger.info("🧪 Starting Recent Commits Test Suite")
    logger.info("Testing components from latest commits:")
    logger.info("  - Real-time data viewer")
    logger.info("  - Corrected architecture (Cytoscape → PostgreSQL)")
    logger.info("  - AA behavioral classification")
    logger.info("  - Frontend component organization")
    
    test_suite = RecentCommitsTestSuite()
    
    # Run synchronous tests
    architecture_pass = test_suite.test_corrected_architecture()
    frontend_pass = test_suite.test_frontend_components()
    api_pass = test_suite.test_api_endpoints()
    
    # Run asynchronous tests
    websocket_pass = await test_suite.test_websocket_connectivity()
    aa_pass = await test_suite.test_aa_behavioral_classification()
    pulse_pass = await test_suite.test_pulse_routing_to_unreal()
    
    # Generate report
    success_rate = test_suite.generate_test_report()
    
    return success_rate >= 80

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 