#!/usr/bin/env python3
"""
Phase 3 Quick Retest - Focus on WebSocket fixes and critical production issues
"""

import asyncio
import websockets
import json
import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase3QuickTester:
    def __init__(self):
        self.test_results = {}
        self.critical_tests = [
            "WebSocket Basic Connectivity",
            "WebSocket Message Handling", 
            "Pulse System Basic Test",
            "API Performance",
            "Service Coordination",
            "Production Simulation (Short)"
        ]
        
    async def run_critical_tests(self):
        """Run critical tests to verify fixes"""
        logger.info("🔄 Phase 3 Quick Retest - Verifying Critical Fixes")
        logger.info("=" * 50)
        
        await self.test_websocket_connectivity()
        await self.test_websocket_message_handling()
        await self.test_pulse_system_basic()
        await self.test_api_performance()
        await self.test_service_coordination()
        await self.test_production_simulation_short()
        
        self.generate_quick_report()
    
    async def test_websocket_connectivity(self):
        """Test basic WebSocket connectivity"""
        test_name = "WebSocket Basic Connectivity"
        
        try:
            async with websockets.connect("ws://localhost:8765") as ws:
                # Connection successful
                self.log_test_result(test_name, True, "Connection established")
        except Exception as e:
            self.log_test_result(test_name, False, f"Connection failed: {e}")
    
    async def test_websocket_message_handling(self):
        """Test WebSocket message handling"""
        test_name = "WebSocket Message Handling"
        
        try:
            async with websockets.connect("ws://localhost:8765") as ws:
                # Send test message
                test_message = {"type": "test", "payload": {"action": "ping"}}
                await ws.send(json.dumps(test_message))
                
                # Wait for response
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                response_data = json.loads(response)
                
                if response_data.get("status") == "received":
                    self.log_test_result(test_name, True, "Message handled correctly")
                else:
                    self.log_test_result(test_name, False, f"Unexpected response: {response_data}")
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_pulse_system_basic(self):
        """Test basic pulse system functionality"""
        test_name = "Pulse System Basic Test"
        
        try:
            async with websockets.connect("ws://localhost:8766") as ws:
                # Send pulse
                pulse_data = {
                    "type": "bid_pulse",
                    "source": "test_node",
                    "urgency": 3
                }
                await ws.send(json.dumps(pulse_data))
                
                # Wait for response
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                response_data = json.loads(response)
                
                if "pulse_received" in response_data.get("status", ""):
                    self.log_test_result(test_name, True, "Pulse processed successfully")
                else:
                    self.log_test_result(test_name, False, f"Pulse not processed: {response_data}")
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_api_performance(self):
        """Test API performance"""
        test_name = "API Performance"
        
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8002/api/status", timeout=5)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200 and response_time < 0.1:
                self.log_test_result(test_name, True, f"Response time: {response_time:.3f}s")
            else:
                self.log_test_result(test_name, False, f"Slow or failed: {response_time:.3f}s, status: {response.status_code}")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_service_coordination(self):
        """Test coordination between API and WebSocket services"""
        test_name = "Service Coordination"
        
        try:
            # Test API
            api_response = requests.post("http://localhost:8002/cosmetic_ac", 
                                       json={"test": "coordination"}, timeout=5)
            
            # Test WebSocket
            async with websockets.connect("ws://localhost:8765") as ws:
                await ws.send('{"type": "coordination_test"}')
                ws_response = await asyncio.wait_for(ws.recv(), timeout=5)
            
            if api_response.status_code == 200 and "received" in ws_response:
                self.log_test_result(test_name, True, "Both services responding")
            else:
                self.log_test_result(test_name, False, "Service coordination issues")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_production_simulation_short(self):
        """Test short production simulation"""
        test_name = "Production Simulation (Short)"
        
        try:
            logger.info("  🔄 Running 10-second production simulation...")
            
            # Run load for 10 seconds
            start_time = time.time()
            request_count = 0
            errors = 0
            
            while time.time() - start_time < 10:
                try:
                    response = requests.post("http://localhost:8002/cosmetic_ac",
                                           json={"simulation": "short"}, timeout=2)
                    if response.status_code == 200:
                        request_count += 1
                    else:
                        errors += 1
                except:
                    errors += 1
                
                await asyncio.sleep(0.1)  # 10 requests per second
            
            error_rate = errors / (request_count + errors) if request_count + errors > 0 else 1
            
            if error_rate < 0.05:  # Less than 5% error rate
                self.log_test_result(test_name, True, f"{request_count} requests, {error_rate:.1%} error rate")
            else:
                self.log_test_result(test_name, False, f"High error rate: {error_rate:.1%}")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    def log_test_result(self, test_name, success, details=""):
        """Log test result"""
        if success:
            status = "✅ PASS"
            logger.info(f"  {status} {test_name}")
            if details:
                logger.info(f"      └─ {details}")
        else:
            status = "❌ FAIL"
            logger.error(f"  {status} {test_name}")
            if details:
                logger.error(f"      └─ {details}")
        
        self.test_results[test_name] = {"success": success, "details": details}
    
    def generate_quick_report(self):
        """Generate quick test report"""
        logger.info("\n" + "=" * 50)
        logger.info("📊 PHASE 3 QUICK RETEST RESULTS")
        logger.info("=" * 50)
        
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        total_tests = len(self.critical_tests)
        pass_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"📈 CRITICAL TEST RESULTS:")
        logger.info(f"   Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"   Success Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 83:  # 5/6 or better
            logger.info("\n✅ MAJOR IMPROVEMENT ACHIEVED")
            logger.info("   🎯 Critical WebSocket issues resolved")
            logger.info("   🚀 Ready for production feature implementation")
        elif pass_rate >= 67:  # 4/6
            logger.info("\n⚠️  GOOD PROGRESS - Minor issues remain")
            logger.info("   🔧 Continue addressing remaining issues")
        else:
            logger.info("\n❌ CRITICAL ISSUES PERSIST")
            logger.info("   🚨 Major fixes still needed")
        
        # Show failed tests
        failed_tests = [name for name, result in self.test_results.items() if not result["success"]]
        if failed_tests:
            logger.info(f"\n❌ REMAINING ISSUES:")
            for test in failed_tests:
                details = self.test_results[test]["details"]
                logger.info(f"   • {test}: {details}")
        
        logger.info("\n" + "=" * 50)
        return pass_rate

async def main():
    """Main test execution"""
    tester = Phase3QuickTester()
    await tester.run_critical_tests()

if __name__ == "__main__":
    asyncio.run(main()) 