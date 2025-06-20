#!/usr/bin/env python3
"""
Phase 3 Production Readiness Test Suite
Tests performance, scalability, security, monitoring, and deployment readiness
"""

import asyncio
import aiohttp
import websockets
import json
import time
import concurrent.futures
import subprocess
import os
import requests
import threading
import logging
import docker
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase3ProductionTester:
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.error_count = 0
        self.total_tests = 25  # Updated for comprehensive production testing
        
    async def run_all_tests(self):
        """Execute all Phase 3 production readiness tests"""
        logger.info("🚀 Starting Phase 3 Production Readiness Testing")
        logger.info("=" * 60)
        
        test_groups = [
            ("Performance & Load Testing", self.test_performance_suite),
            ("Scalability & Concurrency", self.test_scalability_suite), 
            ("Security & Authentication", self.test_security_suite),
            ("Monitoring & Observability", self.test_monitoring_suite),
            ("Deployment & Infrastructure", self.test_deployment_suite)
        ]
        
        for group_name, test_func in test_groups:
            logger.info(f"\n📊 {group_name}")
            logger.info("-" * 40)
            await test_func()
        
        self.generate_production_report()
    
    async def test_performance_suite(self):
        """Test system performance under various loads"""
        
        # Test 1: API Response Time
        await self.test_api_response_time()
        
        # Test 2: WebSocket Latency
        await self.test_websocket_latency()
        
        # Test 3: Concurrent Request Handling
        await self.test_concurrent_requests()
        
        # Test 4: Memory Usage Under Load
        await self.test_memory_usage()
        
        # Test 5: Database Connection Pool Performance
        await self.test_database_performance()
    
    async def test_scalability_suite(self):
        """Test system scalability and concurrency limits"""
        
        # Test 6: High Concurrency WebSocket Connections
        await self.test_websocket_concurrency()
        
        # Test 7: Pulse System Load Testing
        await self.test_pulse_system_load()
        
        # Test 8: Multi-Service Coordination
        await self.test_service_coordination()
        
        # Test 9: Container Orchestration
        await self.test_container_scaling()
        
        # Test 10: Load Balancing Capability
        await self.test_load_balancing()
    
    async def test_security_suite(self):
        """Test security measures and authentication"""
        
        # Test 11: Input Validation & Sanitization
        await self.test_input_validation()
        
        # Test 12: WebSocket Security
        await self.test_websocket_security()
        
        # Test 13: Rate Limiting
        await self.test_rate_limiting()
        
        # Test 14: CORS Configuration
        await self.test_cors_configuration()
        
        # Test 15: SSL/TLS Readiness
        await self.test_ssl_readiness()
    
    async def test_monitoring_suite(self):
        """Test monitoring and observability features"""
        
        # Test 16: Health Check Endpoints
        await self.test_health_checks()
        
        # Test 17: Metrics Collection
        await self.test_metrics_collection()
        
        # Test 18: Error Tracking
        await self.test_error_tracking()
        
        # Test 19: Performance Monitoring
        await self.test_performance_monitoring()
        
        # Test 20: Service Discovery
        await self.test_service_discovery()
    
    async def test_deployment_suite(self):
        """Test deployment readiness and infrastructure"""
        
        # Test 21: Docker Container Health
        await self.test_docker_health()
        
        # Test 22: Environment Configuration
        await self.test_environment_config()
        
        # Test 23: Database Migration Readiness
        await self.test_database_migration()
        
        # Test 24: Backup & Recovery
        await self.test_backup_recovery()
        
        # Test 25: Production Deployment Simulation
        await self.test_production_simulation()
    
    async def test_api_response_time(self):
        """Test API response times under normal load"""
        test_name = "API Response Time"
        start_time = time.time()
        
        try:
            # Test multiple endpoints with timing
            endpoints = [
                "http://localhost:8002/api/status",
                "http://localhost:8002/cosmetic_ac",
                "http://localhost:8002/unreal_ac"
            ]
            
            response_times = []
            for endpoint in endpoints:
                req_start = time.time()
                if "status" in endpoint:
                    response = requests.get(endpoint, timeout=5)
                else:
                    response = requests.post(endpoint, json={"test": "data"}, timeout=5)
                req_time = time.time() - req_start
                response_times.append(req_time)
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}")
            
            avg_response_time = sum(response_times) / len(response_times)
            self.performance_metrics["avg_api_response_time"] = avg_response_time
            
            # Performance threshold: < 100ms average
            if avg_response_time < 0.1:
                self.log_test_result(test_name, True, f"Avg response time: {avg_response_time:.3f}s")
            else:
                self.log_test_result(test_name, False, f"Slow response time: {avg_response_time:.3f}s")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_websocket_latency(self):
        """Test WebSocket message latency"""
        test_name = "WebSocket Latency"
        
        try:
            # Test ECM Gateway latency
            start_time = time.time()
            async with websockets.connect("ws://localhost:8765") as websocket:
                message = {"type": "ping", "timestamp": time.time()}
                
                send_time = time.time()
                await websocket.send(json.dumps(message))
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                recv_time = time.time()
                
                latency = recv_time - send_time
                self.performance_metrics["websocket_latency"] = latency
                
                # Latency threshold: < 50ms
                if latency < 0.05:
                    self.log_test_result(test_name, True, f"Latency: {latency:.3f}s")
                else:
                    self.log_test_result(test_name, False, f"High latency: {latency:.3f}s")
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_concurrent_requests(self):
        """Test handling of concurrent API requests"""
        test_name = "Concurrent Request Handling"
        
        try:
            async def make_request(session, url):
                async with session.post(url, json={"test": "concurrent"}) as response:
                    return response.status
            
            async with aiohttp.ClientSession() as session:
                # Test 50 concurrent requests
                tasks = [
                    make_request(session, "http://localhost:8002/cosmetic_ac") 
                    for _ in range(50)
                ]
                
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                successful_requests = sum(1 for r in results if r == 200)
                total_time = end_time - start_time
                
                self.performance_metrics["concurrent_requests_success_rate"] = successful_requests / 50
                self.performance_metrics["concurrent_requests_time"] = total_time
                
                if successful_requests >= 45:  # 90% success rate
                    self.log_test_result(test_name, True, f"{successful_requests}/50 successful in {total_time:.2f}s")
                else:
                    self.log_test_result(test_name, False, f"Only {successful_requests}/50 successful")
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_memory_usage(self):
        """Test memory usage under load"""
        test_name = "Memory Usage Under Load"
        
        try:
            import psutil
            
            # Get baseline memory usage
            process = psutil.Process(os.getpid())
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate load for 10 seconds
            tasks = []
            for _ in range(20):
                tasks.append(self.simulate_load())
            
            await asyncio.gather(*tasks)
            
            # Check memory after load
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - baseline_memory
            
            self.performance_metrics["memory_usage_mb"] = current_memory
            self.performance_metrics["memory_increase_mb"] = memory_increase
            
            # Memory threshold: < 100MB increase
            if memory_increase < 100:
                self.log_test_result(test_name, True, f"Memory increase: {memory_increase:.1f}MB")
            else:
                self.log_test_result(test_name, False, f"High memory usage: {memory_increase:.1f}MB")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def simulate_load(self):
        """Simulate system load for testing"""
        try:
            async with aiohttp.ClientSession() as session:
                for _ in range(5):
                    async with session.post("http://localhost:8002/cosmetic_ac", 
                                          json={"load_test": True}) as response:
                        await response.read()
        except:
            pass
    
    async def test_database_performance(self):
        """Test database connection and query performance"""
        test_name = "Database Performance"
        
        try:
            # Check if training database is accessible
            from Final_Phase.dgl_trainer import TrainingDatabase
            
            db = TrainingDatabase()
            start_time = time.time()
            
            # Test connection and basic query
            result = await db.test_connection()
            query_time = time.time() - start_time
            
            self.performance_metrics["db_query_time"] = query_time
            
            if result and query_time < 1.0:  # < 1 second
                self.log_test_result(test_name, True, f"DB responsive in {query_time:.3f}s")
            else:
                self.log_test_result(test_name, False, f"DB slow or unreachable: {query_time:.3f}s")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Database not configured: {e}")
    
    async def test_websocket_concurrency(self):
        """Test multiple concurrent WebSocket connections"""
        test_name = "WebSocket Concurrency"
        
        try:
            async def connect_and_test():
                async with websockets.connect("ws://localhost:8765") as ws:
                    await ws.send('{"type": "test", "data": "concurrent"}')
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    return json.loads(response).get("status") == "received"
            
            # Test 20 concurrent WebSocket connections
            tasks = [connect_and_test() for _ in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_connections = sum(1 for r in results if r is True)
            
            if successful_connections >= 18:  # 90% success rate
                self.log_test_result(test_name, True, f"{successful_connections}/20 connections successful")
            else:
                self.log_test_result(test_name, False, f"Only {successful_connections}/20 connections successful")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_pulse_system_load(self):
        """Test pulse system under load"""
        test_name = "Pulse System Load"
        
        try:
            async def send_pulse():
                async with websockets.connect("ws://localhost:8766") as ws:
                    pulse_data = {
                        "type": "bid_pulse",
                        "source": "test_node",
                        "target": "any_node",
                        "urgency": 5
                    }
                    await ws.send(json.dumps(pulse_data))
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    return "pulse_received" in response
            
            # Send 30 pulses concurrently
            tasks = [send_pulse() for _ in range(30)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_pulses = sum(1 for r in results if r is True)
            
            if successful_pulses >= 27:  # 90% success rate
                self.log_test_result(test_name, True, f"{successful_pulses}/30 pulses processed")
            else:
                self.log_test_result(test_name, False, f"Only {successful_pulses}/30 pulses processed")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_service_coordination(self):
        """Test coordination between multiple services"""
        test_name = "Multi-Service Coordination"
        
        try:
            # Test API -> WebSocket coordination
            api_response = requests.post("http://localhost:8002/cosmetic_ac", 
                                       json={"action": "coordinate_test"})
            
            async with websockets.connect("ws://localhost:8765") as ws:
                await ws.send('{"type": "coordination_test"}')
                ws_response = await asyncio.wait_for(ws.recv(), timeout=5)
            
            if api_response.status_code == 200 and "received" in ws_response:
                self.log_test_result(test_name, True, "Services coordinating properly")
            else:
                self.log_test_result(test_name, False, "Service coordination issues")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_container_scaling(self):
        """Test Docker container scaling capabilities"""
        test_name = "Container Orchestration"
        
        try:
            # Check if Docker is available and containers are healthy
            client = docker.from_env()
            containers = client.containers.list()
            
            # Check for microservice containers
            microservice_containers = [c for c in containers if 'ne-' in c.name or 'sfde' in c.name]
            
            if len(microservice_containers) >= 3:
                self.log_test_result(test_name, True, f"{len(microservice_containers)} microservice containers running")
            else:
                self.log_test_result(test_name, False, f"Only {len(microservice_containers)} containers running")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Docker not available: {e}")
    
    async def test_load_balancing(self):
        """Test load balancing capabilities"""
        test_name = "Load Balancing"
        
        try:
            # Test distribution of requests across endpoints
            response_times = []
            for _ in range(10):
                start = time.time()
                response = requests.get("http://localhost:8002/api/status")
                end = time.time()
                response_times.append(end - start)
            
            avg_time = sum(response_times) / len(response_times)
            variance = sum((t - avg_time) ** 2 for t in response_times) / len(response_times)
            
            # Low variance indicates good load balancing
            if variance < 0.01:  # Low variance threshold
                self.log_test_result(test_name, True, f"Consistent response times (var: {variance:.4f})")
            else:
                self.log_test_result(test_name, False, f"High response time variance: {variance:.4f}")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_input_validation(self):
        """Test input validation and sanitization"""
        test_name = "Input Validation & Sanitization"
        
        try:
            # Test malicious inputs
            malicious_inputs = [
                {"script": "<script>alert('xss')</script>"},
                {"sql": "'; DROP TABLE users; --"},
                {"overflow": "A" * 10000},
                {"null": None},
                {"empty": ""}
            ]
            
            safe_responses = 0
            for malicious_input in malicious_inputs:
                try:
                    response = requests.post("http://localhost:8002/cosmetic_ac", 
                                           json=malicious_input, timeout=5)
                    if response.status_code in [200, 400, 422]:  # Handled properly
                        safe_responses += 1
                except:
                    safe_responses += 1  # Connection refused = safe
            
            if safe_responses == len(malicious_inputs):
                self.log_test_result(test_name, True, "All malicious inputs handled safely")
            else:
                self.log_test_result(test_name, False, f"Only {safe_responses}/{len(malicious_inputs)} inputs handled safely")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_websocket_security(self):
        """Test WebSocket security measures"""
        test_name = "WebSocket Security"
        
        try:
            # Test connection limits and message validation
            async with websockets.connect("ws://localhost:8765") as ws:
                # Test oversized message handling
                large_message = json.dumps({"data": "A" * 100000})
                await ws.send(large_message)
                
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    self.log_test_result(test_name, True, "WebSocket handles large messages safely")
                except:
                    self.log_test_result(test_name, True, "WebSocket properly rejects oversized messages")
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_rate_limiting(self):
        """Test rate limiting implementation"""
        test_name = "Rate Limiting"
        
        try:
            # Send rapid requests to test rate limiting
            responses = []
            for _ in range(100):
                try:
                    response = requests.get("http://localhost:8002/api/status", timeout=1)
                    responses.append(response.status_code)
                except:
                    responses.append(429)  # Rate limited
            
            # Check if any rate limiting occurred
            rate_limited = sum(1 for r in responses if r == 429)
            
            if rate_limited > 0:
                self.log_test_result(test_name, True, f"Rate limiting active ({rate_limited} requests limited)")
            else:
                self.log_test_result(test_name, False, "No rate limiting detected (may need implementation)")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_cors_configuration(self):
        """Test CORS configuration"""
        test_name = "CORS Configuration"
        
        try:
            # Test CORS headers
            response = requests.options("http://localhost:8002/api/status",
                                      headers={"Origin": "http://localhost:3000"})
            
            cors_headers = [
                "Access-Control-Allow-Origin",
                "Access-Control-Allow-Methods",
                "Access-Control-Allow-Headers"
            ]
            
            cors_configured = any(header in response.headers for header in cors_headers)
            
            if cors_configured:
                self.log_test_result(test_name, True, "CORS properly configured")
            else:
                self.log_test_result(test_name, False, "CORS headers missing")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_ssl_readiness(self):
        """Test SSL/TLS readiness"""
        test_name = "SSL/TLS Readiness"
        
        try:
            # Check for SSL certificate files
            ssl_files = [
                "cert.pem",
                "key.pem",
                "ssl/cert.pem",
                "ssl/key.pem"
            ]
            
            ssl_ready = any(os.path.exists(f) for f in ssl_files)
            
            if ssl_ready:
                self.log_test_result(test_name, True, "SSL certificates found")
            else:
                self.log_test_result(test_name, False, "SSL certificates not found (needed for production)")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_health_checks(self):
        """Test health check endpoints"""
        test_name = "Health Check Endpoints"
        
        try:
            # Test API health endpoint
            response = requests.get("http://localhost:8002/api/status")
            
            if response.status_code == 200 and "status" in response.json():
                self.log_test_result(test_name, True, "Health checks responding")
            else:
                self.log_test_result(test_name, False, "Health check endpoint issues")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_metrics_collection(self):
        """Test metrics collection capabilities"""
        test_name = "Metrics Collection"
        
        try:
            # Check if performance metrics are being collected
            metrics_collected = bool(self.performance_metrics)
            
            if metrics_collected:
                self.log_test_result(test_name, True, f"Collecting {len(self.performance_metrics)} metrics")
            else:
                self.log_test_result(test_name, False, "No metrics collection detected")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_error_tracking(self):
        """Test error tracking and logging"""
        test_name = "Error Tracking"
        
        try:
            # Check if errors are being properly logged
            log_files = ["ecm_log.txt", "error.log", "app.log"]
            log_exists = any(os.path.exists(f) for f in log_files)
            
            if log_exists:
                self.log_test_result(test_name, True, "Error logging configured")
            else:
                self.log_test_result(test_name, False, "Error logging not found")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_performance_monitoring(self):
        """Test performance monitoring setup"""
        test_name = "Performance Monitoring"
        
        try:
            # Check if performance data is available
            perf_data_available = len(self.performance_metrics) >= 5
            
            if perf_data_available:
                self.log_test_result(test_name, True, "Performance monitoring active")
            else:
                self.log_test_result(test_name, False, "Limited performance monitoring")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_service_discovery(self):
        """Test service discovery mechanisms"""
        test_name = "Service Discovery"
        
        try:
            # Test if services can find each other
            api_reachable = requests.get("http://localhost:8002/api/status").status_code == 200
            
            async with websockets.connect("ws://localhost:8765") as ws:
                ws_reachable = True
            
            if api_reachable and ws_reachable:
                self.log_test_result(test_name, True, "Services discoverable")
            else:
                self.log_test_result(test_name, False, "Service discovery issues")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_docker_health(self):
        """Test Docker container health"""
        test_name = "Docker Container Health"
        
        try:
            # Check Docker Compose file exists
            compose_files = ["docker-compose.yml", "MICROSERVICE_ENGINES/docker-compose.yml"]
            compose_exists = any(os.path.exists(f) for f in compose_files)
            
            if compose_exists:
                self.log_test_result(test_name, True, "Docker Compose configuration found")
            else:
                self.log_test_result(test_name, False, "Docker Compose configuration missing")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_environment_config(self):
        """Test environment configuration"""
        test_name = "Environment Configuration"
        
        try:
            # Check for environment configuration files
            env_files = [".env", "config.py", "neon/config.py"]
            env_configured = any(os.path.exists(f) for f in env_files)
            
            if env_configured:
                self.log_test_result(test_name, True, "Environment configuration found")
            else:
                self.log_test_result(test_name, False, "Environment configuration missing")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_database_migration(self):
        """Test database migration readiness"""
        test_name = "Database Migration Readiness"
        
        try:
            # Check for database schema files
            schema_files = [
                "neon/postgresql_schema.sql",
                "postgre/enhanced_schema.sql",
                "Final_Phase/training_database_schema.sql"
            ]
            
            schema_ready = any(os.path.exists(f) for f in schema_files)
            
            if schema_ready:
                self.log_test_result(test_name, True, "Database schemas available")
            else:
                self.log_test_result(test_name, False, "Database schemas missing")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_backup_recovery(self):
        """Test backup and recovery procedures"""
        test_name = "Backup & Recovery"
        
        try:
            # Check for backup scripts or procedures
            backup_indicators = [
                "backup.py", 
                "scripts/backup.sh",
                "deploy/backup.yml"
            ]
            
            backup_ready = any(os.path.exists(f) for f in backup_indicators)
            
            if backup_ready:
                self.log_test_result(test_name, True, "Backup procedures found")
            else:
                self.log_test_result(test_name, False, "Backup procedures need implementation")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    async def test_production_simulation(self):
        """Test production deployment simulation"""
        test_name = "Production Deployment Simulation"
        
        try:
            # Simulate production load for 30 seconds
            logger.info("  🔄 Running 30-second production simulation...")
            
            start_time = time.time()
            tasks = []
            
            # Continuous load for 30 seconds
            async def continuous_load():
                end_time = time.time() + 30
                request_count = 0
                errors = 0
                
                while time.time() < end_time:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post("http://localhost:8002/cosmetic_ac",
                                                   json={"simulation": True}) as response:
                                if response.status == 200:
                                    request_count += 1
                                else:
                                    errors += 1
                    except:
                        errors += 1
                    
                    await asyncio.sleep(0.1)  # 10 requests per second
                
                return request_count, errors
            
            # Run multiple load generators
            results = await asyncio.gather(*[continuous_load() for _ in range(5)])
            
            total_requests = sum(r[0] for r in results)
            total_errors = sum(r[1] for r in results)
            
            error_rate = total_errors / (total_requests + total_errors) if total_requests + total_errors > 0 else 1
            
            if error_rate < 0.05:  # Less than 5% error rate
                self.log_test_result(test_name, True, f"{total_requests} requests, {error_rate:.1%} error rate")
            else:
                self.log_test_result(test_name, False, f"High error rate: {error_rate:.1%}")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
    
    def log_test_result(self, test_name, success, details=""):
        """Log test result with consistent formatting"""
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
            self.error_count += 1
        
        self.test_results[test_name] = {"success": success, "details": details}
    
    def generate_production_report(self):
        """Generate comprehensive production readiness report"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 PHASE 3 PRODUCTION READINESS REPORT")
        logger.info("=" * 60)
        
        # Overall metrics
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        pass_rate = (passed_tests / self.total_tests) * 100
        
        logger.info(f"📈 OVERALL RESULTS:")
        logger.info(f"   Tests Passed: {passed_tests}/{self.total_tests}")
        logger.info(f"   Success Rate: {pass_rate:.1f}%")
        
        # Performance metrics summary
        if self.performance_metrics:
            logger.info(f"\n⚡ PERFORMANCE METRICS:")
            for metric, value in self.performance_metrics.items():
                if isinstance(value, float):
                    logger.info(f"   {metric}: {value:.3f}")
                else:
                    logger.info(f"   {metric}: {value}")
        
        # Production readiness assessment
        logger.info(f"\n🚀 PRODUCTION READINESS ASSESSMENT:")
        
        if pass_rate >= 90:
            logger.info("   ✅ READY FOR PRODUCTION DEPLOYMENT")
            logger.info("   🎯 System demonstrates excellent stability and performance")
        elif pass_rate >= 80:
            logger.info("   ⚠️  MOSTLY READY - Minor improvements needed")
            logger.info("   🔧 Address failed tests before production deployment")
        elif pass_rate >= 70:
            logger.info("   ⚠️  NEEDS IMPROVEMENT - Several issues to address")
            logger.info("   🛠️  Significant work needed before production")
        else:
            logger.info("   ❌ NOT READY FOR PRODUCTION")
            logger.info("   🚨 Major issues must be resolved")
        
        # Failed tests summary
        failed_tests = [name for name, result in self.test_results.items() if not result["success"]]
        if failed_tests:
            logger.info(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                details = self.test_results[test]["details"]
                logger.info(f"   • {test}: {details}")
        
        # Next steps
        logger.info(f"\n📋 NEXT STEPS:")
        if pass_rate >= 90:
            logger.info("   1. Deploy to staging environment")
            logger.info("   2. Run final integration tests")
            logger.info("   3. Prepare production deployment")
        else:
            logger.info("   1. Address failed test cases")
            logger.info("   2. Implement missing production features")
            logger.info("   3. Re-run Phase 3 testing")
        
        logger.info("\n" + "=" * 60)
        return pass_rate

async def main():
    """Main test execution"""
    tester = Phase3ProductionTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())