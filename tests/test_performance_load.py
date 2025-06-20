#!/usr/bin/env python3
"""
BEM System Performance & Load Testing Suite
Tests scalability, endurance, and system stability under heavy load
"""

import os
import time
import asyncio
import logging
import psutil
import websockets
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import dgl
import torch
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('performance_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceTestSuite:
    """Comprehensive performance and load testing suite"""
    
    def __init__(self):
        """Initialize test suite with configuration"""
        self.config = {
            'ecm_ws_url': 'ws://localhost:8765',
            'db_connection': {
                'host': 'localhost',
                'port': 5432,
                'database': 'bem_training',
                'user': 'bem_user',
                'password': os.getenv('DB_PASSWORD')
            },
            'api_base_url': 'http://localhost:8000',
            'test_duration': 72 * 3600,  # 72 hours in seconds
            'pulse_interval': 30,  # seconds
            'ws_connections': 1000,
            'nodes_count': 10000,
            'edges_count': 50000,
            'concurrent_pulses': 100
        }
        
        # Test metrics
        self.metrics = {
            'ws_latency': [],
            'pulse_latency': [],
            'memory_usage': [],
            'cpu_usage': [],
            'db_connection_status': [],
            'training_metrics': []
        }
    
    async def _monitor_system_resources(self, interval: int = 60):
        """Monitor system resources during tests"""
        while True:
            process = psutil.Process()
            self.metrics['memory_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'memory_percent': process.memory_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024
            })
            
            self.metrics['cpu_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': process.cpu_percent()
            })
            
            await asyncio.sleep(interval)
    
    async def _websocket_client(self, client_id: int):
        """Simulate WebSocket client connection"""
        try:
            async with websockets.connect(self.config['ecm_ws_url']) as ws:
                logger.info(f"Client {client_id} connected")
                
                while True:
                    # Send periodic heartbeat
                    start_time = time.time()
                    await ws.send(f"heartbeat_{client_id}")
                    response = await ws.recv()
                    latency = time.time() - start_time
                    
                    self.metrics['ws_latency'].append({
                        'client_id': client_id,
                        'timestamp': datetime.now().isoformat(),
                        'latency': latency
                    })
                    
                    await asyncio.sleep(5)  # Heartbeat interval
        except Exception as e:
            logger.error(f"WebSocket client {client_id} error: {str(e)}")
    
    async def _trigger_pulse(self, pulse_id: int):
        """Simulate pulse trigger in the system"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(
                    f"{self.config['api_base_url']}/trigger_pulse",
                    json={'pulse_id': pulse_id}
                ) as response:
                    await response.json()
                    latency = time.time() - start_time
                    
                    self.metrics['pulse_latency'].append({
                        'pulse_id': pulse_id,
                        'timestamp': datetime.now().isoformat(),
                        'latency': latency
                    })
        except Exception as e:
            logger.error(f"Pulse trigger {pulse_id} error: {str(e)}")
    
    def _generate_training_data(self) -> dgl.DGLGraph:
        """Generate large graph for training tests"""
        # Create random graph with specified nodes and edges
        src = np.random.randint(0, self.config['nodes_count'], self.config['edges_count'])
        dst = np.random.randint(0, self.config['nodes_count'], self.config['edges_count'])
        
        # Create DGL graph
        g = dgl.graph((src, dst))
        
        # Add random node and edge features
        g.ndata['feat'] = torch.randn(self.config['nodes_count'], 64)
        g.edata['feat'] = torch.randn(self.config['edges_count'], 32)
        
        return g
    
    async def _run_training_epoch(self, epoch: int):
        """Run single training epoch"""
        try:
            # Generate training data
            g = self._generate_training_data()
            
            # Simulate training
            start_time = time.time()
            # Here you would normally run your actual training code
            await asyncio.sleep(10)  # Simulate training time
            duration = time.time() - start_time
            
            self.metrics['training_metrics'].append({
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'duration': duration,
                'graph_size': {
                    'nodes': g.number_of_nodes(),
                    'edges': g.number_of_edges()
                }
            })
            
            logger.info(f"Training epoch {epoch} completed in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Training epoch {epoch} error: {str(e)}")
    
    async def _check_database_connection(self):
        """Check database connection status"""
        while True:
            try:
                with psycopg2.connect(**self.config['db_connection']) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        
                    self.metrics['db_connection_status'].append({
                        'timestamp': datetime.now().isoformat(),
                        'status': 'connected'
                    })
            except Exception as e:
                self.metrics['db_connection_status'].append({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(e)
                })
            
            await asyncio.sleep(60)  # Check every minute
    
    def _save_metrics(self):
        """Save test metrics to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('test_results', exist_ok=True)
        
        # Save each metric type to separate CSV
        for metric_name, data in self.metrics.items():
            if data:
                df = pd.DataFrame(data)
                df.to_csv(f"test_results/{metric_name}_{timestamp}.csv", index=False)
                logger.info(f"Saved {metric_name} metrics to CSV")
    
    async def run_scalability_test(self):
        """Run scalability tests"""
        logger.info("Starting scalability tests...")
        
        # Start WebSocket connections
        ws_clients = [
            self._websocket_client(i) 
            for i in range(self.config['ws_connections'])
        ]
        
        # Start pulse triggers
        pulse_triggers = [
            self._trigger_pulse(i) 
            for i in range(self.config['concurrent_pulses'])
        ]
        
        # Run all tests concurrently
        await asyncio.gather(
            *ws_clients,
            *pulse_triggers,
            self._monitor_system_resources()
        )
    
    async def run_endurance_test(self):
        """Run 72-hour endurance test"""
        logger.info("Starting 72-hour endurance test...")
        start_time = time.time()
        
        tasks = [
            # System monitoring
            self._monitor_system_resources(),
            # Database connection monitoring
            self._check_database_connection()
        ]
        
        # Add periodic pulse triggers
        async def periodic_pulse():
            pulse_id = 0
            while time.time() - start_time < self.config['test_duration']:
                await self._trigger_pulse(pulse_id)
                pulse_id += 1
                await asyncio.sleep(self.config['pulse_interval'])
        
        tasks.append(periodic_pulse())
        
        # Add periodic training
        async def periodic_training():
            epoch = 0
            while time.time() - start_time < self.config['test_duration']:
                await self._run_training_epoch(epoch)
                epoch += 1
                await asyncio.sleep(300)  # Run training every 5 minutes
        
        tasks.append(periodic_training())
        
        try:
            # Run all tasks
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Endurance test error: {str(e)}")
        finally:
            # Save test metrics
            self._save_metrics()
            
            logger.info("Endurance test completed")
            
            # Generate test report
            test_duration = time.time() - start_time
            avg_memory = np.mean([m['memory_mb'] for m in self.metrics['memory_usage']])
            avg_cpu = np.mean([c['cpu_percent'] for c in self.metrics['cpu_usage']])
            avg_ws_latency = np.mean([l['latency'] for l in self.metrics['ws_latency']])
            avg_pulse_latency = np.mean([l['latency'] for l in self.metrics['pulse_latency']])
            
            report = f"""
            Endurance Test Report
            ====================
            Duration: {test_duration/3600:.2f} hours
            Average Memory Usage: {avg_memory:.2f} MB
            Average CPU Usage: {avg_cpu:.2f}%
            Average WebSocket Latency: {avg_ws_latency*1000:.2f} ms
            Average Pulse Latency: {avg_pulse_latency*1000:.2f} ms
            Training Epochs: {len(self.metrics['training_metrics'])}
            """
            
            with open(f"test_results/endurance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
                f.write(report)
            
            logger.info("Test report generated")

async def main():
    """Run performance and load tests"""
    test_suite = PerformanceTestSuite()
    
    # Run scalability tests
    logger.info("Running scalability tests...")
    await test_suite.run_scalability_test()
    
    # Run endurance test
    logger.info("Running endurance tests...")
    await test_suite.run_endurance_test()

if __name__ == "__main__":
    asyncio.run(main()) 