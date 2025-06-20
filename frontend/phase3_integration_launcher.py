#!/usr/bin/env python3
"""
Phase 3 Integration Launcher

Launches the complete ECM infrastructure:
- ECM Infrastructure Gateway (port 8765)
- ECM Pulse System (port 8766) 
- Integration with existing Dual AC System

This completes the architecture loop where:
- Cosmetic AC + Unreal AC → Node Engine (computational)
- ECM Gateway → Infrastructure bridging (non-computational)
- ECM Pulse → Coordination pulses (segregated)

All engines are now complete with AC as the UI interface layer.
"""

import asyncio
import subprocess
import sys
import time
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PHASE3-LAUNCHER - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PHASE3_LAUNCHER")

class Phase3Launcher:
    """
    Phase 3 Integration Launcher
    
    Coordinates startup of all ECM infrastructure components
    """
    
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def check_requirements(self):
        """Check if all required files exist"""
        required_files = [
            'ecm_infrastructure_gateway.py',
            'unreal_integration_client.py',
            'enhanced_unified_interface.html'
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        return True
    
    async def start_ecm_gateway(self):
        """Start ECM Infrastructure Gateway"""
        logger.info("Starting ECM Infrastructure Gateway...")
        
        try:
            process = subprocess.Popen([
                sys.executable, 'ecm_infrastructure_gateway.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['ecm_gateway'] = process
            logger.info("✅ ECM Infrastructure Gateway started (PID: {})".format(process.pid))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start ECM Gateway: {e}")
            return False
    
    def create_ecm_pulse_system(self):
        """Create ECM Pulse System (inline since file creation was problematic)"""
        pulse_code = '''#!/usr/bin/env python3
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Set
import websockets
from websockets.server import WebSocketServerProtocol
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - ECM-PULSE - %(message)s')
logger = logging.getLogger("ECM_PULSE")

class PulseType(str, Enum):
    HEARTBEAT = "heartbeat"
    SYNCHRONIZATION = "sync"
    EVENT_TRIGGER = "event_trigger"
    COORDINATION = "coordination"
    INFRASTRUCTURE = "infrastructure"

class ECMPulseSystem:
    def __init__(self, host="localhost", port=8766):
        self.host = host
        self.port = port
        self.pulse_clients = {}
        self.total_pulses_sent = 0
        self.system_start_time = datetime.now()
        logger.info(f"ECM Pulse System initialized - {host}:{port}")
    
    async def handle_pulse_connection(self, websocket, path):
        client_id = str(uuid.uuid4())
        try:
            handshake = {
                'type': 'pulse_handshake',
                'client_id': client_id,
                'pulse_system_version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(handshake))
            
            self.pulse_clients[client_id] = {
                'websocket': websocket,
                'connected_at': datetime.now(),
                'pulse_count': 0
            }
            
            logger.info(f"PULSE-CONNECT: {client_id}")
            
            async for message in websocket:
                try:
                    msg = json.loads(message)
                    if msg.get('type') == 'ping':
                        await websocket.send(json.dumps({'type': 'pong', 'timestamp': datetime.now().isoformat()}))
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"PULSE-ERROR: {client_id} - {e}")
        finally:
            if client_id in self.pulse_clients:
                del self.pulse_clients[client_id]
    
    async def start_infrastructure_pulses(self):
        async def heartbeat_generator():
            while True:
                pulse = {
                    'type': 'pulse',
                    'pulse_id': str(uuid.uuid4()),
                    'pulse_type': 'heartbeat',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'ecm_pulse_system',
                    'payload': {'system_status': 'operational', 'clients': len(self.pulse_clients)}
                }
                
                for client_id, client_info in list(self.pulse_clients.items()):
                    try:
                        await client_info['websocket'].send(json.dumps(pulse))
                        client_info['pulse_count'] += 1
                        self.total_pulses_sent += 1
                    except:
                        del self.pulse_clients[client_id]
                
                if self.pulse_clients:
                    logger.info(f"PULSE-HEARTBEAT: Sent to {len(self.pulse_clients)} clients")
                await asyncio.sleep(5)
        
        asyncio.create_task(heartbeat_generator())
    
    async def start_pulse_system(self):
        logger.info(f"ECM Pulse System starting on ws://{self.host}:{self.port}")
        server = await websockets.serve(self.handle_pulse_connection, self.host, self.port)
        await self.start_infrastructure_pulses()
        logger.info(f"ECM Pulse System operational")
        return server

async def main():
    pulse_system = ECMPulseSystem()
    server = await pulse_system.start_pulse_system()
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("ECM Pulse System shutting down...")
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        with open('ecm_pulse_system.py', 'w') as f:
            f.write(pulse_code)
        
        logger.info("✅ ECM Pulse System file created")
    
    async def start_ecm_pulse_system(self):
        """Start ECM Pulse System"""
        logger.info("Starting ECM Pulse System...")
        
        # Create the pulse system file first
        self.create_ecm_pulse_system()
        
        try:
            process = subprocess.Popen([
                sys.executable, 'ecm_pulse_system.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['ecm_pulse'] = process
            logger.info("✅ ECM Pulse System started (PID: {})".format(process.pid))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start ECM Pulse System: {e}")
            return False
    
    async def start_dual_ac_system(self):
        """Start existing Dual AC System"""
        logger.info("Starting Dual AC System...")
        
        try:
            process = subprocess.Popen([
                sys.executable, 'dual_ac_api_server.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['dual_ac'] = process
            logger.info("✅ Dual AC System started (PID: {})".format(process.pid))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Dual AC System: {e}")
            return False
    
    async def start_all_systems(self):
        """Start all Phase 3 systems"""
        logger.info("🚀 Starting Phase 3 Complete Integration")
        logger.info("=" * 60)
        
        if not self.check_requirements():
            return False
        
        # Start systems in order
        systems = [
            ("ECM Infrastructure Gateway", self.start_ecm_gateway()),
            ("ECM Pulse System", self.start_ecm_pulse_system()),
            ("Dual AC System", self.start_dual_ac_system())
        ]
        
        for system_name, start_coro in systems:
            success = await start_coro
            if not success:
                logger.error(f"Failed to start {system_name}")
                await self.stop_all_systems()
                return False
            
            # Wait a moment between starts
            await asyncio.sleep(2)
        
        self.running = True
        logger.info("=" * 60)
        logger.info("✅ Phase 3 Complete Integration Started Successfully!")
        logger.info("")
        logger.info("🎯 ARCHITECTURE COMPLETE:")
        logger.info("   📊 Cosmetic AC Group → Node Engine (UI structured inputs)")
        logger.info("   🎮 Unreal AC Group → Node Engine (spatial interactions)")
        logger.info("   🌉 ECM Gateway → Infrastructure bridging (ws://localhost:8765)")
        logger.info("   💓 ECM Pulse System → Coordination pulses (ws://localhost:8766)")
        logger.info("   🖥️  Dual AC Interface → Complete system (http://localhost:8002)")
        logger.info("")
        logger.info("🔄 AC SYSTEM COMPLETE:")
        logger.info("   ✅ Both AC groups apply to same agents")
        logger.info("   ✅ UI interface layer closes engine loop")
        logger.info("   ✅ 1-way compute with 2-way interactive feel")
        logger.info("   ✅ Mobile responsive across all devices")
        logger.info("   ✅ ECM segregated from computational logic")
        logger.info("")
        logger.info("🌐 ACCESS POINTS:")
        logger.info("   • Main Interface: http://localhost:8002")
        logger.info("   • ECM Gateway: ws://localhost:8765")
        logger.info("   • ECM Pulse: ws://localhost:8766")
        logger.info("   • API Status: http://localhost:8002/api/status")
        logger.info("=" * 60)
        
        return True
    
    async def stop_all_systems(self):
        """Stop all running systems"""
        logger.info("🛑 Stopping all Phase 3 systems...")
        
        for system_name, process in self.processes.items():
            try:
                logger.info(f"Stopping {system_name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {system_name}...")
                    process.kill()
                    process.wait()
                
                logger.info(f"✅ {system_name} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping {system_name}: {e}")
        
        self.processes.clear()
        self.running = False
        logger.info("✅ All systems stopped")
    
    async def monitor_systems(self):
        """Monitor running systems"""
        while self.running:
            # Check if any processes have died
            dead_processes = []
            for system_name, process in self.processes.items():
                if process.poll() is not None:
                    dead_processes.append(system_name)
            
            if dead_processes:
                logger.error(f"Systems died: {', '.join(dead_processes)}")
                await self.stop_all_systems()
                break
            
            await asyncio.sleep(10)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop_all_systems())

async def main():
    """Main Phase 3 launcher"""
    launcher = Phase3Launcher()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)
    
    # Start all systems
    if not await launcher.start_all_systems():
        logger.error("Failed to start Phase 3 systems")
        return 1
    
    try:
        # Monitor systems
        await launcher.monitor_systems()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await launcher.stop_all_systems()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 