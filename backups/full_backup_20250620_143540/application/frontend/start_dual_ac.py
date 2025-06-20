#!/usr/bin/env python3
"""
Enhanced Dual Agent Coefficient System Startup Script

Launches the dual AC system with:
- Cosmetic AC Group (UI inputs)
- Unreal AC Group (spatial actions)
- Unified Node Engine processing
- Real-time WebSocket updates
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'websockets',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install fastapi uvicorn websockets pydantic")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'dual_ac_api_server.py',
        'enhanced_unified_interface.html'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    return True

def start_dual_ac_system():
    """Start the enhanced dual AC system"""
    print("🎯 Starting Enhanced Dual Agent Coefficient System")
    print("=" * 60)
    print("📊 Cosmetic AC Group: Structured UI inputs")
    print("   - Budget sliders, investment levels")
    print("   - Component priorities, timeline controls")
    print("   - Agent registration forms")
    print()
    print("🎮 Unreal AC Group: Spatial environment actions")
    print("   - Wall selections, room placement")
    print("   - Zone interactions, geometry measurements")
    print("   - Component property analysis")
    print()
    print("🧠 Unified Node Engine:")
    print("   - Processes both AC types through same pipeline")
    print("   - 1-way compute with 2-way interactive feel")
    print("   - Real-time coefficient tracking and emergence")
    print()
    print("🌐 Interface Access:")
    print("   - Main Interface: http://localhost:8002")
    print("   - API Status: http://localhost:8002/api/status")
    print("   - Live Coefficients: http://localhost:8002/api/coefficients")
    print("=" * 60)
    
    # Start the server
    try:
        process = subprocess.Popen([
            sys.executable, 'dual_ac_api_server.py'
        ], cwd=Path(__file__).parent)
        
        print("✅ Dual AC System launched successfully!")
        print("🔄 Processing both Cosmetic and Unreal Agent Coefficients")
        print("📡 WebSocket connections available for real-time updates")
        print("\n⏹️  Press Ctrl+C to stop the system")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Dual AC System...")
        process.terminate()
        process.wait()
        print("✅ System shutdown complete")
    
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        return False
    
    return True

def show_usage():
    """Show usage information"""
    print("Enhanced Dual Agent Coefficient System")
    print("=" * 50)
    print()
    print("🎯 ARCHITECTURE OVERVIEW:")
    print("   Cosmetic AC Group → ")
    print("   Unreal AC Group   → Node Engine → Controlled Emergence")
    print()
    print("📊 COSMETIC AC (Structured Inputs):")
    print("   • Budget & Investment sliders")
    print("   • Quality & Timeline controls")
    print("   • Component priority dropdowns")
    print("   • Agent registration forms")
    print()
    print("🎮 UNREAL AC (Spatial Actions):")
    print("   • Wall/component selections")
    print("   • Room placement actions")
    print("   • Zone touch interactions")
    print("   • Geometry measurements")
    print()
    print("🧠 NODE ENGINE FEATURES:")
    print("   • Unified coefficient processing")
    print("   • 10 Agent class system")
    print("   • Real-time metrics & emergence")
    print("   • WebSocket live updates")
    print()
    print("🚀 USAGE:")
    print("   python start_dual_ac.py")
    print()
    print("🌐 ENDPOINTS:")
    print("   http://localhost:8002/ - Main Interface")
    print("   http://localhost:8002/api/status - Engine Status")
    print("   http://localhost:8002/api/coefficients - Active AC")
    print("   ws://localhost:8002/ws - WebSocket Updates")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_usage()
        return
    
    print("🎯 Enhanced Dual Agent Coefficient System")
    print("🔍 Checking system requirements...")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check files
    if not check_files():
        print("💡 Please ensure all required files are in the frontend directory")
        sys.exit(1)
    
    print("✅ All requirements satisfied")
    print("🚀 Launching dual AC system...")
    
    # Start the system
    if not start_dual_ac_system():
        sys.exit(1)

if __name__ == "__main__":
    main() 