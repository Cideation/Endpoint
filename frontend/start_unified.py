#!/usr/bin/env python3
"""
Startup script for Neon Unified Interface
Demonstrates 1-Way Compute Architecture: Structured + Spatial → Node Engine → Controlled Emergence
"""

import uvicorn
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    print("🧠 Starting Neon Unified Interface - 1-Way Compute Architecture")
    print("=" * 70)
    print("")
    print("🎯 Architecture Overview:")
    print("   • Cosmetic UI (Left) → Structured inputs → Node Engine")
    print("   • Spatial Environment (Right) → Spatial actions → Node Engine")
    print("   • Node Engine → Controlled emergence → Both environments")
    print("")
    print("🚀 Interface URLs:")
    print("   • Unified Interface: http://localhost:8001")
    print("   • Legacy Interface: http://localhost:8001/legacy")
    print("   • API Documentation: http://localhost:8001/docs")
    print("   • Engine Status: http://localhost:8001/api/v2/engine/status")
    print("")
    print("🎮 Features:")
    print("   • Agent Sign-Up Form (Your original requirement)")
    print("   • 3D Spatial Environment with clickable components")
    print("   • Real-time Node Engine processing visualization")
    print("   • Unified event stream showing all inputs → emergence")
    print("   • WebSocket updates for real-time synchronization")
    print("")
    print("🧩 How to Test:")
    print("   1. Fill out Agent Sign-Up form → Watch emergence in 3D")
    print("   2. Click 3D components → See structured events generated")
    print("   3. Use spatial buttons → Observe Node Engine processing")
    print("   4. All inputs flow through same engine → Unified emergence")
    print("")
    print("✅ Ready for frontend user discussions!")
    print("   This demonstrates the 2-way interactive feel from 1-way compute!")
    print("")
    
    uvicorn.run(
        "unified_api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    ) 