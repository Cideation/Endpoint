#!/usr/bin/env python3
"""
APM Demo Runner
Simple script to run the APM monitoring demo
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from apm_demo import main
    
    if __name__ == '__main__':
        print("🚀 Starting APM Monitoring Demo...")
        print("📋 This demo showcases:")
        print("   • Real-time performance monitoring")
        print("   • Multi-provider APM integration")
        print("   • Performance profiling")
        print("   • Interactive web dashboard")
        print("   • Custom business metrics")
        print("   • Error tracking and alerting")
        print()
        
        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please install required dependencies:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"❌ Error running demo: {e}")
    sys.exit(1)
