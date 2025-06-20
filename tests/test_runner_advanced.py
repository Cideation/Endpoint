#!/usr/bin/env python3
"""
Advanced Test Runner
Runs all 5 advanced graph tests in sequence
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run all advanced graph tests"""
    print("🧠 Starting Advanced Graph Test Suite")
    print("=" * 60)
    
    advanced_tests = [
        ("test_full_graph_pass.py", "Full Graph Pass Test"),
        ("test_edge_callback_logic.py", "Edge Callback Logic Test"),
        ("test_emergent_values.py", "Emergent Values Test"),
        ("test_agent_impact.py", "Agent Impact Test"),
        ("test_trace_path_index.py", "Trace Path Index Test")
    ]
    
    total_failures = 0
    
    for test_file, test_name in advanced_tests:
        print(f"\n🚀 Running {test_name}")
        print("-" * 40)
        
        try:
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED ({result.returncode} failures)")
                total_failures += result.returncode
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            total_failures += 1
    
    print("\n" + "=" * 60)
    if total_failures == 0:
        print("🎉 ALL ADVANCED TESTS PASSED!")
    else:
        print(f"⚠️  {total_failures} TEST FAILURES")
    
    sys.exit(total_failures)

if __name__ == "__main__":
    main()
