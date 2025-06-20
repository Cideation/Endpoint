#!/usr/bin/env python3
"""
Focused Test Runner
Simple script to execute the 5 focused validation tests
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the focused validation tests"""
    print("🎯 Focused Validation Test Runner")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if test file exists
    test_file = "test_focused_validation.py"
    if not Path(test_file).exists():
        print(f"❌ Test file {test_file} not found!")
        sys.exit(1)
    
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🧪 Running test file: {test_file}")
    print()
    
    try:
        # Run the focused tests
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=False, text=True)
        
        print()
        print("=" * 50)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ {result.returncode} test(s) failed!")
        
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
