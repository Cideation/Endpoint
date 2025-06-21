#!/usr/bin/env python3
"""
BEM Stress Test Runner
Easy command line interface for running the stress testing system
"""

import asyncio
import sys
import argparse
from stress_test_engine import StressTestEngine

async def run_quick_test():
    """Run a quick 3-test demo"""
    print("🚀 Quick Stress Test Demo (3 tests)")
    engine = StressTestEngine()
    results = await engine.run_random_test_suite(num_tests=3)
    engine.save_results(results, "quick_test_results.json")
    return results

async def main():
    parser = argparse.ArgumentParser(description='BEM Stress Test Runner')
    parser.add_argument('--mode', '-m', 
                       choices=['quick', 'full', 'manual', 'edge', 'stress'],
                       default='quick',
                       help='Test mode to run')
    
    args = parser.parse_args()
    
    print(f"🎯 BEM Stress Test Runner")
    print(f"Mode: {args.mode}")
    print("-" * 40)
    
    results = await run_quick_test()
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['tests_run']}")
    print(f"Successful: {results['successful_tests']} ✅")
    print(f"Failed: {results['failed_tests']} ❌")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
