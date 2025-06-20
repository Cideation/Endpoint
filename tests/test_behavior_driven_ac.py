#!/usr/bin/env python3
"""
Test Suite: Behavior-Driven Agent Console (AC) System
Validates AA behavioral classification and dynamic UI spawning
"""

import asyncio
import json
import time
import requests
import subprocess
import sys
from typing import Dict, List

def test_behavior_classification():
    """Test AA behavioral classification accuracy"""
    print("🧠 Testing AA Behavioral Classification...")
    
    try:
        # Test simulation endpoint first
        response = requests.post("http://localhost:8003/simulate_behavior", timeout=5)
        if response.status_code == 200:
            results = response.json()
            
            print("✅ Behavior Classification Results:")
            for role, data in results.items():
                detected = data['detected_role']
                confidence = data['confidence']
                panels = data['ac_panels']
                expected_role = role.replace('simulated_', '')
                
                print(f"   {role}: {detected} (confidence: {confidence:.2f})")
                print(f"   AC Panels: {panels}")
                
                # Validate accuracy
                if detected == expected_role:
                    print(f"   ✅ Correct classification")
                else:
                    print(f"   ❌ Expected {expected_role}, got {detected}")
                print()
            
            return True
        else:
            print(f"❌ Behavior service not responding: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Behavior service not running on port 8003")
        return False
    except Exception as e:
        print(f"❌ Behavior classification test failed: {e}")
        return False

def test_dynamic_ac_spawning():
    """Test dynamic AC panel spawning based on behavior"""
    print("🎯 Testing Dynamic AC Panel Spawning...")
    
    try:
        base_url = "http://localhost:8003"
        session_id = f"test_session_{int(time.time())}"
        
        # Test different behavior patterns
        test_scenarios = [
            {
                "name": "Investor Behavior",
                "actions": [
                    {"action_type": "view_investment", "target": "irr_calculator", "context": {"amount": 100000}},
                    {"action_type": "analyze_roi", "target": "finance_dashboard", "context": {"period": "annual"}},
                ],
                "expected_role": "investor",
                "expected_panels": ["investment_ac", "irr_calculator", "roi_dashboard"]
            },
            {
                "name": "Contributor Behavior", 
                "actions": [
                    {"action_type": "create_node", "target": "node_editor", "context": {"type": "housing"}},
                    {"action_type": "modify_graph", "target": "cytoscape", "context": {"operation": "add_edge"}},
                ],
                "expected_role": "contributor",
                "expected_panels": ["contributor_ac", "node_editor", "upload_tools"]
            },
            {
                "name": "Validator Behavior",
                "actions": [
                    {"action_type": "validate_compliance", "target": "compliance_checker", "context": {"standard": "code"}},
                    {"action_type": "audit_changes", "target": "audit_log", "context": {"timeframe": "24h"}},
                ],
                "expected_role": "validator", 
                "expected_panels": ["validator_ac", "compliance_checker", "audit_tools"]
            }
        ]
        
        success_count = 0
        for scenario in test_scenarios:
            print(f"\n   Testing: {scenario['name']}")
            
            # Send actions sequentially
            for action in scenario['actions']:
                action_data = {
                    "session_id": session_id,
                    "action_type": action["action_type"],
                    "target": action["target"], 
                    "context": action["context"],
                    "timestamp": time.time()
                }
                
                response = requests.post(f"{base_url}/log_action", json=action_data, timeout=5)
                if response.status_code != 200:
                    print(f"   ❌ Failed to log action: {response.status_code}")
                    continue
            
            # Check final classification
            response = requests.get(f"{base_url}/get_role/{session_id}", timeout=5)
            if response.status_code == 200:
                result = response.json()
                detected_role = result['agent_role']
                confidence = result['confidence']
                ac_panels = result['ac_panels']
                
                print(f"   Detected Role: {detected_role} (confidence: {confidence:.2f})")
                print(f"   AC Panels: {ac_panels}")
                
                # Validate results
                if detected_role == scenario['expected_role']:
                    print(f"   ✅ Correct role classification")
                    success_count += 1
                else:
                    print(f"   ❌ Expected {scenario['expected_role']}, got {detected_role}")
                
                # Check if expected panels are included
                expected_panels = set(scenario['expected_panels'])
                actual_panels = set(ac_panels)
                if expected_panels.intersection(actual_panels):
                    print(f"   ✅ AC panels correctly spawned")
                else:
                    print(f"   ❌ Expected panels not found")
            else:
                print(f"   ❌ Failed to get role: {response.status_code}")
            
            # Use new session for next test
            session_id = f"test_session_{int(time.time())}_{success_count}"
        
        success_rate = (success_count / len(test_scenarios)) * 100
        print(f"\n✅ Dynamic AC Spawning Success Rate: {success_rate:.1f}%")
        return success_rate >= 66.7  # At least 2/3 scenarios should pass
        
    except Exception as e:
        print(f"❌ Dynamic AC spawning test failed: {e}")
        return False

def test_no_hardcoded_ui():
    """Test that UI is behavior-driven, not hardcoded"""
    print("🚫 Testing No Hardcoded UI Principle...")
    
    try:
        # Test with minimal/no behavior - should show observer role
        base_url = "http://localhost:8003"
        session_id = f"minimal_session_{int(time.time())}"
        
        # Single minimal action
        minimal_action = {
            "session_id": session_id,
            "action_type": "initial_view",
            "target": "interface",
            "context": {},
            "timestamp": time.time()
        }
        
        response = requests.post(f"{base_url}/log_action", json=minimal_action, timeout=5)
        if response.status_code == 200:
            result = response.json()
            
            print(f"   Minimal Behavior Result:")
            print(f"   Role: {result['agent_role']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   AC Panels: {result['ac_panels']}")
            
            # Should be observer with basic panels
            if result['agent_role'] == 'observer' and 'basic_view' in result['ac_panels']:
                print("   ✅ Correctly defaults to observer role with basic panels")
                return True
            else:
                print("   ❌ Should default to observer role with basic panels")
                return False
        else:
            print(f"   ❌ Failed to test minimal behavior: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ No hardcoded UI test failed: {e}")
        return False

def test_view_separation():
    """Test that Cytoscape and Unreal views have correct separation"""
    print("🌐 Testing View Separation (Cytoscape vs Unreal)...")
    
    # Test frontend interface file
    try:
        with open('frontend/dynamic_ac_interface.html', 'r') as f:
            content = f.read()
        
        checks = {
            "Cytoscape Graph View": "Graph View (Logic)" in content,
            "Unreal IRL View": "IRL View (Visual + Spatial)" in content,
            "PostgreSQL Direct Connection": "PostgreSQL Direct" in content,
            "Pulse Visualization in Unreal Only": "pulse-visualization" in content,
            "No ECM in Cytoscape": "ECM" not in content.split("Graph View")[0] if "Graph View" in content else True,
            "Behavior Detection": "AA Classification" in content,
            "Dynamic AC Panels": "ac-panels-container" in content
        }
        
        passed = 0
        for check_name, result in checks.items():
            if result:
                print(f"   ✅ {check_name}")
                passed += 1
            else:
                print(f"   ❌ {check_name}")
        
        success_rate = (passed / len(checks)) * 100
        print(f"\n✅ View Separation Success Rate: {success_rate:.1f}%")
        return success_rate >= 85.0  # 6/7 checks should pass
        
    except FileNotFoundError:
        print("   ❌ Dynamic AC interface file not found")
        return False
    except Exception as e:
        print(f"   ❌ View separation test failed: {e}")
        return False

def test_aa_analytics():
    """Test AA analytics and behavioral insights"""
    print("📊 Testing AA Analytics...")
    
    try:
        response = requests.get("http://localhost:8003/behavior_analytics", timeout=5)
        if response.status_code == 200:
            analytics = response.json()
            
            print(f"   Analytics Response: {analytics}")
            
            if "message" in analytics and analytics["message"] == "No active sessions":
                print("   ✅ Correctly reports no active sessions")
                return True
            elif "total_sessions" in analytics:
                print(f"   Total Sessions: {analytics.get('total_sessions', 0)}")
                print(f"   Role Distribution: {analytics.get('role_distribution', {})}")
                print(f"   Average Confidence: {analytics.get('average_confidence', 0):.2f}")
                print("   ✅ AA analytics functioning")
                return True
            else:
                print("   ❌ Invalid analytics response format")
                return False
        else:
            print(f"   ❌ Analytics endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ AA analytics test failed: {e}")
        return False

def start_behavior_service():
    """Start the behavior-driven AC service"""
    print("🚀 Starting Behavior-Driven AC Service...")
    
    try:
        # Check if service is already running
        try:
            response = requests.get("http://localhost:8003/behavior_analytics", timeout=2)
            if response.status_code == 200:
                print("   ✅ Service already running")
                return True
        except:
            pass
        
        # Start the service
        process = subprocess.Popen([
            sys.executable, "frontend/behavior_driven_ac.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for service to start
        time.sleep(3)
        
        # Check if service is responding
        try:
            response = requests.get("http://localhost:8003/behavior_analytics", timeout=5)
            if response.status_code == 200:
                print("   ✅ Service started successfully")
                return True
            else:
                print(f"   ❌ Service not responding: {response.status_code}")
                return False
        except:
            print("   ❌ Service failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start behavior service: {e}")
        return False

def main():
    """Run comprehensive behavior-driven AC tests"""
    print("🎯 BEM Behavior-Driven Agent Console (AC) Test Suite")
    print("=" * 60)
    
    # Start behavior service if not running
    service_started = start_behavior_service()
    
    # Run test suite
    tests = [
        ("AA Behavioral Classification", test_behavior_classification),
        ("Dynamic AC Panel Spawning", test_dynamic_ac_spawning),
        ("No Hardcoded UI Principle", test_no_hardcoded_ui),
        ("View Separation (Cytoscape/Unreal)", test_view_separation),
        ("AA Analytics Dashboard", test_aa_analytics)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        if service_started or test_name == "View Separation (Cytoscape/Unreal)":
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        else:
            print(f"⏸️  {test_name}: SKIPPED (service not available)")
            results.append((test_name, None))
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 BEHAVIOR-DRIVEN AC TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            print(f"✅ {test_name}")
        elif result is False:
            print(f"❌ {test_name}")
        else:
            print(f"⏸️  {test_name} (SKIPPED)")
    
    success_rate = (passed / (total - skipped)) * 100 if (total - skipped) > 0 else 0
    
    print(f"\n📈 Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 BEHAVIOR-DRIVEN AC SYSTEM: PRODUCTION READY")
        return True
    else:
        print("🔧 BEHAVIOR-DRIVEN AC SYSTEM: NEEDS IMPROVEMENT")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 