#!/usr/bin/env python3
"""
Simple Tests for Global Design Parameters System v2.0
🧪 Quick validation of core functionality
"""

import sys
import os

# Add the shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MICROSERVICE_ENGINES'))

def test_basic_functionality():
    """Test basic GDP functionality"""
    print("🧪 Testing Global Design Parameters v2.0")
    print("=" * 50)
    
    try:
        # Import the module
        from shared.global_design_parameters import (
            global_design_parameters,
            enrich_node_by_type,
            enrich_node_full,
            get_parameters_by_category,
            get_node_mapping,
            validate_global_parameters
        )
        print("✅ Module import successful")
        
        # Test 1: Parameter loading
        assert isinstance(global_design_parameters, dict), "GDP should be a dictionary"
        assert len(global_design_parameters) > 0, "GDP should not be empty"
        print(f"✅ Parameter loading: {len(global_design_parameters)} parameters loaded")
        
        # Test 2: Required parameters exist
        required_params = [
            "form_strategy", "function_flexibility", "facade_style_id",
            "connectivity_validated", "layout_compactness", "aesthetic_tension",
            "precision_tolerance", "daylight_factor", "evolutionary_potential"
        ]
        
        missing_params = []
        for param in required_params:
            if param not in global_design_parameters:
                missing_params.append(param)
        
        assert len(missing_params) == 0, f"Missing required parameters: {missing_params}"
        print(f"✅ Required parameters: All {len(required_params)} parameters present")
        
        # Test 3: Parameter types
        assert isinstance(global_design_parameters["form_strategy"], str), "form_strategy should be string"
        assert isinstance(global_design_parameters["function_flexibility"], (int, float)), "function_flexibility should be numeric"
        assert isinstance(global_design_parameters["connectivity_validated"], bool), "connectivity_validated should be boolean"
        print("✅ Parameter types: Correct data types")
        
        # Test 4: Parameter ranges
        numeric_params = ["function_flexibility", "aesthetic_tension", "daylight_factor", "evolutionary_potential"]
        for param in numeric_params:
            value = global_design_parameters[param]
            assert 0.0 <= value <= 1.0, f"{param} should be between 0 and 1, got {value}"
        print("✅ Parameter ranges: Numeric values within valid ranges")
        
        # Test 5: Node enrichment by type
        test_node = {"id": "test_001", "type": "ProductComponent"}
        enriched_node = enrich_node_by_type(test_node, "V01_ProductComponent")
        
        assert enriched_node["id"] == "test_001", "Original data should be preserved"
        assert enriched_node.get("_gdp_enriched") == True, "Enrichment flag should be set"
        assert "form_strategy" in enriched_node, "Should contain form_strategy parameter"
        print("✅ Node enrichment by type: Working correctly")
        
        # Test 6: Full node enrichment
        full_enriched = enrich_node_full(test_node)
        assert full_enriched.get("_gdp_full_enrichment") == True, "Full enrichment flag should be set"
        assert len(full_enriched) > len(enriched_node), "Full enrichment should add more parameters"
        print("✅ Full node enrichment: Working correctly")
        
        # Test 7: Category-based parameters
        structural_params = get_parameters_by_category("structural")
        assert isinstance(structural_params, dict), "Category params should be dict"
        assert len(structural_params) > 0, "Structural category should have parameters"
        print(f"✅ Category-based parameters: {len(structural_params)} structural parameters")
        
        # Test 8: Node mapping
        mapping = get_node_mapping("V01_ProductComponent")
        assert isinstance(mapping, list), "Node mapping should be list"
        assert len(mapping) > 0, "ProductComponent should have mapped parameters"
        print(f"✅ Node mapping: {len(mapping)} parameters mapped to V01_ProductComponent")
        
        # Test 9: Parameter validation
        validation = validate_global_parameters()
        assert isinstance(validation, dict), "Validation should return dict"
        assert "valid" in validation, "Validation should have 'valid' key"
        assert validation["valid"] == True, "Parameters should be valid"
        print(f"✅ Parameter validation: Valid={validation['valid']}, Errors={len(validation['errors'])}")
        
        # Test 10: Multiple node types
        node_types = ["V01_ProductComponent", "V04_UserEconomicProfile", "V09_OccupancyNode"]
        mappings = [get_node_mapping(nt) for nt in node_types]
        
        # Should have different mappings
        assert mappings[0] != mappings[1], "Different node types should have different mappings"
        assert mappings[1] != mappings[2], "Different node types should have different mappings"
        print("✅ Multiple node types: Different mappings for different types")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance with multiple nodes"""
    print("\n⚡ Performance Test")
    print("-" * 30)
    
    try:
        from shared.global_design_parameters import enrich_node_by_type
        import time
        
        # Create 100 test nodes
        test_nodes = [{"id": f"perf_{i:03d}", "type": "TestComponent"} for i in range(100)]
        
        start_time = time.time()
        enriched_nodes = [enrich_node_by_type(node, "V01_ProductComponent") for node in test_nodes]
        end_time = time.time()
        
        processing_time = end_time - start_time
        nodes_per_second = len(test_nodes) / processing_time
        
        print(f"✅ Processed {len(test_nodes)} nodes in {processing_time:.3f}s")
        print(f"✅ Performance: {nodes_per_second:.1f} nodes/second")
        
        # Verify all nodes are enriched
        enriched_count = sum(1 for node in enriched_nodes if node.get("_gdp_enriched"))
        assert enriched_count == len(test_nodes), "All nodes should be enriched"
        print(f"✅ All {enriched_count} nodes successfully enriched")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_integration_pattern():
    """Test microservice integration pattern"""
    print("\n🔗 Integration Pattern Test")
    print("-" * 35)
    
    try:
        from shared.global_design_parameters import enrich_node_by_type
        
        # Simulate microservice request
        request_data = {
            "components": [
                {"id": "comp_001", "type": "ProductComponent", "name": "Steel Beam"},
                {"id": "comp_002", "type": "UserProfile", "name": "Economic Profile"},
                {"id": "comp_003", "type": "OccupancyNode", "name": "Office Space"}
            ]
        }
        
        # Simulate enrichment process
        components = request_data["components"]
        node_types = ["V01_ProductComponent", "V04_UserEconomicProfile", "V09_OccupancyNode"]
        
        enriched_components = []
        for i, component in enumerate(components):
            enriched = enrich_node_by_type(component.copy(), node_types[i])
            enriched_components.append(enriched)
        
        # Verify results
        assert len(enriched_components) == 3, "Should have 3 enriched components"
        
        for enriched in enriched_components:
            assert enriched.get("_gdp_enriched") == True, "All should be enriched"
            assert "_gdp_node_type" in enriched, "Should have node type metadata"
            assert "_gdp_params_applied" in enriched, "Should have applied params metadata"
        
        print(f"✅ Enriched {len(enriched_components)} components")
        print("✅ Integration pattern working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🏗️ Global Design Parameters v2.0 - Test Suite")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Performance", test_performance),
        ("Integration Pattern", test_integration_pattern)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        if test_func():
            passed += 1
    
    # Final report
    print("\n" + "=" * 60)
    print("📊 Final Test Report")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Global Design Parameters v2.0 is ready for production!")
    else:
        print("❌ Some tests failed. Please review the errors above.")
        sys.exit(1)
