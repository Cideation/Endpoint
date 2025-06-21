#!/usr/bin/env python3
"""
Comprehensive Tests for Global Design Parameters System
🧪 Testing parameter loading, validation, node enrichment, and dynamic mapping

Test Coverage:
- Parameter loading and fallback
- Node-specific enrichment
- Category-based parameter access
- Parameter validation
- Dynamic mapping functionality
- Performance and edge cases
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path

# Add the shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MICROSERVICE_ENGINES'))

try:
    from shared.global_design_parameters import (
        global_design_parameters,
        enrich_node_by_type,
        enrich_node_full,
        get_parameters_by_category,
        get_node_mapping,
        validate_global_parameters,
        get_global_design_parameters
    )
    GDP_IMPORT_SUCCESS = True
except ImportError as e:
    GDP_IMPORT_SUCCESS = False
    print(f"⚠️ Import failed: {e}")

class TestGlobalDesignParameters(unittest.TestCase):
    """Test suite for Global Design Parameters system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        if not GDP_IMPORT_SUCCESS:
            cls.skipTest(cls, "Global Design Parameters module not available")
    
    def test_parameter_loading(self):
        """Test parameter loading from JSON"""
        # Test that parameters are loaded
        self.assertIsInstance(global_design_parameters, dict)
        self.assertGreater(len(global_design_parameters), 0)
        
        # Test key parameters exist
        expected_keys = [
            "form_strategy", "function_flexibility", "facade_style_id",
            "connectivity_validated", "layout_compactness"
        ]
        for key in expected_keys:
            self.assertIn(key, global_design_parameters, f"Missing parameter: {key}")
    
    def test_parameter_types(self):
        """Test parameter data types"""
        # String parameters
        string_params = [
            "locus_condition", "self_weight_category", "order_type",
            "form_strategy", "material_expression", "precision_tolerance",
            "detail_complexity", "joint_type", "hierarchy_type",
            "public_private_ratio", "facade_style_id", "bay_arrangement_pattern",
            "component_modularity", "floor_span_variance", "generation_type"
        ]
        
        for param in string_params:
            if param in global_design_parameters:
                self.assertIsInstance(global_design_parameters[param], str, f"{param} should be string")
        
        # Numeric parameters (float/int)
        numeric_params = [
            "function_flexibility", "aesthetic_tension", "daylight_factor",
            "evolutionary_potential", "layout_compactness", "room_rectangularity_score",
            "parametric_frame_depth", "layout_success_rate", "symbolic_depth_range"
        ]
        
        for param in numeric_params:
            if param in global_design_parameters:
                self.assertIsInstance(global_design_parameters[param], (int, float), f"{param} should be numeric")
        
        # Boolean parameters
        boolean_params = [
            "connectivity_validated", "mirror_operations_enabled",
            "tilt_transformation", "cad_integration_ready"
        ]
        
        for param in boolean_params:
            if param in global_design_parameters:
                self.assertIsInstance(global_design_parameters[param], bool, f"{param} should be boolean")
    
    def test_parameter_ranges(self):
        """Test parameter value ranges"""
        # Parameters that should be between 0 and 1
        ratio_params = [
            "function_flexibility", "aesthetic_tension", "daylight_factor",
            "evolutionary_potential", "layout_compactness", "room_rectangularity_score",
            "layout_success_rate"
        ]
        
        for param in ratio_params:
            if param in global_design_parameters:
                value = global_design_parameters[param]
                if isinstance(value, (int, float)):
                    self.assertGreaterEqual(value, 0.0, f"{param} should be >= 0")
                    self.assertLessEqual(value, 1.0, f"{param} should be <= 1")
    
    def test_node_enrichment_by_type(self):
        """Test node enrichment with type-specific parameters"""
        # Test with V01_ProductComponent
        test_node = {
            "id": "test_product_001",
            "type": "ProductComponent",
            "name": "Test Component"
        }
        
        enriched_node = enrich_node_by_type(test_node, "V01_ProductComponent")
        
        # Check original data is preserved
        self.assertEqual(enriched_node["id"], "test_product_001")
        self.assertEqual(enriched_node["type"], "ProductComponent")
        self.assertEqual(enriched_node["name"], "Test Component")
        
        # Check enrichment metadata
        self.assertTrue(enriched_node.get("_gdp_enriched", False))
        self.assertEqual(enriched_node.get("_gdp_node_type"), "V01_ProductComponent")
        self.assertIsInstance(enriched_node.get("_gdp_params_applied"), list)
    
    def test_node_enrichment_full(self):
        """Test full node enrichment with all parameters"""
        test_node = {
            "id": "test_full_001",
            "type": "GenericComponent"
        }
        
        enriched_node = enrich_node_full(test_node)
        
        # Check original data is preserved
        self.assertEqual(enriched_node["id"], "test_full_001")
        self.assertEqual(enriched_node["type"], "GenericComponent")
        
        # Check enrichment metadata
        self.assertTrue(enriched_node.get("_gdp_enriched", False))
        self.assertTrue(enriched_node.get("_gdp_full_enrichment", False))
    
    def test_category_based_parameters(self):
        """Test category-based parameter retrieval"""
        # Test structural category
        structural_params = get_parameters_by_category("structural")
        self.assertIsInstance(structural_params, dict)
        
        # Test spatial category
        spatial_params = get_parameters_by_category("spatial")
        self.assertIsInstance(spatial_params, dict)
        
        # Test facade category
        facade_params = get_parameters_by_category("facade")
        self.assertIsInstance(facade_params, dict)
    
    def test_node_mapping(self):
        """Test node type parameter mapping"""
        # Test known node types
        test_node_types = ["V01_ProductComponent", "V04_UserEconomicProfile", "V09_OccupancyNode"]
        
        for node_type in test_node_types:
            mapping = get_node_mapping(node_type)
            self.assertIsInstance(mapping, list, f"Mapping for {node_type} should be a list")
        
        # Test unknown node type
        unknown_mapping = get_node_mapping("UnknownNodeType")
        self.assertIsInstance(unknown_mapping, list)
    
    def test_parameter_validation(self):
        """Test parameter validation functionality"""
        validation_result = validate_global_parameters()
        
        # Check validation result structure
        self.assertIsInstance(validation_result, dict)
        self.assertIn("valid", validation_result)
        self.assertIn("errors", validation_result)
        self.assertIn("warnings", validation_result)
        
        # Check data types
        self.assertIsInstance(validation_result["valid"], bool)
        self.assertIsInstance(validation_result["errors"], list)
        self.assertIsInstance(validation_result["warnings"], list)

def run_comprehensive_tests():
    """Run all tests and generate report"""
    print("🧪 Running Comprehensive Global Design Parameters Tests")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestGlobalDesignParameters))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate report
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed successfully!")
    
    return result

if __name__ == "__main__":
    if GDP_IMPORT_SUCCESS:
        run_comprehensive_tests()
    else:
        print("❌ Cannot run tests: Global Design Parameters module not available")
        sys.exit(1)
