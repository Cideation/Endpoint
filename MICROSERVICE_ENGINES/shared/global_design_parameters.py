#!/usr/bin/env python3
"""
Global Design Parameters - Shared Resource v2.0
🏗️ Node-level shared variables for enriching components during parsing, scoring, and UI rendering
Enhanced with dynamic node mapping and flattened parameter structure

Usage:
    from shared.global_design_parameters import global_design_parameters
    
    # Direct access to flattened parameters
    form_strategy = global_design_parameters['form_strategy']
    
    # Dynamic node-specific enrichment
    enriched_node = enrich_node_by_type(node, 'V01_ProductComponent')
    
    # Category-based parameter application
    structural_params = get_parameters_by_category('structural')
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalDesignParameters:
    """Enhanced singleton class for managing global design parameters with dynamic mapping"""
    
    _instance = None
    _parameters = None
    _node_mapping = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalDesignParameters, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._parameters is None:
            self._load_parameters()
            self._load_node_mapping()
    
    def _load_parameters(self):
        """Load global design parameters from JSON file"""
        try:
            # Try multiple possible paths
            possible_paths = [
                Path(__file__).parent.parent / "global_design_parameters.json",
                Path("MICROSERVICE_ENGINES/global_design_parameters.json"),
                Path("global_design_parameters.json")
            ]
            
            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError("global_design_parameters.json not found")
            
            with open(config_path, 'r') as f:
                data = json.load(f)
                self._parameters = data.get('global_design_parameters', {})
            
            logger.info(f"✅ Loaded global design parameters from {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load global design parameters: {e}")
            # Fallback to hardcoded parameters
            self._parameters = self._get_fallback_parameters()
    
    def _load_node_mapping(self):
        """Load node parameter mapping from JSON file"""
        try:
            # Try multiple possible paths for node mapping
            possible_paths = [
                Path(__file__).parent.parent / "node_parameter_mapping.json",
                Path("MICROSERVICE_ENGINES/node_parameter_mapping.json"),
                Path("node_parameter_mapping.json")
            ]
            
            mapping_path = None
            for path in possible_paths:
                if path.exists():
                    mapping_path = path
                    break
            
            if mapping_path is None:
                raise FileNotFoundError("node_parameter_mapping.json not found")
            
            with open(mapping_path, 'r') as f:
                data = json.load(f)
                self._node_mapping = data.get('node_parameter_mapping', {})
            
            logger.info(f"✅ Loaded node parameter mapping from {mapping_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load node parameter mapping: {e}")
            # Fallback to basic mapping
            self._node_mapping = self._get_fallback_mapping()
    
    def _get_fallback_parameters(self) -> Dict[str, Any]:
        """Fallback parameters if JSON file is not available"""
        return {
            "locus_condition": "context_driven",
            "self_weight_category": "lightweight",
            "order_type": "modular_grid",
            "form_strategy": "evolutionary_adaptive",
            "function_flexibility": 0.85,
            "material_expression": "symbolic_natural",
            "aesthetic_tension": 0.6,
            "precision_tolerance": "high",
            "detail_complexity": "prefab_ready",
            "joint_type": "slip_fit",
            "daylight_factor": 0.73,
            "evolutionary_potential": 0.92,
            "connectivity_validated": True,
            "symbolic_depth_range": 5,
            "layout_compactness": 0.72,
            "hierarchy_type": "heterogeneous",
            "mirror_operations_enabled": True,
            "layout_success_rate": 0.99,
            "room_rectangularity_score": 0.84,
            "public_private_ratio": "1:4",
            "facade_style_id": "style_2_bay_variation",
            "parametric_frame_depth": 0.3,
            "bay_arrangement_pattern": "alternating_even_odd",
            "component_modularity": "enabled",
            "floor_span_variance": "medium",
            "tilt_transformation": False,
            "generation_type": "rule_driven",
            "cad_integration_ready": True
        }
    
    def _get_fallback_mapping(self) -> Dict[str, List[str]]:
        """Fallback node mapping if JSON file is not available"""
        return {
            "V01_ProductComponent": ["form_strategy", "precision_tolerance"],
            "V04_UserEconomicProfile": ["function_flexibility", "aesthetic_tension"],
            "V09_OccupancyNode": ["public_private_ratio", "layout_compactness"]
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value by key"""
        return self._parameters.get(key, default)
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all global design parameters"""
        return self._parameters.copy()
    
    def get_node_mapping(self, node_type: str) -> List[str]:
        """Get parameter mapping for specific node type"""
        return self._node_mapping.get(node_type, [])
    
    def enrich_node_by_type(self, node: Dict[str, Any], node_type: str) -> Dict[str, Any]:
        """Enrich a node with parameters specific to its type"""
        enriched_node = node.copy()
        
        # Get parameters for this node type
        relevant_params = self.get_node_mapping(node_type)
        
        # Apply only relevant parameters
        for param_key in relevant_params:
            if param_key in self._parameters:
                enriched_node[param_key] = self._parameters[param_key]
        
        # Add enrichment metadata
        enriched_node['_gdp_enriched'] = True
        enriched_node['_gdp_node_type'] = node_type
        enriched_node['_gdp_params_applied'] = relevant_params
        
        return enriched_node
    
    def enrich_node_full(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a node with all global design parameters"""
        enriched_node = node.copy()
        enriched_node.update(self._parameters)
        enriched_node['_gdp_enriched'] = True
        enriched_node['_gdp_full_enrichment'] = True
        return enriched_node
    
    def get_parameters_by_category(self, category: str) -> Dict[str, Any]:
        """Get parameters by category (structural, spatial, facade, etc.)"""
        # This would be enhanced with category mapping from node_parameter_mapping.json
        category_params = {}
        
        # Basic category mapping (can be enhanced)
        category_mapping = {
            "structural": ["self_weight_category", "precision_tolerance", "joint_type", "material_expression"],
            "spatial": ["locus_condition", "layout_compactness", "room_rectangularity_score", "daylight_factor"],
            "facade": ["facade_style_id", "parametric_frame_depth", "bay_arrangement_pattern", "aesthetic_tension"],
            "manufacturing": ["precision_tolerance", "detail_complexity", "generation_type", "cad_integration_ready"],
            "layout": ["connectivity_validated", "layout_compactness", "hierarchy_type", "mirror_operations_enabled"]
        }
        
        param_keys = category_mapping.get(category, [])
        for key in param_keys:
            if key in self._parameters:
                category_params[key] = self._parameters[key]
        
        return category_params
    
    def validate_parameters(self) -> Dict[str, Any]:
        """Validate parameter values and types"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Type validation
        expected_types = {
            "function_flexibility": (float, int),
            "aesthetic_tension": (float, int),
            "daylight_factor": (float, int),
            "evolutionary_potential": (float, int),
            "layout_compactness": (float, int),
            "room_rectangularity_score": (float, int),
            "parametric_frame_depth": (float, int),
            "connectivity_validated": bool,
            "mirror_operations_enabled": bool,
            "tilt_transformation": bool,
            "cad_integration_ready": bool
        }
        
        for param, expected_type in expected_types.items():
            if param in self._parameters:
                if not isinstance(self._parameters[param], expected_type):
                    validation_results["errors"].append(f"{param}: expected {expected_type}, got {type(self._parameters[param])}")
                    validation_results["valid"] = False
        
        # Range validation for numeric parameters
        numeric_ranges = {
            "function_flexibility": (0.0, 1.0),
            "aesthetic_tension": (0.0, 1.0),
            "daylight_factor": (0.0, 1.0),
            "evolutionary_potential": (0.0, 1.0),
            "layout_compactness": (0.0, 1.0),
            "room_rectangularity_score": (0.0, 1.0),
            "parametric_frame_depth": (0.0, 2.0),
            "layout_success_rate": (0.0, 1.0)
        }
        
        for param, (min_val, max_val) in numeric_ranges.items():
            if param in self._parameters:
                value = self._parameters[param]
                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        validation_results["warnings"].append(f"{param}: value {value} outside recommended range [{min_val}, {max_val}]")
        
        return validation_results
    
    def reload(self):
        """Reload parameters from files"""
        self._parameters = None
        self._node_mapping = None
        self._load_parameters()
        self._load_node_mapping()

# Create singleton instance
_gdp_instance = GlobalDesignParameters()

# Export the global design parameters dictionary (flattened)
global_design_parameters = _gdp_instance._parameters

# Export helper functions
def get_global_design_parameters() -> Dict[str, Any]:
    """Get the complete global design parameters dictionary"""
    return _gdp_instance.get_all_parameters()

def enrich_node_by_type(node: Dict[str, Any], node_type: str) -> Dict[str, Any]:
    """Enrich a node with parameters specific to its type"""
    return _gdp_instance.enrich_node_by_type(node, node_type)

def enrich_node_full(node: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a node with all global design parameters"""
    return _gdp_instance.enrich_node_full(node)

def get_parameters_by_category(category: str) -> Dict[str, Any]:
    """Get parameters by category"""
    return _gdp_instance.get_parameters_by_category(category)

def get_node_mapping(node_type: str) -> List[str]:
    """Get parameter mapping for specific node type"""
    return _gdp_instance.get_node_mapping(node_type)

def validate_global_parameters() -> Dict[str, Any]:
    """Validate global design parameters"""
    return _gdp_instance.validate_parameters()

def reload_global_parameters():
    """Reload global parameters from files"""
    _gdp_instance.reload()
    global global_design_parameters
    global_design_parameters = _gdp_instance._parameters

# Backwards compatibility
def enrich_node_with_global_params(node: Dict[str, Any], category: str = None) -> Dict[str, Any]:
    """Legacy function - now uses full enrichment"""
    return enrich_node_full(node)

# Export for backwards compatibility
GDP = _gdp_instance

if __name__ == "__main__":
    # Test the module
    print("🏗️ Global Design Parameters v2.0 Test")
    print("=" * 60)
    
    print(f"\n📊 Total Parameters: {len(global_design_parameters)}")
    
    print("\n🔧 Sample Parameters:")
    sample_params = ["form_strategy", "function_flexibility", "facade_style_id"]
    for param in sample_params:
        print(f"  {param}: {global_design_parameters.get(param, 'NOT FOUND')}")
    
    print("\n🎯 Node Type Mapping Test:")
    test_node_types = ["V01_ProductComponent", "V04_UserEconomicProfile", "V09_OccupancyNode"]
    for node_type in test_node_types:
        mapping = get_node_mapping(node_type)
        print(f"  {node_type}: {len(mapping)} parameters -> {mapping[:3]}...")
    
    print("\n🧪 Node Enrichment Test:")
    test_node = {"id": "test_001", "type": "ProductComponent"}
    enriched = enrich_node_by_type(test_node, "V01_ProductComponent")
    print(f"  Original keys: {len(test_node)}")
    print(f"  Enriched keys: {len(enriched)}")
    print(f"  Applied params: {enriched.get('_gdp_params_applied', [])}")
    
    print("\n✅ Validation Test:")
    validation = validate_global_parameters()
    print(f"  Valid: {validation['valid']}")
    print(f"  Errors: {len(validation['errors'])}")
    print(f"  Warnings: {len(validation['warnings'])}")
    
    print("\n✅ Global Design Parameters v2.0 loaded successfully!")
