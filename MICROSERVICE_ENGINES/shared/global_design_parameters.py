#!/usr/bin/env python3
"""
Global Design Parameters - Shared Resource
🏗️ Node-level shared variables for enriching components during parsing, scoring, and UI rendering

Usage:
    from shared.global_design_parameters import global_design_parameters
    
    # Enrich node with building component parameters
    node.update(global_design_parameters['building_components'])
    
    # Use in scoring algorithms
    score = calculate_score(component, global_design_parameters)
    
    # UI rendering context
    render_component(component, global_design_parameters['generative_facade_tool'])
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Design Parameters Dictionary
global_design_parameters = {
    # From "Building Components"
    "building_components": {
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
        "evolutionary_potential": 0.92
    },

    # From "Bi-Directional Procedural Model"
    "bidirectional_procedural_model": {
        "connectivity_validated": True,
        "symbolic_depth_range": 5,
        "layout_compactness": 0.72,
        "hierarchy_type": "heterogeneous",
        "mirror_operations_enabled": True,
        "layout_success_rate": 0.99,
        "room_rectangularity_score": 0.84,
        "public_private_ratio": "1:4"
    },

    # From "Generative Facade Tool"
    "generative_facade_tool": {
        "facade_style_id": "style_2_bay_variation",
        "parametric_frame_depth": 0.3,
        "bay_arrangement_pattern": "alternating_even_odd",
        "component_modularity": "enabled",
        "floor_span_variance": "medium",
        "tilt_transformation": False,
        "generation_type": "rule_driven",
        "cad_integration_ready": True
    }
}

def enrich_node_with_global_params(node: Dict[str, Any], category: str = 'building_components') -> Dict[str, Any]:
    """Enrich a node with global design parameters"""
    if category in global_design_parameters:
        node.update(global_design_parameters[category])
    return node

def get_building_component_params() -> Dict[str, Any]:
    """Get building components parameters"""
    return global_design_parameters.get('building_components', {})

def get_procedural_model_params() -> Dict[str, Any]:
    """Get bidirectional procedural model parameters"""
    return global_design_parameters.get('bidirectional_procedural_model', {})

def get_facade_tool_params() -> Dict[str, Any]:
    """Get generative facade tool parameters"""
    return global_design_parameters.get('generative_facade_tool', {})

def get_scoring_context() -> Dict[str, Any]:
    """Get all parameters for scoring algorithms"""
    return global_design_parameters.copy()

def get_ui_rendering_context(category: Optional[str] = None) -> Dict[str, Any]:
    """Get parameters for UI rendering"""
    if category and category in global_design_parameters:
        return global_design_parameters[category]
    return global_design_parameters.copy()

if __name__ == "__main__":
    # Test the module
    print("🏗️ Global Design Parameters Test")
    print("=" * 50)
    
    print("\n�� Building Components:")
    for key, value in get_building_component_params().items():
        print(f"  {key}: {value}")
    
    print("\n🔄 Procedural Model:")
    for key, value in get_procedural_model_params().items():
        print(f"  {key}: {value}")
    
    print("\n🏢 Facade Tool:")
    for key, value in get_facade_tool_params().items():
        print(f"  {key}: {value}")
    
    print("\n✅ Global Design Parameters loaded successfully!")
