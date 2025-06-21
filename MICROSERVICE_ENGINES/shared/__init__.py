"""
Shared Resources Package
🔧 Common utilities and configurations for BEM microservices
"""

from .global_design_parameters import (
    global_design_parameters,
    enrich_node_with_global_params,
    get_building_component_params,
    get_procedural_model_params,
    get_facade_tool_params,
    get_scoring_context,
    get_ui_rendering_context
)

__all__ = [
    'global_design_parameters',
    'enrich_node_with_global_params',
    'get_building_component_params',
    'get_procedural_model_params',
    'get_facade_tool_params',
    'get_scoring_context',
    'get_ui_rendering_context'
]
