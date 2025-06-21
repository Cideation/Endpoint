"""
Shared Resources Package
�� Common utilities and configurations for BEM microservices
"""

from .global_design_parameters import (
    get_global_design_parameters,
    enrich_node_by_type,
    enrich_node_full,
    get_parameters_by_category,
    get_node_mapping,
    validate_global_parameters,
    reload_global_parameters,
    enrich_node_with_global_params
)

__all__ = [
    'get_global_design_parameters',
    'enrich_node_by_type',
    'enrich_node_full',
    'get_parameters_by_category',
    'get_node_mapping',
    'validate_global_parameters',
    'reload_global_parameters',
    'enrich_node_with_global_params'
]
