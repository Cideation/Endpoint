#!/usr/bin/env python3
"""
Neo4j Aura Configuration
Add your Aura credentials here
"""

# Neo4j Aura Configuration
AURA_CONFIG = {
    'uri': 'neo4j+s://your-aura-instance.databases.neo4j.io:7687',
    'username': 'neo4j',
    'password': 'your-password-here'
}

# Aura Database Settings
AURA_SETTINGS = {
    'database': 'neo4j',  # Default database name
    'max_connection_lifetime': 3600,  # 1 hour
    'max_connection_pool_size': 50,
    'connection_timeout': 30,
    'encrypted': True
}

# Graph Processing Settings
GRAPH_SETTINGS = {
    'batch_size': 100,
    'max_retries': 3,
    'retry_delay': 1,  # seconds
    'enable_analytics': True,
    'enable_indexes': True
}

# Sync Configuration
SYNC_CONFIG = {
    'tables_to_sync': [
        'component_enhanced',
        'material_enhanced', 
        'supplier_enhanced',
        'component_hierarchy',
        'material_variants',
        'manufacturing_methods'
    ],
    'relationship_mappings': {
        'component_hierarchy': {
            'source_label': 'Component',
            'target_label': 'Component',
            'relationship_type': 'HAS_CHILD',
            'source_id_field': 'parent_component_id',
            'target_id_field': 'child_component_id'
        },
        'material_variants': {
            'source_label': 'Material',
            'target_label': 'Material',
            'relationship_type': 'HAS_VARIANT',
            'source_id_field': 'base_material_id',
            'target_id_field': 'variant_code'
        }
    }
} 