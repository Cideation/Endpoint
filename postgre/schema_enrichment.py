#!/usr/bin/env python3
"""
Schema Enrichment Script
Uses normalized property mapping to enhance database schema
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaEnricher:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.mapping_df = None
        self.enhanced_schema = {}
        
    def load_mapping(self, mapping_file: str):
        """Load the normalized property mapping"""
        try:
            self.mapping_df = pd.read_csv(mapping_file)
            logger.info(f"Loaded {len(self.mapping_df)} field mappings")
            return True
        except Exception as e:
            logger.error(f"Failed to load mapping file: {e}")
            return False
    
    def analyze_schema_structure(self):
        """Analyze the mapping to understand schema structure"""
        if self.mapping_df is None:
            return False
            
        # Group by node_label to understand table structure
        table_groups = self.mapping_df.groupby('node_label')
        
        for table_name, group in table_groups:
            self.enhanced_schema[table_name] = {
                'fields': {},
                'relationships': [],
                'constraints': []
            }
            
            for _, row in group.iterrows():
                field_name = row['renamed_key']
                field_type = row['type_guess']
                sample_values = eval(row['sample_values']) if pd.notna(row['sample_values']) else []
                
                self.enhanced_schema[table_name]['fields'][field_name] = {
                    'original_name': row['node_label'],
                    'type': field_type,
                    'sample_values': sample_values,
                    'postgres_type': self._map_to_postgres_type(field_type)
                }
        
        logger.info(f"Analyzed {len(self.enhanced_schema)} tables")
        return True
    
    def _map_to_postgres_type(self, python_type: str) -> str:
        """Map Python types to PostgreSQL types"""
        type_mapping = {
            'string': 'TEXT',
            'integer': 'INTEGER',
            'float': 'DOUBLE PRECISION',
            'boolean': 'BOOLEAN'
        }
        return type_mapping.get(python_type, 'TEXT')
    
    def generate_enhanced_schema_sql(self) -> str:
        """Generate SQL for enhanced schema"""
        sql_parts = []
        
        # Create enhanced tables
        for table_name, table_info in self.enhanced_schema.items():
            sql_parts.append(f"\n-- Enhanced table: {table_name}")
            sql_parts.append(f"CREATE TABLE IF NOT EXISTS {table_name.lower()}_enhanced (")
            
            # Add standard fields
            sql_parts.append("    id SERIAL PRIMARY KEY,")
            sql_parts.append("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,")
            sql_parts.append("    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,")
            
            # Add mapped fields
            field_definitions = []
            for field_name, field_info in table_info['fields'].items():
                postgres_type = field_info['postgres_type']
                field_definitions.append(f"    {field_name} {postgres_type}")
            
            sql_parts.append(",\n".join(field_definitions))
            sql_parts.append(");")
            
            # Add indexes for common query fields
            sql_parts.append(f"CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_id ON {table_name.lower()}_enhanced(id);")
        
        # Create relationship tables
        sql_parts.append(self._generate_relationship_tables())
        
        # Create materialized views for analytics
        sql_parts.append(self._generate_analytics_views())
        
        return "\n".join(sql_parts)
    
    def _generate_relationship_tables(self) -> str:
        """Generate tables for complex relationships"""
        sql = """
-- Relationship tables for complex associations
CREATE TABLE IF NOT EXISTS component_hierarchy (
    id SERIAL PRIMARY KEY,
    parent_component_id TEXT,
    child_component_id TEXT,
    relationship_type TEXT,
    hierarchy_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS material_variants (
    id SERIAL PRIMARY KEY,
    base_material_id TEXT,
    variant_code TEXT,
    variant_name TEXT,
    properties JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS manufacturing_methods (
    id SERIAL PRIMARY KEY,
    method_id TEXT,
    method_name TEXT,
    capabilities JSONB,
    tolerances JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS spatial_references (
    id SERIAL PRIMARY KEY,
    reference_id TEXT,
    reference_type TEXT,
    coordinates GEOMETRY,
    properties JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
        return sql
    
    def _generate_analytics_views(self) -> str:
        """Generate materialized views for analytics"""
        sql = """
-- Materialized views for analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS component_analytics AS
SELECT 
    c.component_id,
    c.product_family,
    c.material_id,
    c.bbox_x_mm,
    c.bbox_y_mm,
    c.bbox_z_mm,
    (c.bbox_x_mm * c.bbox_y_mm * c.bbox_z_mm) / 1000.0 as volume_cm3,
    c.is_root_node,
    c.is_reusable
FROM component_enhanced c;

CREATE MATERIALIZED VIEW IF NOT EXISTS supply_chain_analytics AS
SELECT 
    s.supplier_id,
    s.supplier_name,
    s.supplier_type,
    COUNT(i.source_item_id) as item_count,
    AVG(CAST(i.unit_cost AS INTEGER)) as avg_unit_cost
FROM supplier_enhanced s
LEFT JOIN source_item_enhanced i ON s.supplier_id = i.supplier_id
GROUP BY s.supplier_id, s.supplier_name, s.supplier_type;

-- Refresh functions
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW component_analytics;
    REFRESH MATERIALIZED VIEW supply_chain_analytics;
END;
$$ LANGUAGE plpgsql;
"""
        return sql
    
    def apply_enhancements(self, sql_file: str = None):
        """Apply schema enhancements to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Generate and execute enhanced schema
            enhanced_sql = self.generate_enhanced_schema_sql()
            
            if sql_file:
                with open(sql_file, 'w') as f:
                    f.write(enhanced_sql)
                logger.info(f"Enhanced schema saved to {sql_file}")
            
            # Execute the schema
            cursor.execute(enhanced_sql)
            conn.commit()
            
            logger.info("Enhanced schema applied successfully")
            
            # Create data migration functions
            self._create_migration_functions(cursor, conn)
            
        except Exception as e:
            logger.error(f"Failed to apply enhancements: {e}")
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def _create_migration_functions(self, cursor, conn):
        """Create functions to migrate data from existing tables"""
        migration_functions = """
-- Function to migrate data from existing tables to enhanced schema
CREATE OR REPLACE FUNCTION migrate_to_enhanced_schema()
RETURNS void AS $$
BEGIN
    -- Example migration for component table
    INSERT INTO component_enhanced (
        component_id, product_family, material_id, 
        bbox_x_mm, bbox_y_mm, bbox_z_mm, is_root_node, is_reusable
    )
    SELECT 
        component_id, product_family, material_id,
        CAST(bbox_x_mm AS INTEGER), 
        CAST(bbox_y_mm AS INTEGER), 
        CAST(bbox_z_mm AS INTEGER),
        is_root_node::BOOLEAN, 
        is_reusable::BOOLEAN
    FROM component
    ON CONFLICT (component_id) DO NOTHING;
    
    -- Add more table migrations as needed
    
    RAISE NOTICE 'Migration completed successfully';
END;
$$ LANGUAGE plpgsql;
"""
        cursor.execute(migration_functions)
        conn.commit()
        logger.info("Migration functions created")

def main():
    # Load configuration
    try:
        from config import NEON_CONFIG
        DB_CONFIG = NEON_CONFIG
    except ImportError:
        logger.error("config.py not found. Please ensure NEON_CONFIG is defined.")
        return
    
    # Initialize enricher
    enricher = SchemaEnricher(DB_CONFIG)
    
    # Load mapping
    if not enricher.load_mapping('user_drops/normalized_property_mapping_clean.csv'):
        return
    
    # Analyze schema
    if not enricher.analyze_schema_structure():
        return
    
    # Apply enhancements
    enricher.apply_enhancements('enhanced_schema.sql')
    
    logger.info("Schema enrichment completed!")

if __name__ == "__main__":
    main() 