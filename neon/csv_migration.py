#!/usr/bin/env python3
"""
CSV Migration Script for Enhanced CAD Parser
Migrates auraDB Migration CSV files to PostgreSQL schema
"""

import os
import csv
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVMigrator:
    def __init__(self, db_config: Dict[str, str], csv_directory: str):
        """
        Initialize the CSV migrator
        
        Args:
            db_config: Database connection configuration
            csv_directory: Directory containing CSV files
        """
        self.db_config = db_config
        self.csv_directory = csv_directory
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def get_csv_files(self) -> List[str]:
        """Get list of CSV files in the directory"""
        csv_files = []
        for file in os.listdir(self.csv_directory):
            if file.endswith('.csv'):
                csv_files.append(file)
        return sorted(csv_files)
    
    def read_csv_data(self, filename: str) -> List[Dict[str, Any]]:
        """Read CSV file and return data as list of dictionaries"""
        filepath = os.path.join(self.csv_directory, filename)
        data = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Clean empty values
                    cleaned_row = {k: v.strip() if v else None for k, v in row.items()}
                    data.append(cleaned_row)
            
            logger.info(f"Read {len(data)} rows from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return []
    
    def migrate_components(self, data: List[Dict[str, Any]]):
        """Migrate component data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                component_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO components (component_id, component_name, component_type, description)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (component_id) DO NOTHING
                """, (
                    component_id,
                    row.get('component_name', 'Unknown'),
                    row.get('component_type', 'Unknown'),
                    row.get('description', '')
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} components")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating components: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_materials(self, data: List[Dict[str, Any]]):
        """Migrate material data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                material_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO materials (material_id, material_name, base_material, material_variant, material_code, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (material_id) DO NOTHING
                """, (
                    material_id,
                    row.get('material_name', 'Unknown'),
                    row.get('base_material', 'Unknown'),
                    row.get('material_variant', ''),
                    row.get('material_code', ''),
                    row.get('description', '')
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} materials")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating materials: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_suppliers(self, data: List[Dict[str, Any]]):
        """Migrate supplier data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                supplier_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO suppliers (supplier_id, supplier_name, supplier_type, contact_person, email, phone, lead_time_days)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (supplier_id) DO NOTHING
                """, (
                    supplier_id,
                    row.get('supplier_name', 'Unknown'),
                    row.get('supplier_type', 'Unknown'),
                    row.get('contact_person', ''),
                    row.get('email', ''),
                    row.get('phone', ''),
                    int(row.get('lead_time_days', 0)) if row.get('lead_time_days') else None
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} suppliers")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating suppliers: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_functions(self, data: List[Dict[str, Any]]):
        """Migrate function data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                function_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO functions (function_id, function_name, function_type, functor_name, functor_type, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (function_id) DO NOTHING
                """, (
                    function_id,
                    row.get('function_name', 'Unknown'),
                    row.get('function_type', 'Unknown'),
                    row.get('functor_name', ''),
                    row.get('functor_type', ''),
                    row.get('description', '')
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} functions")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating functions: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_families(self, data: List[Dict[str, Any]]):
        """Migrate family data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                family_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO families (family_id, family_name, family_type, product_family, product_family_type, variant, variant_name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (family_id) DO NOTHING
                """, (
                    family_id,
                    row.get('family_name', 'Unknown'),
                    row.get('family_type', 'Unknown'),
                    row.get('product_family', ''),
                    row.get('product_family_type', ''),
                    row.get('variant', ''),
                    row.get('variant_name', '')
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} families")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating families: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_item_identity_nodes(self, data: List[Dict[str, Any]]):
        """Migrate item identity nodes"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                item_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO item_identity_nodes (item_id, product_family, variant, group_id, manufacture_score, spec_tag)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (item_id) DO NOTHING
                """, (
                    item_id,
                    row.get('product_family', ''),
                    row.get('variant', ''),
                    row.get('group_id', None),
                    float(row.get('manufacture_score', 0)) if row.get('manufacture_score') else None,
                    row.get('spec_tag', '')
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} item identity nodes")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating item identity nodes: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_item_identity_connections(self, data: List[Dict[str, Any]]):
        """Migrate item identity connections"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                connection_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO item_identity_connections (connection_id, source_id, target_id, connection_type, connection_tag, connection_role, connection_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (connection_id) DO NOTHING
                """, (
                    connection_id,
                    row.get('source_id', None),
                    row.get('target_id', None),
                    row.get('connection_type', ''),
                    row.get('connection_tag', ''),
                    row.get('connection_role', ''),
                    float(row.get('connection_score', 0)) if row.get('connection_score') else None
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} item identity connections")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating item identity connections: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_dimensions(self, data: List[Dict[str, Any]]):
        """Migrate dimension data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                dimension_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO dimensions (dimension_id, length_mm, width_mm, height_mm, area_m2, volume_cm3, tolerance_mm)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dimension_id) DO NOTHING
                """, (
                    dimension_id,
                    float(row.get('length_mm', 0)) if row.get('length_mm') else None,
                    float(row.get('width_mm', 0)) if row.get('width_mm') else None,
                    float(row.get('height_mm', 0)) if row.get('height_mm') else None,
                    float(row.get('area_m2', 0)) if row.get('area_m2') else None,
                    float(row.get('volume_cm3', 0)) if row.get('volume_cm3') else None,
                    float(row.get('tolerance_mm', 0)) if row.get('tolerance_mm') else None
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} dimensions")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating dimensions: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_spatial_data(self, data: List[Dict[str, Any]]):
        """Migrate spatial data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                spatial_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO spatial_data (spatial_id, centroid_x, centroid_y, centroid_z, bbox_x_mm, bbox_y_mm, bbox_z_mm)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (spatial_id) DO NOTHING
                """, (
                    spatial_id,
                    float(row.get('centroid_x', 0)) if row.get('centroid_x') else None,
                    float(row.get('centroid_y', 0)) if row.get('centroid_y') else None,
                    float(row.get('centroid_z', 0)) if row.get('centroid_z') else None,
                    float(row.get('bbox_x_mm', 0)) if row.get('bbox_x_mm') else None,
                    float(row.get('bbox_y_mm', 0)) if row.get('bbox_y_mm') else None,
                    float(row.get('bbox_z_mm', 0)) if row.get('bbox_z_mm') else None
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} spatial data records")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating spatial data: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_properties(self, data: List[Dict[str, Any]]):
        """Migrate property data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                property_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO properties (property_id, property_key, property_name, property_type, property_value, unit)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (property_id) DO NOTHING
                """, (
                    property_id,
                    row.get('property_key', ''),
                    row.get('property_name', ''),
                    row.get('property_type', ''),
                    row.get('property_value', ''),
                    row.get('unit', '')
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} properties")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating properties: {e}")
            raise
        finally:
            cursor.close()
    
    def migrate_anchors(self, data: List[Dict[str, Any]]):
        """Migrate anchor data"""
        if not data:
            return
        
        cursor = self.connection.cursor()
        try:
            for row in data:
                anchor_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO anchors (anchor_id, anchor_name, anchor_type, anchor_configuration, anchor_constraints, octree_depth, octree_size, topologic_vertex_anchor)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (anchor_id) DO NOTHING
                """, (
                    anchor_id,
                    row.get('anchor_name', ''),
                    row.get('anchor_type', ''),
                    row.get('anchor_configuration', ''),
                    row.get('anchor_constraints', ''),
                    int(row.get('octree_depth', 0)) if row.get('octree_depth') else None,
                    float(row.get('octree_size', 0)) if row.get('octree_size') else None,
                    row.get('topologic_vertex_anchor', 'false').lower() == 'true'
                ))
            
            self.connection.commit()
            logger.info(f"Migrated {len(data)} anchors")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error migrating anchors: {e}")
            raise
        finally:
            cursor.close()
    
    def get_migration_mapping(self) -> Dict[str, callable]:
        """Get mapping of CSV files to migration functions"""
        return {
            'component.csv': self.migrate_components,
            'material.csv': self.migrate_materials,
            'supplier.csv': self.migrate_suppliers,
            'function.csv': self.migrate_functions,
            'family.csv': self.migrate_families,
            'item_identity_nodes.csv': self.migrate_item_identity_nodes,
            'item_identity_connections.csv': self.migrate_item_identity_connections,
            'lengthmm.csv': self.migrate_dimensions,
            'widthmm.csv': self.migrate_dimensions,
            'heightmm.csv': self.migrate_dimensions,
            'aream2.csv': self.migrate_dimensions,
            'boundingboxvolumecm3.csv': self.migrate_dimensions,
            'centroidx.csv': self.migrate_spatial_data,
            'centroidy.csv': self.migrate_spatial_data,
            'centroidz.csv': self.migrate_spatial_data,
            'bboxxmm.csv': self.migrate_spatial_data,
            'bboxymm.csv': self.migrate_spatial_data,
            'bboxzmm.csv': self.migrate_spatial_data,
            'propertykey.csv': self.migrate_properties,
            'propertyname.csv': self.migrate_properties,
            'anchor.csv': self.migrate_anchors,
            'anchorname.csv': self.migrate_anchors,
            'anchortype.csv': self.migrate_anchors
        }
    
    def run_migration(self):
        """Run the complete migration process"""
        logger.info("Starting CSV migration process")
        
        try:
            self.connect()
            migration_mapping = self.get_migration_mapping()
            csv_files = self.get_csv_files()
            
            for csv_file in csv_files:
                if csv_file in migration_mapping:
                    logger.info(f"Processing {csv_file}")
                    data = self.read_csv_data(csv_file)
                    if data:
                        migration_mapping[csv_file](data)
                else:
                    logger.warning(f"No migration function found for {csv_file}")
            
            logger.info("CSV migration completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            self.disconnect()

def main():
    """Main function to run the migration"""
    # Database configuration - update with your Neon credentials
    db_config = {
        'host': os.getenv('NEON_HOST', 'localhost'),
        'port': os.getenv('NEON_PORT', '5432'),
        'database': os.getenv('NEON_DATABASE', 'cad_parser'),
        'user': os.getenv('NEON_USER', 'postgres'),
        'password': os.getenv('NEON_PASSWORD', '')
    }
    
    # CSV directory path
    csv_directory = "postgre/auraDB Migration"
    
    # Create migrator and run migration
    migrator = CSVMigrator(db_config, csv_directory)
    migrator.run_migration()

if __name__ == "__main__":
    main() 