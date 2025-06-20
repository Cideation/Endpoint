#!/usr/bin/env python3
"""
Database Manager for Neon PostgreSQL
Handles connections, pooling, and integration with enhanced CAD parsers
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
import psycopg2
import asyncpg
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeonDBManager:
    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        """
        Initialize Neon database manager
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config or {
            'host': os.getenv('NEON_HOST', 'localhost'),
            'port': os.getenv('NEON_PORT', '5432'),
            'database': os.getenv('NEON_DATABASE', 'cad_parser'),
            'user': os.getenv('NEON_USER', 'postgres'),
            'password': os.getenv('NEON_PASSWORD', '')
        }
        self.pool = None
        self.sync_connection = None
    
    async def create_pool(self, min_size: int = 5, max_size: int = 20):
        """Create connection pool for async operations"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                min_size=min_size,
                max_size=max_size
            )
            logger.info("Async connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    def get_sync_connection(self):
        """Get synchronous database connection"""
        if not self.sync_connection or self.sync_connection.closed:
            try:
                self.sync_connection = psycopg2.connect(**self.db_config)
                logger.info("Synchronous connection established")
            except Exception as e:
                logger.error(f"Failed to establish sync connection: {e}")
                raise
        return self.sync_connection
    
    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections"""
        if not self.pool:
            await self.create_pool()
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        async with self.get_connection() as conn:
            try:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)
                
                return [dict(row) for row in result]
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
    
    async def execute_command(self, command: str, params: Optional[tuple] = None) -> str:
        """Execute a command and return result message"""
        async with self.get_connection() as conn:
            try:
                if params:
                    result = await conn.execute(command, *params)
                else:
                    result = await conn.execute(command)
                
                return result
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                raise
    
    def execute_sync_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a synchronous query and return results"""
        conn = self.get_sync_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Sync query execution failed: {e}")
            raise
        finally:
            cursor.close()
    
    def execute_sync_command(self, command: str, params: Optional[tuple] = None) -> str:
        """Execute a synchronous command and return result message"""
        conn = self.get_sync_connection()
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(command, params)
            else:
                cursor.execute(command)
            
            conn.commit()
            return cursor.statusmessage
        except Exception as e:
            conn.rollback()
            logger.error(f"Sync command execution failed: {e}")
            raise
        finally:
            cursor.close()
    
    # =====================================================
    # PARSER INTEGRATION METHODS
    # =====================================================
    
    async def insert_parsed_file(self, file_data: Dict[str, Any]) -> str:
        """Insert parsed file metadata"""
        query = """
            INSERT INTO parsed_files (file_name, file_path, file_type, file_size, parsing_status, components_extracted, processing_time_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING file_id
        """
        
        result = await self.execute_query(query, (
            file_data.get('file_name'),
            file_data.get('file_path'),
            file_data.get('file_type'),
            file_data.get('file_size', 0),
            file_data.get('parsing_status', 'processing'),
            file_data.get('components_extracted', 0),
            file_data.get('processing_time_ms', 0)
        ))
        
        return result[0]['file_id'] if result else None
    
    async def insert_parsed_component(self, file_id: str, component_data: Dict[str, Any]) -> str:
        """Insert parsed component data"""
        # First insert the component
        component_query = """
            INSERT INTO components (component_name, component_type, description)
            VALUES ($1, $2, $3)
            RETURNING component_id
        """
        
        component_result = await self.execute_query(component_query, (
            component_data.get('name', 'Unknown'),
            component_data.get('type', 'Unknown'),
            component_data.get('description', '')
        ))
        
        component_id = component_result[0]['component_id'] if component_result else None
        
        if component_id:
            # Insert parsed component mapping
            parsed_query = """
                INSERT INTO parsed_components (file_id, component_id, parser_component_id, parser_type, parser_data)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING parsed_component_id
            """
            
            parsed_result = await self.execute_query(parsed_query, (
                file_id,
                component_id,
                component_data.get('id', ''),
                component_data.get('parser_type', ''),
                json.dumps(component_data)
            ))
            
            # Insert spatial data if available
            if component_data.get('geometry', {}).get('has_position'):
                spatial_query = """
                    INSERT INTO spatial_data (component_id, centroid_x, centroid_y, centroid_z)
                    VALUES ($1, $2, $3, $4)
                """
                
                position = component_data['geometry']['position']
                await self.execute_command(spatial_query, (
                    component_id,
                    position[0] if len(position) > 0 else None,
                    position[1] if len(position) > 1 else None,
                    position[2] if len(position) > 2 else None
                ))
            
            # Insert dimensions if available
            if component_data.get('geometry', {}).get('dimensions'):
                dims = component_data['geometry']['dimensions']
                dim_query = """
                    INSERT INTO dimensions (component_id, length_mm, width_mm, height_mm, volume_cm3)
                    VALUES ($1, $2, $3, $4, $5)
                """
                
                await self.execute_command(dim_query, (
                    component_id,
                    dims.get('length'),
                    dims.get('width'),
                    dims.get('height'),
                    dims.get('volume')
                ))
            
            # Insert geometry properties if available
            if component_data.get('geometry', {}).get('properties'):
                geom_props = component_data['geometry']['properties']
                geom_query = """
                    INSERT INTO geometry_properties (component_id, vertex_count, face_count, edge_count, surface_area_m2)
                    VALUES ($1, $2, $3, $4, $5)
                """
                
                await self.execute_command(geom_query, (
                    component_id,
                    geom_props.get('vertex_count'),
                    geom_props.get('face_count'),
                    geom_props.get('edge_count'),
                    geom_props.get('surface_area')
                ))
        
        return component_id
    
    async def update_parsing_status(self, file_id: str, status: str, error_message: Optional[str] = None):
        """Update parsing status for a file"""
        query = """
            UPDATE parsed_files 
            SET parsing_status = $1, error_message = $2, parsed_at = NOW()
            WHERE file_id = $3
        """
        
        await self.execute_command(query, (status, error_message, file_id))
    
    # =====================================================
    # QUERY METHODS
    # =====================================================
    
    async def get_component_summary(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get component summary with all related data"""
        query = """
            SELECT * FROM component_summary 
            ORDER BY created_at DESC 
            LIMIT $1
        """
        
        return await self.execute_query(query, (limit,))
    
    async def get_parser_statistics(self) -> List[Dict[str, Any]]:
        """Get parser statistics"""
        query = "SELECT * FROM parser_statistics"
        return await self.execute_query(query)
    
    async def search_components(self, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search components by name or type"""
        query = """
            SELECT * FROM component_summary 
            WHERE component_name ILIKE $1 OR component_type ILIKE $1
            ORDER BY created_at DESC 
            LIMIT $2
        """
        
        return await self.execute_query(query, (f'%{search_term}%', limit))
    
    async def get_components_by_type(self, component_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get components by type"""
        query = """
            SELECT * FROM component_summary 
            WHERE component_type = $1
            ORDER BY created_at DESC 
            LIMIT $2
        """
        
        return await self.execute_query(query, (component_type, limit))
    
    async def get_spatial_components(self, bbox: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Get components within a bounding box"""
        if bbox:
            query = """
                SELECT c.*, sp.centroid_x, sp.centroid_y, sp.centroid_z
                FROM components c
                JOIN spatial_data sp ON c.component_id = sp.component_id
                WHERE sp.centroid_x BETWEEN $1 AND $2
                AND sp.centroid_y BETWEEN $3 AND $4
                AND sp.centroid_z BETWEEN $5 AND $6
                ORDER BY c.created_at DESC
            """
            
            return await self.execute_query(query, (
                bbox[0], bbox[2],  # x min, x max
                bbox[1], bbox[3],  # y min, y max
                bbox[4], bbox[5]   # z min, z max
            ))
        else:
            query = """
                SELECT c.*, sp.centroid_x, sp.centroid_y, sp.centroid_z
                FROM components c
                JOIN spatial_data sp ON c.component_id = sp.component_id
                ORDER BY c.created_at DESC
            """
            
            return await self.execute_query(query)
    
    # =====================================================
    # UTILITY METHODS
    # =====================================================
    
    async def get_table_count(self, table_name: str) -> int:
        """Get row count for a table"""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = await self.execute_query(query)
        return result[0]['count'] if result else 0
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        tables = [
            'components', 'materials', 'suppliers', 'functions', 
            'families', 'parsed_files', 'parsed_components',
            'spatial_data', 'dimensions', 'properties'
        ]
        
        for table in tables:
            stats[table] = await self.get_table_count(table)
        
        # Get parser statistics
        parser_stats = await self.get_parser_statistics()
        stats['parser_statistics'] = parser_stats
        
        return stats
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old parsed data"""
        query = """
            DELETE FROM parsed_components 
            WHERE created_at < NOW() - INTERVAL '$1 days'
        """
        
        await self.execute_command(query, (days,))
        logger.info(f"Cleaned up data older than {days} days")
    
    def close(self):
        """Close all connections"""
        if self.sync_connection and not self.sync_connection.closed:
            self.sync_connection.close()
            logger.info("Synchronous connection closed")
    
    async def close_async(self):
        """Close async connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Async connection pool closed")

# =====================================================
# USAGE EXAMPLE
# =====================================================

async def example_usage():
    """Example usage of the NeonDBManager"""
    db_manager = NeonDBManager()
    
    try:
        # Get database statistics
        stats = await db_manager.get_database_stats()
        print("Database Statistics:", stats)
        
        # Search for components
        components = await db_manager.search_components("wall", limit=10)
        print(f"Found {len(components)} components matching 'wall'")
        
        # Get parser statistics
        parser_stats = await db_manager.get_parser_statistics()
        print("Parser Statistics:", parser_stats)
        
    finally:
        await db_manager.close_async()
        db_manager.close()

if __name__ == "__main__":
    asyncio.run(example_usage()) 