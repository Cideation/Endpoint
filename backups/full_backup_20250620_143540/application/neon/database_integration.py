"""
Database Integration Layer for Phase 2 Pipeline
Connects JSON transformation pipeline to actual Neon PostgreSQL database
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from .db_manager import NeonDBManager
from .models import (
    Component, SpatialData, GeometryProperties, Material, Dimensions,
    ComponentWithRelations, ParsedFile, ParsedComponent
)
from .json_transformer import JSONTransformer
from .orchestrator import Orchestrator
from .config import DATABASE_URL, NEON_CONFIG

logger = logging.getLogger(__name__)

class DatabaseIntegration:
    """
    Integrates the JSON transformation pipeline with Neon PostgreSQL database
    """
    
    def __init__(self):
        self.db_manager = NeonDBManager(NEON_CONFIG)
        self.transformer = JSONTransformer()
        self.orchestrator = Orchestrator()
        self._pool_initialized = False
    
    async def initialize(self):
        """Initialize database connection pool"""
        if not self._pool_initialized:
            await self.db_manager.create_pool()
            self._pool_initialized = True
            logger.info("Database integration initialized")
    
    async def get_components_with_relations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get components with all related data from database
        """
        await self.initialize()
        
        # Get components with all related data using the view
        query = """
            SELECT 
                c.component_id,
                c.component_name,
                c.component_type,
                c.description,
                c.created_at,
                c.updated_at,
                m.material_name,
                m.base_material,
                m.material_code,
                sp.centroid_x,
                sp.centroid_y,
                sp.centroid_z,
                sp.bbox_x_mm,
                sp.bbox_y_mm,
                sp.bbox_z_mm,
                d.length_mm,
                d.width_mm,
                d.height_mm,
                d.area_m2,
                d.volume_cm3,
                d.tolerance_mm,
                gp.geometry_format,
                gp.geometry_type,
                gp.vertex_count,
                gp.face_count,
                gp.edge_count,
                gp.bounding_box_volume_cm3,
                gp.surface_area_m2
            FROM components c
            LEFT JOIN component_materials cm ON c.component_id = cm.component_id
            LEFT JOIN materials m ON cm.material_id = m.material_id
            LEFT JOIN spatial_data sp ON c.component_id = sp.component_id
            LEFT JOIN dimensions d ON c.component_id = d.component_id
            LEFT JOIN geometry_properties gp ON c.component_id = gp.component_id
            ORDER BY c.created_at DESC
            LIMIT $1
        """
        
        try:
            results = await self.db_manager.execute_query(query, (limit,))
            
            # Transform database results to component dictionaries
            components = []
            for row in results:
                component = self._transform_db_row_to_component(row)
                components.append(component)
            
            logger.info(f"Retrieved {len(components)} components from database")
            return components
            
        except Exception as e:
            logger.error(f"Failed to get components: {e}")
            raise
    
    async def get_components_by_type(self, component_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get components filtered by type
        """
        await self.initialize()
        
        query = """
            SELECT 
                c.component_id,
                c.component_name,
                c.component_type,
                c.description,
                c.created_at,
                c.updated_at,
                m.material_name,
                m.base_material,
                m.material_code,
                sp.centroid_x,
                sp.centroid_y,
                sp.centroid_z,
                sp.bbox_x_mm,
                sp.bbox_y_mm,
                sp.bbox_z_mm,
                d.length_mm,
                d.width_mm,
                d.height_mm,
                d.area_m2,
                d.volume_cm3,
                d.tolerance_mm,
                gp.geometry_format,
                gp.geometry_type,
                gp.vertex_count,
                gp.face_count,
                gp.edge_count,
                gp.bounding_box_volume_cm3,
                gp.surface_area_m2
            FROM components c
            LEFT JOIN component_materials cm ON c.component_id = cm.component_id
            LEFT JOIN materials m ON cm.material_id = m.material_id
            LEFT JOIN spatial_data sp ON c.component_id = sp.component_id
            LEFT JOIN dimensions d ON c.component_id = d.component_id
            LEFT JOIN geometry_properties gp ON c.component_id = gp.component_id
            WHERE c.component_type ILIKE $1
            ORDER BY c.created_at DESC
            LIMIT $2
        """
        
        try:
            results = await self.db_manager.execute_query(query, (f"%{component_type}%", limit))
            
            components = []
            for row in results:
                component = self._transform_db_row_to_component(row)
                components.append(component)
            
            logger.info(f"Retrieved {len(components)} components of type '{component_type}'")
            return components
            
        except Exception as e:
            logger.error(f"Failed to get components by type: {e}")
            raise
    
    async def get_spatial_components(self, bbox: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Get components with spatial data, optionally filtered by bounding box
        """
        await self.initialize()
        
        if bbox and len(bbox) == 6:  # min_x, min_y, min_z, max_x, max_y, max_z
            query = """
                SELECT 
                    c.component_id,
                    c.component_name,
                    c.component_type,
                    c.description,
                    c.created_at,
                    c.updated_at,
                    m.material_name,
                    m.base_material,
                    m.material_code,
                    sp.centroid_x,
                    sp.centroid_y,
                    sp.centroid_z,
                    sp.bbox_x_mm,
                    sp.bbox_y_mm,
                    sp.bbox_z_mm,
                    d.length_mm,
                    d.width_mm,
                    d.height_mm,
                    d.area_m2,
                    d.volume_cm3,
                    d.tolerance_mm,
                    gp.geometry_format,
                    gp.geometry_type,
                    gp.vertex_count,
                    gp.face_count,
                    gp.edge_count,
                    gp.bounding_box_volume_cm3,
                    gp.surface_area_m2
                FROM components c
                LEFT JOIN component_materials cm ON c.component_id = cm.component_id
                LEFT JOIN materials m ON cm.material_id = m.material_id
                LEFT JOIN spatial_data sp ON c.component_id = sp.component_id
                LEFT JOIN dimensions d ON c.component_id = d.component_id
                LEFT JOIN geometry_properties gp ON c.component_id = gp.component_id
                WHERE sp.centroid_x BETWEEN $1 AND $4
                  AND sp.centroid_y BETWEEN $2 AND $5
                  AND sp.centroid_z BETWEEN $3 AND $6
                ORDER BY c.created_at DESC
            """
            params = tuple(bbox)
        else:
            query = """
                SELECT 
                    c.component_id,
                    c.component_name,
                    c.component_type,
                    c.description,
                    c.created_at,
                    c.updated_at,
                    m.material_name,
                    m.base_material,
                    m.material_code,
                    sp.centroid_x,
                    sp.centroid_y,
                    sp.centroid_z,
                    sp.bbox_x_mm,
                    sp.bbox_y_mm,
                    sp.bbox_z_mm,
                    d.length_mm,
                    d.width_mm,
                    d.height_mm,
                    d.area_m2,
                    d.volume_cm3,
                    d.tolerance_mm,
                    gp.geometry_format,
                    gp.geometry_type,
                    gp.vertex_count,
                    gp.face_count,
                    gp.edge_count,
                    gp.bounding_box_volume_cm3,
                    gp.surface_area_m2
                FROM components c
                LEFT JOIN component_materials cm ON c.component_id = cm.component_id
                LEFT JOIN materials m ON cm.material_id = m.material_id
                LEFT JOIN spatial_data sp ON c.component_id = sp.component_id
                LEFT JOIN dimensions d ON c.component_id = d.component_id
                LEFT JOIN geometry_properties gp ON c.component_id = gp.component_id
                WHERE sp.centroid_x IS NOT NULL
                ORDER BY c.created_at DESC
            """
            params = None
        
        try:
            results = await self.db_manager.execute_query(query, params)
            
            components = []
            for row in results:
                component = self._transform_db_row_to_component(row)
                components.append(component)
            
            logger.info(f"Retrieved {len(components)} spatial components")
            return components
            
        except Exception as e:
            logger.error(f"Failed to get spatial components: {e}")
            raise
    
    async def run_pipeline_on_database_components(
        self,
        component_type: Optional[str] = None,
        spatial_filter: Optional[List[float]] = None,
        limit: int = 100,
        affinity_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline on components from the database
        """
        await self.initialize()
        
        # Get components from database
        if component_type:
            components = await self.get_components_by_type(component_type, limit)
        elif spatial_filter:
            components = await self.get_spatial_components(spatial_filter)
        else:
            components = await self.get_components_with_relations(limit)
        
        if not components:
            logger.warning("No components found in database")
            return {
                "status": "no_data",
                "message": "No components found in database",
                "components_count": 0
            }
        
        # Set default affinity types if not provided
        if affinity_types is None:
            affinity_types = ["spatial", "structural", "cost", "energy", "mep", "time"]
        
        # Run the pipeline
        logger.info(f"Running pipeline on {len(components)} components")
        pipeline_result = await self.orchestrator.orchestrate_full_pipeline(
            components=components,
            affinity_types=affinity_types
        )
        
        # Add database metadata
        pipeline_result["database_metadata"] = {
            "component_type_filter": component_type,
            "spatial_filter": spatial_filter,
            "limit": limit,
            "affinity_types": affinity_types,
            "database_url": DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else "neon.tech"
        }
        
        return pipeline_result
    
    async def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about pipeline execution and database
        """
        await self.initialize()
        
        try:
            # Get database statistics
            db_stats = await self.db_manager.get_database_stats()
            
            # Get pipeline execution history
            execution_history = self.orchestrator.get_execution_history()
            
            # Get parser statistics
            parser_stats = await self.db_manager.get_parser_statistics()
            
            return {
                "database_stats": db_stats,
                "pipeline_executions": len(execution_history),
                "parser_stats": parser_stats,
                "last_execution": execution_history[-1] if execution_history else None,
                "pipeline_config": self.orchestrator.export_pipeline_config()
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline statistics: {e}")
            raise
    
    def _transform_db_row_to_component(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a database row to component dictionary format
        """
        component = {
            "component_id": str(row.get("component_id")),
            "component_name": row.get("component_name"),
            "component_type": row.get("component_type"),
            "description": row.get("description"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at")
        }
        
        # Add spatial data if available
        if row.get("centroid_x") is not None:
            component["spatial_data"] = {
                "centroid_x": float(row["centroid_x"]),
                "centroid_y": float(row["centroid_y"]),
                "centroid_z": float(row["centroid_z"]),
                "bbox_x_mm": float(row["bbox_x_mm"]) if row.get("bbox_x_mm") else None,
                "bbox_y_mm": float(row["bbox_y_mm"]) if row.get("bbox_y_mm") else None,
                "bbox_z_mm": float(row["bbox_z_mm"]) if row.get("bbox_z_mm") else None
            }
        
        # Add dimensions if available
        if row.get("length_mm") is not None:
            component["dimensions"] = {
                "length_mm": float(row["length_mm"]),
                "width_mm": float(row["width_mm"]),
                "height_mm": float(row["height_mm"]),
                "area_m2": float(row["area_m2"]) if row.get("area_m2") else None,
                "volume_cm3": float(row["volume_cm3"]) if row.get("volume_cm3") else None,
                "tolerance_mm": float(row["tolerance_mm"]) if row.get("tolerance_mm") else None
            }
        
        # Add geometry properties if available
        if row.get("vertex_count") is not None:
            component["geometry_properties"] = {
                "geometry_format": row.get("geometry_format"),
                "geometry_type": row.get("geometry_type"),
                "vertex_count": int(row["vertex_count"]),
                "face_count": int(row["face_count"]),
                "edge_count": int(row["edge_count"]),
                "bounding_box_volume_cm3": float(row["bounding_box_volume_cm3"]) if row.get("bounding_box_volume_cm3") else None,
                "surface_area_m2": float(row["surface_area_m2"]) if row.get("surface_area_m2") else None
            }
        
        # Add materials if available
        if row.get("material_name"):
            component["materials"] = [{
                "material_name": row["material_name"],
                "base_material": row.get("base_material"),
                "material_code": row.get("material_code")
            }]
        
        return component
    
    async def close(self):
        """Close database connections"""
        if self._pool_initialized:
            await self.db_manager.close_async()
            logger.info("Database integration closed")

# Convenience functions for easy usage
async def run_pipeline_on_all_components(limit: int = 100) -> Dict[str, Any]:
    """Run pipeline on all components in database"""
    integration = DatabaseIntegration()
    try:
        return await integration.run_pipeline_on_database_components(limit=limit)
    finally:
        await integration.close()

async def run_pipeline_by_type(component_type: str, limit: int = 50) -> Dict[str, Any]:
    """Run pipeline on components of specific type"""
    integration = DatabaseIntegration()
    try:
        return await integration.run_pipeline_on_database_components(
            component_type=component_type, 
            limit=limit
        )
    finally:
        await integration.close()

async def run_pipeline_spatial(bbox: List[float]) -> Dict[str, Any]:
    """Run pipeline on components within spatial bounding box"""
    integration = DatabaseIntegration()
    try:
        return await integration.run_pipeline_on_database_components(
            spatial_filter=bbox
        )
    finally:
        await integration.close()

async def get_pipeline_stats() -> Dict[str, Any]:
    """Get pipeline and database statistics"""
    integration = DatabaseIntegration()
    try:
        return await integration.get_pipeline_statistics()
    finally:
        await integration.close() 