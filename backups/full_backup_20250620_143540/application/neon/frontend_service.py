"""
Frontend-Ready API Service for Neon Database Integration
Provides clean, standardized endpoints for frontend consumption
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID

from .db_manager import NeonDBManager
from .config import NEON_CONFIG
from .api_models import (
    ComponentSummary, ComponentDetail, ComponentListResponse, ComponentDetailResponse,
    FileProcessingStatus, FileProcessingResponse, FileStatusResponse, FileListResponse,
    ComponentSearchRequest, FileSearchRequest, DashboardStatistics,
    ComponentStatistics, FileProcessingStatistics, BulkOperationRequest,
    BulkOperationResponse, BulkOperationResult, ProcessingStatus, ComponentType, FileType
)
from .file_processor import FileProcessor

logger = logging.getLogger(__name__)

class FrontendAPIService:
    """
    Frontend-ready API service with standardized endpoints
    """
    
    def __init__(self):
        self.db_manager = NeonDBManager(NEON_CONFIG)
        self.file_processor = FileProcessor()
        self._pool_initialized = False
    
    async def initialize(self):
        """Initialize database connection"""
        if not self._pool_initialized:
            await self.db_manager.create_pool()
            await self.file_processor.initialize()
            self._pool_initialized = True
    
    async def get_components(self, search_request: ComponentSearchRequest) -> ComponentListResponse:
        """Get paginated list of components with search and filters"""
        try:
            await self.initialize()
            
            # Build query with filters
            query, params = self._build_component_search_query(search_request)
            
            # Get total count
            count_query = f"SELECT COUNT(*) FROM ({query}) as subquery"
            count_result = await self.db_manager.execute_query(count_query, params)
            total_count = count_result[0]['count'] if count_result else 0
            
            # Add pagination
            offset = (search_request.page - 1) * search_request.page_size
            paginated_query = f"{query} ORDER BY c.created_at DESC LIMIT {search_request.page_size} OFFSET {offset}"
            
            # Execute query
            results = await self.db_manager.execute_query(paginated_query, params)
            
            # Transform to ComponentSummary
            components = []
            for row in results:
                component = ComponentSummary(
                    component_id=row['component_id'],
                    component_name=row['component_name'] or 'Unnamed Component',
                    component_type=ComponentType(row['component_type'] or 'unknown'),
                    description=row['description'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    has_spatial_data=bool(row['has_spatial_data']),
                    has_materials=bool(row['has_materials']),
                    has_dimensions=bool(row['has_dimensions']),
                    material_count=row['material_count'] or 0
                )
                components.append(component)
            
            total_pages = (total_count + search_request.page_size - 1) // search_request.page_size
            
            return ComponentListResponse(
                success=True,
                data=components,
                total_count=total_count,
                page=search_request.page,
                page_size=search_request.page_size,
                total_pages=total_pages,
                message=f"Retrieved {len(components)} components"
            )
            
        except Exception as e:
            logger.error(f"Error getting components: {e}")
            return ComponentListResponse(
                success=False,
                message="Failed to retrieve components",
                errors=[str(e)]
            )
    
    async def get_component_detail(self, component_id: UUID) -> ComponentDetailResponse:
        """Get detailed component information"""
        try:
            await self.initialize()
            
            # Get component with all relations
            query = """
                SELECT 
                    c.*,
                    sd.spatial_id, sd.centroid_x, sd.centroid_y, sd.centroid_z,
                    sd.bbox_x_mm, sd.bbox_y_mm, sd.bbox_z_mm,
                    d.dimension_id, d.length_mm, d.width_mm, d.height_mm,
                    d.area_m2, d.volume_cm3, d.tolerance_mm,
                    gp.geometry_id, gp.geometry_format, gp.geometry_type,
                    gp.vertex_count, gp.face_count, gp.edge_count,
                    gp.bounding_box_volume_cm3, gp.surface_area_m2
                FROM components c
                LEFT JOIN spatial_data sd ON c.component_id = sd.component_id
                LEFT JOIN dimensions d ON c.component_id = d.component_id
                LEFT JOIN geometry_properties gp ON c.component_id = gp.component_id
                WHERE c.component_id = $1
            """
            
            result = await self.db_manager.execute_query(query, (component_id,))
            
            if not result:
                return ComponentDetailResponse(
                    success=False,
                    message="Component not found",
                    errors=["Component not found"]
                )
            
            row = result[0]
            
            # Get materials
            materials_query = """
                SELECT m.*, cm.quantity, cm.unit
                FROM materials m
                JOIN component_materials cm ON m.material_id = cm.material_id
                WHERE cm.component_id = $1
            """
            materials_result = await self.db_manager.execute_query(materials_query, (component_id,))
            
            # Get parsed files
            files_query = """
                SELECT pf.*
                FROM parsed_files pf
                JOIN parsed_components pc ON pf.file_id = pc.file_id
                WHERE pc.component_id = $1
            """
            files_result = await self.db_manager.execute_query(files_query, (component_id,))
            
            # Build component detail
            component_detail = ComponentDetail(
                component_id=row['component_id'],
                component_name=row['component_name'] or 'Unnamed Component',
                component_type=ComponentType(row['component_type'] or 'unknown'),
                description=row['description'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                spatial_data=self._build_spatial_data_detail(row) if row['spatial_id'] else None,
                dimensions=self._build_dimensions_detail(row) if row['dimension_id'] else None,
                geometry_properties=self._build_geometry_properties_detail(row) if row['geometry_id'] else None,
                materials=[self._build_material_detail(m) for m in materials_result],
                parsed_files=[self._build_parsed_file_summary(f) for f in files_result]
            )
            
            return ComponentDetailResponse(
                success=True,
                data=component_detail,
                message="Component details retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Error getting component detail: {e}")
            return ComponentDetailResponse(
                success=False,
                message="Failed to retrieve component details",
                errors=[str(e)]
            )
    
    async def process_file(self, file_path: str, original_filename: str, file_type: str) -> FileProcessingResponse:
        """Process uploaded file and extract components"""
        try:
            await self.initialize()
            
            # Use file processor to process and discard raw file
            result = await self.file_processor.process_and_discard(
                file_path, original_filename, file_type
            )
            
            return FileProcessingResponse(
                success=result.success,
                file_id=result.file_id,
                components_created=result.components_created,
                processing_time_ms=result.processing_time_ms,
                message=result.message,
                errors=result.errors
            )
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return FileProcessingResponse(
                success=False,
                message="Failed to process file",
                errors=[str(e)]
            )
    
    async def get_dashboard_statistics(self) -> DashboardStatistics:
        """Get comprehensive dashboard statistics"""
        try:
            await self.initialize()
            
            # Component statistics
            component_stats = await self._get_component_statistics()
            
            # File processing statistics
            file_stats = await self._get_file_processing_statistics()
            
            return DashboardStatistics(
                components=component_stats,
                files=file_stats,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting dashboard statistics: {e}")
            # Return empty statistics on error
            return DashboardStatistics(
                components=ComponentStatistics(),
                files=FileProcessingStatistics(),
                last_updated=datetime.now()
            )
    
    def _build_component_search_query(self, search_request: ComponentSearchRequest) -> tuple:
        """Build SQL query for component search with filters"""
        base_query = """
            SELECT 
                c.*,
                CASE WHEN sd.component_id IS NOT NULL THEN true ELSE false END as has_spatial_data,
                CASE WHEN d.component_id IS NOT NULL THEN true ELSE false END as has_dimensions,
                CASE WHEN cm.component_id IS NOT NULL THEN true ELSE false END as has_materials,
                COUNT(DISTINCT cm.material_id) as material_count
            FROM components c
            LEFT JOIN spatial_data sd ON c.component_id = sd.component_id
            LEFT JOIN dimensions d ON c.component_id = d.component_id
            LEFT JOIN component_materials cm ON c.component_id = cm.component_id
        """
        
        conditions = []
        params = []
        param_count = 0
        
        if search_request.query:
            param_count += 1
            conditions.append(f"(c.component_name ILIKE ${param_count} OR c.description ILIKE ${param_count})")
            params.append(f"%{search_request.query}%")
        
        if search_request.component_type:
            param_count += 1
            conditions.append(f"c.component_type = ${param_count}")
            params.append(search_request.component_type.value)
        
        if search_request.has_spatial_data is not None:
            if search_request.has_spatial_data:
                conditions.append("sd.component_id IS NOT NULL")
            else:
                conditions.append("sd.component_id IS NULL")
        
        if search_request.has_materials is not None:
            if search_request.has_materials:
                conditions.append("cm.component_id IS NOT NULL")
            else:
                conditions.append("cm.component_id IS NULL")
        
        if search_request.created_after:
            param_count += 1
            conditions.append(f"c.created_at >= ${param_count}")
            params.append(search_request.created_after)
        
        if search_request.created_before:
            param_count += 1
            conditions.append(f"c.created_at <= ${param_count}")
            params.append(search_request.created_before)
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        base_query += " GROUP BY c.component_id, sd.component_id, d.component_id, cm.component_id"
        
        return base_query, params
    
    async def _get_component_statistics(self) -> ComponentStatistics:
        """Get component statistics"""
        try:
            # Total components
            total_query = "SELECT COUNT(*) FROM components"
            total_result = await self.db_manager.execute_query(total_query)
            total_components = total_result[0]['count'] if total_result else 0
            
            # Components by type
            type_query = "SELECT component_type, COUNT(*) FROM components GROUP BY component_type"
            type_result = await self.db_manager.execute_query(type_query)
            components_by_type = {row['component_type'] or 'unknown': row['count'] for row in type_result}
            
            # Components with spatial data
            spatial_query = "SELECT COUNT(DISTINCT c.component_id) FROM components c JOIN spatial_data sd ON c.component_id = sd.component_id"
            spatial_result = await self.db_manager.execute_query(spatial_query)
            components_with_spatial_data = spatial_result[0]['count'] if spatial_result else 0
            
            # Components with materials
            materials_query = "SELECT COUNT(DISTINCT c.component_id) FROM components c JOIN component_materials cm ON c.component_id = cm.component_id"
            materials_result = await self.db_manager.execute_query(materials_query)
            components_with_materials = materials_result[0]['count'] if materials_result else 0
            
            # Components with dimensions
            dimensions_query = "SELECT COUNT(DISTINCT c.component_id) FROM components c JOIN dimensions d ON c.component_id = d.component_id"
            dimensions_result = await self.db_manager.execute_query(dimensions_query)
            components_with_dimensions = dimensions_result[0]['count'] if dimensions_result else 0
            
            # Recent components
            today = datetime.now().date()
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            today_query = "SELECT COUNT(*) FROM components WHERE DATE(created_at) = $1"
            today_result = await self.db_manager.execute_query(today_query, (today,))
            components_created_today = today_result[0]['count'] if today_result else 0
            
            week_query = "SELECT COUNT(*) FROM components WHERE created_at >= $1"
            week_result = await self.db_manager.execute_query(week_query, (week_ago,))
            components_created_this_week = week_result[0]['count'] if week_result else 0
            
            month_query = "SELECT COUNT(*) FROM components WHERE created_at >= $1"
            month_result = await self.db_manager.execute_query(month_query, (month_ago,))
            components_created_this_month = month_result[0]['count'] if month_result else 0
            
            return ComponentStatistics(
                total_components=total_components,
                components_by_type=components_by_type,
                components_with_spatial_data=components_with_spatial_data,
                components_with_materials=components_with_materials,
                components_with_dimensions=components_with_dimensions,
                components_created_today=components_created_today,
                components_created_this_week=components_created_this_week,
                components_created_this_month=components_created_this_month
            )
            
        except Exception as e:
            logger.error(f"Error getting component statistics: {e}")
            return ComponentStatistics()
    
    async def _get_file_processing_statistics(self) -> FileProcessingStatistics:
        """Get file processing statistics"""
        try:
            # Total files
            total_query = "SELECT COUNT(*) FROM parsed_files"
            total_result = await self.db_manager.execute_query(total_query)
            total_files = total_result[0]['count'] if total_result else 0
            
            # Files by type
            type_query = "SELECT file_type, COUNT(*) FROM parsed_files GROUP BY file_type"
            type_result = await self.db_manager.execute_query(type_query)
            files_by_type = {row['file_type']: row['count'] for row in type_result}
            
            # Files by status
            status_query = "SELECT parsing_status, COUNT(*) FROM parsed_files GROUP BY parsing_status"
            status_result = await self.db_manager.execute_query(status_query)
            files_by_status = {row['parsing_status']: row['count'] for row in status_result}
            
            # Total components extracted
            components_query = "SELECT COALESCE(SUM(components_extracted), 0) FROM parsed_files WHERE parsing_status = 'success'"
            components_result = await self.db_manager.execute_query(components_query)
            total_components_extracted = components_result[0]['sum'] if components_result else 0
            
            # Average processing time
            time_query = "SELECT AVG(processing_time_ms) FROM parsed_files WHERE parsing_status = 'success' AND processing_time_ms IS NOT NULL"
            time_result = await self.db_manager.execute_query(time_query)
            average_processing_time_ms = time_result[0]['avg'] if time_result else None
            
            # Recent files
            today = datetime.now().date()
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            today_query = "SELECT COUNT(*) FROM parsed_files WHERE DATE(created_at) = $1"
            today_result = await self.db_manager.execute_query(today_query, (today,))
            files_processed_today = today_result[0]['count'] if today_result else 0
            
            week_query = "SELECT COUNT(*) FROM parsed_files WHERE created_at >= $1"
            week_result = await self.db_manager.execute_query(week_query, (week_ago,))
            files_processed_this_week = week_result[0]['count'] if week_result else 0
            
            month_query = "SELECT COUNT(*) FROM parsed_files WHERE created_at >= $1"
            month_result = await self.db_manager.execute_query(month_query, (month_ago,))
            files_processed_this_month = month_result[0]['count'] if month_result else 0
            
            return FileProcessingStatistics(
                total_files=total_files,
                files_by_type=files_by_type,
                files_by_status=files_by_status,
                total_components_extracted=total_components_extracted,
                average_processing_time_ms=average_processing_time_ms,
                files_processed_today=files_processed_today,
                files_processed_this_week=files_processed_this_week,
                files_processed_this_month=files_processed_this_month
            )
            
        except Exception as e:
            logger.error(f"Error getting file processing statistics: {e}")
            return FileProcessingStatistics()
    
    def _build_spatial_data_detail(self, row: Dict[str, Any]):
        """Build spatial data detail from database row"""
        from .api_models import SpatialDataDetail
        return SpatialDataDetail(
            spatial_id=row['spatial_id'],
            centroid_x=row['centroid_x'],
            centroid_y=row['centroid_y'],
            centroid_z=row['centroid_z'],
            bbox_x_mm=row['bbox_x_mm'],
            bbox_y_mm=row['bbox_y_mm'],
            bbox_z_mm=row['bbox_z_mm']
        )
    
    def _build_dimensions_detail(self, row: Dict[str, Any]):
        """Build dimensions detail from database row"""
        from .api_models import DimensionsDetail
        return DimensionsDetail(
            dimension_id=row['dimension_id'],
            length_mm=row['length_mm'],
            width_mm=row['width_mm'],
            height_mm=row['height_mm'],
            area_m2=row['area_m2'],
            volume_cm3=row['volume_cm3'],
            tolerance_mm=row['tolerance_mm']
        )
    
    def _build_material_detail(self, row: Dict[str, Any]):
        """Build material detail from database row"""
        from .api_models import MaterialDetail
        return MaterialDetail(
            material_id=row['material_id'],
            material_name=row['material_name'],
            base_material=row['base_material'],
            material_variant=row['material_variant'],
            material_code=row['material_code'],
            description=row['description'],
            quantity=row['quantity'],
            unit=row['unit']
        )
    
    def _build_geometry_properties_detail(self, row: Dict[str, Any]):
        """Build geometry properties detail from database row"""
        from .api_models import GeometryPropertiesDetail, GeometryType
        return GeometryPropertiesDetail(
            geometry_id=row['geometry_id'],
            geometry_format=row['geometry_format'],
            geometry_type=GeometryType(row['geometry_type']) if row['geometry_type'] else None,
            vertex_count=row['vertex_count'],
            face_count=row['face_count'],
            edge_count=row['edge_count'],
            bounding_box_volume_cm3=row['bounding_box_volume_cm3'],
            surface_area_m2=row['surface_area_m2']
        )
    
    def _build_parsed_file_summary(self, row: Dict[str, Any]):
        """Build parsed file summary from database row"""
        from .api_models import ParsedFileSummary
        return ParsedFileSummary(
            file_id=row['file_id'],
            filename=row['file_name'],
            file_type=FileType(row['file_type']),
            status=ProcessingStatus(row['parsing_status']),
            components_extracted=row['components_extracted'] or 0,
            processing_time_ms=row['processing_time_ms'],
            parsed_at=row['created_at']
        )
    
    async def close(self):
        """Close database connections"""
        if self._pool_initialized:
            await self.db_manager.close_async()
            await self.file_processor.close() 