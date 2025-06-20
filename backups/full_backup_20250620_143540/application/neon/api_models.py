"""
Frontend-Ready API Models for Neon Database Integration
Consolidated and standardized models for frontend consumption
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from pydantic import BaseModel, Field, validator, HttpUrl
from enum import Enum

# =====================================================
# FRONTEND-READY ENUMS
# =====================================================

class FileType(str, Enum):
    """Supported file types for parsing"""
    DXF = "DXF"
    DWG = "DWG"
    IFC = "IFC"
    PDF = "PDF"
    OBJ = "OBJ"
    STEP = "STEP"

class ComponentType(str, Enum):
    """Standardized component types"""
    STRUCTURAL = "structural"
    MEP = "mep"
    ARCHITECTURAL = "architectural"
    CIVIL = "civil"
    LANDSCAPE = "landscape"
    FURNITURE = "furniture"
    EQUIPMENT = "equipment"
    UNKNOWN = "unknown"

class ProcessingStatus(str, Enum):
    """File processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"

class GeometryType(str, Enum):
    """Geometry types for components"""
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"
    MESH = "mesh"
    FACE = "face"
    EDGE = "edge"
    VERTEX = "vertex"
    SOLID = "solid"

# =====================================================
# CORE FRONTEND MODELS
# =====================================================

class ComponentSummary(BaseModel):
    """Simplified component model for frontend lists"""
    component_id: UUID
    component_name: str
    component_type: ComponentType
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    # Quick stats
    has_spatial_data: bool = False
    has_materials: bool = False
    has_dimensions: bool = False
    material_count: int = 0
    
    class Config:
        from_attributes = True

class ComponentDetail(BaseModel):
    """Detailed component model for frontend detail views"""
    component_id: UUID
    component_name: str
    component_type: ComponentType
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    # Spatial data
    spatial_data: Optional['SpatialDataDetail'] = None
    
    # Dimensions
    dimensions: Optional['DimensionsDetail'] = None
    
    # Materials
    materials: List['MaterialDetail'] = Field(default_factory=list)
    
    # Geometry properties
    geometry_properties: Optional['GeometryPropertiesDetail'] = None
    
    # Parsing information
    parsed_files: List['ParsedFileSummary'] = Field(default_factory=list)
    
    class Config:
        from_attributes = True

class SpatialDataDetail(BaseModel):
    """Spatial data for component detail view"""
    spatial_id: UUID
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None
    centroid_z: Optional[float] = None
    bbox_x_mm: Optional[float] = None
    bbox_y_mm: Optional[float] = None
    bbox_z_mm: Optional[float] = None
    
    class Config:
        from_attributes = True

class DimensionsDetail(BaseModel):
    """Dimensions for component detail view"""
    dimension_id: UUID
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    area_m2: Optional[float] = None
    volume_cm3: Optional[float] = None
    tolerance_mm: Optional[float] = None
    
    class Config:
        from_attributes = True

class MaterialDetail(BaseModel):
    """Material detail for component view"""
    material_id: UUID
    material_name: str
    base_material: Optional[str] = None
    material_variant: Optional[str] = None
    material_code: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    
    class Config:
        from_attributes = True

class GeometryPropertiesDetail(BaseModel):
    """Geometry properties for component detail view"""
    geometry_id: UUID
    geometry_format: Optional[str] = None
    geometry_type: Optional[GeometryType] = None
    vertex_count: Optional[int] = None
    face_count: Optional[int] = None
    edge_count: Optional[int] = None
    bounding_box_volume_cm3: Optional[float] = None
    surface_area_m2: Optional[float] = None
    
    class Config:
        from_attributes = True

# =====================================================
# FILE PROCESSING MODELS
# =====================================================

class FileUploadRequest(BaseModel):
    """Request model for file upload"""
    filename: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="Type of file to process")
    description: Optional[str] = Field(None, description="Optional file description")

class FileProcessingStatus(BaseModel):
    """File processing status for frontend"""
    file_id: UUID
    filename: str
    file_type: FileType
    status: ProcessingStatus
    progress_percentage: int = Field(0, ge=0, le=100)
    components_extracted: int = 0
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ParsedFileSummary(BaseModel):
    """Summary of parsed file for component view"""
    file_id: UUID
    filename: str
    file_type: FileType
    status: ProcessingStatus
    components_extracted: int
    processing_time_ms: Optional[int] = None
    parsed_at: datetime
    
    class Config:
        from_attributes = True

# =====================================================
# API RESPONSE MODELS
# =====================================================

class ApiResponse(BaseModel):
    """Base API response model"""
    success: bool
    message: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

class ComponentListResponse(ApiResponse):
    """Response for component list requests"""
    data: List[ComponentSummary] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    total_pages: int = 0

class ComponentDetailResponse(ApiResponse):
    """Response for component detail requests"""
    data: Optional[ComponentDetail] = None

class FileProcessingResponse(ApiResponse):
    """Response for file processing requests"""
    file_id: Optional[UUID] = None
    components_created: int = 0
    processing_time_ms: Optional[int] = None

class FileStatusResponse(ApiResponse):
    """Response for file status requests"""
    data: Optional[FileProcessingStatus] = None

class FileListResponse(ApiResponse):
    """Response for file list requests"""
    data: List[FileProcessingStatus] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    total_pages: int = 0

# =====================================================
# SEARCH AND FILTER MODELS
# =====================================================

class ComponentSearchRequest(BaseModel):
    """Request model for component search"""
    query: Optional[str] = Field(None, description="Search query")
    component_type: Optional[ComponentType] = Field(None, description="Filter by component type")
    has_spatial_data: Optional[bool] = Field(None, description="Filter by spatial data presence")
    has_materials: Optional[bool] = Field(None, description="Filter by materials presence")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=1, le=100, description="Page size")

class SpatialBoundsFilter(BaseModel):
    """Spatial bounds filter for components"""
    min_x: Optional[float] = Field(None, description="Minimum X coordinate")
    max_x: Optional[float] = Field(None, description="Maximum X coordinate")
    min_y: Optional[float] = Field(None, description="Minimum Y coordinate")
    max_y: Optional[float] = Field(None, description="Maximum Y coordinate")
    min_z: Optional[float] = Field(None, description="Minimum Z coordinate")
    max_z: Optional[float] = Field(None, description="Maximum Z coordinate")

class FileSearchRequest(BaseModel):
    """Request model for file search"""
    query: Optional[str] = Field(None, description="Search query")
    file_type: Optional[FileType] = Field(None, description="Filter by file type")
    status: Optional[ProcessingStatus] = Field(None, description="Filter by processing status")
    uploaded_after: Optional[datetime] = Field(None, description="Filter by upload date")
    uploaded_before: Optional[datetime] = Field(None, description="Filter by upload date")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=1, le=100, description="Page size")

# =====================================================
# STATISTICS MODELS
# =====================================================

class ComponentStatistics(BaseModel):
    """Component statistics for dashboard"""
    total_components: int = 0
    components_by_type: Dict[str, int] = Field(default_factory=dict)
    components_with_spatial_data: int = 0
    components_with_materials: int = 0
    components_with_dimensions: int = 0
    average_processing_time_ms: Optional[float] = None
    components_created_today: int = 0
    components_created_this_week: int = 0
    components_created_this_month: int = 0

class FileProcessingStatistics(BaseModel):
    """File processing statistics for dashboard"""
    total_files: int = 0
    files_by_type: Dict[str, int] = Field(default_factory=dict)
    files_by_status: Dict[str, int] = Field(default_factory=dict)
    total_components_extracted: int = 0
    average_processing_time_ms: Optional[float] = None
    files_processed_today: int = 0
    files_processed_this_week: int = 0
    files_processed_this_month: int = 0

class DashboardStatistics(BaseModel):
    """Complete dashboard statistics"""
    components: ComponentStatistics
    files: FileProcessingStatistics
    last_updated: datetime = Field(default_factory=datetime.now)

# =====================================================
# VALIDATION MODELS
# =====================================================

class ValidationError(BaseModel):
    """Validation error for API responses"""
    field: str
    message: str
    value: Any

class ValidationResponse(BaseModel):
    """Validation response for API requests"""
    is_valid: bool
    errors: List[ValidationError] = Field(default_factory=list)

# =====================================================
# BULK OPERATION MODELS
# =====================================================

class BulkOperationRequest(BaseModel):
    """Request for bulk operations"""
    operation: str = Field(..., description="Type of bulk operation")
    component_ids: List[UUID] = Field(default_factory=list, description="Component IDs to operate on")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")

class BulkOperationResult(BaseModel):
    """Result of bulk operation"""
    operation: str
    success_count: int = 0
    error_count: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: Optional[int] = None

class BulkOperationResponse(ApiResponse):
    """Response for bulk operations"""
    data: Optional[BulkOperationResult] = None

# =====================================================
# EXPORT MODELS
# =====================================================

class ExportFormat(str, Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    GEOJSON = "geojson"

class ExportRequest(BaseModel):
    """Request for data export"""
    format: ExportFormat = Field(..., description="Export format")
    component_ids: Optional[List[UUID]] = Field(None, description="Specific component IDs to export")
    include_spatial_data: bool = Field(True, description="Include spatial data in export")
    include_materials: bool = Field(True, description="Include materials in export")
    include_dimensions: bool = Field(True, description="Include dimensions in export")
    filters: Optional[ComponentSearchRequest] = Field(None, description="Filters to apply")

class ExportResponse(ApiResponse):
    """Response for export requests"""
    download_url: Optional[HttpUrl] = Field(None, description="Download URL for exported file")
    file_size_bytes: Optional[int] = Field(None, description="Size of exported file")
    expires_at: Optional[datetime] = Field(None, description="When download URL expires") 