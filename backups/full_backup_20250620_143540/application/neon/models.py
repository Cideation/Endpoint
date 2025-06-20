from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum

# =====================================================
# ENUMS
# =====================================================

class FileType(str, Enum):
    DXF = "DXF"
    DWG = "DWG"
    IFC = "IFC"
    PDF = "PDF"
    OBJ = "OBJ"
    STEP = "STEP"

class ParsingStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"

class GeometryType(str, Enum):
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"
    MESH = "mesh"
    FACE = "face"
    EDGE = "edge"
    VERTEX = "vertex"

# =====================================================
# FOUNDATIONAL MODELS
# =====================================================

class ComponentBase(BaseModel):
    component_name: Optional[str] = Field(None, max_length=255)
    component_type: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None

class ComponentCreate(ComponentBase):
    pass

class ComponentUpdate(ComponentBase):
    component_name: Optional[str] = Field(None, max_length=255)
    component_type: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None

class Component(ComponentBase):
    component_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# =====================================================
# SPATIAL AND GEOMETRY MODELS
# =====================================================

class SpatialDataBase(BaseModel):
    centroid_x: Optional[float] = Field(None, ge=-180, le=180)
    centroid_y: Optional[float] = Field(None, ge=-90, le=90)
    centroid_z: Optional[float] = None
    bbox_x_mm: Optional[float] = Field(None, ge=0)
    bbox_y_mm: Optional[float] = Field(None, ge=0)
    bbox_z_mm: Optional[float] = Field(None, ge=0)

class SpatialDataCreate(SpatialDataBase):
    component_id: UUID

class SpatialDataUpdate(SpatialDataBase):
    pass

class SpatialData(SpatialDataBase):
    spatial_id: UUID
    component_id: UUID
    location_geom: Optional[str] = None  # PostGIS geometry as WKT
    bounding_box_geom: Optional[str] = None  # PostGIS geometry as WKT
    created_at: datetime

    class Config:
        from_attributes = True

class GeometryPropertiesBase(BaseModel):
    geometry_format: Optional[str] = Field(None, max_length=50)
    geometry_type: Optional[GeometryType] = None
    vertex_count: Optional[int] = Field(None, ge=0)
    face_count: Optional[int] = Field(None, ge=0)
    edge_count: Optional[int] = Field(None, ge=0)
    bounding_box_volume_cm3: Optional[float] = Field(None, ge=0)
    surface_area_m2: Optional[float] = Field(None, ge=0)

class GeometryPropertiesCreate(GeometryPropertiesBase):
    component_id: UUID

class GeometryPropertiesUpdate(GeometryPropertiesBase):
    pass

class GeometryProperties(GeometryPropertiesBase):
    geometry_id: UUID
    component_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True

# =====================================================
# MATERIAL MODELS
# =====================================================

class MaterialBase(BaseModel):
    material_name: Optional[str] = Field(None, max_length=255)
    base_material: Optional[str] = Field(None, max_length=255)
    material_variant: Optional[str] = Field(None, max_length=255)
    material_code: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None

class MaterialCreate(MaterialBase):
    pass

class MaterialUpdate(MaterialBase):
    pass

class Material(MaterialBase):
    material_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True

class ComponentMaterialBase(BaseModel):
    quantity: Optional[float] = Field(None, ge=0)
    unit: Optional[str] = Field(None, max_length=50)

class ComponentMaterialCreate(ComponentMaterialBase):
    component_id: UUID
    material_id: UUID

class ComponentMaterialUpdate(ComponentMaterialBase):
    pass

class ComponentMaterial(ComponentMaterialBase):
    component_id: UUID
    material_id: UUID

    class Config:
        from_attributes = True

# =====================================================
# DIMENSIONS MODELS
# =====================================================

class DimensionsBase(BaseModel):
    length_mm: Optional[float] = Field(None, ge=0)
    width_mm: Optional[float] = Field(None, ge=0)
    height_mm: Optional[float] = Field(None, ge=0)
    area_m2: Optional[float] = Field(None, ge=0)
    volume_cm3: Optional[float] = Field(None, ge=0)
    tolerance_mm: Optional[float] = Field(None, ge=0)

class DimensionsCreate(DimensionsBase):
    component_id: UUID

class DimensionsUpdate(DimensionsBase):
    pass

class Dimensions(DimensionsBase):
    dimension_id: UUID
    component_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True

# =====================================================
# PARSER INTEGRATION MODELS
# =====================================================

class ParsedFileBase(BaseModel):
    file_name: str = Field(..., max_length=255)
    file_path: str
    file_type: FileType
    file_size: int = Field(..., ge=0)
    parsing_status: ParsingStatus = ParsingStatus.PROCESSING
    error_message: Optional[str] = None
    components_extracted: Optional[int] = Field(None, ge=0)
    processing_time_ms: Optional[int] = Field(None, ge=0)

class ParsedFileCreate(ParsedFileBase):
    pass

class ParsedFileUpdate(BaseModel):
    parsing_status: Optional[ParsingStatus] = None
    error_message: Optional[str] = None
    components_extracted: Optional[int] = Field(None, ge=0)
    processing_time_ms: Optional[int] = Field(None, ge=0)

class ParsedFile(ParsedFileBase):
    file_id: UUID
    parsed_at: datetime

    class Config:
        from_attributes = True

class ParsedComponentBase(BaseModel):
    parser_component_id: str = Field(..., max_length=255)
    parser_type: str = Field(..., max_length=50)
    parser_data: Dict[str, Any] = Field(default_factory=dict)

class ParsedComponentCreate(ParsedComponentBase):
    file_id: UUID
    component_id: UUID

class ParsedComponentUpdate(BaseModel):
    parser_data: Optional[Dict[str, Any]] = None

class ParsedComponent(ParsedComponentBase):
    parsed_component_id: UUID
    file_id: UUID
    component_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True

# =====================================================
# COMPREHENSIVE COMPONENT MODELS
# =====================================================

class ComponentWithRelations(BaseModel):
    """Complete component with all related data"""
    component: Component
    spatial_data: Optional[SpatialData] = None
    geometry_properties: Optional[GeometryProperties] = None
    dimensions: Optional[Dimensions] = None
    materials: List[Material] = Field(default_factory=list)
    parsed_components: List[ParsedComponent] = Field(default_factory=list)

    class Config:
        from_attributes = True

# =====================================================
# SFDE INTEGRATION MODELS
# =====================================================

class SFDEInput(BaseModel):
    """Input model for SFDE calculations"""
    component_id: UUID
    affinity_type: str = Field(..., description="Type of calculation (cost, energy, spatial, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None

class SFDEOutput(BaseModel):
    """Output model for SFDE calculations"""
    component_id: UUID
    affinity_type: str
    results: Dict[str, Any] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = None

class SFDEOnboardingData(BaseModel):
    """Data structure for SFDE onboarding process"""
    components: List[ComponentWithRelations]
    affinity_types: List[str] = Field(default_factory=list)
    calculation_parameters: Dict[str, Any] = Field(default_factory=dict)

# =====================================================
# VALIDATION AND UTILITY MODELS
# =====================================================

class ValidationError(BaseModel):
    field: str
    message: str
    value: Any

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[ValidationError] = Field(default_factory=list)

class BulkOperationResult(BaseModel):
    success_count: int
    error_count: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)

# =====================================================
# RESPONSE MODELS
# =====================================================

class ComponentResponse(BaseModel):
    success: bool
    data: Optional[ComponentWithRelations] = None
    message: Optional[str] = None
    errors: List[str] = Field(default_factory=list)

class ComponentsListResponse(BaseModel):
    success: bool
    data: List[ComponentWithRelations] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    message: Optional[str] = None

class ParserResponse(BaseModel):
    success: bool
    file_id: Optional[UUID] = None
    components_created: int = 0
    processing_time_ms: Optional[int] = None
    message: Optional[str] = None
    errors: List[str] = Field(default_factory=list) 