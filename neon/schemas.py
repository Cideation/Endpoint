"""
JSON Schemas for Neon Database and Microservices Integration
Aligned with containerized architecture: ne-dag-alpha, ne-graph-runtime-engine, SFDE
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
import json

# =====================================================
# MICROSERVICES NODE SCHEMAS
# =====================================================

class CallbackType(str, Enum):
    DAG = "dag"
    RELATIONAL = "relational"
    COMBINATORIAL = "combinatorial"

class Phase(str, Enum):
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"

class ExecutionPhase(str, Enum):
    """Execution phases for the BEM system"""
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    CROSS_PHASE = "cross_phase"

class NodeDictionary(BaseModel):
    """Schema for node_dictionarY.json used by microservices"""
    node_id: str = Field(..., description="Unique node identifier")
    node_label: str = Field(..., description="Human-readable node label")
    phase: Phase = Field(..., description="Processing phase")
    agent: int = Field(..., ge=1, le=10, description="Agent identifier")
    callback_type: CallbackType = Field(..., description="Primary callback type")
    trigger_functor: str = Field(..., description="Triggering functor name")
    dictionary: Dict[str, Any] = Field(default_factory=dict, description="Node-specific data")
    allowed_callback_types: List[CallbackType] = Field(default_factory=list)

class NodeDictionaryCollection(BaseModel):
    """Complete node dictionary collection"""
    nodes: List[NodeDictionary] = Field(default_factory=list)

# =====================================================
# FUNCTOR EDGE SCHEMAS
# =====================================================

class FunctorEdge(BaseModel):
    """Schema for functor edges between nodes"""
    edge_id: str = Field(..., description="Unique edge identifier")
    source_node: str = Field(..., description="Source node ID")
    target_node: str = Field(..., description="Target node ID")
    edge_type: str = Field(..., description="Type of edge relationship")
    weight: Optional[float] = Field(None, ge=0, le=1, description="Edge weight")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FunctorEdgeCollection(BaseModel):
    """Complete functor edge collection"""
    edges: List[FunctorEdge] = Field(default_factory=list)

# =====================================================
# AGENT COEFFICIENTS SCHEMAS
# =====================================================

class AgentCoefficient(BaseModel):
    """Schema for agent coefficients"""
    agent_id: int = Field(..., ge=1, le=10)
    coefficient_name: str = Field(..., description="Name of the coefficient")
    value: float = Field(..., description="Coefficient value")
    phase: Phase = Field(..., description="Applicable phase")
    description: Optional[str] = None

class AgentCoefficientsCollection(BaseModel):
    """Complete agent coefficients collection"""
    coefficients: List[AgentCoefficient] = Field(default_factory=list)

# =====================================================
# SFDE INTEGRATION SCHEMAS
# =====================================================

class SFDEAffinityType(str, Enum):
    COST = "cost"
    ENERGY = "energy"
    SPATIAL = "spatial"
    STRUCTURAL = "structural"
    MEP = "mep"
    TIME = "time"

class SFDEFormula(BaseModel):
    """Schema for SFDE formula definitions"""
    formula_id: str = Field(..., description="Unique formula identifier")
    formula_name: str = Field(..., description="Human-readable formula name")
    affinity_type: SFDEAffinityType = Field(..., description="Formula affinity type")
    formula_expression: str = Field(..., description="Mathematical expression")
    input_parameters: List[str] = Field(default_factory=list, description="Required input parameters")
    output_parameters: List[str] = Field(default_factory=list, description="Output parameters")
    description: Optional[str] = None
    version: str = Field(default="1.0", description="Formula version")

class SFDEFormulaCollection(BaseModel):
    """Complete SFDE formula collection"""
    formulas: List[SFDEFormula] = Field(default_factory=list)

# =====================================================
# DATABASE INTEGRATION SCHEMAS
# =====================================================

class DatabaseComponentSchema(BaseModel):
    """Schema for database component validation"""
    component_id: str = Field(..., description="Component UUID")
    component_name: Optional[str] = Field(None, max_length=255)
    component_type: Optional[str] = Field(None, max_length=100)
    spatial_data: Optional[Dict[str, Any]] = None
    geometry_properties: Optional[Dict[str, Any]] = None
    materials: List[Dict[str, Any]] = Field(default_factory=list)
    dimensions: Optional[Dict[str, Any]] = None

class DatabaseComponentCollection(BaseModel):
    """Collection of database components"""
    components: List[DatabaseComponentSchema] = Field(default_factory=list)
    total_count: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=50)

# =====================================================
# PARSER INTEGRATION SCHEMAS
# =====================================================

class ParserOutputSchema(BaseModel):
    """Schema for parser output validation"""
    file_id: str = Field(..., description="Parsed file UUID")
    file_type: str = Field(..., description="File type (DXF, DWG, IFC, etc.)")
    parsing_status: str = Field(..., description="Parsing status")
    components: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted components")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata")
    processing_time_ms: Optional[int] = Field(None, ge=0)

# =====================================================
# MICROSERVICE COMMUNICATION SCHEMAS
# =====================================================

class MicroserviceMessage(BaseModel):
    """Base schema for microservice communication"""
    message_id: str = Field(..., description="Unique message identifier")
    source_service: str = Field(..., description="Source microservice")
    target_service: str = Field(..., description="Target microservice")
    message_type: str = Field(..., description="Type of message")
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(..., description="ISO timestamp")
    correlation_id: Optional[str] = None

class DAGExecutionRequest(BaseModel):
    """Schema for DAG execution requests"""
    dag_id: str = Field(..., description="DAG identifier")
    input_data: Dict[str, Any] = Field(default_factory=dict)
    node_sequence: List[str] = Field(default_factory=list, description="Node execution sequence")
    callback_type: CallbackType = Field(..., description="Callback type for execution")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class GraphOperationRequest(BaseModel):
    """Schema for graph operation requests"""
    operation_type: str = Field(..., description="Type of graph operation")
    graph_data: Dict[str, Any] = Field(default_factory=dict)
    node_data: Optional[List[NodeDictionary]] = None
    edge_data: Optional[List[FunctorEdge]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

# =====================================================
# SFDE ONBOARDING SCHEMAS
# =====================================================

class SFDEOnboardingRequest(BaseModel):
    """Schema for SFDE onboarding requests"""
    project_id: str = Field(..., description="Project identifier")
    components: List[DatabaseComponentSchema] = Field(default_factory=list)
    affinity_types: List[SFDEAffinityType] = Field(default_factory=list)
    calculation_parameters: Dict[str, Any] = Field(default_factory=dict)
    node_mapping: Dict[str, str] = Field(default_factory=dict, description="Component to node mapping")

class SFDEOnboardingResponse(BaseModel):
    """Schema for SFDE onboarding responses"""
    project_id: str = Field(..., description="Project identifier")
    onboarding_status: str = Field(..., description="Onboarding status")
    processed_components: int = Field(default=0)
    created_nodes: int = Field(default=0)
    created_edges: int = Field(default=0)
    formulas_loaded: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# =====================================================
# CONTAINER CONFIGURATION SCHEMAS
# =====================================================

class ContainerConfig(BaseModel):
    """Schema for container configuration"""
    service_name: str = Field(..., description="Service name")
    container_id: str = Field(..., description="Container identifier")
    input_paths: List[str] = Field(default_factory=list, description="Input file paths")
    output_paths: List[str] = Field(default_factory=list, description="Output file paths")
    shared_paths: List[str] = Field(default_factory=list, description="Shared file paths")
    environment_variables: Dict[str, str] = Field(default_factory=dict)

class ContainerStatus(BaseModel):
    """Schema for container status"""
    service_name: str = Field(..., description="Service name")
    status: str = Field(..., description="Container status")
    health_check: str = Field(..., description="Health check status")
    uptime_seconds: int = Field(default=0)
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

# =====================================================
# VALIDATION UTILITIES
# =====================================================

def validate_node_dictionary(data: Dict[str, Any]) -> bool:
    """Validate node dictionary JSON"""
    try:
        NodeDictionaryCollection(**data)
        return True
    except Exception:
        return False

def validate_functor_edges(data: Dict[str, Any]) -> bool:
    """Validate functor edges JSON"""
    try:
        FunctorEdgeCollection(**data)
        return True
    except Exception:
        return False

def validate_agent_coefficients(data: Dict[str, Any]) -> bool:
    """Validate agent coefficients JSON"""
    try:
        AgentCoefficientsCollection(**data)
        return True
    except Exception:
        return False

def validate_sfde_formulas(data: Dict[str, Any]) -> bool:
    """Validate SFDE formulas JSON"""
    try:
        SFDEFormulaCollection(**data)
        return True
    except Exception:
        return False

# =====================================================
# SCHEMA EXPORT FUNCTIONS
# =====================================================

def export_node_dictionary_schema() -> Dict[str, Any]:
    """Export node dictionary schema as JSON"""
    return NodeDictionaryCollection.schema()

def export_functor_edges_schema() -> Dict[str, Any]:
    """Export functor edges schema as JSON"""
    return FunctorEdgeCollection.schema()

def export_agent_coefficients_schema() -> Dict[str, Any]:
    """Export agent coefficients schema as JSON"""
    return AgentCoefficientsCollection.schema()

def export_sfde_formulas_schema() -> Dict[str, Any]:
    """Export SFDE formulas schema as JSON"""
    return SFDEFormulaCollection.schema()

def export_all_schemas() -> Dict[str, Any]:
    """Export all schemas as JSON"""
    return {
        "node_dictionary": export_node_dictionary_schema(),
        "functor_edges": export_functor_edges_schema(),
        "agent_coefficients": export_agent_coefficients_schema(),
        "sfde_formulas": export_sfde_formulas_schema(),
        "database_components": DatabaseComponentCollection.schema(),
        "parser_output": ParserOutputSchema.schema(),
        "microservice_messages": MicroserviceMessage.schema(),
        "dag_execution": DAGExecutionRequest.schema(),
        "graph_operations": GraphOperationRequest.schema(),
        "sfde_onboarding": SFDEOnboardingRequest.schema(),
        "container_config": ContainerConfig.schema()
    } 