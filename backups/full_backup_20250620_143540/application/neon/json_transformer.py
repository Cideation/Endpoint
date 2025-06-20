"""
JSON Transformer Layer for Phase 2 Microservice Engine
Converts database data to engine-compatible JSON for container orchestration
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

try:
    from .models import (
        Component, SpatialData, GeometryProperties, Material, Dimensions,
        ComponentWithRelations, ParsedFile, ParsedComponent
    )
except ImportError:
    from models import (
        Component, SpatialData, GeometryProperties, Material, Dimensions,
        ComponentWithRelations, ParsedFile, ParsedComponent
    )
from .schemas import (
    NodeDictionary, FunctorEdge, AgentCoefficient, SFDEAffinityType,
    CallbackType, Phase, DAGExecutionRequest, GraphOperationRequest
)

class TransformerType(str, Enum):
    """Types of transformations for different containers"""
    DAG_ALPHA = "dag_alpha"
    FUNCTOR_TYPES = "functor_types"
    CALLBACK_ENGINE = "callback_engine"
    SFDE_ENGINE = "sfde_engine"
    GRAPH_RUNTIME = "graph_runtime"
    DGL_TRAINING = "dgl_training"  # Future container

class JSONTransformer:
    """
    Transforms database components into JSON formats compatible with Phase 2 microservices
    """
    
    def __init__(self):
        self.node_counter = 0
        self.edge_counter = 0
        
    def transform_component_to_node_dictionary(
        self, 
        component_data: Dict[str, Any],
        phase: str = "alpha",
        agent: int = 1,
        callback_type: str = "dag"
    ) -> Dict[str, Any]:
        """
        Transform a database component into a node dictionary for microservices
        """
        self.node_counter += 1
        
        # Extract component properties
        props = self._extract_component_properties(component_data)
        
        # Determine trigger functor based on component type
        trigger_functor = self._determine_trigger_functor(component_data, phase)
        
        # Build node dictionary
        node_dict = {
            "node_id": f"V{self.node_counter:02d}",
            "node_label": component_data.get("component_name", f"Component_{self.node_counter}"),
            "phase": phase,
            "agent": agent,
            "callback_type": callback_type,
            "trigger_functor": trigger_functor,
            "dictionary": props,
            "allowed_callback_types": ["dag", "relational", "combinatorial"]
        }
        
        return node_dict
    
    def transform_components_to_node_collection(
        self, 
        components: List[Dict[str, Any]],
        phase_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Transform multiple components into a complete node collection
        """
        nodes = []
        
        for component in components:
            # Determine phase based on component type or mapping
            phase = self._determine_phase(component, phase_mapping)
            
            # Create node dictionary
            node_dict = self.transform_component_to_node_dictionary(
                component, 
                phase=phase,
                agent=self._determine_agent(component),
                callback_type=self._determine_callback_type(component, phase)
            )
            nodes.append(node_dict)
        
        return {"nodes": nodes}
    
    def create_functor_edges(
        self, 
        source_nodes: List[str], 
        target_nodes: List[str],
        edge_type: str = "data_flow"
    ) -> List[Dict[str, Any]]:
        """
        Create functor edges between nodes
        """
        edges = []
        
        for source in source_nodes:
            for target in target_nodes:
                if source != target:
                    self.edge_counter += 1
                    edge = {
                        "edge_id": f"E{self.edge_counter:02d}",
                        "source_node": source,
                        "target_node": target,
                        "edge_type": edge_type,
                        "weight": 0.5,  # Default weight
                        "metadata": {"created_at": datetime.now().isoformat()}
                    }
                    edges.append(edge)
        
        return edges
    
    def transform_for_dag_alpha(
        self, 
        components: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transform components for ne-dag-alpha container
        """
        # Create node collection
        node_collection = self.transform_components_to_node_collection(
            components, 
            phase_mapping={"structural": "alpha", "spatial": "alpha"}
        )
        
        # Extract input data
        input_data = self._extract_dag_input_data(components)
        
        # Create DAG execution request
        dag_request = {
            "dag_id": str(uuid.uuid4()),
            "input_data": input_data,
            "node_sequence": [node["node_id"] for node in node_collection["nodes"]],
            "callback_type": "dag",
            "parameters": {"phase": "alpha", "execution_mode": "deterministic"}
        }
        
        return dag_request
    
    def transform_for_functor_types(
        self, 
        components: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transform components for ne-functor-types container
        """
        # Extract spatial and aggregation data
        spatial_data = []
        aggregation_data = []
        
        for component in components:
            if component.get("spatial_data"):
                spatial_data.append({
                    "component_id": component.get("component_id"),
                    "centroid": [
                        component["spatial_data"].get("centroid_x", 0),
                        component["spatial_data"].get("centroid_y", 0),
                        component["spatial_data"].get("centroid_z", 0)
                    ],
                    "bbox": [
                        component["spatial_data"].get("bbox_x_mm", 0),
                        component["spatial_data"].get("bbox_y_mm", 0),
                        component["spatial_data"].get("bbox_z_mm", 0)
                    ]
                })
            
            if component.get("dimensions"):
                aggregation_data.append({
                    "component_id": component.get("component_id"),
                    "volume": component["dimensions"].get("volume_cm3", 0),
                    "area": component["dimensions"].get("area_m2", 0),
                    "dimensions": [
                        component["dimensions"].get("length_mm", 0),
                        component["dimensions"].get("width_mm", 0),
                        component["dimensions"].get("height_mm", 0)
                    ]
                })
        
        return {
            "spatial_calculations": spatial_data,
            "aggregation_calculations": aggregation_data,
            "functor_type": "stateless",
            "execution_mode": "cross_phase"
        }
    
    def transform_for_callback_engine(
        self, 
        components: List[Dict[str, Any]],
        phase: str = "beta"
    ) -> Dict[str, Any]:
        """
        Transform components for ne-callback-engine container
        """
        callback_data = []
        
        for component in components:
            callback_item = {
                "component_id": component.get("component_id"),
                "callback_type": "relational" if phase == "beta" else "combinatorial",
                "phase": phase,
                "properties": self._extract_component_properties(component),
                "relationships": self._extract_relationships(component)
            }
            callback_data.append(callback_item)
        
        return {
            "callbacks": callback_data,
            "phase": phase,
            "execution_mode": "event_driven"
        }
    
    def transform_for_sfde_engine(
        self, 
        components: List[Dict[str, Any]],
        affinity_types: List[str]
    ) -> Dict[str, Any]:
        """
        Transform components for sfde-engine container
        """
        sfde_data = []
        
        for component in components:
            for affinity_type in affinity_types:
                sfde_item = {
                    "component_id": component.get("component_id"),
                    "affinity_type": affinity_type,
                    "input_parameters": self._extract_sfde_parameters(component, affinity_type),
                    "formula_context": self._extract_formula_context(component)
                }
                sfde_data.append(sfde_item)
        
        return {
            "sfde_requests": sfde_data,
            "affinity_types": affinity_types,
            "execution_mode": "symbolic_reasoning"
        }
    
    def transform_for_graph_runtime(
        self, 
        components: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transform components for ne-graph-runtime-engine container
        """
        # Create node collection
        node_collection = self.transform_components_to_node_collection(components)
        
        # Create edges
        node_ids = [node["node_id"] for node in node_collection["nodes"]]
        edges = self.create_functor_edges(node_ids, node_ids)
        
        # Create graph operation request
        graph_request = {
            "operation_type": "build_and_execute",
            "graph_data": {
                "nodes": node_collection["nodes"], 
                "edges": edges
            },
            "node_data": node_collection["nodes"],
            "edge_data": edges,
            "parameters": {"graph_type": "networkx", "execution_mode": "runtime"}
        }
        
        return graph_request
    
    def _extract_component_properties(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all properties from a component"""
        props = {
            "component_id": component.get("component_id"),
            "component_name": component.get("component_name"),
            "component_type": component.get("component_type"),
            "description": component.get("description")
        }
        
        # Add spatial data
        if component.get("spatial_data"):
            spatial = component["spatial_data"]
            props.update({
                "centroid_x": spatial.get("centroid_x"),
                "centroid_y": spatial.get("centroid_y"),
                "centroid_z": spatial.get("centroid_z"),
                "bbox_x_mm": spatial.get("bbox_x_mm"),
                "bbox_y_mm": spatial.get("bbox_y_mm"),
                "bbox_z_mm": spatial.get("bbox_z_mm")
            })
        
        # Add geometry properties
        if component.get("geometry_properties"):
            geom = component["geometry_properties"]
            props.update({
                "vertex_count": geom.get("vertex_count"),
                "face_count": geom.get("face_count"),
                "edge_count": geom.get("edge_count"),
                "surface_area_m2": geom.get("surface_area_m2"),
                "volume_cm3": geom.get("bounding_box_volume_cm3")
            })
        
        # Add dimensions
        if component.get("dimensions"):
            dims = component["dimensions"]
            props.update({
                "length_mm": dims.get("length_mm"),
                "width_mm": dims.get("width_mm"),
                "height_mm": dims.get("height_mm"),
                "area_m2": dims.get("area_m2"),
                "volume_cm3": dims.get("volume_cm3")
            })
        
        # Add materials
        if component.get("materials"):
            props["materials"] = [
                {
                    "material_name": mat.get("material_name"),
                    "base_material": mat.get("base_material"),
                    "material_code": mat.get("material_code")
                } for mat in component["materials"]
            ]
        
        return props
    
    def _determine_trigger_functor(self, component: Dict[str, Any], phase: str) -> str:
        """Determine the appropriate trigger functor based on component and phase"""
        component_type = component.get("component_type", "").lower()
        
        if phase == "alpha":
            if "structural" in component_type:
                return "evaluate_structural"
            elif "spatial" in component_type:
                return "evaluate_spatial"
            else:
                return "evaluate_manufacturing"
        
        elif phase == "beta":
            return "check_compliance"
        
        elif phase == "gamma":
            return "generate_bid"
        
        return "evaluate_manufacturing"
    
    def _determine_phase(self, component: Dict[str, Any], phase_mapping: Optional[Dict[str, str]]) -> str:
        """Determine the appropriate phase for a component"""
        if phase_mapping:
            component_type = component.get("component_type", "").lower()
            for key, phase in phase_mapping.items():
                if key.lower() in component_type:
                    return phase
        
        # Default phase determination
        component_type = component.get("component_type", "").lower()
        if any(keyword in component_type for keyword in ["structural", "spatial", "manufacturing"]):
            return "alpha"
        elif any(keyword in component_type for keyword in ["compliance", "regulatory"]):
            return "beta"
        elif any(keyword in component_type for keyword in ["bidding", "investment", "roi"]):
            return "gamma"
        
        return "alpha"
    
    def _determine_agent(self, component: Dict[str, Any]) -> int:
        """Determine the appropriate agent for a component"""
        component_type = component.get("component_type", "").lower()
        if "structural" in component_type:
            return 1
        elif "spatial" in component_type:
            return 1
        elif "compliance" in component_type:
            return 2
        elif "bidding" in component_type:
            return 2
        
        return 1
    
    def _determine_callback_type(self, component: Dict[str, Any], phase: str) -> str:
        """Determine the appropriate callback type"""
        if phase == "alpha":
            return "dag"
        elif phase == "beta":
            return "relational"
        elif phase == "gamma":
            return "combinatorial"
        
        return "dag"
    
    def _extract_dag_input_data(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract input data for DAG execution"""
        spatial_count = len([c for c in components if c.get("spatial_data")])
        structural_count = len([c for c in components if "structural" in (c.get("component_type", "") or "").lower()])
        
        total_volume = sum(
            c.get("dimensions", {}).get("volume_cm3", 0) or 0 
            for c in components if c.get("dimensions")
        )
        total_area = sum(
            c.get("dimensions", {}).get("area_m2", 0) or 0 
            for c in components if c.get("dimensions")
        )
        
        return {
            "components_count": len(components),
            "spatial_components": spatial_count,
            "structural_components": structural_count,
            "total_volume": total_volume,
            "total_area": total_area
        }
    
    def _extract_relationships(self, component: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships for callback processing"""
        relationships = []
        
        # Spatial relationships
        if component.get("spatial_data"):
            spatial = component["spatial_data"]
            relationships.append({
                "type": "spatial",
                "centroid": [
                    spatial.get("centroid_x", 0),
                    spatial.get("centroid_y", 0),
                    spatial.get("centroid_z", 0)
                ]
            })
        
        # Material relationships
        if component.get("materials"):
            relationships.append({
                "type": "material",
                "materials": [
                    mat.get("material_name") 
                    for mat in component["materials"] 
                    if mat.get("material_name")
                ]
            })
        
        return relationships
    
    def _extract_sfde_parameters(self, component: Dict[str, Any], affinity_type: str) -> Dict[str, Any]:
        """Extract parameters for SFDE calculations"""
        params = {}
        
        if affinity_type == "spatial":
            if component.get("spatial_data"):
                spatial = component["spatial_data"]
                params.update({
                    "centroid": [
                        spatial.get("centroid_x", 0),
                        spatial.get("centroid_y", 0),
                        spatial.get("centroid_z", 0)
                    ],
                    "bbox": [
                        spatial.get("bbox_x_mm", 0),
                        spatial.get("bbox_y_mm", 0),
                        spatial.get("bbox_z_mm", 0)
                    ]
                })
        
        elif affinity_type == "structural":
            if component.get("dimensions"):
                dims = component["dimensions"]
                params.update({
                    "length_mm": dims.get("length_mm", 0),
                    "width_mm": dims.get("width_mm", 0),
                    "height_mm": dims.get("height_mm", 0),
                    "volume_cm3": dims.get("volume_cm3", 0)
                })
        
        elif affinity_type == "cost":
            if component.get("dimensions"):
                dims = component["dimensions"]
                params.update({
                    "volume_cm3": dims.get("volume_cm3", 0),
                    "area_m2": dims.get("area_m2", 0)
                })
        
        return params
    
    def _extract_formula_context(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context for formula execution"""
        return {
            "component_type": component.get("component_type"),
            "has_spatial_data": component.get("spatial_data") is not None,
            "has_geometry": component.get("geometry_properties") is not None,
            "has_dimensions": component.get("dimensions") is not None,
            "materials_count": len(component.get("materials", []))
        } 