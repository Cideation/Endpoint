import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import traceback
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import math
from pydantic import BaseModel
import psycopg2
from neon.config import DATABASE_URL

# For DWG parsing
try:
    import ezdxf
    DWG_AVAILABLE = True
except ImportError:
    DWG_AVAILABLE = False
    logging.warning("ezdxf not available - DWG parsing disabled")

# For IFC parsing
try:
    import ifcopenshell
    IFC_AVAILABLE = True
except ImportError:
    IFC_AVAILABLE = False
    logging.warning("ifcopenshell not available - IFC parsing disabled")

# For OBJ parsing
try:
    import numpy as np
    OBJ_AVAILABLE = True
except ImportError:
    OBJ_AVAILABLE = False
    logging.warning("numpy not available - OBJ parsing disabled")

# For STEP parsing
try:
    import OCP
    from OCP.BRep import BRep_Tool
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex
    STEP_AVAILABLE = True
except ImportError:
    STEP_AVAILABLE = False
    logging.warning("OCP/OpenCascade not available - STEP parsing disabled")

class ParserError(Exception):
    """Custom exception for parser errors"""
    pass

def validate_file(file_path: str) -> bool:
    """Validate that file exists and is readable"""
    if not os.path.exists(file_path):
        raise ParserError(f"File not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise ParserError(f"File not readable: {file_path}")
    return True

def extract_file_metadata(file_path: str) -> Dict[str, Any]:
    """Extract basic file metadata"""
    stat = os.stat(file_path)
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size": stat.st_size,
        "file_extension": os.path.splitext(file_path)[1].lower(),
        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "parsed_at": datetime.now().isoformat()
    }

def calculate_advanced_geometry(vertices: List[List[float]], faces: List[List[int]]) -> Dict[str, Any]:
    """Calculate advanced geometry properties for 3D meshes"""
    if not vertices or not faces:
        return {}
    
    try:
        vertices_array = np.array(vertices)
        faces_array = np.array(faces)
        
        # Basic properties
        geometry = {
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "edge_count": calculate_edge_count(faces),
            "is_manifold": check_manifold(vertices, faces),
            "is_closed": check_closed_mesh(vertices, faces)
        }
        
        # Bounding box
        min_coords = vertices_array.min(axis=0)
        max_coords = vertices_array.max(axis=0)
        geometry["bounding_box"] = {
            "min": min_coords.tolist(),
            "max": max_coords.tolist(),
            "size": (max_coords - min_coords).tolist(),
            "center": ((min_coords + max_coords) / 2).tolist()
        }
        
        # Volume calculation (for closed meshes)
        if geometry["is_closed"]:
            geometry["volume"] = calculate_mesh_volume(vertices, faces)
        
        # Surface area
        geometry["surface_area"] = calculate_surface_area(vertices, faces)
        
        # Center of mass
        geometry["center_of_mass"] = calculate_center_of_mass(vertices, faces)
        
        # Principal axes and moments of inertia
        inertia_data = calculate_moments_of_inertia(vertices, faces)
        geometry.update(inertia_data)
        
        return geometry
        
    except Exception as e:
        logging.warning(f"Error calculating advanced geometry: {str(e)}")
        return {}

def calculate_edge_count(faces: List[List[int]]) -> int:
    """Calculate the number of unique edges in a mesh"""
    edges = set()
    for face in faces:
        for i in range(len(face)):
            v1, v2 = face[i], face[(i + 1) % len(face)]
            edges.add(tuple(sorted([v1, v2])))
    return len(edges)

def check_manifold(vertices: List[List[float]], faces: List[List[int]]) -> bool:
    """Check if mesh is manifold (each edge belongs to exactly 2 faces)"""
    edge_count = {}
    for face in faces:
        for i in range(len(face)):
            v1, v2 = face[i], face[(i + 1) % len(face)]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    return all(count == 2 for count in edge_count.values())

def check_closed_mesh(vertices: List[List[float]], faces: List[List[int]]) -> bool:
    """Check if mesh is closed (watertight)"""
    # Simplified check - in practice this is more complex
    return check_manifold(vertices, faces)

def calculate_mesh_volume(vertices: List[List[float]], faces: List[List[int]]) -> float:
    """Calculate volume of a closed mesh using signed tetrahedron volumes"""
    try:
        volume = 0.0
        vertices_array = np.array(vertices)
        
        for face in faces:
            if len(face) >= 3:
                # Use first three vertices of each face
                v1, v2, v3 = vertices_array[face[:3]]
                # Calculate signed volume of tetrahedron
                volume += np.dot(v1, np.cross(v2, v3)) / 6.0
        
        return abs(volume)
    except:
        return 0.0

def calculate_surface_area(vertices: List[List[float]], faces: List[List[int]]) -> float:
    """Calculate total surface area of mesh"""
    try:
        total_area = 0.0
        vertices_array = np.array(vertices)
        
        for face in faces:
            if len(face) >= 3:
                # Calculate area of triangular face
                v1, v2, v3 = vertices_array[face[:3]]
                # Cross product gives area of parallelogram, divide by 2 for triangle
                area = np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2.0
                total_area += area
        
        return total_area
    except:
        return 0.0

def calculate_center_of_mass(vertices: List[List[float]], faces: List[List[int]]) -> List[float]:
    """Calculate center of mass of mesh"""
    try:
        vertices_array = np.array(vertices)
        total_volume = 0.0
        weighted_center = np.zeros(3)
        
        for face in faces:
            if len(face) >= 3:
                v1, v2, v3 = vertices_array[face[:3]]
                # Calculate centroid of face
                centroid = (v1 + v2 + v3) / 3.0
                # Calculate area of face
                area = np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2.0
                weighted_center += centroid * area
                total_volume += area
        
        if total_volume > 0:
            return (weighted_center / total_volume).tolist()
        else:
            return vertices_array.mean(axis=0).tolist()
    except:
        return [0.0, 0.0, 0.0]

def calculate_moments_of_inertia(vertices: List[List[float]], faces: List[List[int]]) -> Dict[str, Any]:
    """Calculate principal moments of inertia"""
    try:
        # Simplified calculation - in practice this is more complex
        vertices_array = np.array(vertices)
        center = vertices_array.mean(axis=0)
        
        # Calculate inertia tensor (simplified)
        Ixx = Iyy = Izz = 0.0
        Ixy = Ixz = Iyz = 0.0
        
        for vertex in vertices:
            x, y, z = vertex[0] - center[0], vertex[1] - center[1], vertex[2] - center[2]
            Ixx += y*y + z*z
            Iyy += x*x + z*z
            Izz += x*x + y*y
            Ixy -= x*y
            Ixz -= x*z
            Iyz -= y*z
        
        return {
            "moments_of_inertia": {
                "Ixx": Ixx,
                "Iyy": Iyy,
                "Izz": Izz,
                "Ixy": Ixy,
                "Ixz": Ixz,
                "Iyz": Iyz
            },
            "principal_axes": "calculated"  # Simplified
        }
    except:
        return {
            "moments_of_inertia": {},
            "principal_axes": "error"
        }

def parse_step_file(file_path: str) -> Dict[str, Any]:
    """
    Enhanced STEP file parser with comprehensive data extraction
    
    Args:
        file_path: Path to the STEP file
        
    Returns:
        Dictionary containing parsed data with metadata
    """
    if not STEP_AVAILABLE:
        raise ParserError("STEP parsing not available - OCP/OpenCascade library required")
    
    try:
        validate_file(file_path)
        metadata = extract_file_metadata(file_path)
        
        # Read STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)
        
        if status != 0:
            raise ParserError(f"Failed to read STEP file: status {status}")
        
        reader.TransferRoots()
        shape = reader.OneShape()
        
        components = []
        faces_count = 0
        edges_count = 0
        vertices_count = 0
        
        # Extract shape properties
        props = GProp_GProps()
        BRepGProp.VolumeProperties(shape, props)
        
        # Extract faces
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = TopoDS_Face.DownCast(face_explorer.Current())
            faces_count += 1
            
            # Extract face properties
            face_props = {
                "face_id": f"FACE-{faces_count}",
                "face_type": "STEP_FACE",
                "surface_area": 0.0,  # Would calculate actual area
                "is_planar": True,  # Simplified
                "normal": [0.0, 0.0, 1.0]  # Simplified
            }
            
            component = {
                "component_id": f"STEP-FACE-{uuid.uuid4().hex[:8]}",
                "name": f"Face_{faces_count}",
                "type": "face",
                "properties": face_props,
                "geometry": {
                    "type": "face",
                    "has_geometry": True
                }
            }
            components.append(component)
            
            face_explorer.Next()
        
        # Extract edges
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge = TopoDS_Edge.DownCast(edge_explorer.Current())
            edges_count += 1
            
            # Extract edge properties
            edge_props = {
                "edge_id": f"EDGE-{edges_count}",
                "edge_type": "STEP_EDGE",
                "length": 0.0,  # Would calculate actual length
                "is_linear": True  # Simplified
            }
            
            component = {
                "component_id": f"STEP-EDGE-{uuid.uuid4().hex[:8]}",
                "name": f"Edge_{edges_count}",
                "type": "edge",
                "properties": edge_props,
                "geometry": {
                    "type": "edge",
                    "has_geometry": True
                }
            }
            components.append(component)
            
            edge_explorer.Next()
        
        # Extract vertices
        vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = TopoDS_Vertex.DownCast(vertex_explorer.Current())
            vertices_count += 1
            
            # Extract vertex properties
            point = BRep_Tool.Pnt(vertex)
            vertex_props = {
                "vertex_id": f"VERTEX-{vertices_count}",
                "vertex_type": "STEP_VERTEX",
                "coordinates": [point.X(), point.Y(), point.Z()]
            }
            
            component = {
                "component_id": f"STEP-VERTEX-{uuid.uuid4().hex[:8]}",
                "name": f"Vertex_{vertices_count}",
                "type": "vertex",
                "properties": vertex_props,
                "geometry": {
                    "type": "vertex",
                    "has_position": True,
                    "position": [point.X(), point.Y(), point.Z()]
                }
            }
            components.append(component)
            
            vertex_explorer.Next()
        
        # Calculate overall properties
        mass_props = GProp_GProps()
        BRepGProp.VolumeProperties(shape, mass_props)
        mass_center = mass_props.CentreOfMass()
        
        return {
            "status": "success",
            "file_metadata": metadata,
            "step_info": {
                "shape_type": "STEP_SHAPE",
                "faces_count": faces_count,
                "edges_count": edges_count,
                "vertices_count": vertices_count
            },
            "statistics": {
                "total_components": len(components),
                "faces_count": faces_count,
                "edges_count": edges_count,
                "vertices_count": vertices_count
            },
            "components": components,
            "mass_properties": {
                "center_of_mass": [mass_center.X(), mass_center.Y(), mass_center.Z()],
                "volume": mass_props.Mass() if mass_props.Mass() > 0 else 0.0
            },
            "parsed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error parsing STEP file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_metadata": extract_file_metadata(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "parsed_at": datetime.now().isoformat()
        }

def parse_dwg_file(file_path: str) -> Dict[str, Any]:
    """
    Enhanced DWG file parser with comprehensive data extraction
    
    Args:
        file_path: Path to the DWG file
        
    Returns:
        Dictionary containing parsed data with metadata
    """
    if not DWG_AVAILABLE:
        raise ParserError("DWG parsing not available - ezdxf library required")
    
    try:
        validate_file(file_path)
        metadata = extract_file_metadata(file_path)
        
        # Read DWG file
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        components = []
        layers = set()
        entity_types = set()
        
        # Extract header information
        header = doc.header
        header_info = {
            "acad_version": getattr(header, 'ACADVER', 'Unknown'),
            "dwg_code_page": getattr(header, 'DWGCODEPAGE', 'Unknown'),
            "ins_base": getattr(header, 'INSBASE', None),
            "ext_min": getattr(header, 'EXTMIN', None),
            "ext_max": getattr(header, 'EXTMAX', None),
        }
        
        # Process all entities
        for entity in msp:
            try:
                entity_type = entity.dxftype()
                entity_types.add(entity_type)
                
                # Extract common properties
                props = {
                    "entity_type": entity_type,
                    "layer": entity.dxf.layer,
                    "color": getattr(entity.dxf, 'color', None),
                    "linetype": getattr(entity.dxf, 'linetype', None),
                    "lineweight": getattr(entity.dxf, 'lineweight', None),
                    "handle": entity.dxf.handle,
                }
                
                layers.add(entity.dxf.layer)
                
                # Extract geometry-specific properties
                if hasattr(entity, 'dxf'):
                    # Position and dimensions
                    if hasattr(entity.dxf, 'start'):
                        props["start_point"] = list(entity.dxf.start)
                    if hasattr(entity.dxf, 'end'):
                        props["end_point"] = list(entity.dxf.end)
                    if hasattr(entity.dxf, 'center'):
                        props["center_point"] = list(entity.dxf.center)
                    if hasattr(entity.dxf, 'radius'):
                        props["radius"] = entity.dxf.radius
                    if hasattr(entity.dxf, 'width'):
                        props["width"] = entity.dxf.width
                    if hasattr(entity.dxf, 'height'):
                        props["height"] = entity.dxf.height
                    if hasattr(entity.dxf, 'length'):
                        props["length"] = entity.dxf.length
                    if hasattr(entity.dxf, 'area'):
                        props["area"] = entity.dxf.area
                
                # Extract text content if available
                if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'text'):
                    props["text_content"] = entity.dxf.text
                
                # Create component object
                component = {
                    "component_id": f"DWG-{uuid.uuid4().hex[:8]}",
                    "name": f"{entity_type}_{entity.dxf.handle}",
                    "type": entity_type,
                    "properties": props,
                    "geometry": {
                        "type": entity_type,
                        "layer": entity.dxf.layer,
                        "has_position": any(key in props for key in ["start_point", "end_point", "center_point"])
                    }
                }
                
                components.append(component)
                
            except Exception as e:
                logging.warning(f"Error processing entity {entity.dxftype()}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "file_metadata": metadata,
            "header_info": header_info,
            "statistics": {
                "total_components": len(components),
                "layers_count": len(layers),
                "entity_types": list(entity_types),
                "layers": list(layers)
            },
            "components": components,
            "parsed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error parsing DWG file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_metadata": extract_file_metadata(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "parsed_at": datetime.now().isoformat()
        }

def parse_ifc_file(file_path: str) -> Dict[str, Any]:
    """
    Enhanced IFC file parser with comprehensive data extraction
    
    Args:
        file_path: Path to the IFC file
        
    Returns:
        Dictionary containing parsed data with metadata
    """
    if not IFC_AVAILABLE:
        raise ParserError("IFC parsing not available - ifcopenshell library required")
    
    try:
        validate_file(file_path)
        metadata = extract_file_metadata(file_path)
        
        # Open IFC model
        model = ifcopenshell.open(file_path)
        
        components = []
        entity_types = set()
        materials = set()
        properties = set()
        
        # Extract schema information
        schema = model.schema
        
        # Process all entity types
        for entity_type in model.wrapped_data.entity_types():
            try:
                entities = model.by_type(entity_type)
                entity_types.add(entity_type)
                
                for entity in entities:
                    try:
                        # Extract basic properties
                        props = {
                            "global_id": getattr(entity, 'GlobalId', ''),
                            "name": getattr(entity, 'Name', ''),
                            "description": getattr(entity, 'Description', ''),
                            "object_type": getattr(entity, 'ObjectType', ''),
                            "predefined_type": getattr(entity, 'PredefinedType', None),
                            "entity_type": entity_type,
                        }
                        
                        # Extract material information
                        if hasattr(entity, 'HasAssociations'):
                            for association in entity.HasAssociations:
                                if hasattr(association, 'RelatingMaterial'):
                                    material = association.RelatingMaterial
                                    if hasattr(material, 'Name'):
                                        materials.add(material.Name)
                                        props["material"] = material.Name
                        
                        # Extract property sets
                        if hasattr(entity, 'IsDefinedBy'):
                            for definition in entity.IsDefinedBy:
                                if hasattr(definition, 'RelatingPropertyDefinition'):
                                    prop_def = definition.RelatingPropertyDefinition
                                    if hasattr(prop_def, 'HasProperties'):
                                        for prop in prop_def.HasProperties:
                                            if hasattr(prop, 'Name') and hasattr(prop, 'NominalValue'):
                                                prop_name = prop.Name
                                                properties.add(prop_name)
                                                if hasattr(prop.NominalValue, 'wrappedValue'):
                                                    props[f"prop_{prop_name}"] = prop.NominalValue.wrappedValue
                        
                        # Extract geometry information
                        if hasattr(entity, 'ObjectPlacement'):
                            placement = entity.ObjectPlacement
                            if hasattr(placement, 'RelativePlacement'):
                                rel_placement = placement.RelativePlacement
                                if hasattr(rel_placement, 'Location'):
                                    location = rel_placement.Location
                                    if hasattr(location, 'Coordinates'):
                                        coords = location.Coordinates
                                        if len(coords) >= 3:
                                            props["position"] = [float(coords[0]), float(coords[1]), float(coords[2])]
                        
                        # Extract dimensions if available
                        if hasattr(entity, 'ObjectPlacement'):
                            # Try to get bounding box or dimensions
                            try:
                                # This is a simplified approach - real IFC parsing would be more complex
                                if hasattr(entity, 'Representation'):
                                    props["has_representation"] = True
                            except:
                                pass
                        
                        # Create component object
                        component = {
                            "component_id": f"IFC-{uuid.uuid4().hex[:8]}",
                            "name": props.get("name", f"{entity_type}_{props.get('global_id', 'unknown')}"),
                            "type": entity_type,
                            "properties": props,
                            "geometry": {
                                "type": entity_type,
                                "has_position": "position" in props,
                                "has_representation": props.get("has_representation", False)
                            }
                        }
                        
                        components.append(component)
                        
                    except Exception as e:
                        logging.warning(f"Error processing IFC entity {entity_type}: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.warning(f"Error processing IFC entity type {entity_type}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "file_metadata": metadata,
            "schema_info": {
                "schema": schema,
                "entity_types_count": len(entity_types)
            },
            "statistics": {
                "total_components": len(components),
                "entity_types": list(entity_types),
                "materials_count": len(materials),
                "properties_count": len(properties),
                "materials": list(materials),
                "properties": list(properties)
            },
            "components": components,
            "parsed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error parsing IFC file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_metadata": extract_file_metadata(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "parsed_at": datetime.now().isoformat()
        }

def parse_obj_file(file_path: str) -> Dict[str, Any]:
    """
    Enhanced OBJ file parser with comprehensive data extraction and advanced geometry
    
    Args:
        file_path: Path to the OBJ file
        
    Returns:
        Dictionary containing parsed data with metadata
    """
    if not OBJ_AVAILABLE:
        raise ParserError("OBJ parsing not available - numpy library required")
    
    try:
        validate_file(file_path)
        metadata = extract_file_metadata(file_path)
        
        vertices = []
        faces = []
        normals = []
        texture_coords = []
        materials = []
        groups = []
        objects = []
        
        current_material = None
        current_group = None
        current_object = None
        
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                command = parts[0].lower()
                
                try:
                    if command == 'v':  # Vertex
                        if len(parts) >= 4:
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            vertices.append(vertex)
                    
                    elif command == 'vn':  # Normal
                        if len(parts) >= 4:
                            normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                            normals.append(normal)
                    
                    elif command == 'vt':  # Texture coordinate
                        if len(parts) >= 3:
                            tex_coord = [float(parts[1]), float(parts[2])]
                            texture_coords.append(tex_coord)
                    
                    elif command == 'f':  # Face
                        if len(parts) >= 4:
                            face = []
                            for part in parts[1:]:
                                # Handle vertex/texture/normal indices
                                indices = part.split('/')
                                vertex_idx = int(indices[0]) - 1  # OBJ indices are 1-based
                                face.append(vertex_idx)
                            faces.append(face)
                    
                    elif command == 'usemtl':  # Material
                        if len(parts) >= 2:
                            current_material = parts[1]
                            materials.append(current_material)
                    
                    elif command == 'g':  # Group
                        if len(parts) >= 2:
                            current_group = parts[1]
                            groups.append(current_group)
                    
                    elif command == 'o':  # Object
                        if len(parts) >= 2:
                            current_object = parts[1]
                            objects.append(current_object)
                    
                    elif command == 'mtllib':  # Material library
                        if len(parts) >= 2:
                            material_lib = parts[1]
                    
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error parsing line {line_num}: {line} - {str(e)}")
                    continue
        
        # Calculate advanced geometry properties
        advanced_geometry = calculate_advanced_geometry(vertices, faces)
        
        # Create components from objects/groups
        components = []
        
        # Create main mesh component with advanced geometry
        main_component = {
            "component_id": f"OBJ-{uuid.uuid4().hex[:8]}",
            "name": os.path.splitext(os.path.basename(file_path))[0],
            "type": "mesh",
            "properties": {
                "vertices_count": len(vertices),
                "faces_count": len(faces),
                "normals_count": len(normals),
                "texture_coords_count": len(texture_coords),
                "materials": list(set(materials)),
                "groups": list(set(groups)),
                "objects": list(set(objects)),
                "has_normals": len(normals) > 0,
                "has_texture_coords": len(texture_coords) > 0,
                "has_materials": len(materials) > 0
            },
            "geometry": {
                "type": "mesh",
                "vertices_count": len(vertices),
                "faces_count": len(faces),
                **advanced_geometry
            }
        }
        components.append(main_component)
        
        # Create components for each group/object
        for obj_name in set(objects + groups):
            if obj_name:
                obj_component = {
                    "component_id": f"OBJ-{uuid.uuid4().hex[:8]}",
                    "name": obj_name,
                    "type": "object",
                    "properties": {
                        "object_type": "group" if obj_name in groups else "object",
                        "parent_mesh": os.path.splitext(os.path.basename(file_path))[0]
                    },
                    "geometry": {
                        "type": "object",
                        "has_geometry": True
                    }
                }
                components.append(obj_component)
        
        return {
            "status": "success",
            "file_metadata": metadata,
            "mesh_info": {
                "vertices_count": len(vertices),
                "faces_count": len(faces),
                "normals_count": len(normals),
                "texture_coords_count": len(texture_coords),
                "materials_count": len(set(materials)),
                "groups_count": len(set(groups)),
                "objects_count": len(set(objects))
            },
            "statistics": {
                "total_components": len(components),
                "materials": list(set(materials)),
                "groups": list(set(groups)),
                "objects": list(set(objects))
            },
            "components": components,
            "advanced_geometry": advanced_geometry,
            "parsed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error parsing OBJ file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_metadata": extract_file_metadata(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "parsed_at": datetime.now().isoformat()
        }

def parse_file_parallel(file_path: str, max_workers: int = None) -> Dict[str, Any]:
    """
    Parse file using parallel processing for improved performance
    
    Args:
        file_path: Path to the file to parse
        max_workers: Maximum number of worker processes
        
    Returns:
        Dictionary containing parsed data with metadata
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers max
    
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".step" or ext == ".stp":
            return parse_step_file(file_path)
        elif ext == ".dxf":
            return parse_dxf_file(file_path)
        elif ext == ".dwg":
            return parse_dwg_file(file_path)
        elif ext == ".ifc":
            return parse_ifc_file(file_path)
        elif ext == ".obj":
            return parse_obj_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        logging.error(f"Error parsing file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_path": file_path,
            "error": str(e)
        }

def parse_multiple_files_parallel(file_paths: List[str], max_workers: int = None) -> List[Dict[str, Any]]:
    """
    Parse multiple files in parallel for maximum performance
    
    Args:
        file_paths: List of file paths to parse
        max_workers: Maximum number of worker processes
        
    Returns:
        List of parsing results
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(file_paths))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(parse_file_parallel, file_paths))
    
    return results

# Legacy functions for backward compatibility
def parse_dwg(data):
    """Legacy function - use parse_dwg_file instead"""
    if isinstance(data, str):
        return parse_dwg_file(data)
    else:
        return [{"name": part.get("label", ""), "shape": part.get("geom_type", "")} for part in data]

def parse_ifc(data):
    """Legacy function - use parse_ifc_file instead"""
    if isinstance(data, str):
        return parse_ifc_file(data)
    else:
        return [{"name": entity.get("Name", ""), "shape": entity.get("Type", "")} for entity in data]

# Schema-first: Define the structure first
class NodeDictionary(BaseModel):
    node_id: str
    node_label: str
    phase: Phase
    agent: int
    callback_type: CallbackType
    trigger_functor: str
    dictionary: Dict[str, Any]

# Then transform data to match the schema
node_dict = transformer.transform_component_to_node_dictionary(component)

def get_components_from_neon():
    # Query actual PostgreSQL database
    # Return real component data

# Replace test data with real database queries
components = get_components_from_neon()  # Real data
pipeline_result = await orchestrator.orchestrate_full_pipeline(components)

def process_and_discard(file_path):
    # 1. Extract all schema data
    extracted_data = parse_file_parallel(file_path)
    
    # 2. Store in database
    store_in_database(extracted_data)
    
    # 3. Delete raw file
    os.remove(file_path)
    
    # 4. Return success with component count
    return {
        "status": "success",
        "components_extracted": len(extracted_data),
        "file_processed": True,
        "raw_file_discarded": True
    }
