import ezdxf
import uuid
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

def validate_dxf_file(file_path: str) -> bool:
    """Validate that DXF file exists and is readable"""
    if not os.path.exists(file_path):
        raise ValueError(f"DXF file not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise ValueError(f"DXF file not readable: {file_path}")
    return True

def extract_dxf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract basic file metadata"""
    stat = os.stat(file_path)
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size": stat.st_size,
        "file_extension": ".dxf",
        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "parsed_at": datetime.now().isoformat()
    }

def extract_entity_geometry(entity) -> Dict[str, Any]:
    """Extract geometry information from DXF entity"""
    geometry = {
        "type": entity.dxftype(),
        "layer": entity.dxf.layer,
        "has_position": False
    }
    
    try:
        # Extract position and dimension data based on entity type
        if hasattr(entity.dxf, 'start'):
            geometry["start_point"] = list(entity.dxf.start)
            geometry["has_position"] = True
            
        if hasattr(entity.dxf, 'end'):
            geometry["end_point"] = list(entity.dxf.end)
            
        if hasattr(entity.dxf, 'center'):
            geometry["center_point"] = list(entity.dxf.center)
            geometry["has_position"] = True
            
        if hasattr(entity.dxf, 'radius'):
            geometry["radius"] = entity.dxf.radius
            
        if hasattr(entity.dxf, 'width'):
            geometry["width"] = entity.dxf.width
            
        if hasattr(entity.dxf, 'height'):
            geometry["height"] = entity.dxf.height
            
        if hasattr(entity.dxf, 'length'):
            geometry["length"] = entity.dxf.length
            
        if hasattr(entity.dxf, 'area'):
            geometry["area"] = entity.dxf.area
            
        # Extract text content for text entities
        if hasattr(entity.dxf, 'text'):
            geometry["text_content"] = entity.dxf.text
            
        # Extract polyline points
        if hasattr(entity, 'vertices'):
            vertices = []
            for vertex in entity.vertices:
                if hasattr(vertex.dxf, 'location'):
                    vertices.append(list(vertex.dxf.location))
            if vertices:
                geometry["vertices"] = vertices
                geometry["has_position"] = True
                
        # Extract arc angles
        if hasattr(entity.dxf, 'start_angle'):
            geometry["start_angle"] = entity.dxf.start_angle
        if hasattr(entity.dxf, 'end_angle'):
            geometry["end_angle"] = entity.dxf.end_angle
            
        # Extract circle properties
        if hasattr(entity.dxf, 'thickness'):
            geometry["thickness"] = entity.dxf.thickness
            
        # Extract block reference properties
        if hasattr(entity.dxf, 'name'):
            geometry["block_name"] = entity.dxf.name
        if hasattr(entity.dxf, 'insert'):
            geometry["insert_point"] = list(entity.dxf.insert)
            geometry["has_position"] = True
            
        # Extract scale factors
        if hasattr(entity.dxf, 'xscale'):
            geometry["x_scale"] = entity.dxf.xscale
        if hasattr(entity.dxf, 'yscale'):
            geometry["y_scale"] = entity.dxf.yscale
        if hasattr(entity.dxf, 'zscale'):
            geometry["z_scale"] = entity.dxf.zscale
            
        # Extract rotation
        if hasattr(entity.dxf, 'rotation'):
            geometry["rotation"] = entity.dxf.rotation
            
    except Exception as e:
        logging.warning(f"Error extracting geometry from {entity.dxftype()}: {str(e)}")
        
    return geometry

def extract_entity_properties(entity) -> Dict[str, Any]:
    """Extract all properties from DXF entity"""
    props = {
        "entity_type": entity.dxftype(),
        "layer": entity.dxf.layer,
        "handle": entity.dxf.handle,
        "color": getattr(entity.dxf, 'color', None),
        "linetype": getattr(entity.dxf, 'linetype', None),
        "lineweight": getattr(entity.dxf, 'lineweight', None),
        "visible": getattr(entity.dxf, 'visible', True),
    }
    
    try:
        # Extract additional properties based on entity type
        if hasattr(entity.dxf, 'linetype_scale'):
            props["linetype_scale"] = entity.dxf.linetype_scale
            
        if hasattr(entity.dxf, 'transparency'):
            props["transparency"] = entity.dxf.transparency
            
        if hasattr(entity.dxf, 'material'):
            props["material"] = entity.dxf.material
            
        # Extract text properties
        if hasattr(entity.dxf, 'text'):
            props["text_content"] = entity.dxf.text
        if hasattr(entity.dxf, 'text_style'):
            props["text_style"] = entity.dxf.text_style
        if hasattr(entity.dxf, 'text_height'):
            props["text_height"] = entity.dxf.text_height
        if hasattr(entity.dxf, 'text_rotation'):
            props["text_rotation"] = entity.dxf.text_rotation
            
        # Extract dimension properties
        if hasattr(entity.dxf, 'dimension_text'):
            props["dimension_text"] = entity.dxf.dimension_text
        if hasattr(entity.dxf, 'dimension_style'):
            props["dimension_style"] = entity.dxf.dimension_style
            
        # Extract hatch properties
        if hasattr(entity.dxf, 'pattern_name'):
            props["pattern_name"] = entity.dxf.pattern_name
        if hasattr(entity.dxf, 'pattern_scale'):
            props["pattern_scale"] = entity.dxf.pattern_scale
            
        # Extract leader properties
        if hasattr(entity.dxf, 'annotation_handle'):
            props["annotation_handle"] = entity.dxf.annotation_handle
            
    except Exception as e:
        logging.warning(f"Error extracting properties from {entity.dxftype()}: {str(e)}")
        
    return props

def parse_dxf_file(file_path: str) -> Dict[str, Any]:
    """
    Enhanced DXF file parser with comprehensive data extraction
    
    Args:
        file_path: Path to the DXF file
        
    Returns:
        Dictionary containing parsed data with metadata
    """
    try:
        validate_dxf_file(file_path)
        metadata = extract_dxf_metadata(file_path)
        
        # Read DXF file
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        components = []
        layers = set()
        entity_types = set()
        blocks = set()
        text_content = []
        
        # Extract header information
        header = doc.header
        header_info = {
            "acad_version": getattr(header, 'ACADVER', 'Unknown'),
            "dwg_code_page": getattr(header, 'DWGCODEPAGE', 'Unknown'),
            "ins_base": getattr(header, 'INSBASE', None),
            "ext_min": getattr(header, 'EXTMIN', None),
            "ext_max": getattr(header, 'EXTMAX', None),
            "units": getattr(header, 'INSUNITS', None),
        }
        
        # Extract block definitions
        for block in doc.blocks:
            if not block.name.startswith('*'):
                blocks.add(block.name)
        
        # Process all entities in modelspace
        for entity in msp:
            try:
                entity_type = entity.dxftype()
                entity_types.add(entity_type)
                layers.add(entity.dxf.layer)
                
                # Extract properties and geometry
                props = extract_entity_properties(entity)
                geometry = extract_entity_geometry(entity)
                
                # Collect text content
                if "text_content" in props:
                    text_content.append(props["text_content"])
                
                # Create component object
                component = {
                    "component_id": f"DXF-{uuid.uuid4().hex[:8]}",
                    "name": f"{entity_type}_{entity.dxf.handle}",
                    "type": entity_type,
                    "properties": props,
                    "geometry": geometry
                }
                
                components.append(component)
                
            except Exception as e:
                logging.warning(f"Error processing entity {entity.dxftype()}: {str(e)}")
                continue
        
        # Process entities in paper space if available
        psp = doc.layouts.get('Layout1')  # Default paper space
        if psp:
            for entity in psp:
                try:
                    entity_type = entity.dxftype()
                    entity_types.add(entity_type)
                    layers.add(entity.dxf.layer)
                    
                    props = extract_entity_properties(entity)
                    geometry = extract_entity_geometry(entity)
                    
                    if "text_content" in props:
                        text_content.append(props["text_content"])
                    
                    component = {
                        "component_id": f"DXF-PSP-{uuid.uuid4().hex[:8]}",
                        "name": f"{entity_type}_{entity.dxf.handle}",
                        "type": entity_type,
                        "space": "paper",
                        "properties": props,
                        "geometry": geometry
                    }
                    
                    components.append(component)
                    
                except Exception as e:
                    logging.warning(f"Error processing paper space entity {entity.dxftype()}: {str(e)}")
                    continue
        
        # Calculate bounding box from all entities with positions
        positions = []
        for comp in components:
            geom = comp["geometry"]
            if "start_point" in geom:
                positions.append(geom["start_point"])
            if "end_point" in geom:
                positions.append(geom["end_point"])
            if "center_point" in geom:
                positions.append(geom["center_point"])
            if "insert_point" in geom:
                positions.append(geom["insert_point"])
        
        bounding_box = None
        if positions:
            try:
                import numpy as np
                pos_array = np.array(positions)
                min_coords = pos_array.min(axis=0)
                max_coords = pos_array.max(axis=0)
                bounding_box = {
                    "min": min_coords.tolist(),
                    "max": max_coords.tolist(),
                    "size": (max_coords - min_coords).tolist()
                }
            except ImportError:
                logging.warning("NumPy not available - skipping bounding box calculation")
        
        return {
            "status": "success",
            "file_metadata": metadata,
            "header_info": header_info,
            "statistics": {
                "total_components": len(components),
                "layers_count": len(layers),
                "entity_types": list(entity_types),
                "blocks_count": len(blocks),
                "text_entities": len(text_content),
                "layers": list(layers),
                "blocks": list(blocks)
            },
            "components": components,
            "bounding_box": bounding_box,
            "text_content": text_content,
            "parsed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error parsing DXF file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_metadata": extract_dxf_metadata(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "parsed_at": datetime.now().isoformat()
        }

# Legacy function for backward compatibility
def parse_dxf_file_legacy(file_path: str) -> List[Dict[str, Any]]:
    """Legacy DXF parser function for backward compatibility"""
    result = parse_dxf_file(file_path)
    if result["status"] == "success":
        return result["components"]
    else:
        return [] 