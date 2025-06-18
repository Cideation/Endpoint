import ifcopenshell
import uuid
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

def validate_ifc_file(file_path: str) -> bool:
    """Validate that IFC file exists and is readable"""
    if not os.path.exists(file_path):
        raise ValueError(f"IFC file not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise ValueError(f"IFC file not readable: {file_path}")
    return True

def extract_ifc_metadata(file_path: str) -> Dict[str, Any]:
    """Extract basic file metadata"""
    stat = os.stat(file_path)
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size": stat.st_size,
        "file_extension": ".ifc",
        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "parsed_at": datetime.now().isoformat()
    }

def extract_ifc_property_value(prop_value) -> Any:
    """Extract value from IFC property value"""
    try:
        if hasattr(prop_value, 'wrappedValue'):
            return prop_value.wrappedValue
        elif hasattr(prop_value, 'NominalValue'):
            return extract_ifc_property_value(prop_value.NominalValue)
        else:
            return str(prop_value)
    except:
        return str(prop_value)

def extract_entity_properties(entity) -> Dict[str, Any]:
    """Extract all properties from IFC entity"""
    props = {
        "global_id": getattr(entity, 'GlobalId', ''),
        "name": getattr(entity, 'Name', ''),
        "description": getattr(entity, 'Description', ''),
        "object_type": getattr(entity, 'ObjectType', ''),
        "predefined_type": getattr(entity, 'PredefinedType', None),
        "entity_type": entity.is_a(),
    }
    
    try:
        # Extract material information
        if hasattr(entity, 'HasAssociations'):
            for association in entity.HasAssociations:
                if hasattr(association, 'RelatingMaterial'):
                    material = association.RelatingMaterial
                    if hasattr(material, 'Name'):
                        props["material"] = material.Name
                    if hasattr(material, 'Description'):
                        props["material_description"] = material.Description
                        
        # Extract property sets
        if hasattr(entity, 'IsDefinedBy'):
            for definition in entity.IsDefinedBy:
                if hasattr(definition, 'RelatingPropertyDefinition'):
                    prop_def = definition.RelatingPropertyDefinition
                    if hasattr(prop_def, 'Name'):
                        props[f"property_set_{prop_def.Name}"] = {}
                        
                    if hasattr(prop_def, 'HasProperties'):
                        for prop in prop_def.HasProperties:
                            if hasattr(prop, 'Name') and hasattr(prop, 'NominalValue'):
                                prop_name = prop.Name
                                prop_value = extract_ifc_property_value(prop.NominalValue)
                                props[f"prop_{prop_name}"] = prop_value
                                
                                # Add to property set if available
                                if f"property_set_{prop_def.Name}" in props:
                                    props[f"property_set_{prop_def.Name}"][prop_name] = prop_value
                                    
        # Extract quantity sets
        if hasattr(entity, 'IsDefinedBy'):
            for definition in entity.IsDefinedBy:
                if hasattr(definition, 'RelatingPropertyDefinition'):
                    prop_def = definition.RelatingPropertyDefinition
                    if hasattr(prop_def, 'HasProperties'):
                        for prop in prop_def.HasProperties:
                            if hasattr(prop, 'Name') and hasattr(prop, 'LengthValue'):
                                props[f"quantity_{prop.Name}"] = prop.LengthValue
                            elif hasattr(prop, 'Name') and hasattr(prop, 'AreaValue'):
                                props[f"quantity_{prop.Name}"] = prop.AreaValue
                            elif hasattr(prop, 'Name') and hasattr(prop, 'VolumeValue'):
                                props[f"quantity_{prop.Name}"] = prop.VolumeValue
                                
        # Extract classification information
        if hasattr(entity, 'HasAssociations'):
            for association in entity.HasAssociations:
                if hasattr(association, 'RelatingClassification'):
                    classification = association.RelatingClassification
                    if hasattr(classification, 'Name'):
                        props["classification"] = classification.Name
                        
        # Extract spatial structure
        if hasattr(entity, 'ContainedInStructure'):
            for containment in entity.ContainedInStructure:
                if hasattr(containment, 'RelatingStructure'):
                    structure = containment.RelatingStructure
                    if hasattr(structure, 'Name'):
                        props["spatial_structure"] = structure.Name
                        
        # Extract owner information
        if hasattr(entity, 'OwnerHistory'):
            owner = entity.OwnerHistory
            if hasattr(owner, 'OwningUser'):
                user = owner.OwningUser
                if hasattr(user, 'ThePerson'):
                    person = user.ThePerson
                    if hasattr(person, 'GivenName'):
                        props["owner_name"] = person.GivenName
                        
        # Extract creation and modification dates
        if hasattr(entity, 'OwnerHistory'):
            owner = entity.OwnerHistory
            if hasattr(owner, 'CreationDate'):
                props["creation_date"] = str(owner.CreationDate)
            if hasattr(owner, 'LastModifiedDate'):
                props["last_modified_date"] = str(owner.LastModifiedDate)
                
    except Exception as e:
        logging.warning(f"Error extracting properties from {entity.is_a()}: {str(e)}")
        
    return props

def extract_entity_geometry(entity) -> Dict[str, Any]:
    """Extract geometry information from IFC entity"""
    geometry = {
        "type": entity.is_a(),
        "has_position": False,
        "has_representation": False
    }
    
    try:
        # Extract placement information
        if hasattr(entity, 'ObjectPlacement'):
            placement = entity.ObjectPlacement
            if hasattr(placement, 'RelativePlacement'):
                rel_placement = placement.RelativePlacement
                if hasattr(rel_placement, 'Location'):
                    location = rel_placement.Location
                    if hasattr(location, 'Coordinates'):
                        coords = location.Coordinates
                        if len(coords) >= 3:
                            geometry["position"] = [float(coords[0]), float(coords[1]), float(coords[2])]
                            geometry["has_position"] = True
                            
        # Extract representation information
        if hasattr(entity, 'Representation'):
            representation = entity.Representation
            if hasattr(representation, 'Representations'):
                reps = representation.Representations
                if reps:
                    geometry["has_representation"] = True
                    geometry["representation_count"] = len(reps)
                    
                    # Extract representation types
                    rep_types = []
                    for rep in reps:
                        if hasattr(rep, 'RepresentationType'):
                            rep_types.append(rep.RepresentationType)
                    if rep_types:
                        geometry["representation_types"] = rep_types
                        
        # Extract specific geometry for different entity types
        if entity.is_a() == 'IfcWall':
            if hasattr(entity, 'Height'):
                geometry["height"] = entity.Height
            if hasattr(entity, 'Length'):
                geometry["length"] = entity.Length
                
        elif entity.is_a() == 'IfcBeam':
            if hasattr(entity, 'Length'):
                geometry["length"] = entity.Length
                
        elif entity.is_a() == 'IfcColumn':
            if hasattr(entity, 'Height'):
                geometry["height"] = entity.Height
                
        elif entity.is_a() == 'IfcSlab':
            if hasattr(entity, 'Thickness'):
                geometry["thickness"] = entity.Thickness
                
        elif entity.is_a() == 'IfcDoor':
            if hasattr(entity, 'OverallHeight'):
                geometry["overall_height"] = entity.OverallHeight
            if hasattr(entity, 'OverallWidth'):
                geometry["overall_width"] = entity.OverallWidth
                
        elif entity.is_a() == 'IfcWindow':
            if hasattr(entity, 'OverallHeight'):
                geometry["overall_height"] = entity.OverallHeight
            if hasattr(entity, 'OverallWidth'):
                geometry["overall_width"] = entity.OverallWidth
                
        elif entity.is_a() == 'IfcSpace':
            if hasattr(entity, 'IsExternal'):
                geometry["is_external"] = entity.IsExternal
            if hasattr(entity, 'ElevationWithFlooring'):
                geometry["elevation_with_flooring"] = entity.ElevationWithFlooring
                
    except Exception as e:
        logging.warning(f"Error extracting geometry from {entity.is_a()}: {str(e)}")
        
    return geometry

def parse_ifc_file(file_path: str) -> Dict[str, Any]:
    """
    Enhanced IFC file parser with comprehensive data extraction
    
    Args:
        file_path: Path to the IFC file
        
    Returns:
        Dictionary containing parsed data with metadata
    """
    try:
        validate_ifc_file(file_path)
        metadata = extract_ifc_metadata(file_path)
        
        # Open IFC model
        model = ifcopenshell.open(file_path)
        
        components = []
        entity_types = set()
        materials = set()
        properties = set()
        classifications = set()
        spatial_structures = set()
        
        # Extract schema information
        schema = model.schema
        
        # Get all entity types in the model
        all_entity_types = model.wrapped_data.entity_types()
        
        # Process all entity types
        for entity_type in all_entity_types:
            try:
                entities = model.by_type(entity_type)
                entity_types.add(entity_type)
                
                for entity in entities:
                    try:
                        # Extract properties and geometry
                        props = extract_entity_properties(entity)
                        geometry = extract_entity_geometry(entity)
                        
                        # Collect materials
                        if "material" in props:
                            materials.add(props["material"])
                            
                        # Collect properties
                        for key, value in props.items():
                            if key.startswith("prop_"):
                                properties.add(key[5:])  # Remove "prop_" prefix
                                
                        # Collect classifications
                        if "classification" in props:
                            classifications.add(props["classification"])
                            
                        # Collect spatial structures
                        if "spatial_structure" in props:
                            spatial_structures.add(props["spatial_structure"])
                        
                        # Create component object
                        component = {
                            "component_id": f"IFC-{uuid.uuid4().hex[:8]}",
                            "name": props.get("name", f"{entity_type}_{props.get('global_id', 'unknown')}"),
                            "type": entity_type,
                            "properties": props,
                            "geometry": geometry
                        }
                        
                        components.append(component)
                        
                    except Exception as e:
                        logging.warning(f"Error processing IFC entity {entity_type}: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.warning(f"Error processing IFC entity type {entity_type}: {str(e)}")
                continue
        
        # Extract project information
        project_info = {}
        try:
            projects = model.by_type('IfcProject')
            if projects:
                project = projects[0]
                project_info = {
                    "name": getattr(project, 'Name', ''),
                    "description": getattr(project, 'Description', ''),
                    "global_id": getattr(project, 'GlobalId', ''),
                    "object_type": getattr(project, 'ObjectType', '')
                }
        except Exception as e:
            logging.warning(f"Error extracting project information: {str(e)}")
        
        # Extract site information
        site_info = {}
        try:
            sites = model.by_type('IfcSite')
            if sites:
                site = sites[0]
                site_info = {
                    "name": getattr(site, 'Name', ''),
                    "description": getattr(site, 'Description', ''),
                    "global_id": getattr(site, 'GlobalId', ''),
                    "ref_latitude": getattr(site, 'RefLatitude', None),
                    "ref_longitude": getattr(site, 'RefElevation', None)
                }
        except Exception as e:
            logging.warning(f"Error extracting site information: {str(e)}")
        
        # Calculate bounding box from all entities with positions
        positions = []
        for comp in components:
            geom = comp["geometry"]
            if "position" in geom:
                positions.append(geom["position"])
        
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
            "schema_info": {
                "schema": schema,
                "entity_types_count": len(entity_types)
            },
            "project_info": project_info,
            "site_info": site_info,
            "statistics": {
                "total_components": len(components),
                "entity_types": list(entity_types),
                "materials_count": len(materials),
                "properties_count": len(properties),
                "classifications_count": len(classifications),
                "spatial_structures_count": len(spatial_structures),
                "materials": list(materials),
                "properties": list(properties),
                "classifications": list(classifications),
                "spatial_structures": list(spatial_structures)
            },
            "components": components,
            "bounding_box": bounding_box,
            "parsed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error parsing IFC file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_metadata": extract_ifc_metadata(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "parsed_at": datetime.now().isoformat()
        }

# Legacy function for backward compatibility
def parse_ifc_file_legacy(file_path: str) -> List[Dict[str, Any]]:
    """Legacy IFC parser function for backward compatibility"""
    result = parse_ifc_file(file_path)
    if result["status"] == "success":
        return result["components"]
    else:
        return [] 