"""
Run Pipeline Script
Simple interface to run the complete Phase 2 pipeline with database integration
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from .db_manager import NeonDBManager
from .json_transformer import JSONTransformer
from .orchestrator import Orchestrator
from .config import NEON_CONFIG

async def run_pipeline(
    component_type: Optional[str] = None,
    limit: int = 100,
    affinity_types: List[str] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run the complete Phase 2 pipeline with database integration
    
    Args:
        component_type: Filter by component type (e.g., 'structural', 'mep')
        limit: Maximum number of components to process
        affinity_types: List of SFDE affinity types to process
        save_results: Whether to save results to JSON file
    
    Returns:
        Pipeline execution results
    """
    
    print(f"🚀 Starting Phase 2 Pipeline")
    print(f"   Component Type: {component_type or 'All'}")
    print(f"   Limit: {limit}")
    print(f"   Affinity Types: {affinity_types or ['spatial', 'structural', 'cost', 'energy', 'mep', 'time']}")
    print(f"   Database: {NEON_CONFIG['host']}")
    print(f"   Start Time: {datetime.now().isoformat()}")
    
    # Initialize components
    db_manager = NeonDBManager(NEON_CONFIG)
    transformer = JSONTransformer()
    orchestrator = Orchestrator()
    
    try:
        # Connect to database
        await db_manager.create_pool()
        print("✅ Connected to Neon PostgreSQL database")
        
        # Build query based on filters
        if component_type:
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
            params = (f"%{component_type}%", limit)
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
                ORDER BY c.created_at DESC
                LIMIT $1
            """
            params = (limit,)
        
        # Get components from database
        print("📊 Querying database for components...")
        results = await db_manager.execute_query(query, params)
        
        if not results:
            print("⚠️  No components found in database")
            return {
                "status": "no_data",
                "message": "No components found in database",
                "components_count": 0,
                "end_time": datetime.now().isoformat()
            }
        
        # Transform database results to component dictionaries
        components = []
        for row in results:
            component = transform_db_row_to_component(row)
            components.append(component)
        
        print(f"✅ Retrieved {len(components)} components from database")
        
        # Set default affinity types if not provided
        if affinity_types is None:
            affinity_types = ["spatial", "structural", "cost", "energy", "mep", "time"]
        
        # Run the pipeline
        print("🔄 Running Phase 2 pipeline...")
        pipeline_result = await orchestrator.orchestrate_full_pipeline(
            components=components,
            affinity_types=affinity_types
        )
        
        # Add database metadata
        pipeline_result["database_metadata"] = {
            "component_type_filter": component_type,
            "limit": limit,
            "affinity_types": affinity_types,
            "database_host": NEON_CONFIG['host'],
            "components_retrieved": len(components)
        }
        
        pipeline_result["end_time"] = datetime.now().isoformat()
        
        # Print results summary
        print("\n📈 Pipeline Results Summary:")
        print(f"   Status: {pipeline_result['status']}")
        print(f"   Components Processed: {pipeline_result['components_count']}")
        print(f"   Pipeline ID: {pipeline_result['pipeline_id']}")
        
        if pipeline_result['status'] == 'success':
            print("   Container Results:")
            for container_name, result in pipeline_result['results'].items():
                if 'error' not in result:
                    print(f"     ✅ {container_name}: Success")
                else:
                    print(f"     ❌ {container_name}: {result['error']}")
        
        # Save results if requested
        if save_results:
            filename = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(pipeline_result, f, indent=2, default=str)
            print(f"💾 Results saved to: {filename}")
        
        print(f"🏁 Pipeline completed at: {datetime.now().isoformat()}")
        
        return pipeline_result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        }
        print(f"❌ Pipeline failed: {e}")
        return error_result
        
    finally:
        await db_manager.close_async()

def transform_db_row_to_component(row: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a database row to component dictionary format"""
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

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 2 Pipeline with Database Integration")
    parser.add_argument("--type", help="Filter by component type (e.g., structural, mep)")
    parser.add_argument("--limit", type=int, default=100, help="Maximum components to process")
    parser.add_argument("--affinity", nargs="+", help="SFDE affinity types")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Run pipeline
    result = asyncio.run(run_pipeline(
        component_type=args.type,
        limit=args.limit,
        affinity_types=args.affinity,
        save_results=not args.no_save
    ))
    
    # Exit with appropriate code
    sys.exit(0 if result.get('status') == 'success' else 1) 