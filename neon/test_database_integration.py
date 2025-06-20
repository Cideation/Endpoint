"""
Test script for database integration with Neon PostgreSQL
Demonstrates connecting the pipeline to actual database data
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

from .db_manager import NeonDBManager
from .json_transformer import JSONTransformer
from .orchestrator import Orchestrator
from .config import NEON_CONFIG

async def test_database_connection():
    """Test connection to Neon PostgreSQL database"""
    print("=== Testing Neon Database Connection ===")
    
    db_manager = NeonDBManager(NEON_CONFIG)
    
    try:
        # Initialize connection pool
        await db_manager.create_pool()
        print("✅ Database connection pool created successfully")
        
        # Test basic query
        result = await db_manager.execute_query("SELECT version()")
        print(f"✅ Database version: {result[0]['version']}")
        
        # Test table existence
        tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """
        tables = await db_manager.execute_query(tables_query)
        print(f"✅ Found {len(tables)} tables in database:")
        for table in tables:
            print(f"   - {table['table_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    finally:
        await db_manager.close_async()

async def test_component_queries():
    """Test component queries from database"""
    print("\n=== Testing Component Queries ===")
    
    db_manager = NeonDBManager(NEON_CONFIG)
    
    try:
        await db_manager.create_pool()
        
        # Test component count
        count_query = "SELECT COUNT(*) as count FROM components"
        count_result = await db_manager.execute_query(count_query)
        component_count = count_result[0]['count']
        print(f"✅ Total components in database: {component_count}")
        
        if component_count > 0:
            # Get sample components
            sample_query = """
                SELECT 
                    c.component_id,
                    c.component_name,
                    c.component_type,
                    c.description,
                    c.created_at
                FROM components c
                ORDER BY c.created_at DESC
                LIMIT 5
            """
            components = await db_manager.execute_query(sample_query)
            
            print("✅ Sample components:")
            for i, comp in enumerate(components, 1):
                print(f"   {i}. {comp['component_name']} ({comp['component_type']})")
                print(f"      ID: {comp['component_id']}")
                print(f"      Created: {comp['created_at']}")
        
        return component_count
        
    except Exception as e:
        print(f"❌ Component query failed: {e}")
        return 0
    finally:
        await db_manager.close_async()

async def test_pipeline_with_real_data():
    """Test the complete pipeline with real database data"""
    print("\n=== Testing Pipeline with Real Database Data ===")
    
    db_manager = NeonDBManager(NEON_CONFIG)
    transformer = JSONTransformer()
    orchestrator = Orchestrator()
    
    try:
        await db_manager.create_pool()
        
        # Get components from database
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
            LIMIT 10
        """
        
        results = await db_manager.execute_query(query)
        
        if not results:
            print("⚠️  No components found in database")
            return
        
        # Transform database results to component dictionaries
        components = []
        for row in results:
            component = transform_db_row_to_component(row)
            components.append(component)
        
        print(f"✅ Retrieved {len(components)} components from database")
        
        # Test JSON transformation
        print("\n1. Testing JSON Transformation:")
        node_collection = transformer.transform_components_to_node_collection(components)
        print(f"   Created {len(node_collection['nodes'])} nodes")
        
        # Test container-specific transformations
        dag_data = transformer.transform_for_dag_alpha(components)
        print(f"   DAG Alpha: {len(dag_data['node_sequence'])} nodes")
        
        functor_data = transformer.transform_for_functor_types(components)
        print(f"   Functor Types: {len(functor_data['spatial_calculations'])} spatial calculations")
        
        # Test pipeline orchestration
        print("\n2. Testing Pipeline Orchestration:")
        pipeline_result = await orchestrator.orchestrate_full_pipeline(
            components=components,
            affinity_types=["structural", "cost", "energy"],
            execution_phases=["alpha", "cross_phase"]
        )
        
        print(f"   Pipeline ID: {pipeline_result['pipeline_id']}")
        print(f"   Status: {pipeline_result['status']}")
        print(f"   Components processed: {pipeline_result['components_count']}")
        
        # Show results
        print("\n3. Pipeline Results:")
        for container_name, result in pipeline_result['results'].items():
            if 'error' not in result:
                print(f"   ✅ {container_name}: Success")
            else:
                print(f"   ❌ {container_name}: {result['error']}")
        
        return pipeline_result
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return None
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

async def test_complete_database_integration():
    """Test the complete database integration"""
    print("=== Complete Database Integration Test ===")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Test database connection
    connection_success = await test_database_connection()
    if not connection_success:
        print("❌ Cannot proceed without database connection")
        return
    
    # Test component queries
    component_count = await test_component_queries()
    if component_count == 0:
        print("⚠️  No components in database - using sample data")
        # You might want to insert some sample data here
    
    # Test pipeline with real data
    pipeline_result = await test_pipeline_with_real_data()
    
    # Summary
    print("\n=== Database Integration Summary ===")
    print(f"✅ Database Connection: {'Working' if connection_success else 'Failed'}")
    print(f"✅ Component Queries: {'Working' if component_count > 0 else 'No Data'}")
    print(f"✅ Pipeline Execution: {'Working' if pipeline_result else 'Failed'}")
    
    if pipeline_result and pipeline_result['status'] == 'success':
        print(f"✅ Real Data Pipeline: Success")
        print(f"📊 Processed {pipeline_result['components_count']} real components")
        print(f"🔗 Connected to: {NEON_CONFIG['host']}")
    else:
        print(f"❌ Real Data Pipeline: Failed")
    
    print(f"End time: {datetime.now().isoformat()}")
    
    return pipeline_result

def save_database_test_results(result: Dict[str, Any], filename: str = "database_test_results.json"):
    """Save database test results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n💾 Database test results saved to: {filename}")

if __name__ == "__main__":
    # Run the complete database integration test
    result = asyncio.run(test_complete_database_integration())
    
    # Save results
    if result:
        save_database_test_results(result)
    
    print("\n🎉 Database integration test completed!")
    print("Next steps:")
    print("1. Verify database schema matches expectations")
    print("2. Insert sample data if database is empty")
    print("3. Connect to actual Phase 2 containers")
    print("4. Deploy to production environment") 