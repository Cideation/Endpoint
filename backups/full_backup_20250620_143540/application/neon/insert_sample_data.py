"""
Insert sample component data for testing the Phase 2 pipeline
"""

import asyncio
import uuid
from datetime import datetime
from db_manager import NeonDBManager
from config import NEON_CONFIG

async def insert_sample_data():
    """Insert sample component data for testing"""
    print("Inserting sample component data...")
    
    db_manager = NeonDBManager(NEON_CONFIG)
    
    try:
        await db_manager.create_pool()
        
        # Sample components data
        sample_components = [
            {
                "component_id": str(uuid.uuid4()),
                "component_name": "Steel Beam H400",
                "component_type": "structural",
                "description": "H400 steel beam for structural support",
                "material_name": "Steel",
                "base_material": "Carbon Steel",
                "material_code": "S355",
                "centroid_x": 100.0,
                "centroid_y": 200.0,
                "centroid_z": 50.0,
                "bbox_x_mm": 400.0,
                "bbox_y_mm": 200.0,
                "bbox_z_mm": 100.0,
                "length_mm": 4000.0,
                "width_mm": 200.0,
                "height_mm": 400.0,
                "area_m2": 0.8,
                "volume_cm3": 320000.0,
                "tolerance_mm": 2.0,
                "geometry_format": "BIM",
                "geometry_type": "Beam",
                "vertex_count": 8,
                "face_count": 6,
                "edge_count": 12,
                "bounding_box_volume_cm3": 320000.0,
                "surface_area_m2": 3.2
            },
            {
                "component_id": str(uuid.uuid4()),
                "component_name": "Concrete Column C30",
                "component_type": "structural",
                "description": "C30 concrete column for building support",
                "material_name": "Concrete",
                "base_material": "Reinforced Concrete",
                "material_code": "C30/37",
                "centroid_x": 150.0,
                "centroid_y": 250.0,
                "centroid_z": 75.0,
                "bbox_x_mm": 300.0,
                "bbox_y_mm": 300.0,
                "bbox_z_mm": 3000.0,
                "length_mm": 300.0,
                "width_mm": 300.0,
                "height_mm": 3000.0,
                "area_m2": 0.09,
                "volume_cm3": 270000.0,
                "tolerance_mm": 5.0,
                "geometry_format": "BIM",
                "geometry_type": "Column",
                "vertex_count": 8,
                "face_count": 6,
                "edge_count": 12,
                "bounding_box_volume_cm3": 270000.0,
                "surface_area_m2": 3.6
            },
            {
                "component_id": str(uuid.uuid4()),
                "component_name": "HVAC Duct 600mm",
                "component_type": "mep",
                "description": "600mm diameter HVAC duct for air distribution",
                "material_name": "Galvanized Steel",
                "base_material": "Steel",
                "material_code": "GALV",
                "centroid_x": 200.0,
                "centroid_y": 300.0,
                "centroid_z": 100.0,
                "bbox_x_mm": 600.0,
                "bbox_y_mm": 600.0,
                "bbox_z_mm": 2000.0,
                "length_mm": 2000.0,
                "width_mm": 600.0,
                "height_mm": 600.0,
                "area_m2": 1.2,
                "volume_cm3": 720000.0,
                "tolerance_mm": 1.0,
                "geometry_format": "BIM",
                "geometry_type": "Duct",
                "vertex_count": 16,
                "face_count": 8,
                "edge_count": 24,
                "bounding_box_volume_cm3": 720000.0,
                "surface_area_m2": 4.8
            },
            {
                "component_id": str(uuid.uuid4()),
                "component_name": "Electrical Panel 400A",
                "component_type": "electrical",
                "description": "400A electrical distribution panel",
                "material_name": "Steel",
                "base_material": "Carbon Steel",
                "material_code": "STL",
                "centroid_x": 250.0,
                "centroid_y": 350.0,
                "centroid_z": 25.0,
                "bbox_x_mm": 800.0,
                "bbox_y_mm": 400.0,
                "bbox_z_mm": 200.0,
                "length_mm": 800.0,
                "width_mm": 400.0,
                "height_mm": 200.0,
                "area_m2": 0.32,
                "volume_cm3": 64000.0,
                "tolerance_mm": 1.5,
                "geometry_format": "BIM",
                "geometry_type": "Panel",
                "vertex_count": 8,
                "face_count": 6,
                "edge_count": 12,
                "bounding_box_volume_cm3": 64000.0,
                "surface_area_m2": 0.64
            },
            {
                "component_id": str(uuid.uuid4()),
                "component_name": "Glass Window 2x3m",
                "component_type": "envelope",
                "description": "2x3m glass window for natural lighting",
                "material_name": "Glass",
                "base_material": "Float Glass",
                "material_code": "GLASS",
                "centroid_x": 300.0,
                "centroid_y": 400.0,
                "centroid_z": 150.0,
                "bbox_x_mm": 2000.0,
                "bbox_y_mm": 100.0,
                "bbox_z_mm": 3000.0,
                "length_mm": 2000.0,
                "width_mm": 100.0,
                "height_mm": 3000.0,
                "area_m2": 6.0,
                "volume_cm3": 600000.0,
                "tolerance_mm": 0.5,
                "geometry_format": "BIM",
                "geometry_type": "Window",
                "vertex_count": 8,
                "face_count": 6,
                "edge_count": 12,
                "bounding_box_volume_cm3": 600000.0,
                "surface_area_m2": 6.0
            }
        ]
        
        inserted_count = 0
        
        for component_data in sample_components:
            try:
                # Insert component
                component_query = """
                    INSERT INTO components (component_id, component_name, component_type, description, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (component_id) DO NOTHING
                """
                await db_manager.execute_command(component_query, (
                    component_data["component_id"],
                    component_data["component_name"],
                    component_data["component_type"],
                    component_data["description"],
                    datetime.now(),
                    datetime.now()
                ))
                
                # Insert material
                material_query = """
                    INSERT INTO materials (material_name, base_material, material_code)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (material_name) DO NOTHING
                    RETURNING material_id
                """
                material_result = await db_manager.execute_query(material_query, (
                    component_data["material_name"],
                    component_data["base_material"],
                    component_data["material_code"]
                ))
                
                if material_result:
                    material_id = material_result[0]["material_id"]
                    
                    # Link component to material
                    link_query = """
                        INSERT INTO component_materials (component_id, material_id)
                        VALUES ($1, $2)
                        ON CONFLICT (component_id, material_id) DO NOTHING
                    """
                    await db_manager.execute_command(link_query, (
                        component_data["component_id"],
                        material_id
                    ))
                
                # Insert spatial data
                spatial_query = """
                    INSERT INTO spatial_data (component_id, centroid_x, centroid_y, centroid_z, bbox_x_mm, bbox_y_mm, bbox_z_mm)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (component_id) DO NOTHING
                """
                await db_manager.execute_command(spatial_query, (
                    component_data["component_id"],
                    component_data["centroid_x"],
                    component_data["centroid_y"],
                    component_data["centroid_z"],
                    component_data["bbox_x_mm"],
                    component_data["bbox_y_mm"],
                    component_data["bbox_z_mm"]
                ))
                
                # Insert dimensions
                dimensions_query = """
                    INSERT INTO dimensions (component_id, length_mm, width_mm, height_mm, area_m2, volume_cm3, tolerance_mm)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (component_id) DO NOTHING
                """
                await db_manager.execute_command(dimensions_query, (
                    component_data["component_id"],
                    component_data["length_mm"],
                    component_data["width_mm"],
                    component_data["height_mm"],
                    component_data["area_m2"],
                    component_data["volume_cm3"],
                    component_data["tolerance_mm"]
                ))
                
                # Insert geometry properties
                geometry_query = """
                    INSERT INTO geometry_properties (component_id, geometry_format, geometry_type, vertex_count, face_count, edge_count, bounding_box_volume_cm3, surface_area_m2)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (component_id) DO NOTHING
                """
                await db_manager.execute_command(geometry_query, (
                    component_data["component_id"],
                    component_data["geometry_format"],
                    component_data["geometry_type"],
                    component_data["vertex_count"],
                    component_data["face_count"],
                    component_data["edge_count"],
                    component_data["bounding_box_volume_cm3"],
                    component_data["surface_area_m2"]
                ))
                
                inserted_count += 1
                print(f"✅ Inserted: {component_data['component_name']}")
                
            except Exception as e:
                print(f"❌ Failed to insert {component_data['component_name']}: {e}")
        
        print(f"\n🎉 Successfully inserted {inserted_count} sample components")
        
        # Verify insertion
        count_query = "SELECT COUNT(*) as count FROM components"
        count_result = await db_manager.execute_query(count_query)
        total_components = count_result[0]["count"]
        print(f"📊 Total components in database: {total_components}")
        
        return inserted_count
        
    except Exception as e:
        print(f"❌ Error inserting sample data: {e}")
        return 0
    finally:
        await db_manager.close_async()

if __name__ == "__main__":
    asyncio.run(insert_sample_data()) 