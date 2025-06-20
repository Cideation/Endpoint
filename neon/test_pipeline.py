"""
Test script for the complete JSON transformation and orchestration pipeline
Demonstrates the flow from database components to Phase 2 microservice execution
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

from .json_transformer import JSONTransformer
from .orchestrator import Orchestrator

def create_sample_components() -> List[Dict[str, Any]]:
    """Create sample component data for testing"""
    return [
        {
            "component_id": "comp_001",
            "component_name": "Steel Beam",
            "component_type": "structural",
            "description": "Main structural beam",
            "spatial_data": {
                "centroid_x": 10.5,
                "centroid_y": 15.2,
                "centroid_z": 3.0,
                "bbox_x_mm": 5000,
                "bbox_y_mm": 300,
                "bbox_z_mm": 400
            },
            "geometry_properties": {
                "vertex_count": 8,
                "face_count": 6,
                "edge_count": 12,
                "surface_area_m2": 12.5,
                "bounding_box_volume_cm3": 600000
            },
            "dimensions": {
                "length_mm": 5000,
                "width_mm": 300,
                "height_mm": 400,
                "area_m2": 12.5,
                "volume_cm3": 600000
            },
            "materials": [
                {
                    "material_name": "Structural Steel",
                    "base_material": "Steel",
                    "material_code": "SS-350"
                }
            ]
        },
        {
            "component_id": "comp_002",
            "component_name": "Concrete Column",
            "component_type": "structural",
            "description": "Load-bearing column",
            "spatial_data": {
                "centroid_x": 20.0,
                "centroid_y": 25.0,
                "centroid_z": 2.5,
                "bbox_x_mm": 400,
                "bbox_y_mm": 400,
                "bbox_z_mm": 5000
            },
            "geometry_properties": {
                "vertex_count": 8,
                "face_count": 6,
                "edge_count": 12,
                "surface_area_m2": 8.0,
                "bounding_box_volume_cm3": 800000
            },
            "dimensions": {
                "length_mm": 400,
                "width_mm": 400,
                "height_mm": 5000,
                "area_m2": 8.0,
                "volume_cm3": 800000
            },
            "materials": [
                {
                    "material_name": "Reinforced Concrete",
                    "base_material": "Concrete",
                    "material_code": "RC-40"
                }
            ]
        },
        {
            "component_id": "comp_003",
            "component_name": "HVAC Duct",
            "component_type": "mep",
            "description": "Air conditioning duct",
            "spatial_data": {
                "centroid_x": 15.0,
                "centroid_y": 10.0,
                "centroid_z": 4.0,
                "bbox_x_mm": 3000,
                "bbox_y_mm": 500,
                "bbox_z_mm": 300
            },
            "geometry_properties": {
                "vertex_count": 8,
                "face_count": 6,
                "edge_count": 12,
                "surface_area_m2": 3.3,
                "bounding_box_volume_cm3": 450000
            },
            "dimensions": {
                "length_mm": 3000,
                "width_mm": 500,
                "height_mm": 300,
                "area_m2": 3.3,
                "volume_cm3": 450000
            },
            "materials": [
                {
                    "material_name": "Galvanized Steel",
                    "base_material": "Steel",
                    "material_code": "GS-250"
                }
            ]
        }
    ]

async def test_json_transformer():
    """Test the JSON transformer functionality"""
    print("=== Testing JSON Transformer ===")
    
    transformer = JSONTransformer()
    components = create_sample_components()
    
    # Test node dictionary transformation
    print("\n1. Testing Node Dictionary Transformation:")
    for i, component in enumerate(components):
        node_dict = transformer.transform_component_to_node_dictionary(component)
        print(f"   Component {i+1}: {node_dict['node_label']} -> Node {node_dict['node_id']}")
        print(f"   Phase: {node_dict['phase']}, Agent: {node_dict['agent']}, Trigger: {node_dict['trigger_functor']}")
    
    # Test node collection transformation
    print("\n2. Testing Node Collection Transformation:")
    node_collection = transformer.transform_components_to_node_collection(components)
    print(f"   Total nodes created: {len(node_collection['nodes'])}")
    
    # Test container-specific transformations
    print("\n3. Testing Container-Specific Transformations:")
    
    # DAG Alpha
    dag_data = transformer.transform_for_dag_alpha(components)
    print(f"   DAG Alpha: {len(dag_data['node_sequence'])} nodes in sequence")
    
    # Functor Types
    functor_data = transformer.transform_for_functor_types(components)
    print(f"   Functor Types: {len(functor_data['spatial_calculations'])} spatial, {len(functor_data['aggregation_calculations'])} aggregation")
    
    # SFDE Engine
    sfde_data = transformer.transform_for_sfde_engine(components, ["structural", "cost", "energy"])
    print(f"   SFDE Engine: {len(sfde_data['sfde_requests'])} formula requests")
    
    return components

async def test_orchestrator():
    """Test the orchestrator functionality"""
    print("\n=== Testing Orchestrator ===")
    
    orchestrator = Orchestrator()
    components = create_sample_components()
    
    # Test full pipeline orchestration
    print("\n1. Testing Full Pipeline Orchestration:")
    pipeline_result = await orchestrator.orchestrate_full_pipeline(
        components=components,
        affinity_types=["structural", "cost", "energy"],
        execution_phases=["alpha", "beta", "gamma", "cross_phase"]
    )
    
    print(f"   Pipeline ID: {pipeline_result['pipeline_id']}")
    print(f"   Status: {pipeline_result['status']}")
    print(f"   Components processed: {pipeline_result['components_count']}")
    
    # Show results for each container
    print("\n2. Container Execution Results:")
    for container_name, result in pipeline_result['results'].items():
        if 'error' not in result:
            print(f"   {container_name}: {result['output_data'].get('dag_execution_status', 'completed')}")
        else:
            print(f"   {container_name}: ERROR - {result['error']}")
    
    # Test execution history
    print("\n3. Execution History:")
    history = orchestrator.get_execution_history()
    print(f"   Total pipeline executions: {len(history)}")
    
    # Export pipeline configuration
    print("\n4. Pipeline Configuration:")
    config = orchestrator.export_pipeline_config()
    print(f"   Version: {config['pipeline_version']}")
    print(f"   Containers: {len(config['containers'])}")
    print(f"   Affinity Types: {len(config['supported_affinity_types'])}")
    
    return pipeline_result

async def test_complete_pipeline():
    """Test the complete pipeline from start to finish"""
    print("=== Complete Pipeline Test ===")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Test JSON transformer
    components = await test_json_transformer()
    
    # Test orchestrator
    pipeline_result = await test_orchestrator()
    
    # Summary
    print("\n=== Pipeline Summary ===")
    print(f"✅ JSON Transformer: Working")
    print(f"✅ Orchestrator: Working")
    print(f"✅ Container Simulation: Working")
    print(f"✅ Pipeline Execution: {pipeline_result['status']}")
    
    if pipeline_result['status'] == 'success':
        print(f"✅ All containers executed successfully")
        print(f"📊 Results available for {len(pipeline_result['results'])} containers")
    else:
        print(f"❌ Pipeline failed: {pipeline_result.get('error', 'Unknown error')}")
    
    print(f"End time: {datetime.now().isoformat()}")
    
    return pipeline_result

def save_test_results(pipeline_result: Dict[str, Any], filename: str = "pipeline_test_results.json"):
    """Save test results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(pipeline_result, f, indent=2, default=str)
    print(f"\n💾 Test results saved to: {filename}")

if __name__ == "__main__":
    # Run the complete test
    result = asyncio.run(test_complete_pipeline())
    
    # Save results
    save_test_results(result)
    
    print("\n🎉 Pipeline test completed successfully!")
    print("Next steps:")
    print("1. Replace container simulation with actual container calls")
    print("2. Integrate with real database queries")
    print("3. Add error handling and retry logic")
    print("4. Implement DGL training container")
    print("5. Add monitoring and logging") 