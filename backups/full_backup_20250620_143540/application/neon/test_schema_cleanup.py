"""
Test Script to Demonstrate Schema Cleanup Opportunities
Shows current state and identifies areas for frontend readiness improvements
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_manager import NeonDBManager
from config import NEON_CONFIG
from models import Component, ComponentWithRelations
from schemas import NodeDictionary, DatabaseComponentSchema

async def test_current_schema_state():
    """Test current schema state and identify cleanup opportunities"""
    print("🔍 Testing Current Schema State")
    print("=" * 50)
    
    # Initialize database connection
    db_manager = NeonDBManager(NEON_CONFIG)
    await db_manager.create_pool()
    
    try:
        # Test 1: Check current component models
        print("\n1. Current Component Models:")
        print("-" * 30)
        
        # Show current model structure
        sample_component = Component(
            component_id="123e4567-e89b-12d3-a456-426614174000",
            component_name="Test Component",
            component_type="structural",
            description="Test description",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        print(f"✅ Component model: {sample_component.model_dump()}")
        print("⚠️  Issues: No frontend-friendly summary model")
        
        # Test 2: Check database integration
        print("\n2. Database Integration:")
        print("-" * 30)
        
        # Get actual components from database
        query = "SELECT * FROM components LIMIT 3"
        results = await db_manager.execute_query(query)
        
        if results:
            print(f"✅ Found {len(results)} components in database")
            for i, comp in enumerate(results, 1):
                print(f"   Component {i}: {comp['component_name']} ({comp['component_type']})")
        else:
            print("⚠️  No components found in database")
        
        # Test 3: Check schema validation
        print("\n3. Schema Validation:")
        print("-" * 30)
        
        # Test NodeDictionary schema
        try:
            node_dict = NodeDictionary(
                node_id="test_node",
                node_label="Test Node",
                phase="alpha",
                agent=1,
                callback_type="dag",
                trigger_functor="test_functor",
                dictionary={"test": "data"}
            )
            print("✅ NodeDictionary schema validation works")
        except Exception as e:
            print(f"❌ NodeDictionary schema validation failed: {e}")
        
        # Test 4: Check API response models
        print("\n4. API Response Models:")
        print("-" * 30)
        
        # Show current response structure
        current_response = {
            "success": True,
            "data": [comp['component_name'] for comp in results] if results else [],
            "message": "Components retrieved"
        }
        print(f"✅ Current response format: {json.dumps(current_response, indent=2)}")
        print("⚠️  Issues: No standardized response models")
        
        # Test 5: Check search and filtering
        print("\n5. Search and Filtering:")
        print("-" * 30)
        
        # Test basic search
        search_query = "SELECT * FROM components WHERE component_name ILIKE $1 LIMIT 5"
        search_results = await db_manager.execute_query(search_query, ("%",))
        
        print(f"✅ Basic search works: {len(search_results)} results")
        print("⚠️  Issues: No standardized search models")
        
        # Test 6: Check statistics
        print("\n6. Statistics and Analytics:")
        print("-" * 30)
        
        # Get basic statistics
        stats_queries = [
            ("Total components", "SELECT COUNT(*) FROM components"),
            ("Components by type", "SELECT component_type, COUNT(*) FROM components GROUP BY component_type"),
            ("Recent components", "SELECT COUNT(*) FROM components WHERE created_at >= NOW() - INTERVAL '7 days'")
        ]
        
        for stat_name, query in stats_queries:
            try:
                result = await db_manager.execute_query(query)
                if result:
                    print(f"✅ {stat_name}: {result[0]}")
                else:
                    print(f"⚠️  {stat_name}: No data")
            except Exception as e:
                print(f"❌ {stat_name}: Error - {e}")
        
        print("⚠️  Issues: No standardized statistics models")
        
        # Test 7: Check file processing
        print("\n7. File Processing:")
        print("-" * 30)
        
        # Check parsed files
        files_query = "SELECT * FROM parsed_files LIMIT 3"
        files_result = await db_manager.execute_query(files_query)
        
        if files_result:
            print(f"✅ Found {len(files_result)} parsed files")
            for file in files_result:
                print(f"   File: {file['file_name']} ({file['parsing_status']})")
        else:
            print("⚠️  No parsed files found")
        
        print("⚠️  Issues: No standardized file processing models")
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 SCHEMA CLEANUP SUMMARY")
        print("=" * 50)
        
        print("\n✅ What's Working:")
        print("  - Database schema is well-structured")
        print("  - Basic Pydantic models exist")
        print("  - Database integration is functional")
        print("  - Basic CRUD operations work")
        
        print("\n🔧 What Needs Cleanup:")
        print("  - Create frontend-ready API models")
        print("  - Standardize API response formats")
        print("  - Add search and filter models")
        print("  - Create dashboard statistics models")
        print("  - Add file processing status models")
        print("  - Implement bulk operations")
        print("  - Add export functionality")
        print("  - Create comprehensive validation")
        
        print("\n🚀 Next Steps:")
        print("  1. Create api_models.py with frontend-ready models")
        print("  2. Create frontend_service.py with standardized endpoints")
        print("  3. Add search, filter, and pagination support")
        print("  4. Implement dashboard statistics")
        print("  5. Add comprehensive error handling")
        print("  6. Create API documentation")
        
        print("\n💡 Benefits:")
        print("  - Consistent API for frontend development")
        print("  - Better user experience with proper validation")
        print("  - Easier maintenance and testing")
        print("  - Scalable architecture for future features")
        
    except Exception as e:
        print(f"❌ Error during schema testing: {e}")
    
    finally:
        await db_manager.close_async()

async def demonstrate_cleanup_opportunities():
    """Demonstrate specific cleanup opportunities"""
    print("\n🎯 DEMONSTRATING CLEANUP OPPORTUNITIES")
    print("=" * 50)
    
    # Example 1: Frontend-ready component model
    print("\n1. Frontend-Ready Component Model:")
    print("-" * 40)
    
    frontend_component = {
        "component_id": "123e4567-e89b-12d3-a456-426614174000",
        "component_name": "Steel Beam A-1",
        "component_type": "structural",
        "description": "Primary load-bearing beam",
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T10:30:00Z",
        "quick_stats": {
            "has_spatial_data": True,
            "has_materials": True,
            "has_dimensions": True,
            "material_count": 2
        }
    }
    
    print("✅ Proposed frontend model:")
    print(json.dumps(frontend_component, indent=2))
    
    # Example 2: Standardized API response
    print("\n2. Standardized API Response:")
    print("-" * 40)
    
    api_response = {
        "success": True,
        "message": "Components retrieved successfully",
        "data": [frontend_component],
        "pagination": {
            "page": 1,
            "page_size": 50,
            "total_count": 150,
            "total_pages": 3
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
    
    print("✅ Proposed API response format:")
    print(json.dumps(api_response, indent=2))
    
    # Example 3: Search and filter model
    print("\n3. Search and Filter Model:")
    print("-" * 40)
    
    search_request = {
        "query": "steel beam",
        "component_type": "structural",
        "has_spatial_data": True,
        "has_materials": True,
        "created_after": "2024-01-01T00:00:00Z",
        "page": 1,
        "page_size": 25
    }
    
    print("✅ Proposed search request model:")
    print(json.dumps(search_request, indent=2))
    
    # Example 4: Dashboard statistics
    print("\n4. Dashboard Statistics Model:")
    print("-" * 40)
    
    dashboard_stats = {
        "components": {
            "total_components": 1250,
            "components_by_type": {
                "structural": 450,
                "mep": 300,
                "architectural": 500
            },
            "components_with_spatial_data": 1100,
            "components_with_materials": 800,
            "components_created_today": 15,
            "components_created_this_week": 85
        },
        "files": {
            "total_files": 45,
            "files_by_type": {
                "DWG": 20,
                "IFC": 15,
                "PDF": 10
            },
            "files_by_status": {
                "success": 40,
                "processing": 3,
                "error": 2
            },
            "total_components_extracted": 1250
        },
        "last_updated": "2024-01-15T10:30:00Z"
    }
    
    print("✅ Proposed dashboard statistics model:")
    print(json.dumps(dashboard_stats, indent=2))
    
    print("\n🎉 These models would provide:")
    print("  - Consistent API structure")
    print("  - Frontend-friendly data formats")
    print("  - Comprehensive search and filtering")
    print("  - Rich dashboard analytics")
    print("  - Better user experience")

if __name__ == "__main__":
    asyncio.run(test_current_schema_state())
    asyncio.run(demonstrate_cleanup_opportunities()) 