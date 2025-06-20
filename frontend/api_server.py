"""
Frontend API Server for Neon CAD Parser
Serves the frontend and provides API endpoints that connect to the Neon database
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import neon modules
sys.path.append(str(Path(__file__).parent.parent))

from neon.frontend_service import FrontendAPIService
from neon.api_models import (
    ComponentSearchRequest, FileSearchRequest, 
    ComponentListResponse, ComponentDetailResponse,
    FileProcessingResponse, DashboardStatistics
)

app = FastAPI(
    title="Neon CAD Parser API",
    description="Frontend API for Neon Database Integration",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the frontend service
frontend_service = FrontendAPIService()

@app.on_event("startup")
async def startup_event():
    """Initialize the frontend service on startup"""
    await frontend_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    await frontend_service.close()

# =====================================================
# STATIC FILES AND FRONTEND
# =====================================================

@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML file"""
    return FileResponse("frontend/index.html")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/api/v1/dashboard/statistics")
async def get_dashboard_statistics():
    """Get dashboard statistics"""
    try:
        stats = await frontend_service.get_dashboard_statistics()
        return {
            "success": True,
            "data": {
                "components": {
                    "total_components": stats.components.total_components,
                    "components_by_type": stats.components.components_by_type,
                    "components_with_spatial_data": stats.components.components_with_spatial_data,
                    "components_with_materials": stats.components.components_with_materials,
                    "components_with_dimensions": stats.components.components_with_dimensions,
                    "components_created_today": stats.components.components_created_today,
                    "components_created_this_week": stats.components.components_created_this_week,
                },
                "files": {
                    "total_files": stats.files.total_files,
                    "files_by_type": stats.files.files_by_type,
                    "files_by_status": stats.files.files_by_status,
                    "total_components_extracted": stats.files.total_components_extracted,
                    "average_processing_time_ms": stats.files.average_processing_time_ms,
                    "files_processed_today": stats.files.files_processed_today,
                    "files_processed_this_week": stats.files.files_processed_this_week,
                },
                "last_updated": stats.last_updated.isoformat()
            },
            "message": "Dashboard statistics retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/components")
async def get_components(
    query: str = None,
    component_type: str = None,
    has_spatial_data: bool = None,
    has_materials: bool = None,
    page: int = 1,
    page_size: int = 10
):
    """Get paginated list of components with search and filters"""
    try:
        search_request = ComponentSearchRequest(
            query=query,
            component_type=component_type,
            has_spatial_data=has_spatial_data,
            has_materials=has_materials,
            page=page,
            page_size=page_size
        )
        
        result = await frontend_service.get_components(search_request)
        
        return {
            "success": result.success,
            "data": [
                {
                    "component_id": str(comp.component_id),
                    "component_name": comp.component_name,
                    "component_type": comp.component_type.value,
                    "description": comp.description,
                    "created_at": comp.created_at.isoformat(),
                    "updated_at": comp.updated_at.isoformat(),
                    "has_spatial_data": comp.has_spatial_data,
                    "has_materials": comp.has_materials,
                    "has_dimensions": comp.has_dimensions,
                    "material_count": comp.material_count
                }
                for comp in result.data
            ],
            "total_count": result.total_count,
            "page": result.page,
            "page_size": result.page_size,
            "total_pages": result.total_pages,
            "message": result.message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/components/{component_id}")
async def get_component_detail(component_id: str):
    """Get detailed component information"""
    try:
        from uuid import UUID
        result = await frontend_service.get_component_detail(UUID(component_id))
        
        if not result.success:
            raise HTTPException(status_code=404, detail=result.message)
        
        return {
            "success": result.success,
            "data": {
                "component_id": str(result.data.component_id),
                "component_name": result.data.component_name,
                "component_type": result.data.component_type.value,
                "description": result.data.description,
                "created_at": result.data.created_at.isoformat(),
                "updated_at": result.data.updated_at.isoformat(),
                "spatial_data": result.data.spatial_data.model_dump() if result.data.spatial_data else None,
                "dimensions": result.data.dimensions.model_dump() if result.data.dimensions else None,
                "geometry_properties": result.data.geometry_properties.model_dump() if result.data.geometry_properties else None,
                "materials": [mat.model_dump() for mat in result.data.materials],
                "parsed_files": [pf.model_dump() for pf in result.data.parsed_files]
            },
            "message": result.message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form(...)
):
    """Upload and process a CAD file"""
    try:
        # Save uploaded file temporarily
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the file using the frontend service
        result = await frontend_service.process_file(
            str(file_path), 
            file.filename, 
            file_type
        )
        
        return {
            "success": result.success,
            "file_id": str(result.file_id) if result.file_id else None,
            "components_created": result.components_created,
            "processing_time_ms": result.processing_time_ms,
            "message": result.message,
            "errors": result.errors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/files")
async def get_files(
    query: str = None,
    file_type: str = None,
    status: str = None,
    page: int = 1,
    page_size: int = 10
):
    """Get paginated list of processed files"""
    try:
        search_request = FileSearchRequest(
            query=query,
            file_type=file_type,
            status=status,
            page=page,
            page_size=page_size
        )
        
        result = await frontend_service.get_files(search_request)
        
        return {
            "success": result.success,
            "data": [
                {
                    "file_id": str(file.file_id),
                    "filename": file.filename,
                    "file_type": file.file_type.value,
                    "status": file.status.value,
                    "progress_percentage": file.progress_percentage,
                    "components_extracted": file.components_extracted,
                    "processing_time_ms": file.processing_time_ms,
                    "error_message": file.error_message,
                    "created_at": file.created_at.isoformat(),
                    "updated_at": file.updated_at.isoformat()
                }
                for file in result.data
            ],
            "total_count": result.total_count,
            "page": result.page,
            "page_size": result.page_size,
            "total_pages": result.total_pages,
            "message": result.message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agents/signup")
async def agent_signup(
    name: str = Form(...),
    email: str = Form(...),
    agent_type: str = Form(...),
    ingest_file: UploadFile = File(None)
):
    """Create a new agent and optionally ingest initial data"""
    try:
        # For now, this is a placeholder that simulates agent creation
        # In the future, this will integrate with your Phase 2 microservices
        
        agent_data = {
            "name": name,
            "email": email,
            "agent_type": agent_type,
            "created_at": "2024-01-15T10:30:00Z"
        }
        
        # If file is provided, process it
        if ingest_file:
            # Save and process the ingest file
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            
            file_path = upload_dir / ingest_file.filename
            
            with open(file_path, "wb") as f:
                content = await ingest_file.read()
                f.write(content)
            
            # TODO: Process the ingest file and create agent-specific data
            agent_data["ingest_file_processed"] = True
            agent_data["ingest_filename"] = ingest_file.filename
        
        return {
            "success": True,
            "data": agent_data,
            "message": f"Agent {name} ({agent_type}) created successfully!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Neon CAD Parser API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 