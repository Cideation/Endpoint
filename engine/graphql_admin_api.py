#!/usr/bin/env python3
"""
GraphQL Admin API - Structured System Management
Handles configuration, schema, functor management, and administrative operations
Complements Socket.IO streaming with structured data operations
"""

import strawberry
import json
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

logger = logging.getLogger(__name__)

@strawberry.type
class SystemStatus:
    nodes_active: int
    agents_active: int
    phases_running: List[str]
    last_update: str
    system_health: str

@strawberry.type
class PhaseConfiguration:
    phase_name: str
    display_name: str
    color: str
    node_types: List[str]
    primary_signals: List[str]
    behaviors: List[str]
    active: bool

@strawberry.type
class FunctorType:
    functor_id: str
    functor_name: str
    functor_type: str
    phase: str
    data_affinity: List[str]
    description: str
    enabled: bool

@strawberry.input
class PhaseConfigInput:
    phase_name: str
    display_name: Optional[str] = None
    color: Optional[str] = None
    active: Optional[bool] = None

class AdminDataManager:
    def __init__(self):
        self.base_path = os.path.dirname(__file__)
        self.microservices_path = os.path.join(self.base_path, '..', 'MICROSERVICE_ENGINES')
        self.graph_hints_path = os.path.join(self.microservices_path, 'graph_hints')
        
    def get_system_status(self) -> SystemStatus:
        return SystemStatus(
            nodes_active=45,
            agents_active=8,
            phases_running=["alpha", "beta"],
            last_update=datetime.now().isoformat(),
            system_health="healthy"
        )
    
    def get_phase_configurations(self) -> List[PhaseConfiguration]:
        phase_map_path = os.path.join(self.graph_hints_path, 'phase_map.json')
        
        try:
            with open(phase_map_path, 'r') as f:
                phase_data = json.load(f)
            
            phases = []
            for phase_name, config in phase_data.items():
                phases.append(PhaseConfiguration(
                    phase_name=phase_name,
                    display_name=config.get('name', phase_name),
                    color=config.get('color', '#9E9E9E'),
                    node_types=config.get('node_types', []),
                    primary_signals=config.get('primary_signals', []),
                    behaviors=config.get('behaviors', []),
                    active=config.get('active', False)
                ))
            
            return phases
            
        except FileNotFoundError:
            return self._get_default_phases()
    
    def _get_default_phases(self) -> List[PhaseConfiguration]:
        return [
            PhaseConfiguration(
                phase_name="alpha",
                display_name="Alpha - DAG Processing",
                color="#3F51B5",
                node_types=["V01_ProductComponent"],
                primary_signals=["design_signal"],
                behaviors=["sequential_processing"],
                active=True
            )
        ]

admin_data_manager = AdminDataManager()

@strawberry.type
class Query:
    @strawberry.field
    def system_status(self) -> SystemStatus:
        return admin_data_manager.get_system_status()
    
    @strawberry.field
    def phase_configurations(self) -> List[PhaseConfiguration]:
        return admin_data_manager.get_phase_configurations()

@strawberry.type
class Mutation:
    @strawberry.field
    def reload_system_configuration(self) -> bool:
        return True

def create_admin_api_app() -> FastAPI:
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    
    app = FastAPI(
        title="BEM System Admin API",
        description="GraphQL API for system configuration and management",
        version="1.0.0"
    )
    
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    graphql_app = GraphQLRouter(schema)
    app.include_router(graphql_app, prefix="/graphql")
    
    @app.get("/")
    async def root():
        return {"message": "BEM System Admin API", "graphql_endpoint": "/graphql"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    return app

if __name__ == "__main__":
    import uvicorn
    
    app = create_admin_api_app()
    
    print("🚀 Starting BEM System Admin API...")
    print("📊 GraphQL Playground: http://localhost:8001/graphql")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
