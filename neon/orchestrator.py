"""
Orchestration Layer for Phase 2 Microservice Engine
Routes JSON data to correct containers and manages execution flow
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from .schemas import ExecutionPhase
from .json_transformer import JSONTransformer
from .container_client import ContainerClient, ContainerType

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Production Orchestrator with Real Container Communication
    
    Coordinates execution across microservice containers using HTTP communication.
    Replaces simulation with actual container calls.
    """
    
    def __init__(self):
        self.transformer = JSONTransformer()
        self.execution_history = []
        self.container_client = ContainerClient()
        
    async def execute_pipeline(
        self, 
        components: List[Dict[str, Any]], 
        phase: ExecutionPhase = ExecutionPhase.CROSS_PHASE
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline with real container communication
        """
        pipeline_start = datetime.now()
        logger.info(f"Starting pipeline execution with {len(components)} components")
        
        try:
            # Initialize container client
            async with self.container_client:
                # Check container health before starting
                health_status = await self.container_client.get_all_health_status()
                logger.info(f"Container health status: {health_status}")
                
                # Execute pipeline stages
                results = {
                    "pipeline_id": f"pipeline_{int(pipeline_start.timestamp())}",
                    "start_time": pipeline_start.isoformat(),
                    "phase": phase.value,
                    "components_count": len(components),
                    "container_health": health_status,
                    "results": []
                }
                
                # Stage 1: DAG Alpha (if container available)
                if health_status.get(ContainerType.DAG_ALPHA.value, False):
                    dag_result = await self._execute_dag_alpha(components)
                    results["results"].append(dag_result)
                
                # Stage 2: Functor Types (if container available)
                if health_status.get(ContainerType.FUNCTOR_TYPES.value, False):
                    functor_result = await self._execute_functor_types(components)
                    results["results"].append(functor_result)
                
                # Stage 3: Callback Engine (if container available)
                if health_status.get(ContainerType.CALLBACK_ENGINE.value, False):
                    callback_result = await self._execute_callback_engine(components, phase)
                    results["results"].append(callback_result)
                
                # Stage 4: SFDE Engine (if container available)
                if health_status.get(ContainerType.SFDE_ENGINE.value, False):
                    sfde_result = await self._execute_sfde_engine(components, ["structural", "cost", "energy"])
                    results["results"].append(sfde_result)
                
                # Stage 5: Graph Runtime (if container available)
                if health_status.get(ContainerType.GRAPH_RUNTIME.value, False):
                    graph_result = await self._execute_graph_runtime(components)
                    results["results"].append(graph_result)
                
                # Stage 6: DGL Training (if container available and enabled)
                if health_status.get(ContainerType.DGL_TRAINING.value, False):
                    dgl_result = await self._execute_dgl_training(components)
                    results["results"].append(dgl_result)
                
                # Calculate execution summary
                execution_time = (datetime.now() - pipeline_start).total_seconds()
                results.update({
                    "end_time": datetime.now().isoformat(),
                    "execution_time_seconds": execution_time,
                    "containers_executed": len(results["results"]),
                    "status": "completed" if results["results"] else "no_containers_available"
                })
                
                # Store in execution history
                self.execution_history.append(results)
                
                logger.info(f"Pipeline completed in {execution_time:.2f}s with {len(results['results'])} containers")
                return results
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time_seconds": (datetime.now() - pipeline_start).total_seconds()
            }
    
    async def _execute_dag_alpha(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute DAG Alpha container with real communication"""
        try:
            # Transform components for DAG Alpha
            dag_data = self.transformer.transform_for_dag_alpha(components)
            
            # Call real container
            result = await self.container_client.call_container(ContainerType.DAG_ALPHA, dag_data)
            
            return {
                "container": ContainerType.DAG_ALPHA.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": dag_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat(),
                "status": result.get("status", "completed")
            }
            
        except Exception as e:
            logger.error(f"DAG Alpha execution failed: {str(e)}")
            return {
                "container": ContainerType.DAG_ALPHA.value,
                "error": str(e), 
                "status": "failed"
            }
    
    async def _execute_functor_types(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Functor Types container with real communication"""
        try:
            # Transform components for Functor Types
            functor_data = self.transformer.transform_for_functor_types(components)
            
            # Call real container
            result = await self.container_client.call_container(ContainerType.FUNCTOR_TYPES, functor_data)
            
            return {
                "container": ContainerType.FUNCTOR_TYPES.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": functor_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat(),
                "status": result.get("status", "completed")
            }
            
        except Exception as e:
            logger.error(f"Functor Types execution failed: {str(e)}")
            return {
                "container": ContainerType.FUNCTOR_TYPES.value,
                "error": str(e), 
                "status": "failed"
            }
    
    async def _execute_callback_engine(
        self, 
        components: List[Dict[str, Any]], 
        phase: ExecutionPhase
    ) -> Dict[str, Any]:
        """Execute Callback Engine container with real communication"""
        try:
            # Transform components for Callback Engine
            callback_data = self.transformer.transform_for_callback_engine(
                components, 
                phase=phase.value
            )
            
            # Call real container
            result = await self.container_client.call_container(ContainerType.CALLBACK_ENGINE, callback_data)
            
            return {
                "container": ContainerType.CALLBACK_ENGINE.value,
                "phase": phase.value,
                "input_data": callback_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat(),
                "status": result.get("status", "completed")
            }
            
        except Exception as e:
            logger.error(f"Callback Engine execution failed: {str(e)}")
            return {
                "container": ContainerType.CALLBACK_ENGINE.value,
                "error": str(e), 
                "status": "failed"
            }
    
    async def _execute_sfde_engine(
        self, 
        components: List[Dict[str, Any]], 
        affinity_types: List[str]
    ) -> Dict[str, Any]:
        """Execute SFDE Engine container with real communication"""
        try:
            # Transform components for SFDE Engine
            sfde_data = self.transformer.transform_for_sfde_engine(components, affinity_types)
            
            # Call real container
            result = await self.container_client.call_container(ContainerType.SFDE_ENGINE, sfde_data)
            
            return {
                "container": ContainerType.SFDE_ENGINE.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": sfde_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat(),
                "status": result.get("status", "completed")
            }
            
        except Exception as e:
            logger.error(f"SFDE Engine execution failed: {str(e)}")
            return {
                "container": ContainerType.SFDE_ENGINE.value,
                "error": str(e), 
                "status": "failed"
            }
    
    async def _execute_graph_runtime(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Graph Runtime Engine container with real communication"""
        try:
            # Transform components for Graph Runtime
            graph_data = self.transformer.transform_for_graph_runtime(components)
            
            # Call real container
            result = await self.container_client.call_container(ContainerType.GRAPH_RUNTIME, graph_data)
            
            return {
                "container": ContainerType.GRAPH_RUNTIME.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": graph_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat(),
                "status": result.get("status", "completed")
            }
            
        except Exception as e:
            logger.error(f"Graph Runtime execution failed: {str(e)}")
            return {
                "container": ContainerType.GRAPH_RUNTIME.value,
                "error": str(e), 
                "status": "failed"
            }
    
    async def _execute_dgl_training(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute DGL Training Engine container with real communication"""
        try:
            # Prepare DGL training data
            dgl_data = {
                "components": components,
                "training_mode": "postgresql_based",
                "embedding_type": "node_edge",
                "optimization_targets": ["roi", "occupancy", "spec_fit"],
                "graph_data": self.transformer.transform_for_graph_runtime(components)
            }
            
            # Call real container
            result = await self.container_client.call_container(ContainerType.DGL_TRAINING, dgl_data)
            
            return {
                "container": ContainerType.DGL_TRAINING.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": dgl_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat(),
                "status": result.get("status", "completed")
            }
            
        except Exception as e:
            logger.error(f"DGL Training execution failed: {str(e)}")
            return {
                "container": ContainerType.DGL_TRAINING.value,
                "error": str(e), 
                "status": "failed"
            }
    
    async def get_container_health(self) -> Dict[str, Any]:
        """Get current health status of all containers"""
        try:
            async with self.container_client:
                health_status = await self.container_client.get_all_health_status()
                available_containers = await self.container_client.discover_containers()
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "health_status": health_status,
                    "available_containers": available_containers,
                    "total_containers": len(self.container_client.endpoints),
                    "healthy_containers": len(available_containers)
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "health_status": {},
                "available_containers": []
            }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history
    
    async def validate_containers(self) -> Dict[str, Any]:
        """Validate all container endpoints and configurations"""
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validations": []
        }
        
        try:
            async with self.container_client:
                for container_type, endpoint in self.container_client.endpoints.items():
                    validation = {
                        "container": container_type.value,
                        "endpoint": endpoint.url,
                        "health_url": endpoint.health_url,
                        "configured": True,
                        "reachable": False,
                        "healthy": False
                    }
                    
                    # Test reachability and health
                    try:
                        healthy = await self.container_client._check_container_health(endpoint)
                        validation["reachable"] = True
                        validation["healthy"] = healthy
                        
                    except Exception as e:
                        validation["error"] = str(e)
                    
                    validation_results["validations"].append(validation)
                    
        except Exception as e:
            logger.error(f"Container validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results
    
    def export_pipeline_config(self) -> Dict[str, Any]:
        """Export pipeline configuration"""
        return {
            "pipeline_version": "2.0",
            "containers": [ct.value for ct in ContainerType],
            "execution_phases": [ep.value for ep in ExecutionPhase],
            "supported_affinity_types": ["spatial", "structural", "cost", "energy", "mep", "time"],
            "configuration": {
                "async_execution": True,
                "error_handling": True,
                "execution_history": True,
                "dgl_training_ready": True
            }
        } 