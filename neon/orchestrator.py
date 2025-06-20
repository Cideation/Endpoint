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

try:
    from .json_transformer import JSONTransformer
except ImportError:
    from json_transformer import JSONTransformer

class ContainerType(str, Enum):
    """Phase 2 container types"""
    DAG_ALPHA = "ne-dag-alpha"
    FUNCTOR_TYPES = "ne-functor-types"
    CALLBACK_ENGINE = "ne-callback-engine"
    SFDE_ENGINE = "sfde-engine"
    GRAPH_RUNTIME = "ne-graph-runtime-engine"
    DGL_TRAINING = "dgl-training-engine"  # Future container
    API_GATEWAY = "api-gateway"

class ExecutionPhase(str, Enum):
    """Execution phases"""
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    CROSS_PHASE = "cross_phase"

class Orchestrator:
    """
    Orchestrates data flow between database and Phase 2 microservice containers
    """
    
    def __init__(self):
        self.transformer = JSONTransformer()
        self.logger = logging.getLogger(__name__)
        self.execution_history = []
        
    async def orchestrate_full_pipeline(
        self, 
        components: List[Dict[str, Any]],
        affinity_types: List[str] = None,
        execution_phases: List[ExecutionPhase] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate the complete pipeline from database to all containers
        """
        if affinity_types is None:
            affinity_types = ["spatial", "structural", "cost", "energy", "mep", "time"]
        
        if execution_phases is None:
            execution_phases = [ExecutionPhase.ALPHA, ExecutionPhase.BETA, ExecutionPhase.GAMMA]
        
        pipeline_results = {
            "pipeline_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "components_count": len(components),
            "results": {}
        }
        
        try:
            # 1. Transform for DAG Alpha (Alpha phase)
            if ExecutionPhase.ALPHA in execution_phases:
                self.logger.info("Executing DAG Alpha phase...")
                dag_result = await self._execute_dag_alpha(components)
                pipeline_results["results"]["dag_alpha"] = dag_result
            
            # 2. Transform for Functor Types (Cross-phase)
            if ExecutionPhase.CROSS_PHASE in execution_phases:
                self.logger.info("Executing Functor Types...")
                functor_result = await self._execute_functor_types(components)
                pipeline_results["results"]["functor_types"] = functor_result
            
            # 3. Transform for Callback Engine (Beta & Gamma phases)
            for phase in [ExecutionPhase.BETA, ExecutionPhase.GAMMA]:
                if phase in execution_phases:
                    self.logger.info(f"Executing Callback Engine for {phase.value} phase...")
                    callback_result = await self._execute_callback_engine(components, phase)
                    pipeline_results["results"][f"callback_{phase.value}"] = callback_result
            
            # 4. Transform for SFDE Engine (Cross-phase)
            if ExecutionPhase.CROSS_PHASE in execution_phases:
                self.logger.info("Executing SFDE Engine...")
                sfde_result = await self._execute_sfde_engine(components, affinity_types)
                pipeline_results["results"]["sfde_engine"] = sfde_result
            
            # 5. Transform for Graph Runtime (Cross-phase)
            if ExecutionPhase.CROSS_PHASE in execution_phases:
                self.logger.info("Executing Graph Runtime Engine...")
                graph_result = await self._execute_graph_runtime(components)
                pipeline_results["results"]["graph_runtime"] = graph_result
            
            # 6. Future: DGL Training Engine
            # if ExecutionPhase.CROSS_PHASE in execution_phases:
            #     self.logger.info("Executing DGL Training Engine...")
            #     dgl_result = await self._execute_dgl_training(components)
            #     pipeline_results["results"]["dgl_training"] = dgl_result
            
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["status"] = "success"
            
            # Store execution history
            self.execution_history.append(pipeline_results)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            pipeline_results["status"] = "error"
            pipeline_results["error"] = str(e)
            pipeline_results["end_time"] = datetime.now().isoformat()
        
        return pipeline_results
    
    async def _execute_dag_alpha(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute DAG Alpha container"""
        try:
            # Transform components for DAG Alpha
            dag_data = self.transformer.transform_for_dag_alpha(components)
            
            # Simulate container execution (replace with actual container call)
            result = await self._call_container(ContainerType.DAG_ALPHA, dag_data)
            
            return {
                "container": ContainerType.DAG_ALPHA.value,
                "phase": ExecutionPhase.ALPHA.value,
                "input_data": dag_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"DAG Alpha execution failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_functor_types(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Functor Types container"""
        try:
            # Transform components for Functor Types
            functor_data = self.transformer.transform_for_functor_types(components)
            
            # Simulate container execution
            result = await self._call_container(ContainerType.FUNCTOR_TYPES, functor_data)
            
            return {
                "container": ContainerType.FUNCTOR_TYPES.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": functor_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Functor Types execution failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_callback_engine(
        self, 
        components: List[Dict[str, Any]], 
        phase: ExecutionPhase
    ) -> Dict[str, Any]:
        """Execute Callback Engine container"""
        try:
            # Transform components for Callback Engine
            callback_data = self.transformer.transform_for_callback_engine(
                components, 
                phase=phase.value
            )
            
            # Simulate container execution
            result = await self._call_container(ContainerType.CALLBACK_ENGINE, callback_data)
            
            return {
                "container": ContainerType.CALLBACK_ENGINE.value,
                "phase": phase.value,
                "input_data": callback_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Callback Engine execution failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_sfde_engine(
        self, 
        components: List[Dict[str, Any]], 
        affinity_types: List[str]
    ) -> Dict[str, Any]:
        """Execute SFDE Engine container"""
        try:
            # Transform components for SFDE Engine
            sfde_data = self.transformer.transform_for_sfde_engine(components, affinity_types)
            
            # Simulate container execution
            result = await self._call_container(ContainerType.SFDE_ENGINE, sfde_data)
            
            return {
                "container": ContainerType.SFDE_ENGINE.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": sfde_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"SFDE Engine execution failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_graph_runtime(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Graph Runtime Engine container"""
        try:
            # Transform components for Graph Runtime
            graph_data = self.transformer.transform_for_graph_runtime(components)
            
            # Simulate container execution
            result = await self._call_container(ContainerType.GRAPH_RUNTIME, graph_data)
            
            return {
                "container": ContainerType.GRAPH_RUNTIME.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": graph_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Graph Runtime execution failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_dgl_training(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute DGL Training Engine container (future)"""
        try:
            # Future implementation for DGL training
            dgl_data = {
                "components": components,
                "training_mode": "postgresql_based",
                "embedding_type": "node_edge",
                "optimization_targets": ["roi", "occupancy", "spec_fit"]
            }
            
            # Simulate container execution
            result = await self._call_container(ContainerType.DGL_TRAINING, dgl_data)
            
            return {
                "container": ContainerType.DGL_TRAINING.value,
                "phase": ExecutionPhase.CROSS_PHASE.value,
                "input_data": dgl_data,
                "output_data": result,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"DGL Training execution failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _call_container(self, container_type: ContainerType, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific container with data
        This is a placeholder - replace with actual container communication
        """
        # Simulate async container call
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Return mock response based on container type
        if container_type == ContainerType.DAG_ALPHA:
            return {
                "dag_execution_status": "completed",
                "nodes_processed": len(data.get("node_sequence", [])),
                "execution_time_ms": 150,
                "results": {
                    "structural_evaluation": "passed",
                    "spatial_evaluation": "passed",
                    "manufacturing_score": 0.85
                }
            }
        
        elif container_type == ContainerType.FUNCTOR_TYPES:
            return {
                "functor_execution_status": "completed",
                "spatial_calculations": len(data.get("spatial_calculations", [])),
                "aggregation_calculations": len(data.get("aggregation_calculations", [])),
                "results": {
                    "spatial_metrics": {"centroid_distances": [1.2, 0.8, 1.5]},
                    "aggregation_metrics": {"total_volume": 1250.5, "total_area": 45.2}
                }
            }
        
        elif container_type == ContainerType.CALLBACK_ENGINE:
            return {
                "callback_execution_status": "completed",
                "callbacks_processed": len(data.get("callbacks", [])),
                "phase": data.get("phase"),
                "results": {
                    "compliance_check": "passed",
                    "relational_score": 0.92,
                    "combinatorial_score": 0.78
                }
            }
        
        elif container_type == ContainerType.SFDE_ENGINE:
            return {
                "sfde_execution_status": "completed",
                "formulas_executed": len(data.get("sfde_requests", [])),
                "affinity_types": data.get("affinity_types"),
                "results": {
                    "cost_calculations": {"total_cost": 125000, "unit_cost": 250},
                    "energy_calculations": {"total_energy": 4500, "efficiency": 0.85},
                    "structural_calculations": {"safety_factor": 1.8, "load_capacity": 2500}
                }
            }
        
        elif container_type == ContainerType.GRAPH_RUNTIME:
            return {
                "graph_execution_status": "completed",
                "nodes_built": len(data.get("graph_data", {}).get("nodes", [])),
                "edges_built": len(data.get("graph_data", {}).get("edges", [])),
                "results": {
                    "graph_metrics": {"density": 0.45, "diameter": 3, "clustering": 0.67},
                    "execution_path": ["V01", "V02", "V03", "V04"]
                }
            }
        
        elif container_type == ContainerType.DGL_TRAINING:
            return {
                "dgl_training_status": "completed",
                "training_epochs": 100,
                "optimization_targets": data.get("optimization_targets"),
                "results": {
                    "model_accuracy": 0.89,
                    "optimized_coefficients": {"roi_weight": 0.35, "occupancy_weight": 0.28},
                    "graph_embeddings": {"node_embeddings": 64, "edge_embeddings": 32}
                }
            }
        
        return {"status": "unknown_container", "data": data}
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history
    
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