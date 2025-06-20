#!/usr/bin/env python3
"""
Container Client - Real Microservice Communication

Replaces mock container calls with actual HTTP communication to microservice containers.
Handles container discovery, load balancing, error handling, and retry logic.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ContainerType(Enum):
    """Microservice container types"""
    DAG_ALPHA = "ne-dag-alpha"
    FUNCTOR_TYPES = "ne-functor-types"
    CALLBACK_ENGINE = "ne-callback-engine"
    SFDE_ENGINE = "sfde"
    GRAPH_RUNTIME = "ne-graph-runtime-engine"
    OPTIMIZATION_ENGINE = "ne-optimization-engine"
    DGL_TRAINING = "dgl-training"

@dataclass
class ContainerEndpoint:
    """Container endpoint configuration"""
    name: str
    host: str
    port: int
    path: str = "/process"
    health_path: str = "/health"
    timeout: int = 30
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"
    
    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.health_path}"

class ContainerClient:
    """
    Real container communication client
    Handles HTTP requests to microservice containers
    """
    
    def __init__(self):
        self.endpoints = self._configure_endpoints()
        self.session: Optional[aiohttp.ClientSession] = None
        self.health_status: Dict[str, bool] = {}
        self.last_health_check: Dict[str, datetime] = {}
        
    def _configure_endpoints(self) -> Dict[ContainerType, ContainerEndpoint]:
        """Configure container endpoints based on Docker Compose or K8s discovery"""
        return {
            ContainerType.DAG_ALPHA: ContainerEndpoint(
                name="ne-dag-alpha",
                host="ne-dag-alpha",  # Docker Compose service name
                port=5000
            ),
            ContainerType.FUNCTOR_TYPES: ContainerEndpoint(
                name="ne-functor-types", 
                host="ne-functor-types",
                port=5001
            ),
            ContainerType.CALLBACK_ENGINE: ContainerEndpoint(
                name="ne-callback-engine",
                host="ne-callback-engine", 
                port=5002
            ),
            ContainerType.SFDE_ENGINE: ContainerEndpoint(
                name="sfde",
                host="sfde",
                port=5003
            ),
            ContainerType.GRAPH_RUNTIME: ContainerEndpoint(
                name="ne-graph-runtime-engine",
                host="ne-graph-runtime-engine",
                port=5004
            ),
            ContainerType.OPTIMIZATION_ENGINE: ContainerEndpoint(
                name="ne-optimization-engine",
                host="ne-optimization-engine",
                port=5005
            ),
            ContainerType.DGL_TRAINING: ContainerEndpoint(
                name="dgl-training",
                host="dgl-training",
                port=5006
            )
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def call_container(
        self, 
        container_type: ContainerType, 
        data: Dict[str, Any],
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Call specific container with data and handle response
        
        Args:
            container_type: Type of container to call
            data: Input data for container
            retries: Number of retry attempts
            
        Returns:
            Container response data
        """
        endpoint = self.endpoints.get(container_type)
        if not endpoint:
            return {
                "status": "error",
                "error": f"Unknown container type: {container_type}",
                "container": container_type.value
            }
        
        # Check container health first
        if not await self._check_container_health(endpoint):
            return {
                "status": "error", 
                "error": f"Container {endpoint.name} is unhealthy",
                "container": container_type.value
            }
        
        # Attempt container call with retries
        for attempt in range(retries + 1):
            try:
                return await self._make_request(endpoint, data)
                
            except aiohttp.ClientConnectorError:
                logger.error(f"Connection failed to {endpoint.name}, attempt {attempt + 1}")
                if attempt == retries:
                    return {
                        "status": "error",
                        "error": f"Failed to connect to {endpoint.name} after {retries + 1} attempts",
                        "container": container_type.value
                    }
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout calling {endpoint.name}, attempt {attempt + 1}")
                if attempt == retries:
                    return {
                        "status": "error",
                        "error": f"Timeout calling {endpoint.name}",
                        "container": container_type.value
                    }
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error calling {endpoint.name}: {e}")
                if attempt == retries:
                    return {
                        "status": "error",
                        "error": f"Unexpected error: {str(e)}",
                        "container": container_type.value
                    }
                await asyncio.sleep(2 ** attempt)
    
    async def _make_request(
        self, 
        endpoint: ContainerEndpoint, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make HTTP request to container endpoint"""
        if not self.session:
            raise RuntimeError("Session not initialized - use async context manager")
        
        start_time = datetime.now()
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": f"req_{int(start_time.timestamp())}"
        }
        
        request_payload = {
            "data": data,
            "timestamp": start_time.isoformat(),
            "source": "orchestrator"
        }
        
        # Make request
        async with self.session.post(
            endpoint.url,
            json=request_payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
        ) as response:
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if response.status == 200:
                result = await response.json()
                result["execution_time_seconds"] = execution_time
                result["container"] = endpoint.name
                result["status"] = result.get("status", "completed")
                
                logger.info(f"Container {endpoint.name} responded in {execution_time:.2f}s")
                return result
                
            else:
                error_text = await response.text()
                logger.error(f"Container {endpoint.name} returned {response.status}: {error_text}")
                
                return {
                    "status": "error",
                    "error": f"HTTP {response.status}: {error_text}",
                    "container": endpoint.name,
                    "execution_time_seconds": execution_time
                }
    
    async def _check_container_health(self, endpoint: ContainerEndpoint) -> bool:
        """Check if container is healthy"""
        # Use cached health status if recent
        now = datetime.now()
        last_check = self.last_health_check.get(endpoint.name)
        
        if last_check and (now - last_check) < timedelta(minutes=1):
            return self.health_status.get(endpoint.name, False)
        
        # Perform health check
        try:
            if not self.session:
                return False
                
            async with self.session.get(
                endpoint.health_url,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                healthy = response.status == 200
                self.health_status[endpoint.name] = healthy
                self.last_health_check[endpoint.name] = now
                
                if not healthy:
                    logger.warning(f"Container {endpoint.name} health check failed: {response.status}")
                
                return healthy
                
        except Exception as e:
            logger.error(f"Health check failed for {endpoint.name}: {e}")
            self.health_status[endpoint.name] = False
            self.last_health_check[endpoint.name] = now
            return False
    
    async def get_all_health_status(self) -> Dict[str, bool]:
        """Get health status of all containers"""
        health_results = {}
        
        for container_type, endpoint in self.endpoints.items():
            health_results[container_type.value] = await self._check_container_health(endpoint)
        
        return health_results
    
    async def discover_containers(self) -> List[str]:
        """Discover available containers"""
        available = []
        
        for container_type, endpoint in self.endpoints.items():
            if await self._check_container_health(endpoint):
                available.append(container_type.value)
        
        logger.info(f"Discovered {len(available)} healthy containers: {available}")
        return available

# Global client instance
_container_client: Optional[ContainerClient] = None

async def get_container_client() -> ContainerClient:
    """Get singleton container client instance"""
    global _container_client
    
    if _container_client is None:
        _container_client = ContainerClient()
    
    return _container_client

# Convenience functions for backward compatibility
async def call_container(container_type: ContainerType, data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for calling containers"""
    async with ContainerClient() as client:
        return await client.call_container(container_type, data)

async def check_container_health() -> Dict[str, bool]:
    """Convenience function for health checks"""
    async with ContainerClient() as client:
        return await client.get_all_health_status() 