"""
Redis Cache Manager for Neo Bank Performance Optimization
High-performance caching for contributions, SPV data, and agent registrations
"""

import redis
import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from functools import wraps
import asyncio
import aioredis
from dataclasses import asdict

logger = logging.getLogger(__name__)

class CacheConfig:
    """Cache configuration settings"""
    def __init__(self):
        self.redis_url = "redis://localhost:6379/0"
        self.default_ttl = 3600  # 1 hour
        self.max_connections = 20
        self.socket_timeout = 30
        self.socket_connect_timeout = 30
        self.retry_on_timeout = True
        self.health_check_interval = 30

class RedisCacheManager:
    """
    High-performance Redis cache manager for Neo Bank operations
    
    Features:
    - Automatic serialization/deserialization
    - TTL management with intelligent expiration
    - Cache warming and preloading
    - Performance monitoring
    - Failover handling
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_client = None
        self.async_client = None
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'total_operations': 0
        }
        
        self._connect()
        logger.info("Redis Cache Manager initialized")
    
    def _connect(self):
        """Establish Redis connection with retry logic"""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=False  # Handle binary data
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def _connect_async(self):
        """Establish async Redis connection"""
        try:
            self.async_client = await aioredis.from_url(
                self.config.redis_url,
                max_connections=self.config.max_connections
            )
            await self.async_client.ping()
            logger.info("Async Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to async Redis: {e}")
            self.async_client = None
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage"""
        try:
            if isinstance(data, (dict, list)):
                return json.dumps(data, default=str).encode('utf-8')
            elif hasattr(data, '__dict__'):
                return json.dumps(asdict(data), default=str).encode('utf-8')
            else:
                return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis"""
        try:
            # Try JSON first (most common)
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Fall back to pickle
                return pickle.loads(data)
            except Exception as e:
                logger.error(f"Deserialization error: {e}")
                return None
    
    def _generate_cache_key(self, prefix: str, identifier: str, params: Dict = None) -> str:
        """Generate consistent cache keys"""
        key_parts = [prefix, identifier]
        
        if params:
            # Sort params for consistent key generation
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            key_parts.append(param_hash)
        
        return ":".join(key_parts)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            self.stats['total_operations'] += 1
            data = self.redis_client.get(key)
            
            if data:
                self.stats['hits'] += 1
                return self._deserialize_data(data)
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if not self.redis_client:
            return False
        
        try:
            self.stats['total_operations'] += 1
            self.stats['sets'] += 1
            
            serialized_data = self._serialize_data(value)
            ttl = ttl or self.config.default_ttl
            
            result = self.redis_client.setex(key, ttl, serialized_data)
            return bool(result)
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
        
        try:
            self.stats['total_operations'] += 1
            self.stats['deletes'] += 1
            
            result = self.redis_client.delete(key)
            return bool(result)
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {e}")
            return False
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        if not self.redis_client or not keys:
            return {}
        
        try:
            self.stats['total_operations'] += len(keys)
            
            # Use pipeline for better performance
            pipe = self.redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            
            results = pipe.execute()
            
            output = {}
            for key, data in zip(keys, results):
                if data:
                    self.stats['hits'] += 1
                    output[key] = self._deserialize_data(data)
                else:
                    self.stats['misses'] += 1
            
            return output
            
        except Exception as e:
            self.stats['errors'] += len(keys)
            logger.error(f"Cache get_many error: {e}")
            return {}
    
    def set_many(self, data: Dict[str, Any], ttl: int = None) -> bool:
        """Set multiple values in cache"""
        if not self.redis_client or not data:
            return False
        
        try:
            self.stats['total_operations'] += len(data)
            self.stats['sets'] += len(data)
            
            ttl = ttl or self.config.default_ttl
            
            # Use pipeline for better performance
            pipe = self.redis_client.pipeline()
            for key, value in data.items():
                serialized_data = self._serialize_data(value)
                pipe.setex(key, ttl, serialized_data)
            
            results = pipe.execute()
            return all(results)
            
        except Exception as e:
            self.stats['errors'] += len(data)
            logger.error(f"Cache set_many error: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1, ttl: int = None) -> Optional[int]:
        """Increment counter in cache"""
        if not self.redis_client:
            return None
        
        try:
            self.stats['total_operations'] += 1
            
            # Use pipeline to increment and set TTL atomically
            pipe = self.redis_client.pipeline()
            pipe.incr(key, amount)
            if ttl:
                pipe.expire(key, ttl)
            
            results = pipe.execute()
            return results[0]
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate_percent': round(hit_rate, 2),
            'total_operations': self.stats['total_operations'],
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'deletes': self.stats['deletes'],
            'errors': self.stats['errors'],
            'connected': self.redis_client is not None
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check"""
        if not self.redis_client:
            return {
                'healthy': False,
                'error': 'No Redis connection'
            }
        
        try:
            start_time = datetime.now()
            self.redis_client.ping()
            response_time = (datetime.now() - start_time).total_seconds()
            
            info = self.redis_client.info()
            
            return {
                'healthy': True,
                'response_time_seconds': response_time,
                'redis_version': info.get('redis_version'),
                'used_memory_human': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed')
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0

class NeoBankCacheService:
    """
    High-level cache service for Neo Bank operations
    Provides domain-specific caching methods
    """
    
    def __init__(self, cache_manager: RedisCacheManager = None):
        self.cache = cache_manager or RedisCacheManager()
        
        # Cache TTL configurations for different data types
        self.ttl_config = {
            'spv_data': 1800,      # 30 minutes
            'agent_data': 3600,    # 1 hour
            'contribution': 300,   # 5 minutes
            'release_status': 60,  # 1 minute
            'api_response': 120,   # 2 minutes
            'statistics': 300      # 5 minutes
        }
    
    def cache_spv_data(self, spv_id: str, spv_data: Dict) -> bool:
        """Cache SPV data"""
        key = self.cache._generate_cache_key("spv", spv_id)
        return self.cache.set(key, spv_data, self.ttl_config['spv_data'])
    
    def get_spv_data(self, spv_id: str) -> Optional[Dict]:
        """Get cached SPV data"""
        key = self.cache._generate_cache_key("spv", spv_id)
        return self.cache.get(key)
    
    def cache_agent_integration(self, agent_id: str, integration_data: Dict) -> bool:
        """Cache agent integration data"""
        key = self.cache._generate_cache_key("agent", agent_id)
        return self.cache.set(key, integration_data, self.ttl_config['agent_data'])
    
    def get_agent_integration(self, agent_id: str) -> Optional[Dict]:
        """Get cached agent integration data"""
        key = self.cache._generate_cache_key("agent", agent_id)
        return self.cache.get(key)
    
    def cache_contribution_status(self, spv_id: str, status_data: Dict) -> bool:
        """Cache contribution status for SPV"""
        key = self.cache._generate_cache_key("contribution_status", spv_id)
        return self.cache.set(key, status_data, self.ttl_config['contribution'])
    
    def get_contribution_status(self, spv_id: str) -> Optional[Dict]:
        """Get cached contribution status"""
        key = self.cache._generate_cache_key("contribution_status", spv_id)
        return self.cache.get(key)
    
    def cache_api_response(self, endpoint: str, params: Dict, response_data: Any) -> bool:
        """Cache API response"""
        key = self.cache._generate_cache_key("api", endpoint, params)
        return self.cache.set(key, response_data, self.ttl_config['api_response'])
    
    def get_cached_api_response(self, endpoint: str, params: Dict) -> Optional[Any]:
        """Get cached API response"""
        key = self.cache._generate_cache_key("api", endpoint, params)
        return self.cache.get(key)
    
    def increment_contribution_counter(self, spv_id: str) -> Optional[int]:
        """Increment contribution counter for SPV"""
        key = self.cache._generate_cache_key("counter", f"contributions_{spv_id}")
        return self.cache.increment(key, ttl=self.ttl_config['statistics'])
    
    def warm_cache_for_spv(self, spv_id: str, spv_manager, paluwagan_engine) -> bool:
        """Warm cache with SPV-related data"""
        try:
            # Cache SPV details
            spv_details = spv_manager.get_spv_details(spv_id)
            if spv_details['success']:
                self.cache_spv_data(spv_id, spv_details['spv'])
            
            # Cache contribution status
            contribution_status = paluwagan_engine.get_contribution_status(spv_id)
            self.cache_contribution_status(spv_id, contribution_status)
            
            # Cache release conditions
            release_conditions = paluwagan_engine.check_spv_release_conditions(spv_id)
            key = self.cache._generate_cache_key("release_conditions", spv_id)
            self.cache.set(key, release_conditions, self.ttl_config['release_status'])
            
            return True
            
        except Exception as e:
            logger.error(f"Cache warming error for SPV {spv_id}: {e}")
            return False
    
    def invalidate_spv_cache(self, spv_id: str) -> bool:
        """Invalidate all cache entries for an SPV"""
        try:
            patterns = [
                f"spv:{spv_id}*",
                f"contribution_status:{spv_id}*",
                f"release_conditions:{spv_id}*",
                f"counter:contributions_{spv_id}*"
            ]
            
            total_cleared = 0
            for pattern in patterns:
                total_cleared += self.cache.clear_pattern(pattern)
            
            logger.info(f"Invalidated {total_cleared} cache entries for SPV {spv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation error for SPV {spv_id}: {e}")
            return False

def cache_response(ttl: int = None, key_prefix: str = "api"):
    """
    Decorator for caching API responses
    
    Usage:
        @cache_response(ttl=300, key_prefix="spv_status")
        def get_spv_status(spv_id):
            return expensive_operation(spv_id)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_service = NeoBankCacheService()
            
            # Create a unique key from function name and arguments
            func_name = func.__name__
            params = {
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            
            # Try to get from cache first
            cached_result = cache_service.get_cached_api_response(
                f"{key_prefix}_{func_name}", params
            )
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            cache_service.cache_api_response(
                f"{key_prefix}_{func_name}", params, result
            )
            
            return result
        
        return wrapper
    return decorator

# Global cache service instance
cache_service = NeoBankCacheService()

# Convenience functions
def get_cache_stats() -> Dict[str, Any]:
    """Get cache performance statistics"""
    return cache_service.cache.get_stats()

def perform_cache_health_check() -> Dict[str, Any]:
    """Perform cache health check"""
    return cache_service.cache.health_check()

if __name__ == "__main__":
    # Test the cache system
    cache = RedisCacheManager()
    
    # Test basic operations
    print("Testing cache operations...")
    
    # Set and get
    cache.set("test_key", {"message": "Hello Cache!"}, ttl=60)
    result = cache.get("test_key")
    print(f"Cache test result: {result}")
    
    # Performance stats
    stats = cache.get_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")
    
    # Health check
    health = cache.health_check()
    print(f"Cache health: {json.dumps(health, indent=2)}")
