#!/usr/bin/env python3
"""
Production Rate Limiting & Throttling System
Provides request throttling, DDoS protection, and API abuse prevention
"""

import time
import json
import redis
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, g

logger = logging.getLogger(__name__)

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    requests_per_window: int
    window_seconds: int
    burst_limit: Optional[int] = None
    key_generator: str = "ip"  # ip, user, api_key, endpoint
    block_duration: int = 3600  # 1 hour default block

class RateLimiter:
    """Production-grade rate limiter with Redis backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.rules = self._load_default_rules()
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis for rate limiting")
        except redis.ConnectionError:
            logger.error("Failed to connect to Redis - rate limiting will not work")
            raise
    
    def _load_default_rules(self) -> Dict[str, RateLimitRule]:
        """Load default rate limiting rules"""
        return {
            # General API limits
            'api_general': RateLimitRule(
                name="General API",
                requests_per_window=1000,
                window_seconds=3600,  # 1 hour
                burst_limit=100,
                key_generator="ip"
            ),
            
            # Authentication endpoints (more restrictive)
            'auth_endpoints': RateLimitRule(
                name="Authentication",
                requests_per_window=10,
                window_seconds=300,  # 5 minutes
                burst_limit=5,
                key_generator="ip",
                block_duration=900  # 15 minutes
            ),
            
            # File upload endpoints
            'file_upload': RateLimitRule(
                name="File Upload",
                requests_per_window=50,
                window_seconds=3600,  # 1 hour
                burst_limit=10,
                key_generator="user"
            ),
            
            # WebSocket connections
            'websocket_connections': RateLimitRule(
                name="WebSocket Connections",
                requests_per_window=100,
                window_seconds=3600,  # 1 hour
                burst_limit=20,
                key_generator="ip"
            ),
            
            # Database write operations
            'db_writes': RateLimitRule(
                name="Database Writes",
                requests_per_window=500,
                window_seconds=3600,  # 1 hour
                burst_limit=50,
                key_generator="user"
            ),
            
            # AI/ML processing endpoints
            'ai_processing': RateLimitRule(
                name="AI Processing",
                requests_per_window=20,
                window_seconds=3600,  # 1 hour
                burst_limit=5,
                key_generator="user"
            ),
            
            # Admin endpoints (very restrictive)
            'admin_endpoints': RateLimitRule(
                name="Admin Operations",
                requests_per_window=100,
                window_seconds=3600,  # 1 hour
                burst_limit=10,
                key_generator="user",
                block_duration=1800  # 30 minutes
            )
        }
    
    def add_rule(self, rule_key: str, rule: RateLimitRule):
        """Add custom rate limiting rule"""
        self.rules[rule_key] = rule
        logger.info(f"Added rate limiting rule: {rule_key}")
    
    def _generate_key(self, rule: RateLimitRule, request_context: Dict) -> str:
        """Generate rate limiting key based on rule configuration"""
        if rule.key_generator == "ip":
            identifier = request_context.get('ip', 'unknown')
        elif rule.key_generator == "user":
            identifier = request_context.get('user_id', request_context.get('ip', 'anonymous'))
        elif rule.key_generator == "api_key":
            identifier = request_context.get('api_key', request_context.get('ip', 'no_key'))
        elif rule.key_generator == "endpoint":
            identifier = f"{request_context.get('ip')}:{request_context.get('endpoint', 'unknown')}"
        else:
            identifier = request_context.get('ip', 'unknown')
        
        # Create hash for consistent key length
        key_hash = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        return f"rate_limit:{rule.name}:{key_hash}"
    
    def _is_blocked(self, key: str, rule: RateLimitRule) -> bool:
        """Check if key is currently blocked"""
        block_key = f"{key}:blocked"
        return self.redis_client.exists(block_key)
    
    def _block_key(self, key: str, rule: RateLimitRule):
        """Block a key for specified duration"""
        block_key = f"{key}:blocked"
        self.redis_client.setex(block_key, rule.block_duration, "1")
        logger.warning(f"Blocked key {key} for {rule.block_duration} seconds")
    
    def check_rate_limit(self, rule_key: str, request_context: Dict) -> Tuple[bool, Dict]:
        """
        Check if request is within rate limits
        Returns: (is_allowed, rate_limit_info)
        """
        if rule_key not in self.rules:
            logger.warning(f"Unknown rate limit rule: {rule_key}")
            return True, {}
        
        rule = self.rules[rule_key]
        key = self._generate_key(rule, request_context)
        current_time = int(time.time())
        
        # Check if key is blocked
        if self._is_blocked(key, rule):
            return False, {
                'rule': rule.name,
                'blocked': True,
                'block_expires': self.redis_client.ttl(f"{key}:blocked"),
                'reason': 'Rate limit exceeded - temporarily blocked'
            }
        
        # Use sliding window algorithm with Redis
        window_start = current_time - rule.window_seconds
        
        # Clean old entries and count current requests
        pipe = self.redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(current_time): current_time})
        pipe.expire(key, rule.window_seconds)
        
        results = pipe.execute()
        current_requests = results[1]
        
        # Check burst limit (requests in last minute)
        if rule.burst_limit:
            burst_window_start = current_time - 60  # 1 minute
            burst_count = self.redis_client.zcount(key, burst_window_start, current_time)
            
            if burst_count > rule.burst_limit:
                self._block_key(key, rule)
                return False, {
                    'rule': rule.name,
                    'blocked': True,
                    'burst_limit_exceeded': True,
                    'requests_in_burst': burst_count,
                    'burst_limit': rule.burst_limit,
                    'reason': 'Burst limit exceeded'
                }
        
        # Check main rate limit
        if current_requests >= rule.requests_per_window:
            # Block if significantly over limit
            if current_requests > rule.requests_per_window * 1.2:
                self._block_key(key, rule)
            
            return False, {
                'rule': rule.name,
                'rate_limited': True,
                'requests_made': current_requests,
                'requests_allowed': rule.requests_per_window,
                'window_seconds': rule.window_seconds,
                'reset_time': window_start + rule.window_seconds,
                'reason': 'Rate limit exceeded'
            }
        
        # Calculate remaining requests and reset time
        remaining = rule.requests_per_window - current_requests
        reset_time = window_start + rule.window_seconds
        
        return True, {
            'rule': rule.name,
            'allowed': True,
            'requests_made': current_requests,
            'requests_remaining': remaining,
            'requests_limit': rule.requests_per_window,
            'reset_time': reset_time,
            'window_seconds': rule.window_seconds
        }
    
    def get_rate_limit_status(self, rule_key: str, request_context: Dict) -> Dict:
        """Get current rate limit status without incrementing counter"""
        if rule_key not in self.rules:
            return {'error': 'Unknown rule'}
        
        rule = self.rules[rule_key]
        key = self._generate_key(rule, request_context)
        current_time = int(time.time())
        window_start = current_time - rule.window_seconds
        
        # Check if blocked
        if self._is_blocked(key, rule):
            return {
                'blocked': True,
                'block_expires': self.redis_client.ttl(f"{key}:blocked")
            }
        
        # Get current request count
        current_requests = self.redis_client.zcount(key, window_start, current_time)
        remaining = max(0, rule.requests_per_window - current_requests)
        
        return {
            'rule': rule.name,
            'requests_made': current_requests,
            'requests_remaining': remaining,
            'requests_limit': rule.requests_per_window,
            'window_seconds': rule.window_seconds,
            'reset_time': window_start + rule.window_seconds
        }
    
    def reset_rate_limit(self, rule_key: str, request_context: Dict):
        """Reset rate limit for a specific key (admin function)"""
        if rule_key not in self.rules:
            return False
        
        rule = self.rules[rule_key]
        key = self._generate_key(rule, request_context)
        
        # Remove rate limit data and any blocks
        self.redis_client.delete(key)
        self.redis_client.delete(f"{key}:blocked")
        
        logger.info(f"Reset rate limit for key: {key}")
        return True
    
    def get_global_stats(self) -> Dict:
        """Get global rate limiting statistics"""
        stats = {
            'active_keys': 0,
            'blocked_keys': 0,
            'rules_configured': len(self.rules),
            'redis_memory_usage': 0
        }
        
        try:
            # Count active rate limit keys
            rate_limit_keys = self.redis_client.keys("rate_limit:*")
            blocked_keys = [key for key in rate_limit_keys if key.endswith(":blocked")]
            
            stats['active_keys'] = len(rate_limit_keys) - len(blocked_keys)
            stats['blocked_keys'] = len(blocked_keys)
            
            # Get Redis memory info
            info = self.redis_client.info('memory')
            stats['redis_memory_usage'] = info.get('used_memory', 0)
            
        except Exception as e:
            logger.error(f"Failed to get rate limit stats: {e}")
        
        return stats

# Flask integration
class FlaskRateLimiter:
    """Flask integration for rate limiting"""
    
    def __init__(self, app: Flask = None, rate_limiter: RateLimiter = None):
        self.rate_limiter = rate_limiter or RateLimiter()
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize Flask app with rate limiting"""
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        
        # Add rate limit status endpoint
        @app.route('/admin/rate-limits/status')
        def rate_limit_status():
            return jsonify(self.rate_limiter.get_global_stats())
    
    def _get_request_context(self):
        """Extract request context for rate limiting"""
        return {
            'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
            'user_id': getattr(g, 'user_id', None),
            'api_key': request.headers.get('X-API-Key'),
            'endpoint': request.endpoint,
            'method': request.method,
            'path': request.path
        }
    
    def _before_request(self):
        """Check rate limits before processing request"""
        # Skip rate limiting for static files and health checks
        if request.endpoint in ['static', 'health']:
            return
        
        # Determine which rule to apply
        rule_key = self._get_rule_for_endpoint(request.endpoint, request.path)
        
        if rule_key:
            context = self._get_request_context()
            is_allowed, rate_info = self.rate_limiter.check_rate_limit(rule_key, context)
            
            if not is_allowed:
                # Add rate limit headers
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'message': rate_info.get('reason', 'Too many requests'),
                    'rate_limit_info': rate_info
                })
                response.status_code = 429
                
                # Add standard rate limit headers
                if 'reset_time' in rate_info:
                    response.headers['X-RateLimit-Reset'] = str(rate_info['reset_time'])
                if 'requests_remaining' in rate_info:
                    response.headers['X-RateLimit-Remaining'] = str(rate_info['requests_remaining'])
                if 'requests_limit' in rate_info:
                    response.headers['X-RateLimit-Limit'] = str(rate_info['requests_limit'])
                
                return response
            
            # Store rate limit info for after_request
            g.rate_limit_info = rate_info
    
    def _after_request(self, response):
        """Add rate limit headers to response"""
        if hasattr(g, 'rate_limit_info'):
            info = g.rate_limit_info
            
            if 'requests_remaining' in info:
                response.headers['X-RateLimit-Remaining'] = str(info['requests_remaining'])
            if 'requests_limit' in info:
                response.headers['X-RateLimit-Limit'] = str(info['requests_limit'])
            if 'reset_time' in info:
                response.headers['X-RateLimit-Reset'] = str(info['reset_time'])
        
        return response
    
    def _get_rule_for_endpoint(self, endpoint: str, path: str) -> Optional[str]:
        """Determine which rate limiting rule to apply"""
        # Authentication endpoints
        if endpoint in ['login', 'register', 'reset_password'] or '/auth/' in path:
            return 'auth_endpoints'
        
        # File upload endpoints
        if endpoint in ['upload', 'parse'] or '/upload' in path or '/parse' in path:
            return 'file_upload'
        
        # AI processing endpoints
        if '/ai/' in path or '/clean_with_ai' in path or endpoint in ['ai_process', 'clean_with_ai']:
            return 'ai_processing'
        
        # Admin endpoints
        if '/admin/' in path or endpoint.startswith('admin_'):
            return 'admin_endpoints'
        
        # Database write operations
        if endpoint in ['push', 'evaluate_and_push'] or path.endswith('/push'):
            return 'db_writes'
        
        # Default to general API rate limit
        return 'api_general'

# Decorator for function-level rate limiting
def rate_limit(rule_key: str, rate_limiter: RateLimiter = None):
    """Decorator for rate limiting specific functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = rate_limiter or RateLimiter()
            
            # Extract context (works for Flask requests)
            context = {}
            try:
                context = {
                    'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
                    'user_id': getattr(g, 'user_id', None),
                    'endpoint': func.__name__
                }
            except:
                # Fallback for non-Flask contexts
                context = {'ip': 'unknown', 'endpoint': func.__name__}
            
            is_allowed, rate_info = limiter.check_rate_limit(rule_key, context)
            
            if not is_allowed:
                if 'flask' in str(type(func)):
                    # Return Flask response
                    response = jsonify({
                        'error': 'Rate limit exceeded',
                        'rate_limit_info': rate_info
                    })
                    response.status_code = 429
                    return response
                else:
                    # Raise exception for other contexts
                    raise Exception(f"Rate limit exceeded: {rate_info.get('reason')}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test the rate limiter
    limiter = RateLimiter()
    
    # Test context
    test_context = {
        'ip': '192.168.1.100',
        'user_id': 'test_user',
        'endpoint': '/api/test'
    }
    
    # Test rate limiting
    for i in range(15):
        allowed, info = limiter.check_rate_limit('auth_endpoints', test_context)
        print(f"Request {i+1}: {'✅ Allowed' if allowed else '❌ Blocked'} - {info}")
        
        if not allowed:
            break
        
        time.sleep(0.1)
    
    # Print stats
    stats = limiter.get_global_stats()
    print(f"\nRate Limiter Stats: {json.dumps(stats, indent=2)}") 