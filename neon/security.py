"""
BEM System Security Module
Implements authentication, authorization, rate limiting, and security monitoring
"""

import os
import jwt
import time
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from functools import wraps
from dataclasses import dataclass
from collections import defaultdict
import redis
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
SECURITY_CONFIG = {
    'JWT_SECRET': os.getenv('JWT_SECRET', secrets.token_hex(32)),
    'JWT_ALGORITHM': 'HS256',
    'JWT_EXPIRATION_HOURS': 24,
    'RATE_LIMIT_WINDOW': 60,  # 1 minute
    'RATE_LIMIT_MAX_REQUESTS': 60,  # 60 requests per minute
    'PASSWORD_MIN_LENGTH': 12,
    'PASSWORD_REQUIRE_SPECIAL': True,
    'SESSION_TIMEOUT': 3600,  # 1 hour
    'MAX_FAILED_LOGINS': 5,
    'LOCKOUT_DURATION': 900,  # 15 minutes
    'API_KEY_LENGTH': 32,
}

@dataclass
class UserRole:
    """User role with associated permissions"""
    name: str
    permissions: List[str]

# Define system roles
ROLES = {
    'admin': UserRole('admin', ['read', 'write', 'delete', 'manage_users']),
    'operator': UserRole('operator', ['read', 'write']),
    'viewer': UserRole('viewer', ['read']),
}

class SecurityManager:
    """Manages authentication, authorization, and security monitoring"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize security manager with Redis for rate limiting and session management"""
        self.redis_client = redis.from_url(redis_url)
        self.security = HTTPBearer()
        self.failed_logins = defaultdict(int)
        self.lockout_until = defaultdict(float)
        
        # Initialize encryption for sensitive data
        self.fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(SECURITY_CONFIG['JWT_SECRET'].encode()).digest()))
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(SECURITY_CONFIG['API_KEY_LENGTH'])
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash password using PBKDF2 with SHA256"""
        if not salt:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        password_hash = kdf.derive(password.encode())
        return password_hash, salt
    
    def verify_password(self, password: str, stored_hash: bytes, salt: bytes) -> bool:
        """Verify password against stored hash"""
        password_hash, _ = self.hash_password(password, salt)
        return secrets.compare_digest(password_hash, stored_hash)
    
    def create_jwt_token(self, user_id: str, role: str) -> str:
        """Create JWT token for authenticated user"""
        expiration = datetime.utcnow() + timedelta(hours=SECURITY_CONFIG['JWT_EXPIRATION_HOURS'])
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': expiration
        }
        return jwt.encode(payload, SECURITY_CONFIG['JWT_SECRET'], algorithm=SECURITY_CONFIG['JWT_ALGORITHM'])
    
    def verify_jwt_token(self, token: str) -> Dict:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, SECURITY_CONFIG['JWT_SECRET'], algorithms=[SECURITY_CONFIG['JWT_ALGORITHM']])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> Dict:
        """FastAPI dependency for getting current user from token"""
        token = credentials.credentials
        return self.verify_jwt_token(token)
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        current = int(time.time())
        key = f"rate_limit:{user_id}:{current // SECURITY_CONFIG['RATE_LIMIT_WINDOW']}"
        
        pipe = self.redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, SECURITY_CONFIG['RATE_LIMIT_WINDOW'])
        result = pipe.execute()
        
        request_count = result[0]
        return request_count <= SECURITY_CONFIG['RATE_LIMIT_MAX_REQUESTS']
    
    def require_permissions(self, required_permissions: List[str]):
        """Decorator to check if user has required permissions"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = kwargs.get('current_user')
                if not user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                user_role = ROLES.get(user['role'])
                if not user_role:
                    raise HTTPException(status_code=403, detail="Invalid role")
                
                if not all(perm in user_role.permissions for perm in required_permissions):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                if not self.check_rate_limit(user['user_id']):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using Fernet symmetric encryption"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data using Fernet symmetric encryption"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements"""
        if len(password) < SECURITY_CONFIG['PASSWORD_MIN_LENGTH']:
            return False
        
        if SECURITY_CONFIG['PASSWORD_REQUIRE_SPECIAL']:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False
            
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        return has_upper and has_lower and has_digit
    
    def check_failed_logins(self, user_id: str) -> bool:
        """Check if user is locked out due to failed login attempts"""
        if time.time() < self.lockout_until[user_id]:
            return False
        
        if self.failed_logins[user_id] >= SECURITY_CONFIG['MAX_FAILED_LOGINS']:
            self.lockout_until[user_id] = time.time() + SECURITY_CONFIG['LOCKOUT_DURATION']
            return False
        
        return True
    
    def record_failed_login(self, user_id: str):
        """Record failed login attempt"""
        self.failed_logins[user_id] += 1
    
    def reset_failed_logins(self, user_id: str):
        """Reset failed login attempts after successful login"""
        self.failed_logins[user_id] = 0
        self.lockout_until[user_id] = 0
    
    def create_audit_log(self, user_id: str, action: str, resource: str, status: str, details: Optional[str] = None):
        """Create security audit log entry"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'status': status,
            'details': details
        }
        
        logger.info(f"AUDIT: {log_entry}")
        # In a production environment, you would want to store this in a secure audit log system
        
    def scan_for_threats(self, request_data: Union[str, Dict]) -> bool:
        """Basic threat detection for incoming requests"""
        threat_patterns = [
            "../../",  # Path traversal
            "SELECT",  # SQL injection
            "<script",  # XSS
            "eval(",   # Code injection
        ]
        
        data_str = str(request_data).lower()
        return any(pattern.lower() in data_str for pattern in threat_patterns)

# Example usage in FastAPI endpoints:
"""
from fastapi import FastAPI, Depends
from neon.security import SecurityManager, ROLES

app = FastAPI()
security_manager = SecurityManager()

@app.post("/api/secure_endpoint")
@security_manager.require_permissions(['write'])
async def secure_endpoint(
    data: Dict,
    current_user: Dict = Depends(security_manager.get_current_user)
):
    # Check for threats
    if security_manager.scan_for_threats(data):
        security_manager.create_audit_log(
            current_user['user_id'],
            'threat_detected',
            '/api/secure_endpoint',
            'blocked',
            str(data)
        )
        raise HTTPException(status_code=400, detail="Potential security threat detected")
    
    # Process request
    result = process_data(data)
    
    # Audit log success
    security_manager.create_audit_log(
        current_user['user_id'],
        'data_process',
        '/api/secure_endpoint',
        'success'
    )
    
    return result
""" 