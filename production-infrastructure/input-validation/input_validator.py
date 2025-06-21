#!/usr/bin/env python3
"""
Production Input Validation & Sanitization System
Provides comprehensive input validation, sanitization, and security protection
"""

import re
import json
import html
import logging
import bleach
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from functools import wraps
from datetime import datetime
import uuid
from urllib.parse import urlparse
from flask import request, jsonify

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Input validation rule configuration"""
    field_name: str
    field_type: str  # string, integer, float, email, url, uuid, json, file
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    sanitize: bool = True
    allow_html: bool = False

class InputValidator:
    """Production-grade input validation and sanitization"""
    
    def __init__(self):
        self.validation_schemas = {}
        self.sanitization_config = self._get_sanitization_config()
        self.security_patterns = self._load_security_patterns()
        
    def _get_sanitization_config(self) -> Dict:
        """Configure sanitization settings"""
        return {
            'allowed_html_tags': [
                'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
            ],
            'allowed_html_attributes': {
                '*': ['class', 'id'],
                'a': ['href', 'title'],
                'img': ['src', 'alt', 'width', 'height']
            },
            'strip_comments': True,
            'strip_scripts': True
        }
    
    def _load_security_patterns(self) -> Dict:
        """Load security threat detection patterns"""
        return {
            'sql_injection': [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
                r"(['\"]\s*(OR|AND)\s*['\"]\s*=\s*['\"])",
                r"(--|\#|/\*|\*/)"
            ],
            'xss_patterns': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"vbscript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"onclick\s*=",
                r"eval\s*\(",
                r"expression\s*\("
            ],
            'command_injection': [
                r"(\||&|;|\$\(|\`)",
                r"(rm\s+|del\s+|format\s+)",
                r"(\.\.\/|\.\.\\)",
                r"(/etc/passwd|/etc/shadow)"
            ],
            'path_traversal': [
                r"\.\.\/",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ],
            'ldap_injection': [
                r"(\*|\(|\)|\||&)",
                r"(objectClass=|cn=|ou=|dc=)"
            ]
        }
    
    def register_schema(self, schema_name: str, rules: List[ValidationRule]):
        """Register validation schema"""
        self.validation_schemas[schema_name] = {rule.field_name: rule for rule in rules}
        logger.info(f"Registered validation schema: {schema_name}")
    
    def validate_input(self, data: Dict, schema_name: str) -> Dict:
        """
        Validate input data against schema
        Returns: {'valid': bool, 'errors': list, 'sanitized_data': dict}
        """
        if schema_name not in self.validation_schemas:
            return {
                'valid': False,
                'errors': [f"Unknown validation schema: {schema_name}"],
                'sanitized_data': {}
            }
        
        schema = self.validation_schemas[schema_name]
        errors = []
        sanitized_data = {}
        
        # Check required fields
        for field_name, rule in schema.items():
            if rule.required and field_name not in data:
                errors.append(f"Required field missing: {field_name}")
        
        # Validate each field in data
        for field_name, value in data.items():
            if field_name in schema:
                rule = schema[field_name]
                field_errors, sanitized_value = self._validate_field(field_name, value, rule)
                
                if field_errors:
                    errors.extend(field_errors)
                else:
                    sanitized_data[field_name] = sanitized_value
            else:
                # Unknown field - sanitize basic strings only
                if isinstance(value, str):
                    sanitized_data[field_name] = self._sanitize_string(value, allow_html=False)
                else:
                    sanitized_data[field_name] = value
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'sanitized_data': sanitized_data
        }
    
    def _validate_field(self, field_name: str, value: Any, rule: ValidationRule) -> tuple:
        """Validate individual field"""
        errors = []
        
        # Type validation
        if not self._validate_type(value, rule.field_type):
            errors.append(f"{field_name}: Invalid type, expected {rule.field_type}")
            return errors, value
        
        # Convert value if needed
        converted_value = self._convert_type(value, rule.field_type)
        
        # Length validation (for strings)
        if rule.field_type == 'string' and isinstance(converted_value, str):
            if rule.min_length is not None and len(converted_value) < rule.min_length:
                errors.append(f"{field_name}: Minimum length {rule.min_length}, got {len(converted_value)}")
            
            if rule.max_length is not None and len(converted_value) > rule.max_length:
                errors.append(f"{field_name}: Maximum length {rule.max_length}, got {len(converted_value)}")
        
        # Value range validation
        if rule.field_type in ['integer', 'float']:
            if rule.min_value is not None and converted_value < rule.min_value:
                errors.append(f"{field_name}: Minimum value {rule.min_value}, got {converted_value}")
            
            if rule.max_value is not None and converted_value > rule.max_value:
                errors.append(f"{field_name}: Maximum value {rule.max_value}, got {converted_value}")
        
        # Pattern validation
        if rule.pattern and isinstance(converted_value, str):
            if not re.match(rule.pattern, converted_value):
                errors.append(f"{field_name}: Does not match required pattern")
        
        # Allowed values validation
        if rule.allowed_values and converted_value not in rule.allowed_values:
            errors.append(f"{field_name}: Value not in allowed list: {rule.allowed_values}")
        
        # Custom validator
        if rule.custom_validator:
            try:
                if not rule.custom_validator(converted_value):
                    errors.append(f"{field_name}: Failed custom validation")
            except Exception as e:
                errors.append(f"{field_name}: Custom validation error: {str(e)}")
        
        # Security validation
        security_errors = self._check_security_threats(field_name, converted_value)
        errors.extend(security_errors)
        
        # Sanitization
        if rule.sanitize and not errors:
            sanitized_value = self._sanitize_value(converted_value, rule)
        else:
            sanitized_value = converted_value
        
        return errors, sanitized_value
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'integer':
            return isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        elif expected_type == 'float':
            return isinstance(value, (int, float)) or (isinstance(value, str) and self._is_float(value))
        elif expected_type == 'email':
            return isinstance(value, str) and self._is_valid_email(value)
        elif expected_type == 'url':
            return isinstance(value, str) and self._is_valid_url(value)
        elif expected_type == 'uuid':
            return isinstance(value, str) and self._is_valid_uuid(value)
        elif expected_type == 'json':
            return isinstance(value, (dict, list)) or (isinstance(value, str) and self._is_valid_json(value))
        elif expected_type == 'file':
            return hasattr(value, 'filename') or isinstance(value, str)
        else:
            return True
    
    def _convert_type(self, value: Any, expected_type: str) -> Any:
        """Convert value to expected type"""
        try:
            if expected_type == 'integer' and isinstance(value, str):
                return int(value)
            elif expected_type == 'float' and isinstance(value, str):
                return float(value)
            elif expected_type == 'json' and isinstance(value, str):
                return json.loads(value)
            else:
                return value
        except (ValueError, json.JSONDecodeError):
            return value
    
    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_valid_uuid(self, uuid_str: str) -> bool:
        """Validate UUID format"""
        try:
            uuid.UUID(uuid_str)
            return True
        except ValueError:
            return False
    
    def _is_valid_json(self, json_str: str) -> bool:
        """Validate JSON format"""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    def _check_security_threats(self, field_name: str, value: Any) -> List[str]:
        """Check for security threats in input"""
        if not isinstance(value, str):
            return []
        
        errors = []
        value_lower = value.lower()
        
        # Check each security pattern category
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value_lower, re.IGNORECASE):
                    errors.append(f"{field_name}: Potential {threat_type.replace('_', ' ')} detected")
                    logger.warning(f"Security threat detected in {field_name}: {threat_type}")
                    break  # One detection per threat type is enough
        
        return errors
    
    def _sanitize_value(self, value: Any, rule: ValidationRule) -> Any:
        """Sanitize value based on rule configuration"""
        if isinstance(value, str):
            return self._sanitize_string(value, rule.allow_html)
        elif isinstance(value, dict):
            return {k: self._sanitize_string(str(v), False) if isinstance(v, str) else v 
                   for k, v in value.items()}
        elif isinstance(value, list):
            return [self._sanitize_string(str(item), False) if isinstance(item, str) else item 
                   for item in value]
        else:
            return value
    
    def _sanitize_string(self, text: str, allow_html: bool = False) -> str:
        """Sanitize string input"""
        if not text:
            return text
        
        # HTML escape if HTML not allowed
        if not allow_html:
            text = html.escape(text)
        else:
            # Use bleach for HTML sanitization
            text = bleach.clean(
                text,
                tags=self.sanitization_config['allowed_html_tags'],
                attributes=self.sanitization_config['allowed_html_attributes'],
                strip=True,
                strip_comments=self.sanitization_config['strip_comments']
            )
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '')
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_flask_validator(self, schema_name: str):
        """Create Flask decorator for request validation"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get request data
                if request.is_json:
                    data = request.get_json() or {}
                else:
                    data = request.form.to_dict()
                    # Add file data if present
                    for file_key in request.files:
                        data[file_key] = request.files[file_key]
                
                # Validate input
                validation_result = self.validate_input(data, schema_name)
                
                if not validation_result['valid']:
                    return jsonify({
                        'error': 'Input validation failed',
                        'validation_errors': validation_result['errors']
                    }), 400
                
                # Replace request data with sanitized data
                request.validated_data = validation_result['sanitized_data']
                
                return func(*args, **kwargs)
            return wrapper
        return decorator

# Pre-defined validation schemas
def setup_default_schemas(validator: InputValidator):
    """Setup default validation schemas for common endpoints"""
    
    # User registration schema
    validator.register_schema('user_registration', [
        ValidationRule('email', 'email', required=True, max_length=255),
        ValidationRule('password', 'string', required=True, min_length=8, max_length=128),
        ValidationRule('name', 'string', required=True, min_length=2, max_length=100),
        ValidationRule('phone', 'string', required=False, pattern=r'^\+?[1-9]\d{1,14}$'),
    ])
    
    # File upload schema
    validator.register_schema('file_upload', [
        ValidationRule('file', 'file', required=True),
        ValidationRule('description', 'string', required=False, max_length=500),
        ValidationRule('tags', 'json', required=False),
        ValidationRule('public', 'string', required=False, allowed_values=['true', 'false']),
    ])
    
    # CAD parsing schema
    validator.register_schema('cad_parsing', [
        ValidationRule('component_name', 'string', required=True, min_length=1, max_length=200),
        ValidationRule('format', 'string', required=True, allowed_values=['dwg', 'dxf', 'ifc', 'pdf']),
        ValidationRule('parse_options', 'json', required=False),
        ValidationRule('ai_cleaning', 'string', required=False, allowed_values=['true', 'false']),
    ])
    
    # Database operations schema
    validator.register_schema('database_operation', [
        ValidationRule('component_id', 'string', required=True, pattern=r'^[a-zA-Z0-9_-]+$'),
        ValidationRule('operation', 'string', required=True, allowed_values=['insert', 'update', 'delete']),
        ValidationRule('data', 'json', required=True),
        ValidationRule('validate_schema', 'string', required=False, allowed_values=['true', 'false']),
    ])
    
    # Agent coefficient schema
    validator.register_schema('agent_coefficients', [
        ValidationRule('agent_type', 'string', required=True, max_length=50),
        ValidationRule('coefficients', 'json', required=True),
        ValidationRule('priority', 'integer', required=False, min_value=1, max_value=10),
        ValidationRule('expiry', 'string', required=False),  # ISO datetime
    ])
    
    # WebSocket message schema
    validator.register_schema('websocket_message', [
        ValidationRule('message_type', 'string', required=True, max_length=50),
        ValidationRule('data', 'json', required=True),
        ValidationRule('target', 'string', required=False, max_length=100),
        ValidationRule('priority', 'integer', required=False, min_value=1, max_value=5),
    ])

# Custom validators
def validate_component_id(component_id: str) -> bool:
    """Custom validator for component IDs"""
    # Must start with letter, contain only alphanumeric and underscore
    return re.match(r'^[A-Za-z][A-Za-z0-9_]*$', component_id) is not None

def validate_coefficient_data(data: dict) -> bool:
    """Custom validator for coefficient data"""
    if not isinstance(data, dict):
        return False
    
    # Check for required coefficient fields
    required_fields = ['budget', 'quality', 'timeline']
    return all(field in data for field in required_fields)

# Middleware integration
class ValidationMiddleware:
    """Middleware for automatic input validation"""
    
    def __init__(self, app, validator: InputValidator):
        self.app = app
        self.validator = validator
        self.endpoint_schemas = {}
    
    def register_endpoint_schema(self, endpoint: str, schema_name: str):
        """Register validation schema for specific endpoint"""
        self.endpoint_schemas[endpoint] = schema_name
    
    def __call__(self, environ, start_response):
        """WSGI middleware implementation"""
        # This would integrate with WSGI applications
        return self.app(environ, start_response)

if __name__ == "__main__":
    # Test the input validator
    validator = InputValidator()
    setup_default_schemas(validator)
    
    # Test data
    test_data = {
        'email': 'test@example.com',
        'password': 'securepassword123',
        'name': 'John Doe',
        'phone': '+1234567890'
    }
    
    # Test malicious data
    malicious_data = {
        'email': 'test@example.com',
        'password': 'password',
        'name': '<script>alert("xss")</script>John',
        'comment': "'; DROP TABLE users; --"
    }
    
    # Validate inputs
    result1 = validator.validate_input(test_data, 'user_registration')
    result2 = validator.validate_input(malicious_data, 'user_registration')
    
    print("Valid data result:", result1)
    print("\nMalicious data result:", result2) 