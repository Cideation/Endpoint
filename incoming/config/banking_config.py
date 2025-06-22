"""
Banking API Configuration
Secure configuration management for banking operations
"""

import os
from typing import Dict, Any

class BankingConfig:
    """Banking API configuration management"""
    
    # API Endpoints
    BANKING_API_BASE_URL = os.getenv('BANKING_API_BASE_URL', 'https://api.bank.example.com')
    PAYMENT_GATEWAY_URL = os.getenv('PAYMENT_GATEWAY_URL', 'https://payments.example.com')
    
    # Authentication
    API_KEY = os.getenv('BANKING_API_KEY', 'your-api-key-here')
    API_SECRET = os.getenv('BANKING_API_SECRET', 'your-api-secret-here')
    CLIENT_ID = os.getenv('BANKING_CLIENT_ID', 'your-client-id')
    
    # Security
    ENCRYPTION_KEY = os.getenv('BANKING_ENCRYPTION_KEY', 'your-encryption-key')
    SSL_VERIFY = os.getenv('BANKING_SSL_VERIFY', 'true').lower() == 'true'
    
    # Transaction Limits
    MAX_TRANSACTION_AMOUNT = float(os.getenv('MAX_TRANSACTION_AMOUNT', '10000.00'))
    MIN_TRANSACTION_AMOUNT = float(os.getenv('MIN_TRANSACTION_AMOUNT', '0.01'))
    DAILY_LIMIT = float(os.getenv('DAILY_TRANSACTION_LIMIT', '50000.00'))
    
    # Timeouts
    API_TIMEOUT = int(os.getenv('BANKING_API_TIMEOUT', '30'))
    CONNECTION_TIMEOUT = int(os.getenv('BANKING_CONNECTION_TIMEOUT', '10'))
    
    # Retry Configuration
    MAX_RETRIES = int(os.getenv('BANKING_MAX_RETRIES', '3'))
    RETRY_DELAY = int(os.getenv('BANKING_RETRY_DELAY', '5'))
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get complete banking configuration"""
        return {
            'api': {
                'base_url': cls.BANKING_API_BASE_URL,
                'payment_gateway': cls.PAYMENT_GATEWAY_URL,
                'timeout': cls.API_TIMEOUT,
                'connection_timeout': cls.CONNECTION_TIMEOUT,
                'ssl_verify': cls.SSL_VERIFY
            },
            'auth': {
                'api_key': cls.API_KEY,
                'api_secret': cls.API_SECRET,
                'client_id': cls.CLIENT_ID
            },
            'limits': {
                'max_amount': cls.MAX_TRANSACTION_AMOUNT,
                'min_amount': cls.MIN_TRANSACTION_AMOUNT,
                'daily_limit': cls.DAILY_LIMIT
            },
            'retry': {
                'max_retries': cls.MAX_RETRIES,
                'delay': cls.RETRY_DELAY
            },
            'security': {
                'encryption_key': cls.ENCRYPTION_KEY
            }
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate banking configuration"""
        required_vars = [
            'BANKING_API_BASE_URL',
            'BANKING_API_KEY',
            'BANKING_API_SECRET',
            'BANKING_CLIENT_ID'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True
