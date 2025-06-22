"""
NEOBANK Adapter - Interface to NEOBANK SPV management system
Handles communication with NEOBANK for SPV operations and title releases
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import hmac

class NeobankAdapter:
    """
    Adapter for NEOBANK SPV management system
    
    Handles:
    - SPV registration and management
    - Title release triggers
    - SPV status monitoring
    - Secure API communication
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'https://api.neobank.com')
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.timeout = config.get('timeout', 30)
        
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key or not self.api_secret:
            raise ValueError("NEOBANK API credentials are required")
        
        self.logger.info("NEOBANK Adapter initialized")
    
    def register_spv(self, spv_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new SPV with NEOBANK
        
        Args:
            spv_data: SPV registration data including property details
            
        Returns:
            Registration result with SPV ID
        """
        try:
            endpoint = f"{self.base_url}/api/v1/spv/register"
            
            payload = {
                'property_id': spv_data['property_id'],
                'property_address': spv_data['property_address'],
                'property_value': spv_data['property_value'],
                'target_amount': spv_data['target_amount'],
                'title_document_hash': spv_data['title_document_hash'],
                'registration_timestamp': datetime.now().isoformat()
            }
            
            headers = self._get_auth_headers(payload)
            
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 201:
                result = response.json()
                self.logger.info(f"SPV registered successfully: {result.get('spv_id')}")
                return {
                    'success': True,
                    'spv_id': result['spv_id'],
                    'status': result['status'],
                    'registration_date': result['registration_date']
                }
            else:
                self.logger.error(f"SPV registration failed: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"Registration failed: {response.status_code}",
                    'details': response.text
                }
                
        except Exception as e:
            self.logger.error(f"Error registering SPV: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def trigger_title_release(self, spv_id: str, release_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger title release for an SPV
        
        Args:
            spv_id: SPV identifier
            release_data: Release trigger data and verification
            
        Returns:
            Release trigger result
        """
        try:
            endpoint = f"{self.base_url}/api/v1/spv/{spv_id}/release"
            
            payload = {
                'spv_id': spv_id,
                'trigger_reason': release_data.get('trigger_reason', 'conditions_met'),
                'total_contributions': release_data['total_contributions'],
                'participant_count': release_data['participant_count'],
                'verification_hash': release_data['verification_hash'],
                'release_timestamp': datetime.now().isoformat(),
                'authorized_by': release_data.get('authorized_by', 'system')
            }
            
            headers = self._get_auth_headers(payload)
            
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Title release triggered for SPV {spv_id}")
                return {
                    'success': True,
                    'release_id': result['release_id'],
                    'status': result['status'],
                    'estimated_completion': result.get('estimated_completion'),
                    'tracking_number': result.get('tracking_number')
                }
            else:
                self.logger.error(f"Title release failed: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"Release failed: {response.status_code}",
                    'details': response.text
                }
                
        except Exception as e:
            self.logger.error(f"Error triggering title release for SPV {spv_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_spv_status(self, spv_id: str) -> Dict[str, Any]:
        """
        Get current status of an SPV
        
        Args:
            spv_id: SPV identifier
            
        Returns:
            SPV status information
        """
        try:
            endpoint = f"{self.base_url}/api/v1/spv/{spv_id}/status"
            headers = self._get_auth_headers()
            
            response = requests.get(
                endpoint,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'spv_id': spv_id,
                    'status': result['status'],
                    'property_id': result['property_id'],
                    'title_status': result['title_status'],
                    'last_updated': result['last_updated'],
                    'release_progress': result.get('release_progress', 0)
                }
            else:
                return {
                    'success': False,
                    'error': f"Status check failed: {response.status_code}",
                    'details': response.text
                }
                
        except Exception as e:
            self.logger.error(f"Error getting SPV status {spv_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_spvs(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        List SPVs with optional filters
        
        Args:
            filters: Optional filters for SPV listing
            
        Returns:
            List of SPVs
        """
        try:
            endpoint = f"{self.base_url}/api/v1/spv/list"
            headers = self._get_auth_headers()
            
            params = filters or {}
            
            response = requests.get(
                endpoint,
                headers=headers,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'spvs': result['spvs'],
                    'total_count': result['total_count'],
                    'page': result.get('page', 1)
                }
            else:
                return {
                    'success': False,
                    'error': f"List SPVs failed: {response.status_code}",
                    'details': response.text
                }
                
        except Exception as e:
            self.logger.error(f"Error listing SPVs: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_spv_contribution(self, spv_id: str, contribution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update SPV with new contribution data
        
        Args:
            spv_id: SPV identifier
            contribution_data: Contribution update data
            
        Returns:
            Update result
        """
        try:
            endpoint = f"{self.base_url}/api/v1/spv/{spv_id}/contribution"
            
            payload = {
                'contribution_id': contribution_data['contribution_id'],
                'amount': contribution_data['amount'],
                'agent_id': contribution_data['agent_id'],
                'transaction_ref': contribution_data['transaction_ref'],
                'timestamp': contribution_data['timestamp']
            }
            
            headers = self._get_auth_headers(payload)
            
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'spv_id': spv_id,
                    'updated_total': result['updated_total'],
                    'progress_percentage': result['progress_percentage']
                }
            else:
                return {
                    'success': False,
                    'error': f"Contribution update failed: {response.status_code}",
                    'details': response.text
                }
                
        except Exception as e:
            self.logger.error(f"Error updating SPV contribution {spv_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_title_ownership(self, property_id: str, title_hash: str) -> Dict[str, Any]:
        """
        Verify title ownership with NEOBANK
        
        Args:
            property_id: Property identifier
            title_hash: Hash of title document
            
        Returns:
            Verification result
        """
        try:
            endpoint = f"{self.base_url}/api/v1/title/verify"
            
            payload = {
                'property_id': property_id,
                'title_hash': title_hash,
                'verification_timestamp': datetime.now().isoformat()
            }
            
            headers = self._get_auth_headers(payload)
            
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'verified': result['verified'],
                    'owner_info': result.get('owner_info'),
                    'title_status': result['title_status']
                }
            else:
                return {
                    'success': False,
                    'error': f"Title verification failed: {response.status_code}",
                    'details': response.text
                }
                
        except Exception as e:
            self.logger.error(f"Error verifying title ownership: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_auth_headers(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate authentication headers for NEOBANK API"""
        timestamp = str(int(datetime.now().timestamp()))
        
        # Create signature
        if payload:
            message = f"{timestamp}{json.dumps(payload, sort_keys=True)}"
        else:
            message = timestamp
            
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key,
            'X-Timestamp': timestamp,
            'X-Signature': signature,
            'User-Agent': 'Paluwagan-Engine/1.0'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check NEOBANK API health"""
        try:
            endpoint = f"{self.base_url}/api/v1/health"
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'success': False,
                    'status': 'unhealthy',
                    'error': f"Health check failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': 'unreachable',
                'error': str(e)
            }
