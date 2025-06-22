"""
SPV Manager - Manages Special Purpose Vehicles for home title custody
Handles SPV lifecycle, contribution tracking, and release conditions
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class SPVStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FUNDING = "funding"
    READY_FOR_RELEASE = "ready_for_release"
    RELEASED = "released"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

@dataclass
class PropertyInfo:
    """Property information held by SPV"""
    property_id: str
    address: str
    city: str
    province: str
    postal_code: str
    property_type: str
    estimated_value: float
    title_number: str
    title_hash: str
    legal_description: str

@dataclass
class SPVConfig:
    """SPV configuration and trigger conditions"""
    target_amount: float
    minimum_contribution: float
    maximum_contribution: Optional[float]
    required_participants: int
    funding_deadline: Optional[datetime]
    auto_release: bool
    manual_approval_required: bool
    contribution_currency: str

@dataclass
class SPV:
    """Special Purpose Vehicle for home title custody"""
    spv_id: str
    property_info: PropertyInfo
    config: SPVConfig
    status: SPVStatus
    created_at: datetime
    updated_at: datetime
    current_amount: float
    participant_count: int
    contributions: List[str]  # List of contribution IDs
    neobank_reference: Optional[str]
    release_conditions_met: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SPV to dictionary for JSON serialization"""
        return {
            'spv_id': self.spv_id,
            'property_info': asdict(self.property_info),
            'config': {
                **asdict(self.config),
                'funding_deadline': self.config.funding_deadline.isoformat() if self.config.funding_deadline else None
            },
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'current_amount': self.current_amount,
            'participant_count': self.participant_count,
            'contributions': self.contributions,
            'neobank_reference': self.neobank_reference,
            'release_conditions_met': self.release_conditions_met
        }

class SPVManager:
    """
    Manages SPVs for the Paluwagan system
    
    Responsibilities:
    1. Create and configure SPVs for properties
    2. Track contribution progress
    3. Evaluate release conditions
    4. Interface with NEOBANK for title custody
    5. Manage SPV lifecycle
    """
    
    def __init__(self, data_path: str = "incoming/neo_bank/data"):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
        
        # Load SPV data
        self.spvs = self._load_spvs()
        
        self.logger.info("SPV Manager initialized")
    
    def create_spv(self, property_info: PropertyInfo, config: SPVConfig) -> Dict[str, Any]:
        """
        Create a new SPV for a property
        
        Args:
            property_info: Property details
            config: SPV configuration
            
        Returns:
            SPV creation result
        """
        try:
            # Generate SPV ID
            spv_id = f"SPV-{uuid.uuid4().hex[:8].upper()}"
            
            # Create SPV instance
            spv = SPV(
                spv_id=spv_id,
                property_info=property_info,
                config=config,
                status=SPVStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                current_amount=0.0,
                participant_count=0,
                contributions=[],
                neobank_reference=None,
                release_conditions_met=False
            )
            
            # Validate SPV configuration
            validation_result = self._validate_spv_config(spv)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': 'SPV configuration validation failed',
                    'details': validation_result['errors']
                }
            
            # Store SPV
            self.spvs[spv_id] = spv
            self._save_spvs()
            
            self.logger.info(f"SPV created: {spv_id} for property {property_info.property_id}")
            
            return {
                'success': True,
                'spv_id': spv_id,
                'status': spv.status.value,
                'property_id': property_info.property_id,
                'target_amount': config.target_amount
            }
            
        except Exception as e:
            self.logger.error(f"Error creating SPV: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_contribution(self, spv_id: str, contribution_id: str, amount: float, agent_id: str) -> Dict[str, Any]:
        """
        Update SPV with new contribution
        
        Args:
            spv_id: SPV identifier
            contribution_id: Contribution identifier
            amount: Contribution amount
            agent_id: Contributing agent ID
            
        Returns:
            Update result
        """
        if spv_id not in self.spvs:
            return {
                'success': False,
                'error': f'SPV {spv_id} not found'
            }
        
        try:
            spv = self.spvs[spv_id]
            
            # Check if SPV can accept contributions
            if spv.status not in [SPVStatus.ACTIVE, SPVStatus.FUNDING]:
                return {
                    'success': False,
                    'error': f'SPV {spv_id} is not accepting contributions (status: {spv.status.value})'
                }
            
            # Validate contribution amount
            if amount < spv.config.minimum_contribution:
                return {
                    'success': False,
                    'error': f'Contribution amount {amount} below minimum {spv.config.minimum_contribution}'
                }
            
            if spv.config.maximum_contribution and amount > spv.config.maximum_contribution:
                return {
                    'success': False,
                    'error': f'Contribution amount {amount} exceeds maximum {spv.config.maximum_contribution}'
                }
            
            # Check funding deadline
            if spv.config.funding_deadline and datetime.now() > spv.config.funding_deadline:
                spv.status = SPVStatus.EXPIRED
                self._save_spvs()
                return {
                    'success': False,
                    'error': f'SPV {spv_id} funding deadline has passed'
                }
            
            # Update SPV with contribution
            spv.contributions.append(contribution_id)
            spv.current_amount += amount
            
            # Update participant count (unique agents)
            # This is simplified - in production, track unique participants properly
            spv.participant_count = len(set(spv.contributions))  # Simplified
            
            spv.updated_at = datetime.now()
            
            # Check if ready for funding status
            if spv.status == SPVStatus.ACTIVE:
                spv.status = SPVStatus.FUNDING
            
            # Check release conditions
            conditions_check = self._check_release_conditions(spv)
            spv.release_conditions_met = conditions_check['all_met']
            
            if spv.release_conditions_met:
                spv.status = SPVStatus.READY_FOR_RELEASE
            
            # Save updated SPV
            self._save_spvs()
            
            self.logger.info(f"SPV {spv_id} updated with contribution {contribution_id}: {amount}")
            
            return {
                'success': True,
                'spv_id': spv_id,
                'current_amount': spv.current_amount,
                'target_amount': spv.config.target_amount,
                'progress_percentage': (spv.current_amount / spv.config.target_amount) * 100,
                'participant_count': spv.participant_count,
                'status': spv.status.value,
                'release_conditions_met': spv.release_conditions_met,
                'conditions_check': conditions_check
            }
            
        except Exception as e:
            self.logger.error(f"Error updating SPV {spv_id} with contribution: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_release_conditions(self, spv_id: str) -> Dict[str, Any]:
        """
        Check if SPV meets release conditions
        
        Args:
            spv_id: SPV identifier
            
        Returns:
            Condition check results
        """
        if spv_id not in self.spvs:
            return {
                'success': False,
                'error': f'SPV {spv_id} not found'
            }
        
        spv = self.spvs[spv_id]
        return self._check_release_conditions(spv)
    
    def activate_spv(self, spv_id: str, neobank_reference: str) -> Dict[str, Any]:
        """
        Activate SPV after NEOBANK registration
        
        Args:
            spv_id: SPV identifier
            neobank_reference: NEOBANK reference ID
            
        Returns:
            Activation result
        """
        if spv_id not in self.spvs:
            return {
                'success': False,
                'error': f'SPV {spv_id} not found'
            }
        
        try:
            spv = self.spvs[spv_id]
            
            if spv.status != SPVStatus.PENDING:
                return {
                    'success': False,
                    'error': f'SPV {spv_id} cannot be activated (current status: {spv.status.value})'
                }
            
            spv.status = SPVStatus.ACTIVE
            spv.neobank_reference = neobank_reference
            spv.updated_at = datetime.now()
            
            self._save_spvs()
            
            self.logger.info(f"SPV {spv_id} activated with NEOBANK reference {neobank_reference}")
            
            return {
                'success': True,
                'spv_id': spv_id,
                'status': spv.status.value,
                'neobank_reference': neobank_reference
            }
            
        except Exception as e:
            self.logger.error(f"Error activating SPV {spv_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def release_spv(self, spv_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Mark SPV as released
        
        Args:
            spv_id: SPV identifier
            force: Force release even if conditions not met
            
        Returns:
            Release result
        """
        if spv_id not in self.spvs:
            return {
                'success': False,
                'error': f'SPV {spv_id} not found'
            }
        
        try:
            spv = self.spvs[spv_id]
            
            # Check conditions unless forced
            if not force:
                conditions_check = self._check_release_conditions(spv)
                if not conditions_check['all_met']:
                    return {
                        'success': False,
                        'error': 'Release conditions not met',
                        'conditions_check': conditions_check
                    }
            
            # Update SPV status
            spv.status = SPVStatus.RELEASED
            spv.updated_at = datetime.now()
            
            self._save_spvs()
            
            self.logger.info(f"SPV {spv_id} marked as released")
            
            return {
                'success': True,
                'spv_id': spv_id,
                'status': spv.status.value,
                'released_at': spv.updated_at.isoformat(),
                'final_amount': spv.current_amount,
                'participant_count': spv.participant_count
            }
            
        except Exception as e:
            self.logger.error(f"Error releasing SPV {spv_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_spv_details(self, spv_id: str) -> Dict[str, Any]:
        """Get detailed SPV information"""
        if spv_id not in self.spvs:
            return {
                'success': False,
                'error': f'SPV {spv_id} not found'
            }
        
        spv = self.spvs[spv_id]
        conditions_check = self._check_release_conditions(spv)
        
        return {
            'success': True,
            'spv': spv.to_dict(),
            'conditions_check': conditions_check,
            'progress_percentage': (spv.current_amount / spv.config.target_amount) * 100 if spv.config.target_amount > 0 else 0
        }
    
    def list_spvs(self, status_filter: Optional[SPVStatus] = None) -> Dict[str, Any]:
        """List all SPVs with optional status filter"""
        try:
            spv_list = []
            
            for spv_id, spv in self.spvs.items():
                if status_filter is None or spv.status == status_filter:
                    spv_summary = {
                        'spv_id': spv_id,
                        'property_id': spv.property_info.property_id,
                        'property_address': spv.property_info.address,
                        'status': spv.status.value,
                        'current_amount': spv.current_amount,
                        'target_amount': spv.config.target_amount,
                        'progress_percentage': (spv.current_amount / spv.config.target_amount) * 100 if spv.config.target_amount > 0 else 0,
                        'participant_count': spv.participant_count,
                        'created_at': spv.created_at.isoformat(),
                        'release_conditions_met': spv.release_conditions_met
                    }
                    spv_list.append(spv_summary)
            
            return {
                'success': True,
                'spvs': spv_list,
                'total_count': len(spv_list)
            }
            
        except Exception as e:
            self.logger.error(f"Error listing SPVs: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Private helper methods
    def _check_release_conditions(self, spv: SPV) -> Dict[str, Any]:
        """Check if SPV meets all release conditions"""
        conditions = []
        
        # Amount condition
        amount_met = spv.current_amount >= spv.config.target_amount
        conditions.append({
            'condition': 'target_amount',
            'met': amount_met,
            'current': spv.current_amount,
            'required': spv.config.target_amount,
            'progress': (spv.current_amount / spv.config.target_amount) * 100 if spv.config.target_amount > 0 else 0
        })
        
        # Participant condition
        participants_met = spv.participant_count >= spv.config.required_participants
        conditions.append({
            'condition': 'required_participants',
            'met': participants_met,
            'current': spv.participant_count,
            'required': spv.config.required_participants
        })
        
        # Deadline condition
        deadline_met = True
        if spv.config.funding_deadline:
            deadline_met = datetime.now() <= spv.config.funding_deadline
            conditions.append({
                'condition': 'funding_deadline',
                'met': deadline_met,
                'deadline': spv.config.funding_deadline.isoformat(),
                'time_remaining': str(spv.config.funding_deadline - datetime.now()) if deadline_met else "expired"
            })
        
        all_met = all(condition['met'] for condition in conditions)
        
        return {
            'all_met': all_met,
            'conditions': conditions,
            'auto_release': spv.config.auto_release,
            'manual_approval_required': spv.config.manual_approval_required
        }
    
    def _validate_spv_config(self, spv: SPV) -> Dict[str, Any]:
        """Validate SPV configuration"""
        errors = []
        
        # Validate amounts
        if spv.config.target_amount <= 0:
            errors.append("Target amount must be positive")
        
        if spv.config.minimum_contribution <= 0:
            errors.append("Minimum contribution must be positive")
        
        if spv.config.maximum_contribution and spv.config.maximum_contribution < spv.config.minimum_contribution:
            errors.append("Maximum contribution cannot be less than minimum contribution")
        
        # Validate participants
        if spv.config.required_participants <= 0:
            errors.append("Required participants must be positive")
        
        # Validate deadline
        if spv.config.funding_deadline and spv.config.funding_deadline <= datetime.now():
            errors.append("Funding deadline must be in the future")
        
        # Validate property info
        if not spv.property_info.property_id:
            errors.append("Property ID is required")
        
        if not spv.property_info.title_hash:
            errors.append("Title hash is required")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    # Data persistence methods
    def _load_spvs(self) -> Dict[str, SPV]:
        """Load SPVs from storage"""
        try:
            with open(f"{self.data_path}/spv_registry.json", 'r') as f:
                data = json.load(f)
                
            spvs = {}
            for spv_id, spv_data in data.items():
                # Reconstruct SPV objects from stored data
                property_info = PropertyInfo(**spv_data['property_info'])
                
                config_data = spv_data['config'].copy()
                if config_data['funding_deadline']:
                    config_data['funding_deadline'] = datetime.fromisoformat(config_data['funding_deadline'])
                config = SPVConfig(**config_data)
                
                spv = SPV(
                    spv_id=spv_data['spv_id'],
                    property_info=property_info,
                    config=config,
                    status=SPVStatus(spv_data['status']),
                    created_at=datetime.fromisoformat(spv_data['created_at']),
                    updated_at=datetime.fromisoformat(spv_data['updated_at']),
                    current_amount=spv_data['current_amount'],
                    participant_count=spv_data['participant_count'],
                    contributions=spv_data['contributions'],
                    neobank_reference=spv_data['neobank_reference'],
                    release_conditions_met=spv_data['release_conditions_met']
                )
                
                spvs[spv_id] = spv
                
            return spvs
            
        except FileNotFoundError:
            return {}
        except Exception as e:
            self.logger.error(f"Error loading SPVs: {e}")
            return {}
    
    def _save_spvs(self):
        """Save SPVs to storage"""
        try:
            data = {spv_id: spv.to_dict() for spv_id, spv in self.spvs.items()}
            
            with open(f"{self.data_path}/spv_registry.json", 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving SPVs: {e}")
            raise
