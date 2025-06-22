"""
Paluwagan Engine - Core contribution routing and verification system
Routes verified contributions and triggers SPV home title releases
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ContributionStatus(Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    ROUTED = "routed"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"

class TriggerCondition(Enum):
    AMOUNT_THRESHOLD = "amount_threshold"
    TIME_BASED = "time_based"
    PARTICIPANT_COUNT = "participant_count"
    MANUAL_APPROVAL = "manual_approval"

@dataclass
class Contribution:
    """Individual contribution record"""
    contribution_id: str
    agent_id: str
    spv_id: str
    amount: float
    currency: str
    timestamp: datetime
    fintech_provider: str  # GCash, PayMaya, etc.
    transaction_ref: str
    status: ContributionStatus
    verification_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contribution_id': self.contribution_id,
            'agent_id': self.agent_id,
            'spv_id': self.spv_id,
            'amount': self.amount,
            'currency': self.currency,
            'timestamp': self.timestamp.isoformat(),
            'fintech_provider': self.fintech_provider,
            'transaction_ref': self.transaction_ref,
            'status': self.status.value,
            'verification_data': self.verification_data
        }

@dataclass
class SPVTrigger:
    """SPV release trigger configuration"""
    spv_id: str
    property_id: str
    target_amount: float
    current_amount: float
    participant_count: int
    required_participants: int
    deadline: Optional[datetime]
    conditions: List[TriggerCondition]
    status: str
    
class PaluwagangEngine:
    """
    Core engine for routing verified contributions and managing SPV triggers
    
    Key Functions:
    1. Route verified contributions from agent fintech APIs
    2. Track contribution progress toward SPV trigger conditions
    3. Release home titles when conditions are met
    4. Manage agent API registrations
    """
    
    def __init__(self, data_path: str = "incoming/neo_bank/data"):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
        
        # Load persistent data
        self.contributions = self._load_contributions()
        self.spv_registry = self._load_spv_registry()
        self.trigger_log = self._load_trigger_log()
        self.agent_integrations = self._load_agent_integrations()
        
        self.logger.info("Paluwagan Engine initialized")
    
    def route_contribution(self, contribution: Contribution) -> Dict[str, Any]:
        """
        Route a verified contribution to the appropriate SPV
        
        Args:
            contribution: Verified contribution from agent's fintech API
            
        Returns:
            Routing result with status and next actions
        """
        try:
            # Validate contribution
            if not self._validate_contribution(contribution):
                return {
                    'success': False,
                    'error': 'Contribution validation failed',
                    'contribution_id': contribution.contribution_id
                }
            
            # Check if agent is registered
            if not self._is_agent_registered(contribution.agent_id):
                return {
                    'success': False,
                    'error': 'Agent not registered or invalid API connection',
                    'contribution_id': contribution.contribution_id
                }
            
            # Route to SPV
            routing_result = self._route_to_spv(contribution)
            
            # Update contribution status
            contribution.status = ContributionStatus.ROUTED
            self._save_contribution(contribution)
            
            # Check trigger conditions
            trigger_result = self._check_trigger_conditions(contribution.spv_id)
            
            result = {
                'success': True,
                'contribution_id': contribution.contribution_id,
                'routing_result': routing_result,
                'trigger_check': trigger_result,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Contribution routed successfully: {contribution.contribution_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error routing contribution {contribution.contribution_id}: {e}")
            contribution.status = ContributionStatus.FAILED
            self._save_contribution(contribution)
            
            return {
                'success': False,
                'error': str(e),
                'contribution_id': contribution.contribution_id
            }
    
    def check_spv_release_conditions(self, spv_id: str) -> Dict[str, Any]:
        """
        Check if SPV meets conditions for home title release
        
        Args:
            spv_id: SPV identifier
            
        Returns:
            Condition check results and release eligibility
        """
        if spv_id not in self.spv_registry:
            return {
                'eligible': False,
                'error': f'SPV {spv_id} not found in registry'
            }
        
        spv_data = self.spv_registry[spv_id]
        trigger = SPVTrigger(**spv_data['trigger'])
        
        # Check all conditions
        conditions_met = []
        
        # Amount threshold
        if TriggerCondition.AMOUNT_THRESHOLD in trigger.conditions:
            amount_met = trigger.current_amount >= trigger.target_amount
            conditions_met.append({
                'condition': 'amount_threshold',
                'met': amount_met,
                'current': trigger.current_amount,
                'target': trigger.target_amount,
                'progress': (trigger.current_amount / trigger.target_amount) * 100
            })
        
        # Participant count
        if TriggerCondition.PARTICIPANT_COUNT in trigger.conditions:
            participants_met = trigger.participant_count >= trigger.required_participants
            conditions_met.append({
                'condition': 'participant_count',
                'met': participants_met,
                'current': trigger.participant_count,
                'required': trigger.required_participants
            })
        
        # Time-based deadline
        if TriggerCondition.TIME_BASED in trigger.conditions and trigger.deadline:
            deadline_met = datetime.now() <= trigger.deadline
            conditions_met.append({
                'condition': 'time_based',
                'met': deadline_met,
                'deadline': trigger.deadline.isoformat(),
                'time_remaining': str(trigger.deadline - datetime.now()) if deadline_met else "expired"
            })
        
        # Check if all conditions are met
        all_conditions_met = all(condition['met'] for condition in conditions_met)
        
        return {
            'spv_id': spv_id,
            'eligible': all_conditions_met,
            'conditions': conditions_met,
            'property_id': trigger.property_id,
            'status': trigger.status
        }
    
    def trigger_title_release(self, spv_id: str, manual_override: bool = False) -> Dict[str, Any]:
        """
        Trigger home title release from SPV
        
        Args:
            spv_id: SPV identifier
            manual_override: Override automatic conditions
            
        Returns:
            Release trigger result
        """
        # Check conditions unless manual override
        if not manual_override:
            condition_check = self.check_spv_release_conditions(spv_id)
            if not condition_check['eligible']:
                return {
                    'success': False,
                    'error': 'SPV conditions not met for title release',
                    'condition_check': condition_check
                }
        
        try:
            # Log trigger event
            trigger_event = {
                'spv_id': spv_id,
                'timestamp': datetime.now().isoformat(),
                'manual_override': manual_override,
                'triggered_by': 'system' if not manual_override else 'manual',
                'status': 'initiated'
            }
            
            self._log_trigger_event(trigger_event)
            
            # Update SPV status
            if spv_id in self.spv_registry:
                self.spv_registry[spv_id]['trigger']['status'] = 'released'
                self._save_spv_registry()
            
            # Here you would integrate with actual NEOBANK SPV release API
            # For now, we simulate the release
            release_result = self._simulate_spv_release(spv_id)
            
            self.logger.info(f"Title release triggered for SPV {spv_id}")
            
            return {
                'success': True,
                'spv_id': spv_id,
                'release_result': release_result,
                'trigger_event': trigger_event
            }
            
        except Exception as e:
            self.logger.error(f"Error triggering title release for SPV {spv_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'spv_id': spv_id
            }
    
    def register_agent_api(self, agent_id: str, fintech_provider: str, api_credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register agent's fintech API connection (GCash, PayMaya, etc.)
        
        Args:
            agent_id: Agent or COOP identifier
            fintech_provider: Provider name (GCash, PayMaya, etc.)
            api_credentials: Encrypted API credentials
            
        Returns:
            Registration result
        """
        try:
            # Validate credentials format
            required_fields = ['api_key', 'merchant_id']
            if not all(field in api_credentials for field in required_fields):
                return {
                    'success': False,
                    'error': f'Missing required credential fields: {required_fields}'
                }
            
            # Store agent integration
            integration_data = {
                'agent_id': agent_id,
                'fintech_provider': fintech_provider,
                'api_credentials': api_credentials,  # Should be encrypted in production
                'registered_at': datetime.now().isoformat(),
                'status': 'active',
                'last_verified': None
            }
            
            self.agent_integrations[agent_id] = integration_data
            self._save_agent_integrations()
            
            self.logger.info(f"Agent API registered: {agent_id} -> {fintech_provider}")
            
            return {
                'success': True,
                'agent_id': agent_id,
                'fintech_provider': fintech_provider,
                'status': 'registered'
            }
            
        except Exception as e:
            self.logger.error(f"Error registering agent API {agent_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_contribution_status(self, spv_id: str) -> Dict[str, Any]:
        """Get current contribution status for an SPV"""
        spv_contributions = [
            contrib for contrib in self.contributions.values() 
            if contrib.get('spv_id') == spv_id
        ]
        
        total_amount = sum(contrib['amount'] for contrib in spv_contributions)
        participant_count = len(set(contrib['agent_id'] for contrib in spv_contributions))
        
        return {
            'spv_id': spv_id,
            'total_contributions': len(spv_contributions),
            'total_amount': total_amount,
            'participant_count': participant_count,
            'contributions': spv_contributions
        }
    
    # Private helper methods
    def _validate_contribution(self, contribution: Contribution) -> bool:
        """Validate contribution data"""
        if contribution.amount <= 0:
            return False
        if not contribution.agent_id or not contribution.spv_id:
            return False
        if contribution.status != ContributionStatus.VERIFIED:
            return False
        return True
    
    def _is_agent_registered(self, agent_id: str) -> bool:
        """Check if agent has valid API registration"""
        return agent_id in self.agent_integrations and \
               self.agent_integrations[agent_id]['status'] == 'active'
    
    def _route_to_spv(self, contribution: Contribution) -> Dict[str, Any]:
        """Route contribution to SPV"""
        return {
            'routed_to': contribution.spv_id,
            'amount': contribution.amount,
            'routing_timestamp': datetime.now().isoformat()
        }
    
    def _check_trigger_conditions(self, spv_id: str) -> Dict[str, Any]:
        """Check if contribution triggers SPV release conditions"""
        condition_check = self.check_spv_release_conditions(spv_id)
        
        if condition_check['eligible']:
            # Auto-trigger if conditions are met
            return self.trigger_title_release(spv_id)
        
        return condition_check
    
    def _simulate_spv_release(self, spv_id: str) -> Dict[str, Any]:
        """Simulate SPV title release (replace with actual NEOBANK API)"""
        return {
            'release_id': f"REL-{spv_id}-{int(datetime.now().timestamp())}",
            'status': 'released',
            'timestamp': datetime.now().isoformat()
        }
    
    # Data persistence methods
    def _load_contributions(self) -> Dict[str, Any]:
        """Load contributions from storage"""
        try:
            with open(f"{self.data_path}/contributions.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _load_spv_registry(self) -> Dict[str, Any]:
        """Load SPV registry from storage"""
        try:
            with open(f"{self.data_path}/spv_registry.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _load_trigger_log(self) -> List[Dict[str, Any]]:
        """Load trigger log from storage"""
        try:
            with open(f"{self.data_path}/trigger_log.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _load_agent_integrations(self) -> Dict[str, Any]:
        """Load agent integrations from storage"""
        try:
            with open(f"{self.data_path}/agent_integrations.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_contribution(self, contribution: Contribution):
        """Save contribution to storage"""
        self.contributions[contribution.contribution_id] = contribution.to_dict()
        with open(f"{self.data_path}/contributions.json", 'w') as f:
            json.dump(self.contributions, f, indent=2)
    
    def _save_spv_registry(self):
        """Save SPV registry to storage"""
        with open(f"{self.data_path}/spv_registry.json", 'w') as f:
            json.dump(self.spv_registry, f, indent=2)
    
    def _save_agent_integrations(self):
        """Save agent integrations to storage"""
        with open(f"{self.data_path}/agent_integrations.json", 'w') as f:
            json.dump(self.agent_integrations, f, indent=2)
    
    def _log_trigger_event(self, event: Dict[str, Any]):
        """Log trigger event"""
        self.trigger_log.append(event)
        with open(f"{self.data_path}/trigger_log.json", 'w') as f:
            json.dump(self.trigger_log, f, indent=2)
