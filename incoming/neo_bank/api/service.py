"""
Neo Bank API Service - Main API layer
Accepts contributions, provides status, and handles release operations
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime
from typing import Dict, Any
import json
import uuid

# Import core components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.paluwagan_engine import PaluwagangEngine, Contribution, ContributionStatus
from core.neobank_adapter import NeobankAdapter
from core.spv_manager import SPVManager, PropertyInfo, SPVConfig

class NeoBankAPIService:
    """
    Main API service for Neo Bank operations
    
    Endpoints:
    - POST /api/v1/contribution - Accept new contribution
    - GET /api/v1/spv/{spv_id}/status - Get SPV status
    - POST /api/v1/spv/{spv_id}/release - Trigger title release
    - POST /api/v1/agent/register - Register agent API connection
    - GET /api/v1/health - Service health check
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        data_path = self.config.get('data_path', 'incoming/neo_bank/data')
        
        self.paluwagan_engine = PaluwagangEngine(data_path)
        self.spv_manager = SPVManager(data_path)
        
        # Initialize NEOBANK adapter if configured
        neobank_config = self.config.get('neobank', {})
        if neobank_config.get('enabled', False):
            self.neobank_adapter = NeobankAdapter(neobank_config)
        else:
            self.neobank_adapter = None
            self.logger.warning("NEOBANK adapter not configured - using simulation mode")
        
        self.logger.info("Neo Bank API Service initialized")
    
    def create_app(self) -> Flask:
        """Create and configure Flask application"""
        app = Flask(__name__)
        
        # Configure logging
        if not app.debug:
            logging.basicConfig(level=logging.INFO)
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: Flask):
        """Register API routes"""
        
        @app.route('/api/v1/contribution', methods=['POST'])
        def accept_contribution():
            """Accept a new contribution from an agent"""
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['agent_id', 'spv_id', 'amount', 'currency', 'fintech_provider', 'transaction_ref']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return jsonify({
                        'success': False,
                        'error': f'Missing required fields: {missing_fields}'
                    }), 400
                
                # Create contribution object
                contribution = Contribution(
                    contribution_id=f"CONTRIB-{uuid.uuid4().hex[:8].upper()}",
                    agent_id=data['agent_id'],
                    spv_id=data['spv_id'],
                    amount=float(data['amount']),
                    currency=data['currency'],
                    timestamp=datetime.now(),
                    fintech_provider=data['fintech_provider'],
                    transaction_ref=data['transaction_ref'],
                    status=ContributionStatus.VERIFIED,  # Assume pre-verified
                    verification_data=data.get('verification_data', {})
                )
                
                # Route contribution through Paluwagan engine
                routing_result = self.paluwagan_engine.route_contribution(contribution)
                
                if routing_result['success']:
                    # Update SPV with contribution
                    spv_result = self.spv_manager.update_contribution(
                        contribution.spv_id,
                        contribution.contribution_id,
                        contribution.amount,
                        contribution.agent_id
                    )
                    
                    response = {
                        'success': True,
                        'contribution_id': contribution.contribution_id,
                        'routing_result': routing_result,
                        'spv_update': spv_result,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return jsonify(response), 201
                else:
                    return jsonify(routing_result), 400
                
            except Exception as e:
                self.logger.error(f"Error accepting contribution: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'details': str(e)
                }), 500
        
        @app.route('/api/v1/spv/<spv_id>/status', methods=['GET'])
        def get_spv_status(spv_id: str):
            """Get current status of an SPV"""
            try:
                # Get SPV details
                spv_result = self.spv_manager.get_spv_details(spv_id)
                
                if not spv_result['success']:
                    return jsonify(spv_result), 404
                
                # Get contribution summary
                contribution_status = self.paluwagan_engine.get_contribution_status(spv_id)
                
                # Check release conditions
                conditions_check = self.paluwagan_engine.check_spv_release_conditions(spv_id)
                
                response = {
                    'success': True,
                    'spv_id': spv_id,
                    'spv_details': spv_result['spv'],
                    'contribution_summary': contribution_status,
                    'release_conditions': conditions_check,
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(response), 200
                
            except Exception as e:
                self.logger.error(f"Error getting SPV status {spv_id}: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'details': str(e)
                }), 500
        
        @app.route('/api/v1/spv/<spv_id>/release', methods=['POST'])
        def trigger_spv_release(spv_id: str):
            """Trigger title release for an SPV"""
            try:
                data = request.get_json() or {}
                manual_override = data.get('manual_override', False)
                authorized_by = data.get('authorized_by', 'api')
                
                # Trigger release through Paluwagan engine
                release_result = self.paluwagan_engine.trigger_title_release(spv_id, manual_override)
                
                if release_result['success']:
                    # If NEOBANK adapter is available, trigger actual release
                    if self.neobank_adapter:
                        neobank_result = self.neobank_adapter.trigger_title_release(
                            spv_id,
                            {
                                'trigger_reason': 'conditions_met',
                                'total_contributions': release_result.get('total_contributions', 0),
                                'participant_count': release_result.get('participant_count', 0),
                                'verification_hash': 'placeholder_hash',
                                'authorized_by': authorized_by
                            }
                        )
                        release_result['neobank_result'] = neobank_result
                    
                    # Update SPV status
                    spv_release_result = self.spv_manager.release_spv(spv_id, manual_override)
                    release_result['spv_release'] = spv_release_result
                    
                    return jsonify(release_result), 200
                else:
                    return jsonify(release_result), 400
                
            except Exception as e:
                self.logger.error(f"Error triggering SPV release {spv_id}: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'details': str(e)
                }), 500
        
        @app.route('/api/v1/agent/register', methods=['POST'])
        def register_agent():
            """Register agent's fintech API connection"""
            try:
                data = request.get_json()
                
                required_fields = ['agent_id', 'fintech_provider', 'api_credentials']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return jsonify({
                        'success': False,
                        'error': f'Missing required fields: {missing_fields}'
                    }), 400
                
                # Register agent through Paluwagan engine
                registration_result = self.paluwagan_engine.register_agent_api(
                    data['agent_id'],
                    data['fintech_provider'],
                    data['api_credentials']
                )
                
                if registration_result['success']:
                    return jsonify(registration_result), 201
                else:
                    return jsonify(registration_result), 400
                
            except Exception as e:
                self.logger.error(f"Error registering agent: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'details': str(e)
                }), 500
        
        @app.route('/api/v1/spv', methods=['GET'])
        def list_spvs():
            """List all SPVs with optional filtering"""
            try:
                # Get query parameters
                status_filter = request.args.get('status')
                
                # List SPVs
                spv_list = self.spv_manager.list_spvs()
                
                if spv_list['success']:
                    return jsonify(spv_list), 200
                else:
                    return jsonify(spv_list), 500
                
            except Exception as e:
                self.logger.error(f"Error listing SPVs: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'details': str(e)
                }), 500
        
        @app.route('/api/v1/spv', methods=['POST'])
        def create_spv():
            """Create a new SPV"""
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['property_info', 'config']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return jsonify({
                        'success': False,
                        'error': f'Missing required fields: {missing_fields}'
                    }), 400
                
                # Create property info object
                property_info = PropertyInfo(**data['property_info'])
                
                # Create SPV config object
                config_data = data['config'].copy()
                if 'funding_deadline' in config_data and config_data['funding_deadline']:
                    config_data['funding_deadline'] = datetime.fromisoformat(config_data['funding_deadline'])
                
                spv_config = SPVConfig(**config_data)
                
                # Create SPV
                creation_result = self.spv_manager.create_spv(property_info, spv_config)
                
                if creation_result['success']:
                    return jsonify(creation_result), 201
                else:
                    return jsonify(creation_result), 400
                
            except Exception as e:
                self.logger.error(f"Error creating SPV: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'details': str(e)
                }), 500
        
        @app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """Service health check"""
            try:
                health_status = {
                    'service': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'components': {
                        'paluwagan_engine': 'healthy',
                        'spv_manager': 'healthy'
                    }
                }
                
                # Check NEOBANK adapter if available
                if self.neobank_adapter:
                    neobank_health = self.neobank_adapter.health_check()
                    health_status['components']['neobank_adapter'] = neobank_health['status']
                else:
                    health_status['components']['neobank_adapter'] = 'disabled'
                
                return jsonify(health_status), 200
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                return jsonify({
                    'service': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/v1/stats', methods=['GET'])
        def get_stats():
            """Get system statistics"""
            try:
                # Get SPV statistics
                spv_list = self.spv_manager.list_spvs()
                
                stats = {
                    'total_spvs': spv_list.get('total_count', 0),
                    'spv_status_breakdown': {},
                    'total_contributions': 0,
                    'total_amount': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Calculate statistics
                if spv_list['success']:
                    for spv in spv_list['spvs']:
                        status = spv['status']
                        if status not in stats['spv_status_breakdown']:
                            stats['spv_status_breakdown'][status] = 0
                        stats['spv_status_breakdown'][status] += 1
                        
                        stats['total_amount'] += spv['current_amount']
                
                return jsonify(stats), 200
                
            except Exception as e:
                self.logger.error(f"Error getting stats: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'details': str(e)
                }), 500

def create_app(config: Dict[str, Any] = None) -> Flask:
    """Factory function to create Flask app"""
    service = NeoBankAPIService(config)
    return service.create_app()

if __name__ == '__main__':
    # Development server
    config = {
        'data_path': 'incoming/neo_bank/data',
        'neobank': {
            'enabled': False,  # Set to True with real NEOBANK credentials
            'base_url': 'https://api.neobank.com',
            'api_key': 'your-api-key',
            'api_secret': 'your-api-secret'
        }
    }
    
    app = create_app(config)
    app.run(host='0.0.0.0', port=5001, debug=True)
