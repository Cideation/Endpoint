"""
Neo Bank Test Suite
Comprehensive tests for the contribution routing and SPV management system
"""

import unittest
import json
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.paluwagan_engine import PaluwagangEngine, Contribution, ContributionStatus
from core.spv_manager import SPVManager, PropertyInfo, SPVConfig, SPVStatus
from core.neobank_adapter import NeobankAdapter

class TestPaluwagangEngine(unittest.TestCase):
    """Test the core Paluwagan engine functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_path = "incoming/neo_bank/data"
        self.engine = PaluwagangEngine(self.test_data_path)
    
    def test_route_contribution(self):
        """Test contribution routing"""
        contribution = Contribution(
            contribution_id="TEST-001",
            agent_id="agent_001",
            spv_id="SPV-DEMO001", 
            amount=5000.0,
            currency="PHP",
            timestamp=datetime.now(),
            fintech_provider="GCash",
            transaction_ref="GC-TEST-001",
            status=ContributionStatus.VERIFIED,
            verification_data={"test": True}
        )
        
        result = self.engine.route_contribution(contribution)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['contribution_id'], "TEST-001")
        self.assertIn('routing_result', result)
    
    def test_agent_registration(self):
        """Test agent API registration"""
        result = self.engine.register_agent_api(
            "test_agent_001",
            "GCash", 
            {
                "api_key": "test_key",
                "merchant_id": "TEST_MERCHANT"
            }
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['agent_id'], "test_agent_001")

class TestSPVManager(unittest.TestCase):
    """Test SPV management functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_path = "incoming/neo_bank/data"
        self.spv_manager = SPVManager(self.test_data_path)
    
    def test_create_spv(self):
        """Test SPV creation"""
        property_info = PropertyInfo(
            property_id="TEST-PROP-001",
            address="Test Address",
            city="Test City", 
            province="Test Province",
            postal_code="1234",
            property_type="Residential",
            estimated_value=1000000.0,
            title_number="TEST-TCT-001",
            title_hash="test_hash_123",
            legal_description="Test property description"
        )
        
        spv_config = SPVConfig(
            target_amount=200000.0,
            minimum_contribution=1000.0,
            maximum_contribution=None,
            required_participants=5,
            funding_deadline=datetime.now() + timedelta(days=30),
            auto_release=True,
            manual_approval_required=False,
            contribution_currency="PHP"
        )
        
        result = self.spv_manager.create_spv(property_info, spv_config)
        
        self.assertTrue(result['success'])
        self.assertIn('spv_id', result)
        self.assertEqual(result['target_amount'], 200000.0)
    
    def test_update_contribution(self):
        """Test SPV contribution update"""
        # First create an SPV
        property_info = PropertyInfo(
            property_id="TEST-PROP-002",
            address="Test Address 2",
            city="Test City",
            province="Test Province", 
            postal_code="1234",
            property_type="Residential",
            estimated_value=1000000.0,
            title_number="TEST-TCT-002",
            title_hash="test_hash_456",
            legal_description="Test property description 2"
        )
        
        spv_config = SPVConfig(
            target_amount=100000.0,
            minimum_contribution=500.0,
            maximum_contribution=None,
            required_participants=3,
            funding_deadline=datetime.now() + timedelta(days=30),
            auto_release=True,
            manual_approval_required=False,
            contribution_currency="PHP"
        )
        
        create_result = self.spv_manager.create_spv(property_info, spv_config)
        self.assertTrue(create_result['success'])
        
        spv_id = create_result['spv_id']
        
        # Activate the SPV
        activate_result = self.spv_manager.activate_spv(spv_id, "NB-TEST-001")
        self.assertTrue(activate_result['success'])
        
        # Add contribution
        update_result = self.spv_manager.update_contribution(
            spv_id, "CONTRIB-TEST-001", 10000.0, "agent_001"
        )
        
        self.assertTrue(update_result['success'])
        self.assertEqual(update_result['current_amount'], 10000.0)

class TestAPIIntegration(unittest.TestCase):
    """Test API integration scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock API configuration
        self.api_config = {
            'data_path': 'incoming/neo_bank/data',
            'neobank': {
                'enabled': False,  # Use simulation mode for tests
                'base_url': 'https://test.neobank.com',
                'api_key': 'test_key',
                'api_secret': 'test_secret'
            }
        }
    
    def test_contribution_flow(self):
        """Test complete contribution flow"""
        # This would test the full API workflow:
        # 1. Agent registers API connection
        # 2. SPV is created
        # 3. Contribution is submitted
        # 4. Contribution is routed and processed
        # 5. SPV conditions are checked
        # 6. Release is triggered if conditions met
        
        # For now, just verify the components exist
        from api.service import NeoBankAPIService
        
        service = NeoBankAPIService(self.api_config)
        self.assertIsNotNone(service.paluwagan_engine)
        self.assertIsNotNone(service.spv_manager)

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and validation"""
    
    def test_agent_integrations_format(self):
        """Test agent integrations data format"""
        with open('incoming/neo_bank/data/agent_integrations.json', 'r') as f:
            data = json.load(f)
        
        # Verify each agent has required fields
        for agent_id, agent_data in data.items():
            self.assertIn('agent_id', agent_data)
            self.assertIn('fintech_provider', agent_data)
            self.assertIn('api_credentials', agent_data)
            self.assertIn('status', agent_data)
            
            # Verify credentials structure based on provider
            provider = agent_data['fintech_provider']
            credentials = agent_data['api_credentials']
            
            if provider == 'GCash':
                self.assertIn('gcash_merchant_id', credentials)
                self.assertIn('gcash_api_key', credentials)
            elif provider == 'PayMaya':
                self.assertIn('paymaya_merchant_id', credentials)
                self.assertIn('paymaya_api_key', credentials)
            elif provider == 'Maya':
                self.assertIn('maya_merchant_id', credentials)
                self.assertIn('maya_api_key', credentials)
    
    def test_spv_registry_format(self):
        """Test SPV registry data format"""
        with open('incoming/neo_bank/data/spv_registry.json', 'r') as f:
            data = json.load(f)
        
        # Verify each SPV has required fields
        for spv_id, spv_data in data.items():
            self.assertIn('spv_id', spv_data)
            self.assertIn('property_info', spv_data)
            self.assertIn('config', spv_data)
            self.assertIn('status', spv_data)
            self.assertIn('current_amount', spv_data)
            self.assertIn('target_amount', spv_data['config'])

def run_demo_scenario():
    """Run a complete demo scenario"""
    print("\n" + "="*60)
    print("🏦 RUNNING NEO BANK DEMO SCENARIO")
    print("="*60)
    
    # Initialize components
    engine = PaluwagangEngine("incoming/neo_bank/data")
    spv_manager = SPVManager("incoming/neo_bank/data")
    
    print("\n📊 Current System Status:")
    
    # Show agent registrations
    print("\n👥 Registered Agents:")
    for agent_id, agent_data in engine.agent_integrations.items():
        print(f"  • {agent_id}: {agent_data['fintech_provider']} ({agent_data['status']})")
    
    # Show SPVs
    spv_list = spv_manager.list_spvs()
    print(f"\n🏠 SPVs in System: {spv_list['total_count']}")
    for spv in spv_list['spvs']:
        print(f"  • {spv['spv_id']}: {spv['property_address']} - {spv['progress_percentage']:.1f}% funded")
    
    # Show contributions
    print(f"\n💰 Total Contributions: {len(engine.contributions)}")
    for contrib_id, contrib in engine.contributions.items():
        print(f"  • {contrib_id}: ₱{contrib['amount']:,.2f} from {contrib['agent_id']}")
    
    print("\n✅ Demo scenario completed successfully!")

if __name__ == '__main__':
    print("🧪 Neo Bank Test Suite")
    print("=" * 50)
    
    # Run demo scenario first
    try:
        run_demo_scenario()
    except Exception as e:
        print(f"❌ Demo scenario failed: {e}")
    
    # Run unit tests
    print("\n🔬 Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)
    
    print("\n🎯 All tests completed!")
