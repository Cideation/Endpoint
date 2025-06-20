"""
Integration Tests for VaaS Billing System
Tests emergence detection, billing triggers, and customer management
"""

import asyncio
import json
import pytest
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

import httpx
import asyncpg
from fastapi.testclient import TestClient

from main import app as billing_app, EMERGENCE_CONFIG
from emergence_webhook import app as webhook_app
from crm_dashboard import app as crm_app

# Test configuration
TEST_DATABASE_URL = "postgresql://bem_user:bem_password@localhost:5432/bem_vaas_test"
TEST_REDIS_URL = "redis://localhost:6379/1"

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
async def test_db():
    """Create test database connection"""
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    
    # Clean up test data
    await conn.execute("TRUNCATE customers, emergence_events, billing_transactions, credit_topups CASCADE")
    
    yield conn
    await conn.close()

@pytest.fixture
def billing_client():
    """Test client for billing service"""
    return TestClient(billing_app)

@pytest.fixture  
def webhook_client():
    """Test client for webhook service"""
    return TestClient(webhook_app)

@pytest.fixture
def crm_client():
    """Test client for CRM dashboard"""
    return TestClient(crm_app)

@pytest.fixture
async def test_customer(test_db):
    """Create test customer"""
    customer_data = {
        "id": "test-customer-123",
        "email": "test@example.com",
        "name": "Test Customer",
        "credit_balance": 200,
        "total_spent": Decimal(0),
        "signup_date": datetime.utcnow(),
        "last_activity": datetime.utcnow(),
        "tier_usage": "{}"
    }
    
    await test_db.execute("""
        INSERT INTO customers (id, email, name, credit_balance, total_spent, signup_date, last_activity, tier_usage)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """, *customer_data.values())
    
    return customer_data

# =============================================================================
# EMERGENCE DETECTION TESTS
# =============================================================================

class TestEmergenceDetection:
    """Test emergence detection logic"""
    
    def test_emergence_config_loaded(self):
        """Test that emergence configuration is properly loaded"""
        assert EMERGENCE_CONFIG["model"] == "value_as_a_service"
        assert EMERGENCE_CONFIG["billing_trigger"] == "on_emergence"
        assert len(EMERGENCE_CONFIG["tiers"]) == 5
        
        # Check specific tier
        blueprint_tier = next(t for t in EMERGENCE_CONFIG["tiers"] if t["id"] == "blueprint_package")
        assert blueprint_tier["credits_required"] == 50
        assert blueprint_tier["fiat_price"] == 500
    
    async def test_blueprint_emergence_detection(self, billing_client, test_customer):
        """Test detection of blueprint package emergence"""
        emergence_request = {
            "project_id": "test-project-123",
            "node_id": "test-node-456",
            "fit_score": 0.95,
            "node_data": {
                "finalized": True,
                "emergence_status": "ready",
                "geometry_resolved": True,
                "specs_complete": True
            }
        }
        
        response = billing_client.post("/emergence/check", json=emergence_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["triggered"] == True
        assert data["tier_id"] == "blueprint_package"
        assert data["credits_required"] == 50
        assert data["fiat_price"] == 500
        assert data["can_afford"] == True  # Customer has 200 credits
    
    async def test_insufficient_fit_score(self, billing_client):
        """Test that low fit score doesn't trigger emergence"""
        emergence_request = {
            "project_id": "test-project-123",
            "node_id": "test-node-456", 
            "fit_score": 0.80,  # Below threshold of 0.90
            "node_data": {
                "finalized": True,
                "emergence_status": "ready",
                "geometry_resolved": True,
                "specs_complete": True
            }
        }
        
        response = billing_client.post("/emergence/check", json=emergence_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["triggered"] == False
    
    async def test_bom_emergence_detection(self, billing_client, test_customer):
        """Test BOM + suppliers emergence"""
        emergence_request = {
            "project_id": "test-project-123",
            "node_id": "test-node-789",
            "fit_score": 0.92,
            "node_data": {
                "finalized": True,
                "emergence_status": "ready",
                "materials_resolved": True,
                "components_sourced": True
            }
        }
        
        response = billing_client.post("/emergence/check", json=emergence_request)
        data = response.json()
        
        assert data["triggered"] == True
        assert data["tier_id"] == "bom_with_suppliers"
        assert data["credits_required"] == 30
        assert data["fiat_price"] == 250

# =============================================================================
# BILLING PROCESS TESTS
# =============================================================================

class TestBillingProcess:
    """Test billing transaction processing"""
    
    async def test_successful_billing(self, billing_client, test_customer, test_db):
        """Test successful billing transaction"""
        emergence_event = {
            "customer_id": test_customer["id"],
            "project_id": "test-project-123",
            "node_id": "test-node-456",
            "emergence_type": "blueprint_package",
            "fit_score": 0.95,
            "metadata": {"geometry_resolved": True, "specs_complete": True}
        }
        
        # Process billing
        response = billing_client.post(
            f"/billing/process?customer_id={test_customer['id']}", 
            json=emergence_event
        )
        assert response.status_code == 200
        
        transaction = response.json()
        assert transaction["credits_charged"] == 50
        assert transaction["fiat_equivalent"] == 500
        assert transaction["status"] == "completed"
        
        # Verify customer balance was deducted
        customer_row = await test_db.fetchrow("SELECT credit_balance FROM customers WHERE id = $1", test_customer["id"])
        assert customer_row["credit_balance"] == 150  # 200 - 50
    
    async def test_insufficient_credits_billing(self, billing_client, test_db):
        """Test billing with insufficient credits"""
        # Create customer with low balance
        low_balance_customer = {
            "id": "low-balance-customer",
            "email": "low@example.com", 
            "name": "Low Balance Customer",
            "credit_balance": 20  # Less than 50 required for blueprint
        }
        
        await test_db.execute("""
            INSERT INTO customers (id, email, name, credit_balance, total_spent, signup_date, last_activity, tier_usage)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, low_balance_customer["id"], low_balance_customer["email"], low_balance_customer["name"],
        low_balance_customer["credit_balance"], Decimal(0), datetime.utcnow(), datetime.utcnow(), "{}")
        
        emergence_event = {
            "customer_id": low_balance_customer["id"],
            "project_id": "test-project-123",
            "node_id": "test-node-456",
            "emergence_type": "blueprint_package",
            "fit_score": 0.95,
            "metadata": {}
        }
        
        response = billing_client.post(
            f"/billing/process?customer_id={low_balance_customer['id']}",
            json=emergence_event
        )
        assert response.status_code == 402  # Payment Required
        assert "Insufficient credits" in response.json()["detail"]

# =============================================================================
# WEBHOOK INTEGRATION TESTS
# =============================================================================

class TestWebhookIntegration:
    """Test webhook integration with main BEM system"""
    
    async def test_node_update_webhook(self, webhook_client, test_customer):
        """Test node update webhook processing"""
        node_update = {
            "node_id": "test-node-123",
            "project_id": "test-project-123", 
            "customer_id": test_customer["id"],
            "node_type": "geometry_node",
            "fit_score": 0.94,
            "finalized": True,
            "emergence_status": "ready",
            "node_data": {
                "geometry_resolved": True,
                "specs_complete": True
            }
        }
        
        response = webhook_client.post("/webhook/node-update", json=node_update)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "emergence_detected"
        assert data["tier_id"] == "blueprint_package"
        assert "emergence_event_id" in data
    
    async def test_pulse_event_webhook(self, webhook_client):
        """Test pulse event webhook processing"""
        pulse_event = {
            "pulse_type": "fit_pulse",
            "source_node": "node-123",
            "target_node": "node-456", 
            "pulse_data": {
                "fit_score": 0.93,
                "confidence": 0.87
            }
        }
        
        response = webhook_client.post("/webhook/pulse", json=pulse_event)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "pulse_logged"
        assert data["pulse_type"] == "fit_pulse"

# =============================================================================
# CUSTOMER MANAGEMENT TESTS
# =============================================================================

class TestCustomerManagement:
    """Test customer management functionality"""
    
    async def test_create_customer(self, billing_client):
        """Test customer creation"""
        response = billing_client.post("/customers?email=new@example.com&name=New Customer")
        assert response.status_code == 200
        
        customer = response.json()
        assert customer["email"] == "new@example.com"
        assert customer["name"] == "New Customer"
        assert customer["credit_balance"] == 100  # Default balance
    
    async def test_credit_topup(self, billing_client, test_customer):
        """Test credit top-up process"""
        response = billing_client.post(
            f"/customers/{test_customer['id']}/topup?credits=100&amount_paid=1000"
        )
        assert response.status_code == 200
        
        topup = response.json()
        assert topup["credits_purchased"] == 100
        assert topup["amount_paid"] == 1000
        assert topup["status"] == "completed"
    
    async def test_customer_balance(self, billing_client, test_customer):
        """Test getting customer balance"""
        response = billing_client.get(f"/customers/{test_customer['id']}/balance")
        assert response.status_code == 200
        
        balance = response.json()
        assert balance["customer_id"] == test_customer["id"]
        assert balance["credit_balance"] == test_customer["credit_balance"]
        assert isinstance(balance["recent_transactions"], list)

# =============================================================================
# CRM DASHBOARD TESTS
# =============================================================================

class TestCRMDashboard:
    """Test CRM dashboard functionality"""
    
    async def test_metrics_endpoint(self, crm_client):
        """Test metrics API endpoint"""
        response = crm_client.get("/api/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        assert "daily_revenue" in metrics
        assert "monthly_revenue" in metrics
        assert "total_customers" in metrics
        assert "active_customers" in metrics
    
    async def test_customers_api(self, crm_client):
        """Test customers API endpoint"""
        response = crm_client.get("/api/customers")
        assert response.status_code == 200
        
        data = response.json()
        assert "customers" in data
        assert "page" in data
        assert isinstance(data["customers"], list)
    
    async def test_tier_analytics(self, crm_client):
        """Test tier analytics API"""
        response = crm_client.get("/api/tier-analytics")
        assert response.status_code == 200
        
        analytics = response.json()
        assert isinstance(analytics, list)

# =============================================================================
# INTEGRATION FLOW TESTS
# =============================================================================

class TestEndToEndFlow:
    """Test complete end-to-end emergence billing flow"""
    
    async def test_complete_emergence_billing_flow(self, webhook_client, billing_client, test_customer, test_db):
        """Test complete flow from node update to billing completion"""
        
        # Step 1: Node update triggers emergence detection
        node_update = {
            "node_id": "flow-test-node",
            "project_id": "flow-test-project",
            "customer_id": test_customer["id"],
            "node_type": "geometry_node",
            "fit_score": 0.96,
            "finalized": True,
            "emergence_status": "ready",
            "node_data": {
                "geometry_resolved": True,
                "specs_complete": True
            }
        }
        
        webhook_response = webhook_client.post("/webhook/node-update", json=node_update)
        assert webhook_response.status_code == 200
        webhook_data = webhook_response.json()
        assert webhook_data["status"] == "emergence_detected"
        
        # Step 2: Check emergence event was stored
        emergence_event_id = webhook_data["emergence_event_id"]
        emergence_row = await test_db.fetchrow(
            "SELECT * FROM emergence_events WHERE id = $1", 
            emergence_event_id
        )
        assert emergence_row is not None
        assert emergence_row["customer_id"] == test_customer["id"]
        assert emergence_row["emergence_type"] == "blueprint_package"
        
        # Step 3: Process billing for the emergence event
        emergence_event = {
            "id": emergence_event_id,
            "customer_id": test_customer["id"],
            "project_id": "flow-test-project",
            "node_id": "flow-test-node",
            "emergence_type": "blueprint_package",
            "fit_score": 0.96,
            "metadata": node_update["node_data"]
        }
        
        billing_response = billing_client.post(
            f"/billing/process?customer_id={test_customer['id']}",
            json=emergence_event
        )
        assert billing_response.status_code == 200
        
        transaction = billing_response.json()
        assert transaction["credits_charged"] == 50
        assert transaction["status"] == "completed"
        
        # Step 4: Verify customer balance updated
        customer_row = await test_db.fetchrow(
            "SELECT credit_balance, total_spent FROM customers WHERE id = $1",
            test_customer["id"]
        )
        assert customer_row["credit_balance"] == 150  # 200 - 50
        assert customer_row["total_spent"] == 500     # ₱500 for blueprint package
        
        # Step 5: Verify transaction recorded
        transaction_row = await test_db.fetchrow(
            "SELECT * FROM billing_transactions WHERE emergence_event_id = $1",
            emergence_event_id
        )
        assert transaction_row is not None
        assert transaction_row["customer_id"] == test_customer["id"]
        assert transaction_row["tier_id"] == "blueprint_package"
        assert transaction_row["status"] == "completed"

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test system performance under load"""
    
    async def test_concurrent_emergence_checks(self, billing_client):
        """Test handling multiple concurrent emergence checks"""
        async def check_emergence():
            emergence_request = {
                "project_id": f"perf-test-{asyncio.current_task().get_name()}",
                "node_id": f"node-{asyncio.current_task().get_name()}",
                "fit_score": 0.95,
                "node_data": {
                    "finalized": True,
                    "emergence_status": "ready",
                    "geometry_resolved": True,
                    "specs_complete": True
                }
            }
            
            with httpx.Client() as client:
                response = client.post("http://testserver/emergence/check", json=emergence_request)
                return response.status_code == 200
        
        # Run 50 concurrent emergence checks
        tasks = [check_emergence() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"]) 