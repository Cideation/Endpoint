"""
VaaS (Value-as-a-Service) Billing System
Emergence Billing: Users are charged only when system delivers actionable, production-ready value

Key Features:
- Freemium exploration (no cost for testing/iteration)
- Emergence-triggered billing (only when outputs become deployable/buildable/fundable)
- Credit-based system with fiat conversion
- Real-time billing triggers based on node completion and fit scores
- Customer management with transaction history
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aioredis
import asyncpg
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# EMERGENCE BILLING CONFIGURATION
# =============================================================================

EMERGENCE_CONFIG = {
    "model": "value_as_a_service",
    "billing_trigger": "on_emergence",
    "conditions": {
        "node_finalized": True,
        "fit_score_threshold": 0.90,
        "emergence_flag": "ready"
    },
    "tiers": [
        {
            "id": "blueprint_package",
            "label": "CAD + Layout Blueprint",
            "includes": ["dxf", "ifc", "pdf_specs"],
            "trigger": "geometry + specs resolved",
            "credits_required": 50,
            "fiat_price": 500  # ₱500
        },
        {
            "id": "bom_with_suppliers",
            "label": "Bill of Materials + Supplier Map",
            "includes": ["bom_json", "supplier_list"],
            "trigger": "material + component logic complete",
            "credits_required": 30,
            "fiat_price": 250  # ₱250
        },
        {
            "id": "compliance_docs",
            "label": "Compliance + Regulatory Package",
            "includes": ["zoning_sheet", "code_match", "certs"],
            "trigger": "region + compliance functors resolved",
            "credits_required": 35,
            "fiat_price": 300  # ₱300
        },
        {
            "id": "investment_packet",
            "label": "Investment & ROI Report",
            "includes": ["roi_summary", "cost_breakdown", "market_match"],
            "trigger": "investment_node resolved + project_roi complete",
            "credits_required": 60,
            "fiat_price": 700  # ₱700
        },
        {
            "id": "full_emergence_bundle",
            "label": "All-in-One Emergence Package",
            "includes": ["blueprint_package", "bom_with_suppliers", "compliance_docs"],
            "trigger": "all modules above complete",
            "credits_required": 120,
            "fiat_price": 1499  # ₱1,499
        }
    ],
    "credits": {
        "unit_value": 10,  # ₱10 per credit
        "default_balance_on_signup": 100,
        "topup_options": [
            {"credits": 100, "price": 1000},   # ₱1,000
            {"credits": 250, "price": 2400},   # ₱2,400
            {"credits": 500, "price": 4500}    # ₱4,500
        ]
    }
}

# =============================================================================
# DATA MODELS
# =============================================================================

class EmergenceStatus(str, Enum):
    EXPLORATION = "exploration"
    PROCESSING = "processing"
    EMERGENCE_DETECTED = "emergence_detected"
    READY_FOR_BILLING = "ready_for_billing"
    BILLED = "billed"
    DELIVERED = "delivered"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

@dataclass
class Customer:
    id: str
    email: str
    name: str
    credit_balance: int
    total_spent: Decimal
    signup_date: datetime
    last_activity: datetime
    tier_usage: Dict[str, int]  # Track usage per tier
    
class EmergenceEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    project_id: str
    node_id: str
    emergence_type: str  # blueprint_package, bom_with_suppliers, etc.
    fit_score: float
    emergence_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: EmergenceStatus = EmergenceStatus.EMERGENCE_DETECTED
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BillingTransaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    emergence_event_id: str
    tier_id: str
    credits_charged: int
    fiat_equivalent: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: PaymentStatus = PaymentStatus.PENDING
    receipt_url: Optional[str] = None
    
class CreditTopup(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    credits_purchased: int
    amount_paid: Decimal
    payment_method: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: PaymentStatus = PaymentStatus.PENDING

# API Request/Response Models
class EmergenceCheckRequest(BaseModel):
    project_id: str
    node_id: str
    fit_score: float
    node_data: Dict[str, Any]

class BillingTriggerResponse(BaseModel):
    triggered: bool
    tier_id: Optional[str] = None
    credits_required: Optional[int] = None
    fiat_price: Optional[Decimal] = None
    customer_balance: Optional[int] = None
    can_afford: Optional[bool] = None

class CustomerBalanceResponse(BaseModel):
    customer_id: str
    credit_balance: int
    fiat_equivalent: Decimal
    recent_transactions: List[BillingTransaction]

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@asynccontextmanager
async def get_db_connection():
    """Get PostgreSQL connection"""
    try:
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            database="bem_vaas",
            user="bem_user",
            password="bem_password"
        )
        yield conn
    finally:
        await conn.close()

@asynccontextmanager 
async def get_redis_connection():
    """Get Redis connection for caching"""
    redis = await aioredis.from_url("redis://localhost:6379")
    try:
        yield redis
    finally:
        await redis.close()

# =============================================================================
# EMERGENCE DETECTION ENGINE
# =============================================================================

class EmergenceDetector:
    """Detects when system outputs reach actionable, billable state"""
    
    def __init__(self):
        self.config = EMERGENCE_CONFIG
        
    async def check_emergence(self, request: EmergenceCheckRequest) -> BillingTriggerResponse:
        """
        Core emergence detection logic
        Determines if node state triggers billing
        """
        logger.info(f"Checking emergence for project {request.project_id}, node {request.node_id}")
        
        # Check basic emergence conditions
        if not self._meets_basic_conditions(request):
            return BillingTriggerResponse(triggered=False)
            
        # Determine which tier this emergence qualifies for
        tier = self._determine_tier(request)
        if not tier:
            return BillingTriggerResponse(triggered=False)
            
        # Get customer and check balance
        async with get_db_connection() as conn:
            customer = await self._get_customer_from_node(conn, request.node_id)
            if not customer:
                raise HTTPException(status_code=404, detail="Customer not found")
                
            can_afford = customer.credit_balance >= tier["credits_required"]
            
            return BillingTriggerResponse(
                triggered=True,
                tier_id=tier["id"],
                credits_required=tier["credits_required"],
                fiat_price=Decimal(tier["fiat_price"]),
                customer_balance=customer.credit_balance,
                can_afford=can_afford
            )
    
    def _meets_basic_conditions(self, request: EmergenceCheckRequest) -> bool:
        """Check if basic emergence conditions are met"""
        conditions = self.config["conditions"]
        
        # Check fit score threshold
        if request.fit_score < conditions["fit_score_threshold"]:
            logger.info(f"Fit score {request.fit_score} below threshold {conditions['fit_score_threshold']}")
            return False
            
        # Check if node is finalized
        node_finalized = request.node_data.get("finalized", False)
        if not node_finalized:
            logger.info("Node not finalized")
            return False
            
        # Check emergence flag
        emergence_flag = request.node_data.get("emergence_status", "")
        if emergence_flag != "ready":
            logger.info(f"Emergence flag '{emergence_flag}' not ready")
            return False
            
        return True
    
    def _determine_tier(self, request: EmergenceCheckRequest) -> Optional[Dict]:
        """Determine which billing tier this emergence qualifies for"""
        node_data = request.node_data
        
        # Check each tier's trigger conditions
        for tier in self.config["tiers"]:
            if self._matches_tier_trigger(tier, node_data):
                logger.info(f"Emergence matches tier: {tier['id']}")
                return tier
                
        return None
    
    def _matches_tier_trigger(self, tier: Dict, node_data: Dict) -> bool:
        """Check if node data matches specific tier trigger conditions"""
        tier_id = tier["id"]
        
        if tier_id == "blueprint_package":
            return (node_data.get("geometry_resolved", False) and 
                   node_data.get("specs_complete", False))
                   
        elif tier_id == "bom_with_suppliers":
            return (node_data.get("materials_resolved", False) and 
                   node_data.get("components_sourced", False))
                   
        elif tier_id == "compliance_docs":
            return (node_data.get("region_validated", False) and 
                   node_data.get("code_compliance_checked", False))
                   
        elif tier_id == "investment_packet":
            return (node_data.get("investment_analysis_complete", False) and 
                   node_data.get("roi_calculated", False))
                   
        elif tier_id == "full_emergence_bundle":
            # Check if all individual tiers are complete
            blueprint_ready = self._matches_tier_trigger(
                {"id": "blueprint_package"}, node_data)
            bom_ready = self._matches_tier_trigger(
                {"id": "bom_with_suppliers"}, node_data)
            compliance_ready = self._matches_tier_trigger(
                {"id": "compliance_docs"}, node_data)
            return blueprint_ready and bom_ready and compliance_ready
            
        return False
    
    async def _get_customer_from_node(self, conn, node_id: str) -> Optional[Customer]:
        """Get customer associated with a node"""
        # This would query the main BEM database to find customer
        # For now, using a placeholder - integrate with actual customer lookup
        query = """
        SELECT customer_id, email, name, credit_balance, total_spent, 
               signup_date, last_activity, tier_usage
        FROM customers 
        WHERE id = (
            SELECT customer_id FROM projects 
            WHERE id = (
                SELECT project_id FROM nodes WHERE id = $1
            )
        )
        """
        try:
            row = await conn.fetchrow(query, node_id)
            if row:
                return Customer(
                    id=row["customer_id"],
                    email=row["email"],
                    name=row["name"],
                    credit_balance=row["credit_balance"],
                    total_spent=row["total_spent"],
                    signup_date=row["signup_date"],
                    last_activity=row["last_activity"],
                    tier_usage=json.loads(row["tier_usage"] or "{}")
                )
        except Exception as e:
            logger.error(f"Error fetching customer: {e}")
            
        return None

# =============================================================================
# BILLING ENGINE
# =============================================================================

class BillingEngine:
    """Handles credit deduction, transaction processing, and receipt generation"""
    
    async def process_emergence_billing(self, customer_id: str, emergence_event: EmergenceEvent) -> BillingTransaction:
        """Process billing when emergence is detected and customer approves"""
        
        async with get_db_connection() as conn:
            # Get tier configuration
            tier = self._get_tier_config(emergence_event.emergence_type)
            if not tier:
                raise HTTPException(status_code=400, detail="Invalid emergence type")
            
            # Start transaction
            async with conn.transaction():
                # Verify customer balance
                customer = await self._get_customer(conn, customer_id)
                if customer.credit_balance < tier["credits_required"]:
                    raise HTTPException(status_code=402, detail="Insufficient credits")
                
                # Create billing transaction
                transaction = BillingTransaction(
                    customer_id=customer_id,
                    emergence_event_id=emergence_event.id,
                    tier_id=tier["id"],
                    credits_charged=tier["credits_required"],
                    fiat_equivalent=Decimal(tier["fiat_price"]),
                    status=PaymentStatus.COMPLETED
                )
                
                # Deduct credits
                await self._deduct_credits(conn, customer_id, tier["credits_required"])
                
                # Record transaction
                await self._record_transaction(conn, transaction)
                
                # Update emergence event status
                emergence_event.status = EmergenceStatus.BILLED
                await self._update_emergence_event(conn, emergence_event)
                
                # Generate receipt
                transaction.receipt_url = await self._generate_receipt(transaction, tier)
                
                logger.info(f"Billing completed for customer {customer_id}, tier {tier['id']}")
                return transaction
    
    def _get_tier_config(self, tier_id: str) -> Optional[Dict]:
        """Get tier configuration by ID"""
        for tier in EMERGENCE_CONFIG["tiers"]:
            if tier["id"] == tier_id:
                return tier
        return None
    
    async def _get_customer(self, conn, customer_id: str) -> Customer:
        """Get customer by ID"""
        query = """
        SELECT id, email, name, credit_balance, total_spent, 
               signup_date, last_activity, tier_usage
        FROM customers WHERE id = $1
        """
        row = await conn.fetchrow(query, customer_id)
        if not row:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        return Customer(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            credit_balance=row["credit_balance"],
            total_spent=row["total_spent"],
            signup_date=row["signup_date"],
            last_activity=row["last_activity"],
            tier_usage=json.loads(row["tier_usage"] or "{}")
        )
    
    async def _deduct_credits(self, conn, customer_id: str, credits: int):
        """Deduct credits from customer balance"""
        query = """
        UPDATE customers 
        SET credit_balance = credit_balance - $2,
            last_activity = NOW()
        WHERE id = $1 AND credit_balance >= $2
        """
        result = await conn.execute(query, customer_id, credits)
        if result == "UPDATE 0":
            raise HTTPException(status_code=402, detail="Insufficient credits or customer not found")
    
    async def _record_transaction(self, conn, transaction: BillingTransaction):
        """Record billing transaction in database"""
        query = """
        INSERT INTO billing_transactions 
        (id, customer_id, emergence_event_id, tier_id, credits_charged, 
         fiat_equivalent, timestamp, status, receipt_url)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        await conn.execute(
            query,
            transaction.id,
            transaction.customer_id,
            transaction.emergence_event_id,
            transaction.tier_id,
            transaction.credits_charged,
            transaction.fiat_equivalent,
            transaction.timestamp,
            transaction.status.value,
            transaction.receipt_url
        )
    
    async def _update_emergence_event(self, conn, event: EmergenceEvent):
        """Update emergence event status"""
        query = """
        UPDATE emergence_events 
        SET status = $2
        WHERE id = $1
        """
        await conn.execute(query, event.id, event.status.value)
    
    async def _generate_receipt(self, transaction: BillingTransaction, tier: Dict) -> str:
        """Generate receipt URL/document for transaction"""
        # This would integrate with a receipt generation service
        # For now, return a placeholder URL
        receipt_id = transaction.id[:8]
        return f"https://bem-billing.vercel.app/receipts/{receipt_id}"

# =============================================================================
# CUSTOMER MANAGEMENT
# =============================================================================

class CustomerManager:
    """Handles customer operations, credit management, and CRM functionality"""
    
    async def create_customer(self, email: str, name: str) -> Customer:
        """Create new customer with default credit balance"""
        customer = Customer(
            id=str(uuid.uuid4()),
            email=email,
            name=name,
            credit_balance=EMERGENCE_CONFIG["credits"]["default_balance_on_signup"],
            total_spent=Decimal(0),
            signup_date=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            tier_usage={}
        )
        
        async with get_db_connection() as conn:
            query = """
            INSERT INTO customers 
            (id, email, name, credit_balance, total_spent, signup_date, last_activity, tier_usage)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            await conn.execute(
                query,
                customer.id,
                customer.email,
                customer.name,
                customer.credit_balance,
                customer.total_spent,
                customer.signup_date,
                customer.last_activity,
                json.dumps(customer.tier_usage)
            )
            
        logger.info(f"Created customer {customer.id} with {customer.credit_balance} credits")
        return customer
    
    async def process_credit_topup(self, customer_id: str, credits: int, amount_paid: Decimal) -> CreditTopup:
        """Process credit top-up for customer"""
        topup = CreditTopup(
            customer_id=customer_id,
            credits_purchased=credits,
            amount_paid=amount_paid,
            payment_method="credit_card",  # Would be determined by payment processor
            status=PaymentStatus.COMPLETED
        )
        
        async with get_db_connection() as conn:
            async with conn.transaction():
                # Add credits to customer balance
                await conn.execute(
                    "UPDATE customers SET credit_balance = credit_balance + $2 WHERE id = $1",
                    customer_id, credits
                )
                
                # Record top-up transaction
                query = """
                INSERT INTO credit_topups 
                (id, customer_id, credits_purchased, amount_paid, payment_method, timestamp, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """
                await conn.execute(
                    query,
                    topup.id,
                    topup.customer_id,
                    topup.credits_purchased,
                    topup.amount_paid,
                    topup.payment_method,
                    topup.timestamp,
                    topup.status.value
                )
        
        logger.info(f"Added {credits} credits to customer {customer_id}")
        return topup
    
    async def get_customer_balance(self, customer_id: str) -> CustomerBalanceResponse:
        """Get customer balance and recent transactions"""
        async with get_db_connection() as conn:
            # Get customer info
            customer_query = "SELECT credit_balance FROM customers WHERE id = $1"
            customer_row = await conn.fetchrow(customer_query, customer_id)
            if not customer_row:
                raise HTTPException(status_code=404, detail="Customer not found")
                
            credit_balance = customer_row["credit_balance"]
            fiat_equivalent = Decimal(credit_balance * EMERGENCE_CONFIG["credits"]["unit_value"])
            
            # Get recent transactions
            transactions_query = """
            SELECT id, emergence_event_id, tier_id, credits_charged, 
                   fiat_equivalent, timestamp, status, receipt_url
            FROM billing_transactions 
            WHERE customer_id = $1 
            ORDER BY timestamp DESC 
            LIMIT 10
            """
            transaction_rows = await conn.fetch(transactions_query, customer_id)
            
            recent_transactions = [
                BillingTransaction(
                    id=row["id"],
                    customer_id=customer_id,
                    emergence_event_id=row["emergence_event_id"],
                    tier_id=row["tier_id"],
                    credits_charged=row["credits_charged"],
                    fiat_equivalent=row["fiat_equivalent"],
                    timestamp=row["timestamp"],
                    status=PaymentStatus(row["status"]),
                    receipt_url=row["receipt_url"]
                )
                for row in transaction_rows
            ]
            
            return CustomerBalanceResponse(
                customer_id=customer_id,
                credit_balance=credit_balance,
                fiat_equivalent=fiat_equivalent,
                recent_transactions=recent_transactions
            )

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="BEM VaaS Billing System",
    description="Emergence-based billing for Value-as-a-Service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
emergence_detector = EmergenceDetector()
billing_engine = BillingEngine()
customer_manager = CustomerManager()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vaas-billing", "timestamp": datetime.utcnow()}

@app.post("/emergence/check", response_model=BillingTriggerResponse)
async def check_emergence(request: EmergenceCheckRequest):
    """Check if node state triggers emergence billing"""
    return await emergence_detector.check_emergence(request)

@app.post("/billing/process")
async def process_billing(customer_id: str, emergence_event: EmergenceEvent):
    """Process billing for an emergence event"""
    return await billing_engine.process_emergence_billing(customer_id, emergence_event)

@app.get("/customers/{customer_id}/balance", response_model=CustomerBalanceResponse)
async def get_customer_balance(customer_id: str):
    """Get customer balance and transaction history"""
    return await customer_manager.get_customer_balance(customer_id)

@app.post("/customers")
async def create_customer(email: str, name: str):
    """Create new customer account"""
    return await customer_manager.create_customer(email, name)

@app.post("/customers/{customer_id}/topup")
async def topup_credits(customer_id: str, credits: int, amount_paid: float):
    """Add credits to customer account"""
    return await customer_manager.process_credit_topup(
        customer_id, credits, Decimal(amount_paid)
    )

@app.get("/tiers")
async def get_billing_tiers():
    """Get all available billing tiers"""
    return EMERGENCE_CONFIG["tiers"]

@app.get("/customers/{customer_id}/emergence-events")
async def get_customer_emergence_events(customer_id: str):
    """Get customer's emergence events history"""
    async with get_db_connection() as conn:
        query = """
        SELECT ee.* FROM emergence_events ee
        JOIN projects p ON ee.project_id = p.id
        WHERE p.customer_id = $1
        ORDER BY ee.emergence_timestamp DESC
        """
        rows = await conn.fetch(query, customer_id)
        return [dict(row) for row in rows]

# =============================================================================
# PROMETHEUS METRICS (for monitoring)
# =============================================================================

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # This would return Prometheus-formatted metrics
    # Integration with existing monitoring infrastructure
    return "# Placeholder for Prometheus metrics"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004) 