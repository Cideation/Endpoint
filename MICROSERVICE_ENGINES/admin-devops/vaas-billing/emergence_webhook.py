"""
Emergence Webhook Integration
Receives real-time emergence events from the main BEM system
Triggers billing when nodes reach actionable status
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
import asyncpg
import aioredis
from contextlib import asynccontextmanager

from main import EmergenceDetector, BillingEngine, EmergenceEvent, EmergenceStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# WEBHOOK DATA MODELS
# =============================================================================

class NodeStateUpdate(BaseModel):
    """Incoming node state update from BEM system"""
    node_id: str
    project_id: str
    customer_id: str
    node_type: str
    fit_score: float
    finalized: bool
    emergence_status: str  # 'ready', 'processing', 'pending'
    node_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PulseEvent(BaseModel):
    """Pulse event from ECM system"""
    pulse_type: str  # 'bid_pulse', 'fit_pulse', 'investment_pulse', etc.
    source_node: str
    target_node: str
    pulse_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EmergenceWebhookPayload(BaseModel):
    """Standard webhook payload from BEM system"""
    event_type: str  # 'node_update', 'pulse_event', 'emergence_detected'
    data: Dict[str, Any]
    source: str = "bem_system"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# =============================================================================
# EMERGENCE WEBHOOK PROCESSOR
# =============================================================================

class EmergenceWebhookProcessor:
    """Processes incoming webhook events and triggers billing when appropriate"""
    
    def __init__(self):
        self.emergence_detector = EmergenceDetector()
        self.billing_engine = BillingEngine()
        
    async def process_webhook(self, payload: EmergenceWebhookPayload) -> Dict[str, Any]:
        """Main webhook processing logic"""
        logger.info(f"Processing webhook event: {payload.event_type}")
        
        try:
            if payload.event_type == "node_update":
                return await self._process_node_update(payload.data)
            elif payload.event_type == "pulse_event":
                return await self._process_pulse_event(payload.data)
            elif payload.event_type == "emergence_detected":
                return await self._process_emergence_detected(payload.data)
            else:
                logger.warning(f"Unknown event type: {payload.event_type}")
                return {"status": "ignored", "reason": "unknown_event_type"}
                
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _process_node_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process node state update and check for emergence"""
        try:
            node_update = NodeStateUpdate(**data)
            
            # Check if this update triggers emergence billing
            emergence_check_request = EmergenceCheckRequest(
                project_id=node_update.project_id,
                node_id=node_update.node_id,
                fit_score=node_update.fit_score,
                node_data=node_update.node_data
            )
            
            billing_response = await self.emergence_detector.check_emergence(emergence_check_request)
            
            if billing_response.triggered:
                # Create emergence event
                emergence_event = EmergenceEvent(
                    customer_id=node_update.customer_id,
                    project_id=node_update.project_id,
                    node_id=node_update.node_id,
                    emergence_type=billing_response.tier_id,
                    fit_score=node_update.fit_score,
                    metadata=node_update.node_data
                )
                
                # Store emergence event
                await self._store_emergence_event(emergence_event)
                
                # Send notification to customer about billing trigger
                await self._notify_customer_emergence(emergence_event, billing_response)
                
                logger.info(f"Emergence detected for customer {node_update.customer_id}, tier {billing_response.tier_id}")
                
                return {
                    "status": "emergence_detected",
                    "emergence_event_id": emergence_event.id,
                    "tier_id": billing_response.tier_id,
                    "credits_required": billing_response.credits_required,
                    "customer_can_afford": billing_response.can_afford
                }
            
            return {"status": "no_emergence", "fit_score": node_update.fit_score}
            
        except Exception as e:
            logger.error(f"Error processing node update: {e}")
            raise
    
    async def _process_pulse_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process pulse event and update node tracking"""
        try:
            pulse_event = PulseEvent(**data)
            
            # Log pulse for analytics
            async with self._get_redis() as redis:
                await redis.lpush(
                    f"pulse_events:{pulse_event.target_node}",
                    json.dumps({
                        "pulse_type": pulse_event.pulse_type,
                        "source": pulse_event.source_node,
                        "data": pulse_event.pulse_data,
                        "timestamp": pulse_event.timestamp.isoformat()
                    })
                )
                
                # Keep only last 100 pulse events per node
                await redis.ltrim(f"pulse_events:{pulse_event.target_node}", 0, 99)
            
            # Check if pulse affects emergence status
            if pulse_event.pulse_type in ["investment_pulse", "fit_pulse", "compliancy_pulse"]:
                # These pulses might trigger emergence - need to check node state
                return await self._check_node_emergence_after_pulse(pulse_event)
            
            return {"status": "pulse_logged", "pulse_type": pulse_event.pulse_type}
            
        except Exception as e:
            logger.error(f"Error processing pulse event: {e}")
            raise
    
    async def _process_emergence_detected(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process explicit emergence detection event"""
        try:
            emergence_event = EmergenceEvent(**data)
            
            # Store the emergence event
            await self._store_emergence_event(emergence_event)
            
            # Check billing requirements
            async with self._get_db() as conn:
                customer = await self._get_customer(conn, emergence_event.customer_id)
                tier_config = self._get_tier_config(emergence_event.emergence_type)
                
                if not tier_config:
                    return {"status": "error", "error": "invalid_emergence_type"}
                
                can_afford = customer.credit_balance >= tier_config["credits_required"]
                
                if can_afford:
                    # Auto-bill if customer has sufficient credits and has auto-billing enabled
                    auto_billing_enabled = await self._check_auto_billing_preference(customer.id)
                    
                    if auto_billing_enabled:
                        transaction = await self.billing_engine.process_emergence_billing(
                            customer.id, emergence_event
                        )
                        return {
                            "status": "auto_billed",
                            "transaction_id": transaction.id,
                            "credits_charged": transaction.credits_charged
                        }
                    else:
                        # Send billing notification for manual approval
                        await self._notify_customer_emergence(emergence_event, None)
                        return {"status": "awaiting_approval", "credits_required": tier_config["credits_required"]}
                else:
                    # Insufficient credits - notify customer to top up
                    await self._notify_insufficient_credits(customer, tier_config["credits_required"])
                    return {"status": "insufficient_credits", "credits_needed": tier_config["credits_required"] - customer.credit_balance}
            
        except Exception as e:
            logger.error(f"Error processing emergence detected: {e}")
            raise
    
    async def _store_emergence_event(self, event: EmergenceEvent):
        """Store emergence event in database"""
        async with self._get_db() as conn:
            query = """
            INSERT INTO emergence_events 
            (id, customer_id, project_id, node_id, emergence_type, fit_score, 
             emergence_timestamp, status, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            await conn.execute(
                query,
                event.id,
                event.customer_id,
                event.project_id,
                event.node_id,
                event.emergence_type,
                event.fit_score,
                event.emergence_timestamp,
                event.status.value,
                json.dumps(event.metadata)
            )
    
    async def _notify_customer_emergence(self, event: EmergenceEvent, billing_response: Optional[Any]):
        """Send notification to customer about emergence detection"""
        # This would integrate with notification service (email, SMS, push)
        logger.info(f"Notifying customer {event.customer_id} about emergence {event.emergence_type}")
        
        # Store notification in Redis for real-time updates
        async with self._get_redis() as redis:
            notification = {
                "type": "emergence_detected",
                "emergence_event_id": event.id,
                "emergence_type": event.emergence_type,
                "fit_score": event.fit_score,
                "timestamp": event.emergence_timestamp.isoformat(),
                "billing_required": billing_response is not None
            }
            
            await redis.lpush(
                f"notifications:{event.customer_id}",
                json.dumps(notification)
            )
            
            # Keep only last 50 notifications
            await redis.ltrim(f"notifications:{event.customer_id}", 0, 49)
    
    async def _notify_insufficient_credits(self, customer: Any, credits_needed: int):
        """Notify customer about insufficient credits"""
        async with self._get_redis() as redis:
            notification = {
                "type": "insufficient_credits",
                "credits_needed": credits_needed,
                "current_balance": customer.credit_balance,
                "topup_suggestions": self._get_topup_suggestions(credits_needed),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await redis.lpush(
                f"notifications:{customer.id}",
                json.dumps(notification)
            )
    
    def _get_topup_suggestions(self, credits_needed: int) -> List[Dict]:
        """Get suggested credit topup amounts"""
        from main import EMERGENCE_CONFIG
        
        suggestions = []
        for option in EMERGENCE_CONFIG["credits"]["topup_options"]:
            if option["credits"] >= credits_needed:
                suggestions.append(option)
        
        return suggestions
    
    async def _check_auto_billing_preference(self, customer_id: str) -> bool:
        """Check if customer has auto-billing enabled"""
        async with self._get_db() as conn:
            query = "SELECT auto_billing_enabled FROM customer_preferences WHERE customer_id = $1"
            result = await conn.fetchval(query, customer_id)
            return result or False  # Default to False if no preference set
    
    @asynccontextmanager
    async def _get_db(self):
        """Get database connection"""
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            database="bem_vaas",
            user="bem_user",
            password="bem_password"
        )
        try:
            yield conn
        finally:
            await conn.close()
    
    @asynccontextmanager
    async def _get_redis(self):
        """Get Redis connection"""
        redis = await aioredis.from_url("redis://localhost:6379")
        try:
            yield redis
        finally:
            await redis.close()

# =============================================================================
# WEBHOOK FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="BEM Emergence Webhook",
    description="Receives emergence events from BEM system",
    version="1.0.0"
)

# Initialize webhook processor
webhook_processor = EmergenceWebhookProcessor()

@app.post("/webhook/emergence")
async def handle_emergence_webhook(
    payload: EmergenceWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Main webhook endpoint for emergence events"""
    
    # Log the incoming webhook
    client_ip = request.client.host
    logger.info(f"Received webhook from {client_ip}: {payload.event_type}")
    
    # Process webhook in background to return quickly
    background_tasks.add_task(
        webhook_processor.process_webhook,
        payload
    )
    
    return {"status": "received", "event_type": payload.event_type}

@app.post("/webhook/node-update")
async def handle_node_update(node_update: NodeStateUpdate):
    """Dedicated endpoint for node updates"""
    payload = EmergenceWebhookPayload(
        event_type="node_update",
        data=node_update.dict()
    )
    
    return await webhook_processor.process_webhook(payload)

@app.post("/webhook/pulse")
async def handle_pulse_event(pulse_event: PulseEvent):
    """Dedicated endpoint for pulse events"""
    payload = EmergenceWebhookPayload(
        event_type="pulse_event",
        data=pulse_event.dict()
    )
    
    return await webhook_processor.process_webhook(payload)

@app.get("/webhook/health")
async def webhook_health():
    """Health check for webhook service"""
    return {
        "status": "healthy",
        "service": "emergence_webhook",
        "timestamp": datetime.utcnow()
    }

@app.get("/notifications/{customer_id}")
async def get_customer_notifications(customer_id: str, limit: int = 10):
    """Get recent notifications for a customer"""
    async with webhook_processor._get_redis() as redis:
        notifications = await redis.lrange(f"notifications:{customer_id}", 0, limit - 1)
        return {
            "customer_id": customer_id,
            "notifications": [json.loads(n) for n in notifications]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006) 