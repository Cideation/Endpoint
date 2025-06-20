"""
VaaS CRM Dashboard
Customer Relationship Management interface for VaaS billing system
Provides analytics, customer management, and billing oversight
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import asyncpg
from contextlib import asynccontextmanager

# =============================================================================
# DATA MODELS FOR CRM
# =============================================================================

class CustomerSummary(BaseModel):
    id: str
    email: str
    name: str
    credit_balance: int
    total_spent: Decimal
    signup_date: datetime
    last_activity: datetime
    total_transactions: int
    unique_tiers_used: int
    lifetime_credits_spent: int
    activity_status: str

class RevenueMetrics(BaseModel):
    daily_revenue: Decimal
    monthly_revenue: Decimal
    total_customers: int
    active_customers: int
    avg_customer_value: Decimal
    top_tier: str

class TierAnalytics(BaseModel):
    tier_id: str
    purchase_count: int
    tier_revenue: Decimal
    avg_price: Decimal
    unique_customers: int

class CustomerActivity(BaseModel):
    customer_id: str
    recent_emergences: List[Dict]
    recent_transactions: List[Dict]
    usage_patterns: Dict

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

# =============================================================================
# CRM SERVICE
# =============================================================================

class CRMService:
    """Core CRM functionality for customer and billing management"""
    
    async def get_customer_summaries(self, limit: int = 50, offset: int = 0) -> List[CustomerSummary]:
        """Get paginated customer summaries"""
        async with get_db_connection() as conn:
            query = """
            SELECT * FROM customer_summary
            ORDER BY last_activity DESC
            LIMIT $1 OFFSET $2
            """
            rows = await conn.fetch(query, limit, offset)
            
            return [
                CustomerSummary(
                    id=row["id"],
                    email=row["email"],
                    name=row["name"],
                    credit_balance=row["credit_balance"],
                    total_spent=row["total_spent"],
                    signup_date=row["signup_date"],
                    last_activity=row["last_activity"],
                    total_transactions=row["total_transactions"],
                    unique_tiers_used=row["unique_tiers_used"],
                    lifetime_credits_spent=row["lifetime_credits_spent"],
                    activity_status=row["activity_status"]
                )
                for row in rows
            ]
    
    async def get_revenue_metrics(self) -> RevenueMetrics:
        """Get overall revenue and customer metrics"""
        async with get_db_connection() as conn:
            # Daily revenue
            daily_query = """
            SELECT COALESCE(SUM(fiat_equivalent), 0) as daily_revenue
            FROM billing_transactions
            WHERE DATE(timestamp) = CURRENT_DATE AND status = 'completed'
            """
            daily_result = await conn.fetchrow(daily_query)
            
            # Monthly revenue
            monthly_query = """
            SELECT COALESCE(SUM(fiat_equivalent), 0) as monthly_revenue
            FROM billing_transactions
            WHERE timestamp >= DATE_TRUNC('month', CURRENT_DATE) AND status = 'completed'
            """
            monthly_result = await conn.fetchrow(monthly_query)
            
            # Customer metrics
            customer_query = """
            SELECT 
                COUNT(*) as total_customers,
                COUNT(CASE WHEN last_activity > NOW() - INTERVAL '7 days' THEN 1 END) as active_customers,
                COALESCE(AVG(total_spent), 0) as avg_customer_value
            FROM customers
            """
            customer_result = await conn.fetchrow(customer_query)
            
            # Top tier
            tier_query = """
            SELECT tier_id
            FROM tier_popularity
            ORDER BY purchase_count DESC
            LIMIT 1
            """
            tier_result = await conn.fetchrow(tier_query)
            
            return RevenueMetrics(
                daily_revenue=daily_result["daily_revenue"],
                monthly_revenue=monthly_result["monthly_revenue"],
                total_customers=customer_result["total_customers"],
                active_customers=customer_result["active_customers"],
                avg_customer_value=customer_result["avg_customer_value"],
                top_tier=tier_result["tier_id"] if tier_result else "None"
            )
    
    async def get_tier_analytics(self) -> List[TierAnalytics]:
        """Get tier performance analytics"""
        async with get_db_connection() as conn:
            query = "SELECT * FROM tier_popularity ORDER BY purchase_count DESC"
            rows = await conn.fetch(query)
            
            return [
                TierAnalytics(
                    tier_id=row["tier_id"],
                    purchase_count=row["purchase_count"],
                    tier_revenue=row["tier_revenue"],
                    avg_price=row["avg_price"],
                    unique_customers=row["unique_customers"]
                )
                for row in rows
            ]
    
    async def get_customer_activity(self, customer_id: str) -> CustomerActivity:
        """Get detailed customer activity and usage patterns"""
        async with get_db_connection() as conn:
            # Recent emergence events
            emergence_query = """
            SELECT emergence_type, fit_score, emergence_timestamp, status
            FROM emergence_events
            WHERE customer_id = $1
            ORDER BY emergence_timestamp DESC
            LIMIT 10
            """
            emergence_rows = await conn.fetch(emergence_query, customer_id)
            
            # Recent transactions
            transaction_query = """
            SELECT tier_id, credits_charged, fiat_equivalent, timestamp, status
            FROM billing_transactions
            WHERE customer_id = $1
            ORDER BY timestamp DESC
            LIMIT 10
            """
            transaction_rows = await conn.fetch(transaction_query, customer_id)
            
            # Usage patterns
            patterns_query = """
            SELECT tier_id, usage_count, total_credits_spent, total_fiat_spent
            FROM tier_usage_history
            WHERE customer_id = $1
            """
            patterns_rows = await conn.fetch(patterns_query, customer_id)
            
            return CustomerActivity(
                customer_id=customer_id,
                recent_emergences=[dict(row) for row in emergence_rows],
                recent_transactions=[dict(row) for row in transaction_rows],
                usage_patterns={row["tier_id"]: dict(row) for row in patterns_rows}
            )
    
    async def search_customers(self, query: str) -> List[CustomerSummary]:
        """Search customers by email or name"""
        async with get_db_connection() as conn:
            search_query = """
            SELECT * FROM customer_summary
            WHERE email ILIKE $1 OR name ILIKE $1
            ORDER BY last_activity DESC
            LIMIT 20
            """
            rows = await conn.fetch(search_query, f"%{query}%")
            
            return [
                CustomerSummary(
                    id=row["id"],
                    email=row["email"],
                    name=row["name"],
                    credit_balance=row["credit_balance"],
                    total_spent=row["total_spent"],
                    signup_date=row["signup_date"],
                    last_activity=row["last_activity"],
                    total_transactions=row["total_transactions"],
                    unique_tiers_used=row["unique_tiers_used"],
                    lifetime_credits_spent=row["lifetime_credits_spent"],
                    activity_status=row["activity_status"]
                )
                for row in rows
            ]
    
    async def get_revenue_timeline(self, days: int = 30) -> List[Dict]:
        """Get daily revenue for the past N days"""
        async with get_db_connection() as conn:
            query = """
            SELECT day, daily_revenue, transactions_count, unique_customers
            FROM revenue_analytics
            WHERE day >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY day DESC
            """ % days
            
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

# =============================================================================
# FASTAPI CRM APPLICATION
# =============================================================================

app = FastAPI(
    title="BEM VaaS CRM Dashboard",
    description="Customer Relationship Management for VaaS Billing",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize CRM service
crm_service = CRMService()

# =============================================================================
# WEB ROUTES (HTML PAGES)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main CRM dashboard page"""
    # Get summary metrics for dashboard
    metrics = await crm_service.get_revenue_metrics()
    tier_analytics = await crm_service.get_tier_analytics()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "metrics": metrics,
        "tier_analytics": tier_analytics
    })

@app.get("/customers", response_class=HTMLResponse)
async def customers_page(request: Request, page: int = 1, search: str = ""):
    """Customer management page"""
    limit = 20
    offset = (page - 1) * limit
    
    if search:
        customers = await crm_service.search_customers(search)
    else:
        customers = await crm_service.get_customer_summaries(limit, offset)
    
    return templates.TemplateResponse("customers.html", {
        "request": request,
        "customers": customers,
        "current_page": page,
        "search_query": search
    })

@app.get("/customer/{customer_id}", response_class=HTMLResponse)
async def customer_detail(request: Request, customer_id: str):
    """Individual customer detail page"""
    activity = await crm_service.get_customer_activity(customer_id)
    
    return templates.TemplateResponse("customer_detail.html", {
        "request": request,
        "customer_id": customer_id,
        "activity": activity
    })

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics and reporting page"""
    metrics = await crm_service.get_revenue_metrics()
    tier_analytics = await crm_service.get_tier_analytics()
    revenue_timeline = await crm_service.get_revenue_timeline(30)
    
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "metrics": metrics,
        "tier_analytics": tier_analytics,
        "revenue_timeline": revenue_timeline
    })

# =============================================================================
# API ROUTES (JSON DATA)
# =============================================================================

@app.get("/api/metrics")
async def get_metrics():
    """Get revenue and customer metrics"""
    return await crm_service.get_revenue_metrics()

@app.get("/api/customers")
async def get_customers(page: int = 1, limit: int = 20, search: str = ""):
    """Get customer list with pagination"""
    offset = (page - 1) * limit
    
    if search:
        customers = await crm_service.search_customers(search)
    else:
        customers = await crm_service.get_customer_summaries(limit, offset)
    
    return {"customers": customers, "page": page, "limit": limit}

@app.get("/api/customer/{customer_id}")
async def get_customer_activity_api(customer_id: str):
    """Get customer activity data"""
    return await crm_service.get_customer_activity(customer_id)

@app.get("/api/tier-analytics")
async def get_tier_analytics_api():
    """Get tier performance analytics"""
    return await crm_service.get_tier_analytics()

@app.get("/api/revenue-timeline")
async def get_revenue_timeline_api(days: int = 30):
    """Get revenue timeline data"""
    return await crm_service.get_revenue_timeline(days)

@app.post("/api/customer/{customer_id}/credit-adjustment")
async def adjust_customer_credits(customer_id: str, credits: int, reason: str):
    """Manually adjust customer credit balance (admin function)"""
    async with get_db_connection() as conn:
        # Update customer balance
        update_query = """
        UPDATE customers 
        SET credit_balance = credit_balance + $2, updated_at = NOW()
        WHERE id = $1
        """
        await conn.execute(update_query, customer_id, credits)
        
        # Log the adjustment
        log_query = """
        INSERT INTO usage_analytics (customer_id, event_type, event_data)
        VALUES ($1, 'credit_adjustment', $2)
        """
        await conn.execute(log_query, customer_id, json.dumps({
            "credits_adjusted": credits,
            "reason": reason,
            "admin_action": True
        }))
    
    return {"status": "success", "message": f"Adjusted {credits} credits for customer {customer_id}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vaas-crm", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 