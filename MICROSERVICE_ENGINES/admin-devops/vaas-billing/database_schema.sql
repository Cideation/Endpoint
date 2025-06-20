-- VaaS Billing System Database Schema
-- PostgreSQL database schema for emergence billing and customer management

-- =============================================================================
-- CUSTOMERS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS customers (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    credit_balance INTEGER DEFAULT 100,
    total_spent DECIMAL(10,2) DEFAULT 0.00,
    signup_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tier_usage JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(status);
CREATE INDEX IF NOT EXISTS idx_customers_last_activity ON customers(last_activity);

-- =============================================================================
-- EMERGENCE EVENTS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS emergence_events (
    id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) NOT NULL REFERENCES customers(id),
    project_id VARCHAR(36) NOT NULL,
    node_id VARCHAR(36) NOT NULL,
    emergence_type VARCHAR(50) NOT NULL, -- blueprint_package, bom_with_suppliers, etc.
    fit_score DECIMAL(3,2) NOT NULL,
    emergence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'emergence_detected',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_emergence_customer ON emergence_events(customer_id);
CREATE INDEX IF NOT EXISTS idx_emergence_project ON emergence_events(project_id);
CREATE INDEX IF NOT EXISTS idx_emergence_type ON emergence_events(emergence_type);
CREATE INDEX IF NOT EXISTS idx_emergence_status ON emergence_events(status);
CREATE INDEX IF NOT EXISTS idx_emergence_timestamp ON emergence_events(emergence_timestamp);

-- =============================================================================
-- BILLING TRANSACTIONS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS billing_transactions (
    id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) NOT NULL REFERENCES customers(id),
    emergence_event_id VARCHAR(36) NOT NULL REFERENCES emergence_events(id),
    tier_id VARCHAR(50) NOT NULL,
    credits_charged INTEGER NOT NULL,
    fiat_equivalent DECIMAL(10,2) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending',
    receipt_url TEXT,
    refund_amount DECIMAL(10,2) DEFAULT 0.00,
    refund_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_billing_customer ON billing_transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_billing_emergence ON billing_transactions(emergence_event_id);
CREATE INDEX IF NOT EXISTS idx_billing_tier ON billing_transactions(tier_id);
CREATE INDEX IF NOT EXISTS idx_billing_status ON billing_transactions(status);
CREATE INDEX IF NOT EXISTS idx_billing_timestamp ON billing_transactions(timestamp);

-- =============================================================================
-- CREDIT TOPUPS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS credit_topups (
    id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) NOT NULL REFERENCES customers(id),
    credits_purchased INTEGER NOT NULL,
    amount_paid DECIMAL(10,2) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    payment_processor_id TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending',
    failure_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_topups_customer ON credit_topups(customer_id);
CREATE INDEX IF NOT EXISTS idx_topups_status ON credit_topups(status);
CREATE INDEX IF NOT EXISTS idx_topups_timestamp ON credit_topups(timestamp);

-- =============================================================================
-- USAGE ANALYTICS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS usage_analytics (
    id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) NOT NULL REFERENCES customers(id),
    event_type VARCHAR(50) NOT NULL, -- 'emergence_check', 'billing_trigger', 'credit_topup', etc.
    event_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_analytics_customer ON usage_analytics(customer_id);
CREATE INDEX IF NOT EXISTS idx_analytics_type ON usage_analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON usage_analytics(timestamp);

-- =============================================================================
-- TIER USAGE HISTORY
-- =============================================================================

CREATE TABLE IF NOT EXISTS tier_usage_history (
    id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) NOT NULL REFERENCES customers(id),
    tier_id VARCHAR(50) NOT NULL,
    usage_count INTEGER DEFAULT 1,
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_credits_spent INTEGER DEFAULT 0,
    total_fiat_spent DECIMAL(10,2) DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(customer_id, tier_id)
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_tier_usage_customer ON tier_usage_history(customer_id);
CREATE INDEX IF NOT EXISTS idx_tier_usage_tier ON tier_usage_history(tier_id);

-- =============================================================================
-- ADMIN/CRM VIEWS
-- =============================================================================

-- Customer summary view
CREATE OR REPLACE VIEW customer_summary AS
SELECT 
    c.id,
    c.email,
    c.name,
    c.credit_balance,
    c.total_spent,
    c.signup_date,
    c.last_activity,
    COUNT(bt.id) as total_transactions,
    COUNT(DISTINCT ee.emergence_type) as unique_tiers_used,
    COALESCE(SUM(bt.credits_charged), 0) as lifetime_credits_spent,
    CASE 
        WHEN c.last_activity > NOW() - INTERVAL '7 days' THEN 'active'
        WHEN c.last_activity > NOW() - INTERVAL '30 days' THEN 'dormant'
        ELSE 'inactive'
    END as activity_status
FROM customers c
LEFT JOIN billing_transactions bt ON c.id = bt.customer_id AND bt.status = 'completed'
LEFT JOIN emergence_events ee ON bt.emergence_event_id = ee.id
GROUP BY c.id, c.email, c.name, c.credit_balance, c.total_spent, c.signup_date, c.last_activity;

-- Revenue analytics view
CREATE OR REPLACE VIEW revenue_analytics AS
SELECT 
    DATE_TRUNC('day', bt.timestamp) as day,
    COUNT(*) as transactions_count,
    SUM(bt.fiat_equivalent) as daily_revenue,
    SUM(bt.credits_charged) as credits_charged,
    COUNT(DISTINCT bt.customer_id) as unique_customers,
    STRING_AGG(DISTINCT bt.tier_id, ',') as tiers_purchased
FROM billing_transactions bt
WHERE bt.status = 'completed'
GROUP BY DATE_TRUNC('day', bt.timestamp)
ORDER BY day DESC;

-- Tier popularity view
CREATE OR REPLACE VIEW tier_popularity AS
SELECT 
    tier_id,
    COUNT(*) as purchase_count,
    SUM(fiat_equivalent) as tier_revenue,
    AVG(fiat_equivalent) as avg_price,
    COUNT(DISTINCT customer_id) as unique_customers
FROM billing_transactions
WHERE status = 'completed'
GROUP BY tier_id
ORDER BY purchase_count DESC;

-- =============================================================================
-- FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Function to update customer total_spent
CREATE OR REPLACE FUNCTION update_customer_spent()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'completed' AND (OLD.status IS NULL OR OLD.status != 'completed') THEN
        UPDATE customers 
        SET total_spent = total_spent + NEW.fiat_equivalent,
            updated_at = NOW()
        WHERE id = NEW.customer_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update customer spent amount
DROP TRIGGER IF EXISTS trigger_update_customer_spent ON billing_transactions;
CREATE TRIGGER trigger_update_customer_spent
    AFTER INSERT OR UPDATE ON billing_transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_customer_spent();

-- Function to update tier usage history
CREATE OR REPLACE FUNCTION update_tier_usage()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'completed' THEN
        INSERT INTO tier_usage_history (customer_id, tier_id, usage_count, total_credits_spent, total_fiat_spent)
        VALUES (NEW.customer_id, NEW.tier_id, 1, NEW.credits_charged, NEW.fiat_equivalent)
        ON CONFLICT (customer_id, tier_id) 
        DO UPDATE SET
            usage_count = tier_usage_history.usage_count + 1,
            last_used = NOW(),
            total_credits_spent = tier_usage_history.total_credits_spent + NEW.credits_charged,
            total_fiat_spent = tier_usage_history.total_fiat_spent + NEW.fiat_equivalent,
            updated_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update tier usage
DROP TRIGGER IF EXISTS trigger_update_tier_usage ON billing_transactions;
CREATE TRIGGER trigger_update_tier_usage
    AFTER INSERT OR UPDATE ON billing_transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_tier_usage();

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert default emergence tier configurations
-- This is mainly for reference; actual config is in the application code 