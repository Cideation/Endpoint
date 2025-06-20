# 💡 BEM VaaS (Value-as-a-Service) Billing System

## Emergence Billing Model

The BEM platform implements **Emergence Billing** - a revolutionary usage-based pricing model where fees are triggered only when the system reaches high-value, actionable states through computation, simulation, or agent coordination.

### 🎯 Core Philosophy

**Users are not charged for exploration, setup, or iteration—but are billed at the moment when emergent outputs become deployable, buildable, or fundable.**

### ✅ Benefits of Emergence Billing

- **🔁 Aligned to Value, Not Usage** — No charge until the system delivers real outcomes
- **🧠 Tied to System Intelligence** — Billing only occurs when emergent logic passes thresholds  
- **📦 Modular** — Different emergence types can be billed separately
- **💵 Transparent** — Users know exactly what they're paying for, and when

---

## 🏗️ Architecture Overview

### Service Components

| Service | Port | Purpose |
|---------|------|---------|
| **VaaS Billing API** | 8004 | Core emergence detection and billing logic |
| **CRM Dashboard** | 8005 | Customer management and analytics interface |
| **Webhook Service** | 8006 | Real-time emergence event processing |
| **PostgreSQL** | 5433 | Customer data and transaction storage |
| **Redis** | 6380 | Caching and real-time notifications |

### Integration Points

```mermaid
graph TD
    A[BEM Main System] -->|Node Updates| B[Webhook Service]
    B -->|Emergence Detection| C[Billing API]
    C -->|Customer Lookup| D[PostgreSQL]
    C -->|Notifications| E[Redis]
    F[CRM Dashboard] -->|Analytics| D
    G[Admin Users] -->|Management| F
    H[Customers] -->|Self-Service| C
```

---

## 🎯 Emergence Billing Tiers

### 📐 Blueprint Package (50 credits / ₱500)
- **Trigger**: `geometry + specs resolved`
- **Includes**: DXF files, IFC models, PDF specifications
- **Use Case**: Finalized CAD drawings ready for construction

### 📦 BOM + Suppliers (30 credits / ₱250)  
- **Trigger**: `material + component logic complete`
- **Includes**: Bill of Materials JSON, verified supplier list
- **Use Case**: Complete component sourcing for procurement

### ✅ Compliance Package (35 credits / ₱300)
- **Trigger**: `region + compliance functors resolved`
- **Includes**: Zoning compliance, code validation, certifications
- **Use Case**: Regulatory approval documentation

### 💰 Investment Report (60 credits / ₱700)
- **Trigger**: `investment_node resolved + project_roi complete`
- **Includes**: ROI summary, cost breakdown, market analysis
- **Use Case**: Funding and investment decisions

### 🏗️ Full Emergence Bundle (120 credits / ₱1,499)
- **Trigger**: `all modules above complete`
- **Includes**: Complete package of all above tiers
- **Use Case**: Ready-to-build project with full documentation

---

## 🔧 Technical Implementation

### Emergence Detection Logic

```python
def check_emergence(node_data, fit_score):
    """
    Core emergence detection algorithm
    """
    # Basic conditions
    if fit_score < 0.90:
        return False
    
    if not node_data.get("finalized"):
        return False
        
    if node_data.get("emergence_status") != "ready":
        return False
    
    # Tier-specific conditions
    if (node_data.get("geometry_resolved") and 
        node_data.get("specs_complete")):
        return "blueprint_package"
    
    # ... additional tier logic
```

### Billing Trigger Flow

1. **Node State Update** → System detects completion
2. **Emergence Check** → Validates conditions are met  
3. **Customer Lookup** → Verifies credit balance
4. **Notification Sent** → Customer approves billing
5. **Credits Deducted** → Transaction processed
6. **Output Generated** → Deliverable provided

### Database Schema

```sql
-- Core customer table
CREATE TABLE customers (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    credit_balance INTEGER DEFAULT 100,
    total_spent DECIMAL(10,2) DEFAULT 0.00,
    -- ... additional fields
);

-- Emergence events tracking
CREATE TABLE emergence_events (
    id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) REFERENCES customers(id),
    emergence_type VARCHAR(50) NOT NULL,
    fit_score DECIMAL(3,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'emergence_detected',
    -- ... additional fields
);

-- Billing transactions
CREATE TABLE billing_transactions (
    id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) REFERENCES customers(id),
    emergence_event_id VARCHAR(36) REFERENCES emergence_events(id),
    credits_charged INTEGER NOT NULL,
    fiat_equivalent DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    -- ... additional fields
);
```

---

## 🚀 Quick Start

### 1. Start VaaS Services

```bash
cd MICROSERVICE_ENGINES/admin-devops
./start-devops.sh
# Select option 2: Start VaaS billing stack
```

### 2. Initialize Database

```bash
# Select option 10: Initialize VaaS database
```

### 3. Access Services

- **CRM Dashboard**: http://localhost:8005
- **Billing API**: http://localhost:8004  
- **Webhook Endpoint**: http://localhost:8006
- **API Documentation**: http://localhost:8080

### 4. Test Emergence Detection

```bash
curl -X POST http://localhost:8004/emergence/check \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "test-project",
    "node_id": "test-node", 
    "fit_score": 0.95,
    "node_data": {
      "finalized": true,
      "emergence_status": "ready",
      "geometry_resolved": true,
      "specs_complete": true
    }
  }'
```

---

## 📊 Credit System

### Default Configuration

- **Signup Bonus**: 100 credits (₱1,000 value)
- **Credit Value**: ₱10 per credit
- **Minimum Balance**: 0 credits (no overdraft)

### Top-up Options

| Credits | Price | Bonus |
|---------|-------|-------|
| 100 | ₱1,000 | 0% |
| 250 | ₱2,400 | 4% savings |
| 500 | ₱4,500 | 10% savings |

### Usage Tracking

```bash
# Get customer balance
curl http://localhost:8004/customers/{customer_id}/balance

# Process credit top-up
curl -X POST http://localhost:8004/customers/{customer_id}/topup \
  -d "credits=100&amount_paid=1000"
```

---

## 🔗 Integration with Main BEM System

### Webhook Integration

The VaaS system receives real-time events from the main BEM system:

```python
# In your BEM system
def notify_node_completion(node_id, fit_score, node_data):
    webhook_payload = {
        "event_type": "node_update",
        "data": {
            "node_id": node_id,
            "fit_score": fit_score,
            "node_data": node_data,
            # ... additional fields
        }
    }
    
    requests.post(
        "http://localhost:8006/webhook/emergence",
        json=webhook_payload
    )
```

### ECM Pulse Integration

```python
# Pulse events trigger emergence checks
def handle_investment_pulse(pulse_data):
    if pulse_data.get("roi_complete"):
        check_emergence_for_investment_tier(pulse_data["target_node"])
```

---

## 📈 Analytics & CRM

### Customer Metrics

- **Lifetime Value**: Total spending per customer
- **Tier Preferences**: Most used emergence types
- **Activity Patterns**: Usage frequency and timing
- **Credit Utilization**: Balance vs. spending patterns

### Revenue Analytics

- **Daily/Monthly Revenue**: Trend analysis
- **Tier Performance**: Which tiers generate most revenue
- **Customer Segments**: Active vs. dormant users
- **Conversion Rates**: Free exploration to paid emergence

### Dashboard Features

- Real-time revenue tracking
- Customer activity monitoring  
- Tier usage analytics
- Transaction history
- Credit balance management
- Automated billing insights

---

## 🔒 Security & Compliance

### Data Protection

- **Encryption**: All sensitive data encrypted at rest and in transit
- **Access Control**: Role-based permissions for admin functions
- **Audit Logging**: Complete transaction audit trail
- **Privacy**: GDPR-compliant customer data handling

### Financial Security

- **PCI Compliance**: Secure payment processing
- **Fraud Detection**: Automated suspicious activity monitoring
- **Refund Management**: Automated refund processing
- **Receipt Generation**: Detailed transaction receipts

---

## 🧪 Testing

### Run Integration Tests

```bash
cd vaas-billing
pytest integration_test.py -v --asyncio-mode=auto
```

### Test Emergence Detection

```bash
# Test blueprint emergence
python -c "
from main import EmergenceDetector
detector = EmergenceDetector()
# ... test logic
"
```

### Load Testing

```bash
# Test concurrent emergence checks
python integration_test.py::TestPerformance::test_concurrent_emergence_checks
```

---

## 📚 API Reference

### Core Endpoints

#### Emergence Detection
```
POST /emergence/check
```
Check if node state triggers billing

#### Billing Processing  
```
POST /billing/process
```
Process emergence billing transaction

#### Customer Management
```
GET /customers/{id}/balance
POST /customers/{id}/topup
```

#### Analytics
```
GET /api/metrics
GET /api/tier-analytics
```

### Webhook Endpoints

#### Node Updates
```
POST /webhook/node-update
```

#### Pulse Events
```
POST /webhook/pulse
```

#### Emergence Events
```
POST /webhook/emergence
```

---

## 🚀 Deployment

### Local Development

```bash
docker-compose up -d
```

### Production Deployment

See `../deploy/` directory for production configurations:

- **Render.com**: Fully managed deployment
- **AWS**: Container-based deployment  
- **Docker Swarm**: Self-hosted orchestration

### Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port
DEBUG=false
WEBHOOK_SECRET=your-webhook-secret
PAYMENT_PROCESSOR_KEY=your-payment-key
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/emergence-enhancement`)
3. Commit changes (`git commit -am 'Add emergence enhancement'`)
4. Push to branch (`git push origin feature/emergence-enhancement`) 
5. Create Pull Request

### Development Guidelines

- Follow emergence billing principles
- Maintain test coverage above 90%
- Document all API changes
- Ensure backward compatibility
- Test with real emergence scenarios

---

## 📄 License

This VaaS billing system is part of the BEM platform and follows the same licensing terms.

---

## 📞 Support

For VaaS billing support:

1. Check the logs: `docker-compose logs vaas-billing`
2. Review emergence conditions in code
3. Test with known good emergence data  
4. Verify customer credit balances
5. Check webhook integration status

### Common Issues

**Q: Emergence not detected despite high fit score**
A: Verify `node_data.finalized = true` and `emergence_status = "ready"`

**Q: Billing transaction failed**  
A: Check customer credit balance and emergence event status

**Q: Webhook not receiving events**
A: Verify network connectivity and webhook URL configuration 