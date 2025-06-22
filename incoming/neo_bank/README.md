# Neo Bank - Contribution Routing & SPV Management System

## 🏦 System Overview

**"We don't custody anything — we just route verified contributions, and if they meet conditions, we release the home title held by the SPV held by NEOBANKS."**

This system enables:
- **Agent-Led Contributions**: Each agent/COOP registers their own fintech API (GCash, PayMaya, etc.)
- **Group Funding**: Users contribute as a group under pre-agreed rules
- **Automated Release**: When conditions are met, home titles are released from SPV custody
- **Programmable Logic**: Like "Pag-IBIG Fund meets GCash Save" but fully programmable

## 📁 Directory Structure

```
neo_bank/
├── core/
│   ├── paluwagan_engine.py      # Main contribution routing engine
│   ├── neobank_adapter.py       # NEOBANK SPV interface
│   └── spv_manager.py           # SPV lifecycle management
│
├── api/
│   └── service.py               # Main API layer (Flask)
│
├── data/
│   ├── spv_registry.json        # SPV definitions and status
│   ├── contributions.json       # All contribution records
│   ├── trigger_log.json         # Release trigger events
│   └── agent_integrations.json  # Agent fintech API connections
│
├── config/
│   └── bank_api.yaml           # System configuration
│
└── tests/
    └── run_all_tests.py        # Comprehensive test suite
```

## 🔄 How It Works

### 1. Agent Registration
Agents and COOPs register their own fintech API connections:

```json
{
  "agent_001": {
    "gcash_merchant_id": "M123456",
    "gcash_api_key": "sk_live_xxx",
    "gcash_webhook_url": "https://agent001.mycoop.org/gcash/webhook"
  }
}
```

### 2. SPV Creation
Properties are registered with funding targets and release conditions:

```json
{
  "property_info": {
    "property_id": "PROP-001",
    "address": "123 Sample Street, Makati City",
    "estimated_value": 2500000.0,
    "title_number": "TCT-12345"
  },
  "config": {
    "target_amount": 500000.0,
    "required_participants": 10,
    "funding_deadline": "2024-12-31T23:59:59"
  }
}
```

### 3. Contribution Flow
1. **User contributes** through agent's fintech platform (GCash, PayMaya, etc.)
2. **Agent's system** verifies payment and submits to Neo Bank API
3. **Paluwagan Engine** routes contribution to appropriate SPV
4. **SPV Manager** tracks progress toward release conditions
5. **Auto-release** triggers when conditions are met

### 4. Title Release
When SPV conditions are met:
- System triggers release through NEOBANK adapter
- Home title custody is transferred
- All stakeholders are notified

## 🚀 API Endpoints

### Accept Contribution
```bash
POST /api/v1/contribution
{
  "agent_id": "agent_001",
  "spv_id": "SPV-DEMO001",
  "amount": 5000.0,
  "currency": "PHP",
  "fintech_provider": "GCash",
  "transaction_ref": "GC-TXN-123456789"
}
```

### Check SPV Status
```bash
GET /api/v1/spv/SPV-DEMO001/status
```

### Trigger Release
```bash
POST /api/v1/spv/SPV-DEMO001/release
{
  "manual_override": false,
  "authorized_by": "system"
}
```

### Register Agent API
```bash
POST /api/v1/agent/register
{
  "agent_id": "agent_001",
  "fintech_provider": "GCash",
  "api_credentials": {
    "gcash_merchant_id": "M123456",
    "gcash_api_key": "sk_live_xxx"
  }
}
```

## 🎯 Key Features

### For NEOBANK
- **Simple Interface**: Just log, verify, and release titles
- **No Custody Risk**: Contributions flow directly between users and SPVs
- **Automated Compliance**: Pre-programmed release conditions
- **Audit Trail**: Complete transaction and trigger logging

### For Agents/COOPs
- **Own API Control**: Register and manage their own fintech connections
- **Flexible Rules**: Configure SPV conditions per property/group
- **Real-time Status**: Track contribution progress and release conditions
- **Multiple Providers**: Support GCash, PayMaya, Maya, and others

### For Users
- **Group Funding**: Contribute collectively toward property acquisition
- **Transparent Progress**: See real-time funding status
- **Automated Release**: No manual intervention when conditions met
- **Familiar Payment**: Use existing fintech apps (GCash, etc.)

## 🔧 Installation & Setup

### 1. Install Dependencies
```bash
pip install flask requests pyyaml
```

### 2. Configure Environment
```bash
export NEOBANK_API_KEY=your_neobank_key
export NEOBANK_API_SECRET=your_neobank_secret
export ENCRYPTION_KEY=your_encryption_key
```

### 3. Start API Service
```bash
cd incoming/neo_bank
python api/service.py
```

### 4. Run Tests
```bash
python tests/run_all_tests.py
```

## 📊 Example Scenario

**Manila Housing Cooperative** wants to help members acquire homes:

1. **COOP registers** their PayMaya merchant account with Neo Bank
2. **Property owner** creates SPV for ₱2.5M house (target: ₱500K down payment)
3. **10 members** contribute ₱50K each through COOP's PayMaya integration
4. **System verifies** all contributions and tracks progress
5. **Auto-release** triggers when ₱500K target reached
6. **NEOBANK releases** title from SPV custody to new owners

## 🔒 Security & Compliance

- **API Key Authentication**: Secure agent registration
- **Encrypted Credentials**: All fintech API keys encrypted at rest
- **Audit Logging**: Complete transaction trail
- **Rate Limiting**: Prevent abuse and ensure stability
- **Webhook Verification**: Validate fintech provider callbacks

## 🎮 Demo Data

The system includes demo data showing:
- **4 registered agents** with different fintech providers
- **1 active SPV** for a Makati property
- **Sample contributions** from various agents
- **Trigger events** and release scenarios

## 🚀 Production Deployment

For production use:
1. **Enable NEOBANK adapter** with real credentials
2. **Configure SSL/TLS** for all API communications
3. **Set up monitoring** and alerting
4. **Implement backup** and disaster recovery
5. **Configure rate limiting** and security policies

---

**Summary**: This system enables programmable, agent-led group funding for property acquisition with automated title release through NEOBANK SPV custody. It's like "Pag-IBIG Fund meets GCash Save" but fully customizable and agent-controlled.
