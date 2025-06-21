# Render Deployment Guide - BEM System
## Account: jpc@homeqube.com | HomeQube Company Pte. Ltd.

### 🚀 Quick Deployment Commands

#### 1. **Connect Repository to Render**
```bash
# Repository: https://github.com/Cideation/Endpoint.git
# Account: jpc@homeqube.com
# Branch: main (auto-deploy enabled)
```

#### 2. **Service Deployment Order**
```bash
# Deploy in this sequence:
1. PostgreSQL Database → bem-postgresql
2. Redis Cache → bem-redis  
3. ECM Gateway → bem-ecm-gateway
4. GraphQL Engine → bem-graphql-engine
5. Frontend Static → bem-frontend
```

#### 3. **Environment Configuration**
```bash
# Required Environment Variables:
DATABASE_URL=postgresql://bem_user:${POSTGRES_PASSWORD}@bem-postgresql:5432/bem_production
REDIS_URL=redis://bem-redis:6379
GRAPHQL_PORT=8004
FRONTEND_PORT=8005
ENVIRONMENT=production
ENABLE_CORS=true
```

### 🔧 Render Platform Communication

#### **Dashboard Access**
- **URL**: https://dashboard.render.com
- **Account**: jpc@homeqube.com
- **Organization**: HomeQube Company Pte. Ltd.

#### **Service Management**
```bash
# Service Names (in Render dashboard):
- bem-postgresql (Database)
- bem-redis (Cache)
- bem-ecm-gateway (WebSocket Handler)
- bem-graphql-engine (Real-time API)
- bem-frontend (Static Site)
```

#### **Deployment Triggers**
```bash
# Auto-deploy on Git push:
git push origin main
# → Triggers automatic deployment to all services

# Manual deploy via Render CLI:
render deploy --service=bem-graphql-engine
render deploy --service=bem-frontend
```

### 📊 Service Configuration

#### **GraphQL Engine Service**
```yaml
name: bem-graphql-engine
type: web
env: python
buildCommand: pip install -r requirements_realtime.txt
startCommand: python frontend/graphql_realtime_engine.py
plan: starter
region: oregon
healthCheckPath: /health
autoDeploy: true
```

#### **Frontend Static Service**
```yaml
name: bem-frontend
type: static
staticPublishPath: ./frontend
buildCommand: echo "Static frontend build"
autoDeploy: true
customHeaders:
  - path: /*
    name: X-Frame-Options
    value: DENY
```

#### **ECM Gateway Service**
```yaml
name: bem-ecm-gateway
type: web
env: python
buildCommand: pip install -r requirements.txt
startCommand: python Final_Phase/ecm_gateway.py
plan: standard
healthCheckPath: /health
autoDeploy: true
```

### 🔐 Security & Access

#### **Account Security**
- **2FA Enabled**: Recommended for jpc@homeqube.com
- **API Keys**: Generate in Render dashboard → Account Settings
- **Webhook Secrets**: Configure for GitHub integration

#### **Database Security**
```bash
# PostgreSQL Access:
- IP Allowlist: Configure in dashboard
- SSL: Enabled by default
- Backup: Automatic daily backups
```

### 💰 Cost Management

#### **Current Plan Estimates**
```bash
# Monthly Costs:
PostgreSQL (Starter): $7/month
Redis: $7/month
ECM Gateway (Standard): $25/month
GraphQL Engine (Starter): $7/month
Frontend (Static): Free
Total: ~$46/month
```

#### **Scaling Options**
```bash
# Production Scaling:
PostgreSQL → Standard ($20/month)
Services → Standard ($25/month each)
Estimated Production: ~$95/month
```

### 🛠️ Troubleshooting

#### **Common Issues**
```bash
# Service won't start:
1. Check logs in Render dashboard
2. Verify environment variables
3. Check health check endpoint

# Database connection issues:
1. Verify DATABASE_URL format
2. Check PostgreSQL service status
3. Test connection from service logs

# WebSocket issues:
1. Ensure ECM Gateway is on Standard plan
2. Check CORS configuration
3. Verify WebSocket upgrade headers
```

#### **Support Contacts**
```bash
# Render Support:
- Dashboard: https://dashboard.render.com/support
- Email: support@render.com
- Account: jpc@homeqube.com

# BEM System:
- Repository: https://github.com/Cideation/Endpoint
- Issues: GitHub Issues tab
```

### 📈 Monitoring & Health Checks

#### **Service Health URLs**
```bash
# Health Check Endpoints:
https://bem-graphql-engine.onrender.com/health
https://bem-ecm-gateway.onrender.com/health
https://bem-frontend.onrender.com/

# Monitoring:
- Render Dashboard → Service Metrics
- Uptime monitoring via health checks
- Error alerts via email to jpc@homeqube.com
```

#### **Performance Metrics**
```bash
# Key Metrics to Monitor:
- Response time < 200ms
- Uptime > 99.5%
- Memory usage < 80%
- Database connections < 100
```

### 🚀 Deployment Automation

#### **GitHub Actions Integration**
```yaml
# .github/workflows/render-deploy.yml
name: Deploy to Render
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Render
        run: |
          curl -X POST "https://api.render.com/v1/services/srv-xxx/deploys" \
            -H "Authorization: Bearer ${{ secrets.RENDER_API_KEY }}"
```

#### **Webhook Configuration**
```bash
# Render Webhook URL (configure in GitHub):
https://api.render.com/postreceive/github/srv-xxx

# Auto-deploy settings:
- Branch: main
- Auto-deploy: Enabled
- Build on PR: Enabled
```

---

**Account Holder**: jpc@homeqube.com  
**Organization**: HomeQube Company Pte. Ltd.  
**Repository**: https://github.com/Cideation/Endpoint  
**Last Updated**: December 2024 