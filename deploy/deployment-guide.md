# BEM System Git-Powered CI/CD Deployment Guide

## 🚀 Overview

The BEM (Building Environment Management) System uses **Git as the primary driver for continuous integration and deployment (CI/CD)**, enabling seamless production updates and automated deployments.

## 📋 Architecture

### Production Components
- **Behavior-Driven AC System** (Port 8003)
- **ECM Gateway** (WebSocket - Ports 8765/8766)
- **Dual AC API Server** (Port 8002)
- **Frontend Web Interface** (Ports 80/443)
- **PostgreSQL Database** (Port 5432)
- **Microservice Engines** (Ports 8004-8006)
- **Monitoring Stack** (Port 9090)

### Git-Powered CI/CD Flow
```
Git Push → GitHub Actions → Build → Test → Deploy → Monitor
```

## 🔧 Deployment Steps

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Cideation/Endpoint.git
cd Endpoint-1

# Create production environment file
cp deploy/.env.example .env

# Edit environment variables
vim .env
```

### 2. Environment Variables

Create `.env` file with:
```bash
# Database
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=bem_production
POSTGRES_USER=bem_user

# SSL Configuration
SSL_CERT_PATH=/etc/ssl/certs/cert.pem
SSL_KEY_PATH=/etc/ssl/certs/key.pem

# Application URLs
API_URL=https://your-domain.com/api
BEHAVIOR_AC_URL=https://your-domain.com/behavior
ECM_WEBSOCKET_URL=wss://your-domain.com/ws

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
```

### 3. SSL Certificate Setup

```bash
# Generate SSL certificates (or use existing)
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes

# Or copy existing certificates
cp /path/to/existing/cert.pem ssl/
cp /path/to/existing/key.pem ssl/
```

### 4. Production Deployment

```bash
# Deploy using Docker Compose
docker-compose -f deploy/production.yml up -d

# Check service status
docker-compose -f deploy/production.yml ps

# View logs
docker-compose -f deploy/production.yml logs -f
```

### 5. Health Checks

```bash
# Test all endpoints
curl -f http://localhost:8003/behavior_analytics  # Behavior AC
curl -f http://localhost:8002/health              # API Server
curl -f http://localhost/health                   # Frontend

# Test WebSocket connections
wscat -c ws://localhost:8765  # ECM Gateway
wscat -c ws://localhost:8766  # Pulse System
```

## 🔄 CI/CD Pipeline

### Automated Deployment Triggers

1. **Push to main branch** → Full production deployment
2. **Push to develop branch** → Staging deployment
3. **Pull request** → Testing and validation only

### Pipeline Stages

#### Stage 1: Code Quality
- ✅ Code formatting (Black, isort)
- ✅ Linting (flake8)
- ✅ Security scanning (bandit, safety)

#### Stage 2: Component Testing
- ✅ Phase 2 integration tests
- ✅ Phase 3 production tests
- ✅ Behavior-driven AC tests
- ✅ Recent commits validation

#### Stage 3: Microservice Testing
- ✅ Docker image builds
- ✅ Service health checks
- ✅ Integration testing

#### Stage 4: Frontend & AC Testing
- ✅ AA behavioral classification
- ✅ Dynamic UI spawning
- ✅ Frontend validation

#### Stage 5: Database Testing
- ✅ Schema validation
- ✅ Migration testing
- ✅ DGL training database

#### Stage 6: Deployment
- ✅ Staging deployment
- ✅ Smoke tests
- ✅ Production deployment
- ✅ Post-deployment verification

#### Stage 7: Monitoring
- ✅ Performance baselines
- ✅ Security monitoring
- ✅ Health checks

## 📊 Monitoring & Observability

### Health Endpoints
- `/health` - Service health status
- `/metrics` - Prometheus metrics
- `/behavior_analytics` - AA behavioral insights

### Log Aggregation
```bash
# View application logs
docker-compose logs -f behavior-ac
docker-compose logs -f ecm-gateway
docker-compose logs -f dual-ac-api

# Monitor system performance
docker stats
```

### Performance Monitoring
- **API Response Time**: <100ms target
- **AA Classification**: <500ms target
- **WebSocket Latency**: <50ms target

## 🛡️ Security

### Implemented Security Features
- ✅ Rate limiting (30-60 requests/minute)
- ✅ CORS configuration
- ✅ SSL/TLS encryption
- ✅ Container security (non-root users)
- ✅ Database connection security

### Security Monitoring
- Automated vulnerability scanning
- SSL certificate monitoring
- Security policy enforcement

## 🔧 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8003
lsof -i :8003

# Kill conflicting processes
pkill -f behavior_driven_ac.py
```

#### Database Connection Issues
```bash
# Test database connectivity
docker exec -it bem-postgres psql -U bem_user -d bem_production

# Check database health
docker-compose exec postgres pg_isready -U bem_user
```

#### WebSocket Connection Failures
```bash
# Check WebSocket service status
docker logs bem-ecm-gateway

# Test WebSocket connectivity
nc -z localhost 8765
nc -z localhost 8766
```

### Recovery Procedures

#### Service Restart
```bash
# Restart specific service
docker-compose restart behavior-ac

# Restart all services
docker-compose -f deploy/production.yml restart
```

#### Database Recovery
```bash
# Backup database
docker exec bem-postgres pg_dump -U bem_user bem_production > backup.sql

# Restore from backup
docker exec -i bem-postgres psql -U bem_user bem_production < backup.sql
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale behavior AC service
docker-compose -f deploy/production.yml up -d --scale behavior-ac=3

# Scale API server
docker-compose -f deploy/production.yml up -d --scale dual-ac-api=2
```

### Load Balancing
- Nginx reverse proxy included
- Automatic service discovery
- Health check-based routing

## 🎯 Production Readiness Checklist

- ✅ SSL certificates configured
- ✅ Environment variables set
- ✅ Database migrations applied
- ✅ Health checks passing
- ✅ Monitoring active
- ✅ Backup procedures tested
- ✅ Security scans clean
- ✅ Performance baselines established

## 🚀 Go Live

```bash
# Final deployment command
docker-compose -f deploy/production.yml up -d

# Verify all systems
./scripts/health-check.sh

# 🎉 BEM System is LIVE!
```

---

## 📞 Support

For deployment issues or questions:
- Check GitHub Actions logs
- Review container logs
- Consult this deployment guide
- Monitor system health endpoints

**Git-Powered CI/CD**: Every commit triggers automated validation and deployment, ensuring continuous delivery and system reliability. 