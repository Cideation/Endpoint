# CAD Parser Phase 2 - Microservices Development Environment

## 🎯 Overview

This dev container setup provides a complete microservices development environment for Phase 2 of the CAD Parser project. The architecture consists of Python containers for CAD processing and Node.js containers for UI and gateway services.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Admin UI      │    │     Gateway     │    │  CAD Parser     │
│   (Node.js)     │◄──►│   (Node.js)     │◄──►│   (Python)      │
│   Port: 3000    │    │   Port: 8080    │    │   Port: 5000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   PostgreSQL    │    │  Data Processor │
                       │   Port: 5432    │    │   (Python)      │
                       └─────────────────┘    │   Port: 5001    │
                                              └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Redis       │
                       │   Port: 6379    │
                       └─────────────────┘
```

## 🚀 Services

### Python Microservices
- **CAD Parser Service** (`:5000`) - Handles CAD file parsing and processing
- **Data Processor Service** (`:5001`) - Handles data transformation and analysis

### Node.js Services
- **API Gateway** (`:8080`) - Routes requests to appropriate microservices
- **Admin UI** (`:3000`) - Web interface for managing the system

### Infrastructure
- **PostgreSQL** (`:5432`) - Primary database for data storage
- **Redis** (`:6379`) - Caching and message queue

## 🛠️ Getting Started

### 1. Open in Dev Container
1. Open this project in VS Code or PyCharm
2. When prompted, click "Reopen in Container"
3. Wait for the container to build and start

### 2. Start All Services
```bash
./scripts/dev.sh
```

### 3. Verify Services
```bash
# Check service health
curl http://localhost:5000/health  # CAD Parser
curl http://localhost:5001/health  # Data Processor
curl http://localhost:8080/health  # Gateway
curl http://localhost:3000/health  # Admin UI
```

## 📁 Project Structure

```
/workspace/
├── services/              # Python microservices
│   ├── cad-parser/       # CAD parsing service
│   └── data-processor/   # Data processing service
├── gateway/              # Node.js API gateway
├── admin-ui/             # Node.js admin interface
├── shared/               # Shared utilities and models
├── scripts/              # Development and deployment scripts
├── tests/                # Test files
└── docs/                 # Documentation
```

## 🔧 Development Commands

### Python Development
```bash
# Run tests
pytest

# Format code
black . && isort .

# Lint code
flake8

# Install dependencies
pip install -r requirements.txt
```

### Node.js Development
```bash
# Gateway
cd gateway
npm install
npm run dev

# Admin UI
cd admin-ui
npm install
npm run dev
```

### Docker Commands
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild services
docker-compose build --no-cache
```

## 🌐 Service Endpoints

### CAD Parser API
- `POST /parse` - Parse CAD files
- `GET /health` - Health check

### Data Processor
- `POST /process` - Process data
- `GET /health` - Health check

### Gateway
- `POST /api/parse` - Route to CAD parser
- `POST /api/process` - Route to data processor
- `GET /health` - Health check

### Admin UI
- `GET /` - Main interface
- `GET /health` - Health check

## 🔍 Monitoring

### Service Status
```bash
# Check all services
docker-compose ps

# View service logs
docker-compose logs [service-name]
```

### Database Access
```bash
# Connect to PostgreSQL
psql -h localhost -p 5432 -U postgres -d cad_parser

# Connect to Redis
redis-cli -h localhost -p 6379
```

## 🧪 Testing

### Python Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific service tests
pytest tests/services/
```

### Node.js Tests
```bash
# Gateway tests
cd gateway && npm test

# Admin UI tests
cd admin-ui && npm test
```

## 📊 Performance

### Resource Usage
```bash
# Monitor container resources
docker stats

# Check service performance
htop
```

### Scaling
```bash
# Scale services
docker-compose up -d --scale cad-parser=3
docker-compose up -d --scale data-processor=2
```

## 🔒 Security

- All services run in isolated containers
- Network communication is restricted to internal Docker network
- Environment variables for sensitive configuration
- HTTPS endpoints for production deployment

## 🚀 Deployment

### Development
```bash
./scripts/dev.sh
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📝 Notes

- All services are automatically restarted on failure
- Logs are collected and can be viewed with `docker-compose logs`
- The development environment includes hot-reloading for both Python and Node.js
- Database migrations are handled automatically on startup

## 🆘 Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 3000, 5000, 5001, 5432, 6379, 8080 are available
2. **Service not starting**: Check logs with `docker-compose logs [service-name]`
3. **Database connection issues**: Ensure PostgreSQL is running and accessible
4. **Node.js dependency issues**: Run `npm install` in respective service directories

### Reset Environment
```bash
# Stop and remove all containers
docker-compose down -v

# Rebuild from scratch
docker-compose build --no-cache

# Start fresh
./scripts/dev.sh
``` 