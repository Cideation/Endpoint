#!/bin/bash

echo "🎯 CAD Parser Phase 2 - Microservices Development Environment"
echo "=========================================================="

# Check if services are running
echo "🔍 Checking service status..."

# Check PostgreSQL
if pg_isready -h postgres -p 5432 > /dev/null 2>&1; then
    echo "✅ PostgreSQL: Running"
else
    echo "❌ PostgreSQL: Not running"
fi

# Check Redis
if redis-cli -h redis ping > /dev/null 2>&1; then
    echo "✅ Redis: Running"
else
    echo "❌ Redis: Not running"
fi

# Display available services
echo ""
echo "🚀 Available Services:"
echo "  • CAD Parser API:     http://localhost:5000"
echo "  • Data Processor:     http://localhost:5001"
echo "  • Gateway:           http://localhost:8080"
echo "  • Admin UI:          http://localhost:3000"
echo "  • PostgreSQL:        localhost:5432"
echo "  • Redis:             localhost:6379"

echo ""
echo "🔧 Development Commands:"
echo "  • Start all services:    ./scripts/dev.sh"
echo "  • Python tests:          pytest"
echo "  • Format code:           black . && isort ."
echo "  • Lint code:             flake8"
echo "  • Node.js tests:         cd gateway && npm test"

echo ""
echo "📁 Project Structure:"
echo "  • /services/           - Python microservices"
echo "  • /gateway/            - Node.js API gateway"
echo "  • /admin-ui/           - Node.js admin interface"
echo "  • /shared/             - Shared utilities"
echo "  • /scripts/            - Development scripts"
echo "  • /tests/              - Test files"
echo "  • /docs/               - Documentation"

echo ""
echo "🎯 Phase 2 Architecture:"
echo "  • Python containers for CAD processing"
echo "  • Node.js containers for UI and gateway"
echo "  • PostgreSQL for data storage"
echo "  • Redis for caching and messaging"
echo "  • Docker Compose for orchestration"

echo ""
echo "✅ Development environment ready!" 