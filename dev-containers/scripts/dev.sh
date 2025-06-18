#!/bin/bash

echo "🚀 Starting CAD Parser Phase 2 - Microservices Development Environment"
echo "=================================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ] && [ ! -f ".devcontainer/docker-compose.yml" ]; then
    echo "❌ No docker-compose.yml found. Please run this from the project root."
    exit 1
fi

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $port is already in use. Please stop the service using this port."
        return 1
    fi
    return 0
}

# Check all required ports
echo "🔍 Checking port availability..."
ports=(3000 5001 5002 5434 6380 8080)
for port in "${ports[@]}"; do
    if ! check_port $port; then
        echo "❌ Cannot start services. Please free up port $port."
        exit 1
    fi
done

echo "✅ All ports are available!"

# Start services using docker-compose
echo ""
echo "📦 Starting infrastructure services..."

# Start PostgreSQL and Redis first
if [ -f ".devcontainer/docker-compose.yml" ]; then
    docker-compose -f .devcontainer/docker-compose.yml up -d postgres redis
else
    docker-compose up -d postgres redis
fi

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker-compose -f .devcontainer/docker-compose.yml exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
    echo "   Waiting for PostgreSQL..."
    sleep 2
done
echo "✅ PostgreSQL is ready!"

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
until docker-compose -f .devcontainer/docker-compose.yml exec -T redis redis-cli ping > /dev/null 2>&1; do
    echo "   Waiting for Redis..."
    sleep 1
done
echo "✅ Redis is ready!"

echo ""
echo "📦 Starting Python microservices..."

# Start Python services
if [ -f ".devcontainer/docker-compose.yml" ]; then
    docker-compose -f .devcontainer/docker-compose.yml up -d cad-parser data-processor
else
    docker-compose up -d cad-parser data-processor
fi

echo ""
echo "📦 Starting Node.js services..."

# Start Node.js services
if [ -f ".devcontainer/docker-compose.yml" ]; then
    docker-compose -f .devcontainer/docker-compose.yml up -d gateway admin-ui
else
    docker-compose up -d gateway admin-ui
fi

echo ""
echo "⏳ Waiting for all services to be ready..."
sleep 5

# Check service health
echo ""
echo "🔍 Checking service health..."

services=(
    "http://localhost:5001/health"
    "http://localhost:5002/health"
    "http://localhost:8080/health"
    "http://localhost:3000/health"
)

for service in "${services[@]}"; do
    if curl -s "$service" > /dev/null; then
        echo "✅ $service - Healthy"
    else
        echo "❌ $service - Not responding"
    fi
done

echo ""
echo "🎉 CAD Parser Phase 2 Development Environment is ready!"
echo ""
echo "🌐 Available Services:"
echo "  • CAD Parser API:     http://localhost:5001"
echo "  • Data Processor:     http://localhost:5002"
echo "  • API Gateway:        http://localhost:8080"
echo "  • Admin UI:           http://localhost:3000"
echo "  • PostgreSQL:         localhost:5434"
echo "  • Redis:              localhost:6380"
echo ""
echo "🔧 Useful Commands:"
echo "  • View logs:          docker-compose logs -f"
echo "  • Stop services:      docker-compose down"
echo "  • Restart services:   docker-compose restart"
echo "  • Check status:       docker-compose ps"
echo ""
echo "🚀 Happy coding! 🚀" 