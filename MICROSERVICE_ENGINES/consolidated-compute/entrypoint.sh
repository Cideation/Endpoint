#!/bin/bash

# Consolidated Compute Cluster Entrypoint
# Starts DAG Alpha (5000), SFDE Engine (5003), and Graph Runtime (5004)

set -e

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to start a service
start_service() {
    local service_name=$1
    local port=$2
    local module=$3
    
    log "Starting $service_name on port $port"
    
    # Start service in background
    cd /app/services
    python -c "
import sys
import os
sys.path.insert(0, '/app/shared')
sys.path.insert(0, '/app/services')

# Import the specific service module
if '$module' == 'dag_alpha':
    from dag_alpha import app
elif '$module' == 'sfde_engine':
    from sfde_engine import app  
elif '$module' == 'graph_runtime':
    from graph_runtime import app
else:
    raise ValueError('Unknown module: $module')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=$port, debug=False, threaded=True)
" &
    
    local pid=$!
    echo $pid > /tmp/${service_name}.pid
    log "$service_name started with PID $pid"
}

# Function to check if service is healthy
check_health() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    log "Checking health for $service_name on port $port"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:$port/health > /dev/null 2>&1; then
            log "$service_name is healthy"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service_name not ready yet"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log "ERROR: $service_name failed to become healthy"
    return 1
}

# Function to cleanup on exit
cleanup() {
    log "Shutting down services..."
    
    # Kill all services
    for service in dag_alpha sfde_engine graph_runtime; do
        if [ -f /tmp/${service}.pid ]; then
            local pid=$(cat /tmp/${service}.pid)
            if kill -0 $pid > /dev/null 2>&1; then
                log "Stopping $service (PID: $pid)"
                kill $pid
                wait $pid 2>/dev/null || true
            fi
            rm -f /tmp/${service}.pid
        fi
    done
    
    log "All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

log "=== BEM Compute Cluster Starting ==="
log "Container role: $NODE_ROLE"
log "Cache enabled: $CACHE_ENABLED"
log "Redis URL: $REDIS_URL"

# Wait for dependencies
if [ "$CACHE_ENABLED" = "true" ] && [ -n "$REDIS_URL" ]; then
    log "Waiting for Redis cache..."
    until redis-cli -u $REDIS_URL ping > /dev/null 2>&1; do
        log "Redis not ready, waiting..."
        sleep 2
    done
    log "Redis cache connected"
fi

# Start all compute services
start_service "dag_alpha" 5000 "dag_alpha"
start_service "sfde_engine" 5003 "sfde_engine" 
start_service "graph_runtime" 5004 "graph_runtime"

# Wait for services to become healthy
sleep 5

check_health "dag_alpha" 5000
check_health "sfde_engine" 5003
check_health "graph_runtime" 5004

log "=== All Compute Services Ready ==="
log "DAG Alpha:     http://localhost:5000"
log "SFDE Engine:   http://localhost:5003" 
log "Graph Runtime: http://localhost:5004"

# Keep container running and monitor processes
while true; do
    # Check if any service died
    for service in dag_alpha sfde_engine graph_runtime; do
        if [ -f /tmp/${service}.pid ]; then
            local pid=$(cat /tmp/${service}.pid)
            if ! kill -0 $pid > /dev/null 2>&1; then
                log "ERROR: $service (PID: $pid) has died"
                cleanup
            fi
        fi
    done
    
    sleep 10
done 