#!/bin/bash

# Consolidated Logic Cluster Entrypoint
# Starts Functor Types (5001), Callback Engine (5002), and Optimization Engine (5005)

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
if '$module' == 'functor_types':
    from functor_types import app
elif '$module' == 'callback_engine':
    from callback_engine import app  
elif '$module' == 'optimization_engine':
    from optimization_engine import app
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
    local max_attempts=20
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
    log "Shutting down logic services..."
    
    # Kill all services
    for service in functor_types callback_engine optimization_engine; do
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
    
    log "All logic services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

log "=== BEM Logic Cluster Starting ==="
log "Container role: $NODE_ROLE"
log "Data persistence: $DATA_PERSISTENCE"

# Start all logic services
start_service "functor_types" 5001 "functor_types"
start_service "callback_engine" 5002 "callback_engine" 
start_service "optimization_engine" 5005 "optimization_engine"

# Wait for services to become healthy
sleep 5

check_health "functor_types" 5001
check_health "callback_engine" 5002
check_health "optimization_engine" 5005

log "=== All Logic Services Ready ==="
log "Functor Types:     http://localhost:5001"
log "Callback Engine:   http://localhost:5002" 
log "Optimization:      http://localhost:5005"

# Keep container running and monitor processes
while true; do
    # Check if any service died
    for service in functor_types callback_engine optimization_engine; do
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