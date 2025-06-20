#!/bin/bash

# BEM System DevOps Stack Startup Script
# Manages admin/monitoring containers

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 BEM DevOps Stack Manager${NC}"
echo "================================"

# Function to check if docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
}

# Function to create required directories
create_directories() {
    echo -e "${YELLOW}📁 Creating required directories...${NC}"
    mkdir -p prometheus grafana/provisioning/datasources grafana/provisioning/dashboards grafana/dashboards
    mkdir -p loki promtail traefik/dynamic swagger certs
}

# Function to generate default configs if missing
generate_configs() {
    # Grafana datasource for Prometheus
    if [ ! -f "grafana/provisioning/datasources/prometheus.yml" ]; then
        cat > grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    fi

    # Grafana dashboard provisioning
    if [ ! -f "grafana/provisioning/dashboards/dashboards.yml" ]; then
        cat > grafana/provisioning/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'BEM System'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    fi

    # Loki config
    if [ ! -f "loki/loki-config.yml" ]; then
        cat > loki/loki-config.yml << EOF
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://localhost:9093
EOF
    fi

    # Promtail config
    if [ ! -f "promtail/promtail-config.yml" ]; then
        cat > promtail/promtail-config.yml << EOF
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log
    pipeline_stages:
      - json:
          expressions:
            output: log
            stream: stream
            attrs:
      - json:
          expressions:
            tag:
          source: attrs
      - regex:
          expression: (?P<container_name>(?:[^|]*))\|(?P<image_name>(?:[^|]*))
          source: tag
      - timestamp:
          format: RFC3339Nano
          source: time
      - labels:
          stream:
          container_name:
          image_name:
      - output:
          source: output
EOF
    fi

    # Traefik config
    if [ ! -f "traefik/traefik.yml" ]; then
        cat > traefik/traefik.yml << EOF
api:
  dashboard: true
  debug: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
  file:
    directory: /etc/traefik/dynamic
    watch: true

log:
  level: INFO

accessLog: {}
EOF
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Select an option:"
    echo "1) Start core services (Portainer, Prometheus, Grafana, Swagger)"
    echo "2) Start all services (including Loki, Traefik)"
    echo "3) Start with debug profile (includes Loki)"
    echo "4) Start with public profile (includes Traefik)"
    echo "5) Stop all services"
    echo "6) View service status"
    echo "7) View logs"
    echo "8) Open service URLs"
    echo "9) Exit"
    echo ""
}

# Start core services
start_core() {
    echo -e "${GREEN}🚀 Starting core DevOps services...${NC}"
    docker-compose up -d portainer prometheus grafana swagger-ui node-exporter cadvisor
    echo -e "${GREEN}✅ Core services started!${NC}"
    show_urls
}

# Start all services
start_all() {
    echo -e "${GREEN}🚀 Starting all DevOps services...${NC}"
    docker-compose --profile debug --profile public up -d
    echo -e "${GREEN}✅ All services started!${NC}"
    show_urls
}

# Start with debug profile
start_debug() {
    echo -e "${GREEN}🚀 Starting services with debug profile...${NC}"
    docker-compose --profile debug up -d
    echo -e "${GREEN}✅ Debug services started!${NC}"
    show_urls
}

# Start with public profile
start_public() {
    echo -e "${GREEN}🚀 Starting services with public profile...${NC}"
    docker-compose --profile public up -d
    echo -e "${GREEN}✅ Public services started!${NC}"
    show_urls
}

# Stop all services
stop_all() {
    echo -e "${YELLOW}🛑 Stopping all DevOps services...${NC}"
    docker-compose --profile debug --profile public down
    echo -e "${GREEN}✅ All services stopped!${NC}"
}

# Show service status
show_status() {
    echo -e "${YELLOW}📊 Service Status:${NC}"
    docker-compose --profile debug --profile public ps
}

# Show logs
show_logs() {
    echo "Select service to view logs:"
    echo "1) Portainer"
    echo "2) Prometheus"
    echo "3) Grafana"
    echo "4) Swagger UI"
    echo "5) All services"
    read -p "Enter choice: " log_choice
    
    case $log_choice in
        1) docker-compose logs -f portainer ;;
        2) docker-compose logs -f prometheus ;;
        3) docker-compose logs -f grafana ;;
        4) docker-compose logs -f swagger-ui ;;
        5) docker-compose logs -f ;;
        *) echo "Invalid choice" ;;
    esac
}

# Show service URLs
show_urls() {
    echo ""
    echo -e "${GREEN}📌 Service URLs:${NC}"
    echo "  • Portainer:   http://localhost:9000"
    echo "  • Prometheus:  http://localhost:9090"
    echo "  • Grafana:     http://localhost:3000 (admin/admin)"
    echo "  • Swagger UI:  http://localhost:8080"
    echo "  • Node Metrics: http://localhost:9100/metrics"
    echo "  • cAdvisor:    http://localhost:8082"
    if docker ps | grep -q bem-loki; then
        echo "  • Loki:        http://localhost:3100"
    fi
    if docker ps | grep -q bem-traefik; then
        echo "  • Traefik:     http://localhost:8081"
    fi
    echo ""
}

# Main execution
check_docker
create_directories
generate_configs

while true; do
    show_menu
    read -p "Enter your choice: " choice
    
    case $choice in
        1) start_core ;;
        2) start_all ;;
        3) start_debug ;;
        4) start_public ;;
        5) stop_all ;;
        6) show_status ;;
        7) show_logs ;;
        8) show_urls ;;
        9) echo -e "${GREEN}👋 Goodbye!${NC}"; exit 0 ;;
        *) echo -e "${RED}Invalid option. Please try again.${NC}" ;;
    esac
done 