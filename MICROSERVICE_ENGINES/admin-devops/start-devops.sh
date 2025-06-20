#!/bin/bash

# BEM System DevOps Stack Startup Script
# Manages admin/monitoring containers and VaaS billing infrastructure

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 BEM DevOps + VaaS Billing Stack Manager${NC}"
echo "================================================"

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
    mkdir -p vaas-billing/templates vaas-billing/static
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

    # Prometheus config with VaaS metrics
    if [ ! -f "prometheus/prometheus.yml" ]; then
        cat > prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'vaas-billing'
    static_configs:
      - targets: ['vaas-billing:8004']
    metrics_path: '/metrics'

  - job_name: 'vaas-crm'
    static_configs:
      - targets: ['vaas-crm:8005']
    metrics_path: '/metrics'

  - job_name: 'vaas-webhook'
    static_configs:
      - targets: ['vaas-webhook:8006']
    metrics_path: '/metrics'

  - job_name: 'bem-ecm'
    static_configs:
      - targets: ['ecm-gateway:8765']
    metrics_path: '/metrics'

  - job_name: 'bem-aa'
    static_configs:
      - targets: ['automated-admin:8003']
    metrics_path: '/metrics'
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
    echo "2) Start VaaS billing stack (Billing, CRM, Webhook + Database)"
    echo "3) Start complete stack (Core + VaaS + Debug tools)"
    echo "4) Start with debug profile (includes Loki)"
    echo "5) Start with public profile (includes Traefik)"
    echo "6) Stop all services"
    echo "7) View service status"
    echo "8) View logs"
    echo "9) Open service URLs"
    echo "10) Initialize VaaS database"
    echo "11) Exit"
    echo ""
}

# Start core services
start_core() {
    echo -e "${GREEN}🚀 Starting core DevOps services...${NC}"
    docker-compose up -d portainer prometheus grafana swagger-ui node-exporter cadvisor
    echo -e "${GREEN}✅ Core services started!${NC}"
    show_urls
}

# Start VaaS billing stack
start_vaas() {
    echo -e "${PURPLE}💰 Starting VaaS billing stack...${NC}"
    docker-compose up -d postgres-vaas redis-vaas vaas-billing vaas-crm vaas-webhook
    
    # Wait for database to be ready
    echo -e "${YELLOW}⏳ Waiting for VaaS database to be ready...${NC}"
    sleep 10
    
    echo -e "${GREEN}✅ VaaS billing stack started!${NC}"
    show_vaas_urls
}

# Start complete stack
start_complete() {
    echo -e "${GREEN}🚀 Starting complete DevOps + VaaS stack...${NC}"
    docker-compose up -d
    
    # Wait for services to start
    echo -e "${YELLOW}⏳ Waiting for all services to start...${NC}"
    sleep 15
    
    echo -e "${GREEN}✅ Complete stack started!${NC}"
    show_all_urls
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
    echo "4) VaaS Billing"
    echo "5) VaaS CRM"
    echo "6) VaaS Webhook"
    echo "7) VaaS Database"
    echo "8) All services"
    read -p "Enter choice: " log_choice
    
    case $log_choice in
        1) docker-compose logs -f portainer ;;
        2) docker-compose logs -f prometheus ;;
        3) docker-compose logs -f grafana ;;
        4) docker-compose logs -f vaas-billing ;;
        5) docker-compose logs -f vaas-crm ;;
        6) docker-compose logs -f vaas-webhook ;;
        7) docker-compose logs -f postgres-vaas ;;
        8) docker-compose logs -f ;;
        *) echo "Invalid choice" ;;
    esac
}

# Show core service URLs
show_urls() {
    echo ""
    echo -e "${GREEN}📌 Core Service URLs:${NC}"
    echo "  • Portainer:   http://localhost:9000"
    echo "  • Prometheus:  http://localhost:9090"
    echo "  • Grafana:     http://localhost:3000 (admin/admin)"
    echo "  • Swagger UI:  http://localhost:8080"
    echo "  • Node Metrics: http://localhost:9100/metrics"
    echo "  • cAdvisor:    http://localhost:8082"
}

# Show VaaS service URLs
show_vaas_urls() {
    echo ""
    echo -e "${PURPLE}💰 VaaS Billing Service URLs:${NC}"
    echo "  • Billing API:     http://localhost:8004"
    echo "  • CRM Dashboard:   http://localhost:8005"
    echo "  • Webhook Endpoint: http://localhost:8006"
    echo "  • PostgreSQL:      localhost:5433 (bem_vaas/bem_user)"
    echo "  • Redis:           localhost:6380"
    echo ""
    echo -e "${BLUE}🎯 Emergence Billing Features:${NC}"
    echo "  • Freemium exploration (no cost for testing)"
    echo "  • Value-triggered billing (only when outputs are actionable)"
    echo "  • Real-time emergence detection"
    echo "  • Customer credit management"
    echo "  • Transaction history and analytics"
}

# Show all service URLs
show_all_urls() {
    show_urls
    show_vaas_urls
    
    if docker ps | grep -q bem-loki; then
        echo "  • Loki:        http://localhost:3100"
    fi
    if docker ps | grep -q bem-traefik; then
        echo "  • Traefik:     http://localhost:8081"
    fi
    echo ""
}

# Initialize VaaS database
init_vaas_database() {
    echo -e "${PURPLE}🗄️ Initializing VaaS database...${NC}"
    
    # Start database if not running
    docker-compose up -d postgres-vaas
    
    # Wait for database to be ready
    echo -e "${YELLOW}⏳ Waiting for database to be ready...${NC}"
    sleep 5
    
    # Run database initialization
    docker-compose exec postgres-vaas psql -U bem_user -d bem_vaas -c "
        SELECT 'Database initialized successfully!' as status;
        SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public';
    "
    
    echo -e "${GREEN}✅ VaaS database initialized!${NC}"
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
        2) start_vaas ;;
        3) start_complete ;;
        4) start_debug ;;
        5) start_public ;;
        6) stop_all ;;
        7) show_status ;;
        8) show_logs ;;
        9) show_all_urls ;;
        10) init_vaas_database ;;
        11) echo -e "${GREEN}👋 Goodbye!${NC}"; exit 0 ;;
        *) echo -e "${RED}Invalid option. Please try again.${NC}" ;;
    esac
done 