# BEM System Admin/DevOps Infrastructure

This directory contains the administrative and monitoring infrastructure for the BEM system, providing comprehensive observability, management, and documentation capabilities.

## 🚀 Quick Start

```bash
# Start core monitoring services
./start-devops.sh
# Select option 1 for core services
```

## 📦 Container Stack

### Core Services (Always Recommended)

| Container | Purpose | Port | URL |
|-----------|---------|------|-----|
| **Portainer** | Visual Docker management | 9000 | http://localhost:9000 |
| **Prometheus** | Metrics collection | 9090 | http://localhost:9090 |
| **Grafana** | Dashboards & visualization | 3000 | http://localhost:3000 |
| **Swagger UI** | API documentation | 8080 | http://localhost:8080 |

### Optional Services

| Container | Purpose | Port | Profile |
|-----------|---------|------|---------|
| **Loki** | Log aggregation | 3100 | debug |
| **Promtail** | Log collector | - | debug |
| **Traefik** | Reverse proxy | 80, 443, 8081 | public |

### Support Services

| Container | Purpose | Port |
|-----------|---------|------|
| **Node Exporter** | System metrics | 9100 |
| **cAdvisor** | Container metrics | 8082 |

## 🔧 Configuration

### Directory Structure
```
admin-devops/
├── docker-compose.yml      # Main compose file
├── start-devops.sh         # Interactive startup script
├── prometheus/
│   ├── prometheus.yml      # Prometheus config
│   └── alerts.yml          # Alert rules
├── grafana/
│   ├── provisioning/       # Auto-provisioning
│   └── dashboards/         # Custom dashboards
├── loki/
│   └── loki-config.yml     # Loki configuration
├── promtail/
│   └── promtail-config.yml # Log collection config
└── traefik/
    └── traefik.yml         # Reverse proxy config
```

### Service Profiles

1. **Core Profile** (Default)
   - Essential monitoring and management
   - Minimal resource usage
   - Recommended for development

2. **Debug Profile**
   - Includes Loki for log aggregation
   - Useful for troubleshooting
   - Higher resource usage

3. **Public Profile**
   - Includes Traefik for external access
   - SSL/TLS termination
   - Production-ready

## 📊 Monitoring Features

### Prometheus Metrics
- Container health and resource usage
- Application-specific metrics
- Custom business metrics
- Alert rules for critical conditions

### Grafana Dashboards
- System overview
- Container performance
- Application metrics
- Custom visualizations

### Available Metrics Endpoints
- ECM Gateway: `:8765/metrics`
- AA Service: `:8003/metrics`
- Microservices: `:800X/metrics`
- Node Exporter: `:9100/metrics`
- cAdvisor: `:8082/metrics`

## 🔐 Security Considerations

1. **Portainer**: Change default admin password on first login
2. **Grafana**: Default login is `admin/admin` - change immediately
3. **Prometheus**: No authentication by default - use reverse proxy
4. **Traefik**: Configure SSL certificates for production

## 🚦 Usage Examples

### Start Services
```bash
# Interactive mode
./start-devops.sh

# Direct commands
docker-compose up -d                          # Core services only
docker-compose --profile debug up -d          # With logging
docker-compose --profile public up -d         # With reverse proxy
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f grafana
```

### Stop Services
```bash
docker-compose down
docker-compose --profile debug --profile public down  # Stop all
```

## 📈 Grafana Dashboard Setup

1. Access Grafana at http://localhost:3000
2. Login with `admin/admin`
3. Prometheus datasource is auto-configured
4. Import dashboards:
   - Node Exporter Full: Dashboard ID `1860`
   - Docker Container: Dashboard ID `193`
   - Prometheus Stats: Dashboard ID `2`

## 🔍 Prometheus Queries

Useful queries for monitoring:

```promql
# Container CPU usage
rate(container_cpu_usage_seconds_total[5m])

# Memory usage
container_memory_usage_bytes

# HTTP request rate
rate(http_requests_total[5m])

# WebSocket connections
websocket_connections_active

# Error rate
rate(http_requests_total{status=~"5.."}[5m])
```

## 🐛 Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs <service-name>

# Verify network
docker network ls

# Check volumes
docker volume ls
```

### Can't access service
- Ensure ports aren't already in use
- Check firewall settings
- Verify Docker network connectivity

### High resource usage
- Adjust retention periods in Prometheus
- Limit log verbosity
- Use profiles to disable unnecessary services

## 🔗 Integration with Main System

The DevOps stack connects to the main BEM network to monitor services:

```yaml
networks:
  bem-network:
    external: true
```

Ensure the main application stack is running and the network exists:
```bash
docker network create bem-network
```

## 📝 Notes

- All data is persisted in Docker volumes
- Configurations are mounted as read-only
- Services auto-restart unless stopped
- Use profiles to control which services start

## 🆘 Support

For issues or questions:
1. Check service logs
2. Verify configurations
3. Ensure Docker resources are adequate
4. Review alert rules in Prometheus 