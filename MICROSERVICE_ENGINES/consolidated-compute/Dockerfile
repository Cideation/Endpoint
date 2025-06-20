FROM python:3.11-slim

# Consolidated Compute Cluster
# Combines: DAG Alpha, SFDE Engine, Graph Runtime Engine
# Optimized for high-performance mathematical computations

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements for all services
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy service implementations
COPY services/ /app/services/
COPY shared/ /app/shared/
COPY entrypoint.sh /app/

# Copy original service files for compatibility
COPY ../ne-dag-alpha/main.py /app/services/dag_alpha.py
COPY ../sfde/main.py /app/services/sfde_engine.py  
COPY ../ne-graph-runtime-engine/main.py /app/services/graph_runtime.py
COPY ../sfde_utility_foundation_extended.py /app/shared/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Create cache directory
RUN mkdir -p /cache

# Expose service ports
EXPOSE 5000 5003 5004

# Environment variables
ENV PYTHONPATH=/app/shared:/app/services:/shared
ENV NODE_ROLE=compute
ENV CACHE_ENABLED=true
ENV FLASK_ENV=production

# Health check for all services
HEALTHCHECK --interval=30s --timeout=15s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health && \
        curl -f http://localhost:5003/health && \
        curl -f http://localhost:5004/health

# Start all services via supervisor-like entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 