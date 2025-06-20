FROM python:3.11-slim

# Consolidated Logic Cluster  
# Combines: Functor Types, Callback Engine, Optimization Engine
# Optimized for business logic and type management

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy service implementations
COPY services/ /app/services/
COPY shared/ /app/shared/
COPY entrypoint.sh /app/

# Copy original service files for compatibility
COPY ../ne-functor-types/main.py /app/services/functor_types.py
COPY ../ne-callback-engine/main.py /app/services/callback_engine.py
COPY ../ne-optimization-engine/main.py /app/services/optimization_engine.py

# Copy configuration files
COPY ../functor_types_with_affinity.json /app/shared/
COPY ../functor_data_affinity.json /app/shared/
COPY ../allowed_functor_types_verbose_by_phase.json /app/shared/
COPY ../callback_registry.json /app/shared/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Create data directory
RUN mkdir -p /data

# Expose service ports
EXPOSE 5001 5002 5005

# Environment variables
ENV PYTHONPATH=/app/shared:/app/services:/shared
ENV NODE_ROLE=logic
ENV DATA_PERSISTENCE=true
ENV FLASK_ENV=production

# Health check for all logic services
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5001/health && \
        curl -f http://localhost:5002/health && \
        curl -f http://localhost:5005/health

# Start all logic services
ENTRYPOINT ["/app/entrypoint.sh"] 