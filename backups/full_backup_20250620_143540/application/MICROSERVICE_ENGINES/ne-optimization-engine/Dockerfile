FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities
COPY ../shared /app/shared

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app:/app/shared

# Expose port for monitoring
EXPOSE 8080

# Run the optimization engine
CMD ["python", "main.py"] 