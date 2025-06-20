FROM python:3.11-slim

LABEL maintainer="BEM System"
LABEL description="Fabrication Bundle Generator - CAD, BOM, and Compliance Automation"

# Install system dependencies for CAD processing
RUN apt-get update && apt-get install -y \
    # CAD and geometry processing
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # PDF generation
    wkhtmltopdf \
    # 3D processing libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Network and API dependencies
    curl \
    wget \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p /app/outputs/{blueprints,bom,compliance,bundles}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5006/health || exit 1

# Expose port
EXPOSE 5006

# Run the application
CMD ["python", "main.py"] 