FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    tree \
    jq \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && pip install black flake8 isort pylint pytest pytest-cov

# Install global Node.js packages
RUN npm install -g nodemon typescript ts-node @types/node

# Create workspace directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONPATH=/workspace
ENV NODE_PATH=/workspace/node_modules
ENV PATH="/workspace/node_modules/.bin:$PATH"

# Create development scripts directory
RUN mkdir -p /workspace/scripts

# Set default shell
SHELL ["/bin/bash", "-c"] 