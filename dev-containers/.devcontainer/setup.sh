#!/bin/bash

echo "🚀 Setting up CAD Parser Phase 2 Development Environment..."

# Create necessary directories
mkdir -p /workspace/services
mkdir -p /workspace/gateway
mkdir -p /workspace/admin-ui
mkdir -p /workspace/shared
mkdir -p /workspace/scripts
mkdir -p /workspace/tests
mkdir -p /workspace/docs

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create Node.js package.json for gateway
echo "📦 Setting up Node.js gateway..."
cat > /workspace/gateway/package.json << 'EOF'
{
  "name": "cad-parser-gateway",
  "version": "1.0.0",
  "description": "Microservice Gateway for CAD Parser",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "nodemon src/index.js",
    "build": "tsc",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "morgan": "^1.10.0",
    "axios": "^1.4.0",
    "redis": "^4.6.7",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "@types/express": "^4.17.17",
    "@types/cors": "^2.8.13",
    "@types/morgan": "^1.9.4",
    "@types/node": "^20.4.1",
    "typescript": "^5.1.6",
    "nodemon": "^3.0.1",
    "jest": "^29.6.1"
  }
}
EOF

# Create Node.js package.json for admin-ui
echo "📦 Setting up Node.js admin UI..."
cat > /workspace/admin-ui/package.json << 'EOF'
{
  "name": "cad-parser-admin-ui",
  "version": "1.0.0",
  "description": "Admin UI for CAD Parser",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "nodemon src/index.js",
    "build": "webpack --mode production",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "axios": "^1.4.0",
    "socket.io": "^4.7.2"
  },
  "devDependencies": {
    "@types/express": "^4.17.17",
    "@types/cors": "^2.8.13",
    "@types/node": "^20.4.1",
    "typescript": "^5.1.6",
    "nodemon": "^3.0.1",
    "webpack": "^5.88.1",
    "webpack-cli": "^5.1.4",
    "jest": "^29.6.1"
  }
}
EOF

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd /workspace/gateway && npm install
cd /workspace/admin-ui && npm install

# Create development scripts
echo "🔧 Creating development scripts..."
cat > /workspace/scripts/dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting CAD Parser Development Environment..."

# Start all services in background
echo "📦 Starting PostgreSQL..."
docker-compose up -d postgres

echo "📦 Starting Redis..."
docker-compose up -d redis

echo "📦 Starting Python services..."
docker-compose up -d cad-parser data-processor

echo "📦 Starting Node.js services..."
docker-compose up -d gateway admin-ui

echo "✅ All services started!"
echo "🌐 Gateway: http://localhost:8080"
echo "🎨 Admin UI: http://localhost:3000"
echo "🐍 CAD Parser API: http://localhost:5000"
echo "📊 Data Processor: http://localhost:5001"
EOF

chmod +x /workspace/scripts/dev.sh

# Create service templates
echo "📁 Creating service templates..."
mkdir -p /workspace/services/cad-parser
mkdir -p /workspace/services/data-processor

# CAD Parser Service
cat > /workspace/services/cad-parser/app.py << 'EOF'
from flask import Flask, request, jsonify
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "cad-parser"})

@app.route('/parse', methods=['POST'])
def parse_cad():
    try:
        data = request.get_json()
        # CAD parsing logic here
        return jsonify({"status": "success", "message": "CAD file parsed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

# Data Processor Service
cat > /workspace/services/data-processor/app.py << 'EOF'
from flask import Flask, request, jsonify
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "data-processor"})

@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        # Data processing logic here
        return jsonify({"status": "success", "message": "Data processed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
EOF

# Gateway Service
cat > /workspace/gateway/src/index.js << 'EOF'
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const axios = require('axios');

const app = express();
const PORT = process.env.API_PORT || 8080;

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', service: 'gateway' });
});

// Route to CAD Parser service
app.post('/api/parse', async (req, res) => {
    try {
        const response = await axios.post('http://cad-parser:5000/parse', req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Route to Data Processor service
app.post('/api/process', async (req, res) => {
    try {
        const response = await axios.post('http://data-processor:5001/process', req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Gateway service running on port ${PORT}`);
});
EOF

# Admin UI Service
cat > /workspace/admin-ui/src/index.js << 'EOF'
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.UI_PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', service: 'admin-ui' });
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`🎨 Admin UI running on port ${PORT}`);
});
EOF

echo "✅ Setup complete! Development environment ready for Phase 2."
echo "🚀 Run './scripts/dev.sh' to start all services" 