#!/usr/bin/env python3
"""
Enhanced CAD Parser API with PostgreSQL Integration
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

# Import parsers
from dwg_cad_ifc_parser import parse_dwg_file, parse_ifc_file
from parse_dxf import parse_dxf_file
from parse_pdf import parse_pdf_file
from openai_cleaner import clean_data_with_ai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'ep-white-waterfall-a85g0dgx-pooler.eastus2.azure.neon.tech'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'neondb'),
    'user': os.getenv('DB_USER', 'neondb_owner'),
    'password': os.getenv('DB_PASSWORD', 'npg_CcgA0kKeYVU2'),
    'sslmode': 'require'
}

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def write_to_postgresql(data):
    """Write data to PostgreSQL database"""
    try:
        conn = get_db_connection()
        if not conn:
            return {"error": "Could not connect to PostgreSQL"}
        
        cursor = conn.cursor()
        
        # Create components table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS components (
                id SERIAL PRIMARY KEY,
                component_id TEXT UNIQUE,
                component_type TEXT,
                properties JSONB,
                geometry JSONB,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert or update component data
        cursor.execute("""
            INSERT INTO components (component_id, component_type, properties, geometry, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (component_id) 
            DO UPDATE SET 
                component_type = EXCLUDED.component_type,
                properties = EXCLUDED.properties,
                geometry = EXCLUDED.geometry,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
        """, (
            data.get('component_id'),
            data.get('component_type'),
            json.dumps(data.get('properties', {})),
            json.dumps(data.get('geometry', {})),
            json.dumps(data.get('metadata', {}))
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": "Data written to PostgreSQL successfully",
            "component_id": data.get('component_id')
        }
        
    except Exception as e:
        logger.error(f"Failed to write to PostgreSQL: {str(e)}")
        return {"error": f"Failed to write to PostgreSQL: {str(e)}"}

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            db_status = "Connected"
    else:
            db_status = "Connection Failed"
    except Exception as e:
        db_status = f"Error: {str(e)[:100]}"
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'postgresql_status': db_status,
        'version': '2.0.0'
    })

@app.route('/parse', methods=['POST'])
def parse_file():
    """Parse uploaded CAD/BIM file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        
        # Determine file type and parse
        filename = file.filename.lower()
        
        if filename.endswith('.dwg'):
            result = parse_dwg_file(temp_path)
        elif filename.endswith('.ifc'):
            result = parse_ifc_file(temp_path)
        elif filename.endswith('.dxf'):
            result = parse_dxf_file(temp_path)
        elif filename.endswith('.pdf'):
            result = parse_pdf_file(temp_path)
            else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Clean up temp file
        os.remove(temp_path)
        
        if result.get('error'):
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in parse_file: {str(e)}")
        return jsonify({'error': f'Parsing failed: {str(e)}'}), 500

@app.route('/push', methods=['POST'])
def push_to_postgresql():
    """Push parsed data to PostgreSQL"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Generate component ID if not provided
        if 'component_id' not in data:
            data['component_id'] = f"COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Write to PostgreSQL
        result = write_to_postgresql(data)
        
        if result.get('error'):
            return jsonify(result), 500
        
        app.logger.info(f'Successfully pushed to PostgreSQL: {data["component_id"]}')
        return jsonify(result)
        
    except Exception as e:
        logger.error(f'Error in push_to_postgresql: {str(e)}')
        return jsonify({'error': f'Push failed: {str(e)}'}), 500

@app.route('/push_enhanced', methods=['POST'])
def push_enhanced_data():
    """Push enhanced data with AI cleaning to PostgreSQL"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Step 1: Clean data with AI
        app.logger.info('Cleaning data with AI')
        cleaned_data = clean_data_with_ai(data)
        
        if cleaned_data.get('error'):
            return jsonify(cleaned_data), 500
        
        # Step 2: Write to PostgreSQL
        app.logger.info('Writing cleaned data to PostgreSQL')
        result = write_to_postgresql(cleaned_data)
        
        if result.get('error'):
            return jsonify(result), 500
        
        app.logger.info('Enhanced data push successful')
        return jsonify({
            'success': True,
            'message': 'Enhanced data pushed to PostgreSQL successfully',
            'component_id': cleaned_data.get('component_id'),
            'postgresql_status': 'success'
        })
            
    except Exception as e:
        logger.error(f'Error in push_enhanced_data: {str(e)}')
        return jsonify({'error': f'Enhanced push failed: {str(e)}'}), 500

@app.route('/db_data', methods=['GET'])
def get_db_data():
    """Get data from PostgreSQL database"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM components ORDER BY created_at DESC LIMIT 10")
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Convert to list of dicts
        data = [dict(row) for row in rows]
        
        return jsonify({
            'success': True,
            'data': data,
            'count': len(data)
        })
        
    except Exception as e:
        logger.error(f'Error getting database data: {str(e)}')
        return jsonify({'error': f'Failed to get data: {str(e)}'}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for debugging"""
    try:
        # Test PostgreSQL connection
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            postgresql_status = "Connected"
        else:
            postgresql_status = "Connection failed"
            version = "Unknown"
        
        return jsonify({
            'status': 'test_successful',
            'timestamp': datetime.now().isoformat(),
            'postgresql_status': postgresql_status,
            'postgresql_version': version,
            'environment': {
                'db_host': DB_CONFIG['host'],
                'db_name': DB_CONFIG['database'],
                'db_user': DB_CONFIG['user']
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'test_failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 