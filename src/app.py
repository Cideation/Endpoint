from flask import Flask, send_from_directory, request, jsonify, render_template, current_app, send_file
import os
from werkzeug.utils import secure_filename
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
import time
from datetime import datetime
import uuid
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# Import your modules
from parse_dxf import parse_dxf_file
from dwg_cad_ifc_parser import parse_dwg_file, parse_ifc_file
from parse_pdf import parse_pdf_file
from openai_cleaner import clean_with_ai, gpt_clean_and_validate
from neo_writer import write_to_neo4j, push_to_neo4j
from generate_ids import assign_ids
from db import init_db, push_to_db, get_all_components

# Initialize Sentry for error tracking
sentry_sdk.init(
    dsn=os.environ.get('SENTRY_DSN'),
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0,
    environment=os.environ.get('FLASK_ENV', 'development')
)

# Create Flask app
app = Flask(__name__,
           static_folder='static',
           static_url_path='/static',
           template_folder='templates')

# Initialize database
init_db(app)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('CAD Parser startup')

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'dxf', 'dwg', 'ifc', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_request():
    """Log request details"""
    app.logger.info(f'Request: {request.method} {request.url}')
    app.logger.info(f'Headers: {dict(request.headers)}')
    if request.is_json:
        app.logger.info(f'JSON Data: {request.get_json()}')

def log_response(response):
    """Log response details"""
    app.logger.info(f'Response: {response.status_code}')
    return response

@app.before_request
def before_request():
    """Log request details before processing"""
    log_request()
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Log response details after processing"""
    duration = time.time() - request.start_time
    app.logger.info(f'Request duration: {duration:.2f}s')
    return log_response(response)

@app.route('/test')
def test():
    return 'Hello, world!', 200

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_index(path):
    # Serve index.html for root and any unknown route (SPA support)
    static_folder = app.static_folder
    index_path = os.path.join(static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(static_folder, 'index.html')
    else:
        return 'index.html not found', 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            app.logger.error('No file part in request')
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error('No selected file')
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.info(f'File uploaded successfully: {filename}')
            return jsonify({'filename': filename, 'message': 'File uploaded successfully'})
        
        app.logger.error(f'File type not allowed: {file.filename}')
        return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        app.logger.error(f'Error in upload_file: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/parse', methods=['POST'])
def parse_file():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            app.logger.error('Invalid input data for parsing')
            return jsonify({'error': 'Invalid input data'}), 400
        
        results = []
        for item in data:
            filename = item.get('name')
            if not filename:
                continue
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                app.logger.error(f'File not found: {filename}')
                continue
            
            app.logger.info(f'Parsing file: {filename}')
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'dxf':
                result = parse_dxf_file(filepath)
            elif ext == 'dwg':
                result = parse_dwg_file(filepath)
            elif ext == 'ifc':
                result = parse_ifc_file(filepath)
            elif ext == 'pdf':
                result = parse_pdf_file(filepath)
            else:
                app.logger.error(f'Unsupported file type: {ext}')
                continue
            
            results.append(result)
            app.logger.info(f'Successfully parsed: {filename}')
        
        return jsonify(results)
    except Exception as e:
        app.logger.error(f'Error in parse_file: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/clean_with_ai', methods=['POST'])
def clean_data():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            app.logger.error('Invalid input data for AI cleaning')
            return jsonify({'error': 'Invalid input data'}), 400
        
        app.logger.info('Starting AI cleaning process')
        cleaned_data = clean_with_ai(data)
        app.logger.info('AI cleaning completed successfully')
        return jsonify(cleaned_data)
    except Exception as e:
        app.logger.error(f'Error in clean_data: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_and_push', methods=['POST'])
def evaluate_and_push():
    try:
        data = request.json
        if not data or not isinstance(data, dict):
            app.logger.error('Invalid input data for evaluation')
            return jsonify({'error': 'Invalid input data'}), 400
        
        component_id = data.get('component_id', f"CMP-{uuid.uuid4().hex[:8]}")
        quantity = data.get('quantity', 1)
        unit_price = 1200  # Example fixed price
        estimated_cost = quantity * unit_price
        
        app.logger.info(f'Evaluating component: {component_id}')
        result = write_to_neo4j({
            'component_id': component_id,
            'quantity': quantity,
            'estimated_cost': estimated_cost
        })
        
        app.logger.info(f'Successfully pushed to Neo4j: {component_id}')
        return jsonify(result)
    except Exception as e:
        app.logger.error(f'Error in evaluate_and_push: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/push', methods=['POST'])
def push_to_neo4j():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            app.logger.error('Invalid input data for database push')
            return jsonify({'error': 'Invalid input data'}), 400
        
        app.logger.info('Pushing data to database')
        success, message = push_to_db(data)
        if success:
            app.logger.info('Successfully pushed to database')
            return jsonify({'status': 'success', 'message': message})
        else:
            app.logger.error(f'Error pushing to database: {message}')
            return jsonify({'status': 'error', 'message': message}), 500
    except Exception as e:
        app.logger.error(f'Error in push_to_neo4j: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/components', methods=['GET'])
def get_components():
    """Get all components from database"""
    try:
        components = get_all_components()
        return jsonify(components)
    except Exception as e:
        app.logger.error(f'Error getting components: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port) 