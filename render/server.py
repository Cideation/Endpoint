from flask import Flask, send_from_directory, request, jsonify, render_template
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

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules
from parse_dxf import parse_dxf_file
from dwg_cad_ifc_parser import parse_dwg_file, parse_ifc_file
from parse_pdf import parse_pdf_file
from openai_cleaner import clean_with_ai, gpt_clean_and_validate
from neo_writer import write_to_neo4j, push_to_neo4j
from generate_ids import assign_ids

# Initialize Sentry for error tracking
sentry_sdk.init(
    dsn=os.environ.get('SENTRY_DSN'),
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0,
    environment=os.environ.get('FLASK_ENV', 'development')
)

# Get the absolute path to the render directory
RENDER_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"RENDER_DIR: {RENDER_DIR}")  # Debug print

app = Flask(__name__,
           static_folder=os.path.join(RENDER_DIR, 'static'),
           template_folder=os.path.join(RENDER_DIR, 'templates'))

# Debug prints for paths
print(f"Static folder: {app.static_folder}")
print(f"Template folder: {app.template_folder}")
print(f"Files in template folder: {os.listdir(app.template_folder)}")

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
UPLOAD_FOLDER = os.path.join(RENDER_DIR, 'uploads')
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

@app.route('/')
def serve_index():
    try:
        app.logger.info("Attempting to serve index.html")
        app.logger.info(f"Template folder contents: {os.listdir(app.template_folder)}")
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error serving index.html: {str(e)}")
        return str(e), 500

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
            app.logger.error('Invalid input data for Neo4j push')
            return jsonify({'error': 'Invalid input data'}), 400
        
        app.logger.info('Pushing data to Neo4j')
        result = write_to_neo4j(data)
        app.logger.info('Successfully pushed to Neo4j')
        return jsonify(result)
    except Exception as e:
        app.logger.error(f'Error in push_to_neo4j: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'service': 'CAD Parser API',
        'features': {
            'openai': True,
            'neo4j': True,
            'id_generator': True
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 