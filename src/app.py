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

# Import your modules - Updated for Render deployment
from parse_dxf import parse_dxf_file
from dwg_cad_ifc_parser import parse_dwg_file, parse_ifc_file
from parse_pdf import parse_pdf_file
from openai_cleaner import clean_with_ai, gpt_clean_and_validate
from neo_writer import write_to_neo4j, push_to_neo4j, get_neo4j_connection
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
def push_to_neo4j_route():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            app.logger.error('Invalid input data for database push')
            return jsonify({'error': 'Invalid input data'}), 400
        
        app.logger.info('Pushing data to Neo4j')
        result = push_to_neo4j(data)
        if 'error' not in result:
            app.logger.info('Successfully pushed to Neo4j')
            return jsonify({'status': 'success', 'message': result.get('message', 'Data pushed successfully')})
        else:
            app.logger.error(f'Error pushing to Neo4j: {result.get("error")}')
            return jsonify({'status': 'error', 'message': result.get('error')}), 500
    except Exception as e:
        app.logger.error(f'Error in push_to_neo4j_route: {str(e)}')
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

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data from both databases"""
    try:
        # Get PostgreSQL data
        postgres_components = get_all_components()
        
        # Calculate analytics
        total_components = len(postgres_components)
        total_cost = sum(c.get('estimated_cost', 0) for c in postgres_components)
        avg_cost = total_cost / total_components if total_components > 0 else 0
        
        # Check Neo4j connection safely
        neo4j_status = "Not Connected"
        try:
            graph = get_neo4j_connection()
            if graph:
                neo4j_status = "Connected"
            else:
                neo4j_status = "Connection Failed"
        except Exception as neo4j_error:
            neo4j_status = f"Error: {str(neo4j_error)[:100]}"
        
        analytics = {
            'postgres_data': {
                'total_components': total_components,
                'total_cost': total_cost,
                'average_cost': avg_cost,
                'components': postgres_components
            },
            'neo4j_status': neo4j_status,
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(analytics)
    except Exception as e:
        app.logger.error(f'Error in analytics: {str(e)}')
        return jsonify({
            'error': str(e),
            'postgres_data': {'total_components': 0, 'total_cost': 0, 'average_cost': 0, 'components': []},
            'neo4j_status': 'Error checking connection',
            'generated_at': datetime.now().isoformat()
        }), 500

@app.route('/process_enhanced', methods=['POST'])
def process_enhanced():
    """Enhanced processing pipeline: Parse → Clean → Store → Graph"""
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Step 1: Parse files (enhanced)
        parsed_results = []
        for item in data:
            filename = item.get('name')
            if filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(filepath):
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
                        continue
                    parsed_results.append(result)
        
        # Step 2: Extract components from parsed results
        all_components = []
        for result in parsed_results:
            if result.get('status') == 'success':
                all_components.extend(result.get('components', []))
        
        # Step 3: Clean with AI
        cleaned_result = clean_with_ai(all_components)
        
        # Step 4: Store in PostgreSQL
        if cleaned_result.get('status') == 'success':
            cleaned_components = cleaned_result.get('cleaned_components', [])
            success, message = push_to_db(cleaned_components)
            
            # Step 5: Push to Neo4j
            neo4j_result = push_to_neo4j(cleaned_components)
            
            return jsonify({
                'status': 'success',
                'pipeline': {
                    'parsed_files': len(parsed_results),
                    'components_extracted': len(all_components),
                    'components_cleaned': len(cleaned_components),
                    'postgres_status': 'success' if success else 'failed',
                    'neo4j_status': neo4j_result.get('message', 'failed'),
                    'total_processed': len(cleaned_components)
                },
                'cleaned_data': cleaned_components,
                'processed_at': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'AI cleaning failed'}), 500
            
    except Exception as e:
        app.logger.error(f'Error in enhanced processing: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/db_data', methods=['GET'])
def get_db_data():
    """Get all data from PostgreSQL database"""
    try:
        components = get_all_components()
        return jsonify({
            'status': 'success',
            'count': len(components),
            'data': components
        })
    except Exception as e:
        app.logger.error(f'Error getting database data: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/neo4j_health', methods=['GET'])
def neo4j_health_check():
    """Simple Neo4j health check"""
    try:
        from neo_writer import get_neo4j_connection
        graph = get_neo4j_connection()
        if graph:
            return jsonify({
                'status': 'healthy',
                'neo4j_connection': 'Connected',
                'message': 'Neo4j is accessible'
            })
        else:
            return jsonify({
                'status': 'unhealthy',
                'neo4j_connection': 'Failed',
                'message': 'Could not connect to Neo4j'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'neo4j_connection': 'Error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port) 