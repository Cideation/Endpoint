from flask import Flask, send_from_directory, request, jsonify
import os
from werkzeug.utils import secure_filename
import sys
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules
from parse_dxf import parse_dxf_file
from dwg_cad_ifc_parser import parse_dwg_file, parse_ifc_file
from parse_pdf import parse_pdf_file
from openai_cleaner import clean_with_ai
from neo_writer import write_to_neo4j

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'dxf', 'dwg', 'ifc', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'filename': filename, 'message': 'File uploaded successfully'})
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/parse', methods=['POST'])
def parse_file():
    data = request.json
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        # Process each file in the input
        results = []
        for item in data:
            filename = item.get('name')
            if not filename:
                continue
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                continue
            
            # Parse based on file extension
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
            
            results.append(result)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clean_with_ai', methods=['POST'])
def clean_data():
    data = request.json
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        cleaned_data = clean_with_ai(data)
        return jsonify(cleaned_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_and_push', methods=['POST'])
def evaluate_and_push():
    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        # Evaluate the data
        component_id = data.get('component_id')
        quantity = data.get('quantity', 1)
        
        # Push to Neo4j
        result = write_to_neo4j({
            'component_id': component_id,
            'quantity': quantity
        })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/push', methods=['POST'])
def push_to_neo4j():
    data = request.json
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        result = write_to_neo4j(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 