from flask import Flask, request, jsonify
import uuid
import os

# Try to import optional modules, with fallbacks
try:
    from openai_cleaner import gpt_clean_and_validate
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI cleaner not available")

try:
    from generate_ids import assign_ids
    ID_GENERATOR_AVAILABLE = True
except ImportError:
    ID_GENERATOR_AVAILABLE = False
    print("Warning: ID generator not available")

try:
    from neo_writer import push_to_neo4j
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: Neo4j writer not available")

app = Flask(__name__)


def clean_and_id(data):
    cleaned = []
    for i, item in enumerate(data):
        name = item.get("name", "").strip().title()
        shape = item.get("shape", "").strip()
        cleaned.append({
            "name": name,
            "shape": shape,
            "component_id": f"CMP-{uuid.uuid4().hex[:8]}",
            "node_id": f"N-{i+1:03d}"
        })
    return cleaned


@app.route('/parse', methods=['POST'])
def parse():
    raw = request.get_json()
    result = clean_and_id(raw)
    return jsonify(result)


@app.route('/evaluate_and_push', methods=['POST'])
def evaluate_and_push():
    data = request.get_json()
    component_id = data.get("component_id", f"CMP-{uuid.uuid4().hex[:8]}")
    quantity = data.get("quantity", 0)
    unit_price = 1200  # Example fixed price
    estimated_cost = quantity * unit_price

    return jsonify({
        "status": "pushed",
        "component_id": component_id,
        "quantity": quantity,
        "estimated_cost": estimated_cost
    })


@app.route('/push', methods=['POST'])
def push():
    data = request.get_json()
    if not NEO4J_AVAILABLE:
        return jsonify({"status": "error", "message": "Neo4j not available"}), 500
    
    try:
        # Use the existing neo_writer module
        push_to_neo4j(data)
        return jsonify({"status": "pushed", "count": len(data)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/clean_with_ai', methods=['POST'])
def clean_with_ai():
    """Use OpenAI to clean and validate data"""
    if not OPENAI_AVAILABLE:
        return jsonify({"status": "error", "message": "OpenAI not available"}), 500
    
    raw_data = request.get_json()
    try:
        cleaned_data = gpt_clean_and_validate(raw_data)
        # Add IDs to cleaned data
        if ID_GENERATOR_AVAILABLE:
            result = assign_ids(cleaned_data)
        else:
            result = clean_and_id(cleaned_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "CAD Parser API",
        "features": {
            "openai": OPENAI_AVAILABLE,
            "neo4j": NEO4J_AVAILABLE,
            "id_generator": ID_GENERATOR_AVAILABLE
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
