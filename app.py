from flask import Flask, request, jsonify
import uuid

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


if __name__ == '__main__':
    app.run(debug=True)

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
    for item in data:
        component_id = item.get("component_id")
        name = item.get("name", "Unknown")
        print(f"[Neo4j] {component_id} → {name}")
    return jsonify({"status": "pushed", "count": len(data)})
