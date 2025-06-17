from flask import Flask, request, jsonify
import traceback

app = Flask(__name__)

@app.route('/parse_and_dispatch', methods=['POST'])
def parse_and_dispatch():
    try:
        data = request.get_json()

        if data is None:
            raise ValueError("No JSON body received")

        file_name = data.get("file_name", "unnamed_file")
        raw_data = data.get("raw_data", "")

        print(f"[INFO] Received file_name: {file_name}")
        print(f"[INFO] Received raw_data preview: {raw_data[:100]}")

        # Simulate output
        return jsonify({
            "status": "ok",
            "message": f"Successfully parsed {file_name}",
            "summary": {
                "length": len(raw_data),
                "preview": raw_data[:50]
            }
        })

    except Exception as e:
        print("❌ ERROR OCCURRED:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return "Endpoint is live", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
