from flask import Flask, request, jsonify
from openai_cleaner import gpt_clean_and_validate
from transformer_main import parse_data
from neo_push import push_to_neo

app = Flask(__name__)

@app.route('/parse_and_dispatch', methods=['POST'])
def parse_and_dispatch():
    try:
        data = request.get_json()
        agent_id = data.get("agent_id", "unknown")
        phase = data.get("phase", "Alpha")
        original_payload = data.get("original_payload", {})

        # Step 1: Clean it
        cleaned = gpt_clean_and_validate(original_payload)

        # Step 2: Parse it
        parsed = parse_data(cleaned)

        # Step 3: Push to Neo4j
        push_to_neo(parsed)

        return jsonify({
            "status": "success",
            "agent_id": agent_id,
            "cleaned": cleaned,
            "parsed": parsed
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
