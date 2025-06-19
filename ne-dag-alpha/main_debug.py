import sys, os

print("🔍 PYTHONPATH:", sys.path)
print("🔍 Current Dir:", os.getcwd())

# Add shared path
sys.path.append("/app/shared")

try:
    import phase_2_runtime_modules
    print("✅ Import succeeded.")
except ModuleNotFoundError as e:
    print("❌ Import failed:", e)

import json

def load_config():
    with open('../inputs/node_dictionarY.json') as f:
        node_dict = json.load(f)
    with open('../shared/agent_coefficients.json') as f:
        agent_coeffs = json.load(f)
    return node_dict, agent_coeffs

def main():
    from phase_2_runtime_modules import AlphaDAGRunner
    node_dict, agent_coeffs = load_config()
    runner = AlphaDAGRunner(node_dict, agent_coeffs)
    runner.run()
    print('[✓] ne-dag-alpha booted and ran successfully')

if __name__ == '__main__':
    main() 