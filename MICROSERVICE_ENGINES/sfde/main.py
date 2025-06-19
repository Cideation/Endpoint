import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
from sfde_utility_foundation_extended import SFDEngine

def load_config():
    with open('../inputs/node_dictionarY.json') as f:
        node_dict = json.load(f)
    with open('../shared/agent_coefficients.json') as f:
        agent_coeffs = json.load(f)
    return node_dict, agent_coeffs

def main():
    node_dict, agent_coeffs = load_config()
    engine = SFDEngine(node_dict, agent_coeffs)
    engine.run()
    print('[✓] SFDE container booted and ran successfully')

if __name__ == '__main__':
    main() 