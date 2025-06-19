import os
import sys
import json

def load_config():
    with open('../inputs/node_dictionarY.json') as f:
        node_dict = json.load(f)
    with open('../shared/agent_coefficients.json') as f:
        agent_coeffs = json.load(f)
    return node_dict, agent_coeffs

def main():
    print('DEBUG sys.path:', sys.path)
    print('DEBUG /shared contents:', os.listdir('/shared'))
    sys.path.append("/shared")
    from sfde_utility_foundation_extended import SFDEngine
    node_dict, agent_coeffs = load_config()
    engine = SFDEngine(node_dict, agent_coeffs)
    engine.run()
    print('[✓] sfde booted and ran successfully')

if __name__ == '__main__':
    main() 