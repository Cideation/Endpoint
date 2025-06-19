import sys
import os
import json
sys.path.append("/shared")
from phase_2_runtime_modules import FunctorTypesRunner

def load_config():
    with open('../inputs/node_dictionarY.json') as f:
        node_dict = json.load(f)
    with open('../shared/agent_coefficients.json') as f:
        agent_coeffs = json.load(f)
    return node_dict, agent_coeffs

def main():
    node_dict, agent_coeffs = load_config()
    runner = FunctorTypesRunner(node_dict, agent_coeffs)
    runner.run()
    print('[✓] ne-functor-types booted and ran successfully')

if __name__ == '__main__':
    main() 