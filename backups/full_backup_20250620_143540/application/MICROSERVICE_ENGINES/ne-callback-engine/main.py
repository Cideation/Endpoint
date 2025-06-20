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
    from phase_2_runtime_modules import CallbackEngineRunner
    node_dict, agent_coeffs = load_config()
    runner = CallbackEngineRunner(node_dict, agent_coeffs)
    runner.run()
    print('[✓] ne-callback-engine booted and ran successfully')

if __name__ == '__main__':
    main() 