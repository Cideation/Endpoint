import json
from network_init import build_graph

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    node_dict = load_json('/inputs/runtime_node_dictionary.json')
    edge_dict = load_json('/inputs/runtime_functor_edges.json')
    try:
        agent_coeffs = load_json('/inputs/agent_coefficients.json')
    except FileNotFoundError:
        agent_coeffs = None
    G = build_graph(node_dict, edge_dict, agent_coeffs)
    print(f"[ne-graph-runtime-engine] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    # Optionally, save graph state or results to /outputs

if __name__ == '__main__':
    main() 