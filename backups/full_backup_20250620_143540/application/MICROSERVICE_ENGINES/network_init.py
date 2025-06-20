
import json
import networkx as nx

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def initialize_graph():
    G = nx.DiGraph()

    # Load nodes and edges
    nodes = load_json("data/runtime_node_dictionary.json")
    edges = load_json("data/runtime_functor_edges.json")

    # Add nodes
    for node in nodes:
        node_id = node.get("node_id")
        if node_id:
            G.add_node(node_id, **node)

    # Add edges
    for edge in edges:
        source = edge.get("source_node")
        target = edge.get("target_node")
        if source and target:
            G.add_edge(source, target, **edge)

    return G

if __name__ == "__main__":
    graph = initialize_graph()
    print(f"Graph initialized with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
