import networkx as nx

def build_graph(node_dict, edge_dict, agent_coeffs=None):
    G = nx.DiGraph()
    for node in node_dict.get('nodes', []):
        G.add_node(node['id'], **node)
    for edge in edge_dict.get('edges', []):
        G.add_edge(edge['source'], edge['target'], **edge)
    # Optionally use agent_coeffs for node/edge attributes
    return G 