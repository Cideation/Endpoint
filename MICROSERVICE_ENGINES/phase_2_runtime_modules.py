# ============================================
# 📦 Phase 2 Runtime Modules (Unified Version)
# ============================================

import json
import networkx as nx

# -----------------------
# graph_init
# -----------------------
def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def init_graph(node_file, edge_file):
    G = nx.DiGraph()
    nodes = load_json(node_file)
    edges = load_json(edge_file)

    for node in nodes:
        G.add_node(node["id"], **node)

    for edge in edges:
        G.add_edge(edge["source"], edge["target"], **edge)

    return G

# -----------------------
# dispatcher
# -----------------------
def dispatch_by_phase(graph, phase):
    return [n for n, attr in graph.nodes(data=True) if attr.get("phase") == phase]

# -----------------------
# edge_resolver
# -----------------------
def resolve_edges(graph, functor_registry):
    for u, v, data in graph.edges(data=True):
        source_functor = graph.nodes[u].get("primary_functor")
        target_functor = graph.nodes[v].get("primary_functor")
        if source_functor and target_functor:
            data["resolved"] = f"{source_functor} -> {target_functor}"

# -----------------------
# ac_injector
# -----------------------
def inject_ac(graph, ac_dict):
    for node_id, ac_data in ac_dict.items():
        if node_id in graph.nodes:
            graph.nodes[node_id]["ac_injected"] = ac_data

# -----------------------
# runtime_executor
# -----------------------
def execute_functors(graph, functor_registry):
    for node_id, node_data in graph.nodes(data=True):
        functor = node_data.get("primary_functor")
        if functor and functor in functor_registry:
            print(f"Executing {functor} for node {node_id}")

# -----------------------
# validator
# -----------------------
def validate_graph(graph):
    missing_phase = [n for n, d in graph.nodes(data=True) if "phase" not in d]
    missing_functor = [n for n, d in graph.nodes(data=True) if "primary_functor" not in d]
    return {
        "missing_phase": missing_phase,
        "missing_functor": missing_functor
    }

# ============================================
# End of Phase 2 Runtime Modules
# ============================================
