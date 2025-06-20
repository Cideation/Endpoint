#!/usr/bin/env python3
"""
Graph Runtime Engine Microservice - Real Implementation
Handles NetworkX graph operations and runtime computations
"""

from flask import Flask, request, jsonify
import json
import logging
import networkx as nx
from datetime import datetime
import sys
import os

# Add shared modules to path
sys.path.append('/shared')
sys.path.append('/app/shared')

try:
    from network_graph import load_graph, get_graph_metrics
except ImportError:
    def load_graph():
        # Fallback graph if shared module not available
        G = nx.DiGraph()
        G.add_node("root", phase="runtime", primary_functor="graph")
        return G
    
    def get_graph_metrics():
        return {"nodes": 0, "edges": 0}

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global graph instance
runtime_graph = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global runtime_graph
    
    if runtime_graph is None:
        runtime_graph = load_graph()
    
    return jsonify({
        "status": "healthy",
        "service": "ne-graph-runtime-engine",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "graph_loaded": runtime_graph is not None,
        "graph_nodes": runtime_graph.number_of_nodes() if runtime_graph else 0
    }), 200

@app.route('/process', methods=['POST'])
def process_graph_runtime():
    """
    Process graph runtime operations with real NetworkX computations
    """
    global runtime_graph
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        request_data = data.get('data', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"Processing Graph Runtime request at {timestamp}")
        
        # Load graph if not already loaded
        if runtime_graph is None:
            runtime_graph = load_graph()
        
        # Extract graph data from request
        graph_data = request_data.get('graph_data', {})
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # Process graph operations
        results = {
            "graph_execution_status": "completed",
            "timestamp": datetime.now().isoformat(),
            "nodes_built": len(nodes),
            "edges_built": len(edges),
            "runtime_graph_nodes": runtime_graph.number_of_nodes(),
            "runtime_graph_edges": runtime_graph.number_of_edges()
        }
        
        # Build temporary graph from request data
        temp_graph = build_temporary_graph(nodes, edges)
        
        # Perform graph analysis
        graph_analysis = perform_graph_analysis(temp_graph, runtime_graph)
        results["analysis"] = graph_analysis
        
        # Calculate graph metrics
        graph_metrics = calculate_graph_metrics(temp_graph, runtime_graph)
        results["graph_metrics"] = graph_metrics
        
        # Find execution paths
        execution_paths = find_execution_paths(temp_graph, runtime_graph)
        results["execution_paths"] = execution_paths
        
        # Perform graph algorithms
        algorithms_results = run_graph_algorithms(temp_graph, runtime_graph)
        results["algorithms"] = algorithms_results
        
        logger.info(f"Graph Runtime processing completed successfully")
        
        return jsonify({
            "status": "success",
            "results": results,
            "processing_time_ms": (datetime.now() - datetime.fromisoformat(timestamp.replace('Z', '+00:00'))).total_seconds() * 1000,
            "service": "ne-graph-runtime-engine"
        }), 200
        
    except Exception as e:
        logger.error(f"Graph Runtime processing failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "service": "ne-graph-runtime-engine",
            "timestamp": datetime.now().isoformat()
        }), 500

def build_temporary_graph(nodes, edges):
    """Build a temporary graph from request data"""
    try:
        temp_graph = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            node_id = node.get('id') or node.get('node_id')
            if node_id:
                temp_graph.add_node(node_id, **node)
        
        # Add edges
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                temp_graph.add_edge(source, target, **edge)
        
        logger.info(f"Built temporary graph: {temp_graph.number_of_nodes()} nodes, {temp_graph.number_of_edges()} edges")
        return temp_graph
        
    except Exception as e:
        logger.error(f"Failed to build temporary graph: {e}")
        return nx.DiGraph()

def perform_graph_analysis(temp_graph, runtime_graph):
    """Perform comprehensive graph analysis"""
    try:
        analysis = {}
        
        # Analyze temporary graph
        if temp_graph.number_of_nodes() > 0:
            analysis["temp_graph"] = {
                "is_connected": nx.is_weakly_connected(temp_graph),
                "is_dag": nx.is_directed_acyclic_graph(temp_graph),
                "density": nx.density(temp_graph),
                "number_of_components": nx.number_weakly_connected_components(temp_graph)
            }
        
        # Analyze runtime graph
        if runtime_graph.number_of_nodes() > 0:
            analysis["runtime_graph"] = {
                "is_connected": nx.is_weakly_connected(runtime_graph),
                "is_dag": nx.is_directed_acyclic_graph(runtime_graph),
                "density": nx.density(runtime_graph),
                "number_of_components": nx.number_weakly_connected_components(runtime_graph)
            }
        
        # Compare graphs
        if temp_graph.number_of_nodes() > 0 and runtime_graph.number_of_nodes() > 0:
            common_nodes = set(temp_graph.nodes()) & set(runtime_graph.nodes())
            analysis["comparison"] = {
                "common_nodes": len(common_nodes),
                "unique_temp_nodes": len(set(temp_graph.nodes()) - set(runtime_graph.nodes())),
                "unique_runtime_nodes": len(set(runtime_graph.nodes()) - set(temp_graph.nodes())),
                "similarity_score": len(common_nodes) / max(1, len(set(temp_graph.nodes()) | set(runtime_graph.nodes())))
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Graph analysis failed: {e}")
        return {"error": str(e)}

def calculate_graph_metrics(temp_graph, runtime_graph):
    """Calculate detailed graph metrics"""
    try:
        metrics = {}
        
        # Metrics for temporary graph
        if temp_graph.number_of_nodes() > 0:
            metrics["temp_graph"] = {
                "nodes": temp_graph.number_of_nodes(),
                "edges": temp_graph.number_of_edges(),
                "density": nx.density(temp_graph),
                "average_degree": sum(dict(temp_graph.degree()).values()) / temp_graph.number_of_nodes() if temp_graph.number_of_nodes() > 0 else 0
            }
            
            # Additional metrics for connected graphs
            if nx.is_weakly_connected(temp_graph) and temp_graph.number_of_nodes() > 1:
                metrics["temp_graph"]["diameter"] = nx.diameter(temp_graph.to_undirected())
                metrics["temp_graph"]["radius"] = nx.radius(temp_graph.to_undirected())
                metrics["temp_graph"]["clustering"] = nx.average_clustering(temp_graph.to_undirected())
        
        # Metrics for runtime graph
        if runtime_graph.number_of_nodes() > 0:
            metrics["runtime_graph"] = {
                "nodes": runtime_graph.number_of_nodes(),
                "edges": runtime_graph.number_of_edges(),
                "density": nx.density(runtime_graph),
                "average_degree": sum(dict(runtime_graph.degree()).values()) / runtime_graph.number_of_nodes() if runtime_graph.number_of_nodes() > 0 else 0
            }
            
            # Additional metrics for connected graphs
            if nx.is_weakly_connected(runtime_graph) and runtime_graph.number_of_nodes() > 1:
                metrics["runtime_graph"]["diameter"] = nx.diameter(runtime_graph.to_undirected())
                metrics["runtime_graph"]["radius"] = nx.radius(runtime_graph.to_undirected())
                metrics["runtime_graph"]["clustering"] = nx.average_clustering(runtime_graph.to_undirected())
        
        return metrics
        
    except Exception as e:
        logger.error(f"Graph metrics calculation failed: {e}")
        return {"error": str(e)}

def find_execution_paths(temp_graph, runtime_graph):
    """Find optimal execution paths in graphs"""
    try:
        paths = {}
        
        # Find paths in temporary graph
        if temp_graph.number_of_nodes() > 1:
            nodes = list(temp_graph.nodes())
            if len(nodes) >= 2:
                try:
                    shortest_path = nx.shortest_path(temp_graph, nodes[0], nodes[-1])
                    paths["temp_graph_shortest"] = shortest_path
                except nx.NetworkXNoPath:
                    paths["temp_graph_shortest"] = "No path found"
        
        # Find paths in runtime graph
        if runtime_graph.number_of_nodes() > 1:
            nodes = list(runtime_graph.nodes())
            if len(nodes) >= 2:
                try:
                    shortest_path = nx.shortest_path(runtime_graph, nodes[0], nodes[-1])
                    paths["runtime_graph_shortest"] = shortest_path
                except nx.NetworkXNoPath:
                    paths["runtime_graph_shortest"] = "No path found"
        
        # Topological sort for DAGs
        if temp_graph.number_of_nodes() > 0 and nx.is_directed_acyclic_graph(temp_graph):
            try:
                paths["temp_graph_topological"] = list(nx.topological_sort(temp_graph))
            except:
                paths["temp_graph_topological"] = "Not a DAG"
        
        if runtime_graph.number_of_nodes() > 0 and nx.is_directed_acyclic_graph(runtime_graph):
            try:
                paths["runtime_graph_topological"] = list(nx.topological_sort(runtime_graph))
            except:
                paths["runtime_graph_topological"] = "Not a DAG"
        
        return paths
        
    except Exception as e:
        logger.error(f"Execution path finding failed: {e}")
        return {"error": str(e)}

def run_graph_algorithms(temp_graph, runtime_graph):
    """Run various graph algorithms"""
    try:
        algorithms = {}
        
        # Centrality measures for temporary graph
        if temp_graph.number_of_nodes() > 0:
            algorithms["temp_graph_centrality"] = {}
            
            # Degree centrality
            algorithms["temp_graph_centrality"]["degree"] = dict(nx.degree_centrality(temp_graph))
            
            # Betweenness centrality (for connected graphs)
            if nx.is_weakly_connected(temp_graph) and temp_graph.number_of_nodes() > 2:
                algorithms["temp_graph_centrality"]["betweenness"] = dict(nx.betweenness_centrality(temp_graph))
            
            # PageRank
            try:
                algorithms["temp_graph_centrality"]["pagerank"] = dict(nx.pagerank(temp_graph))
            except:
                algorithms["temp_graph_centrality"]["pagerank"] = "Failed to calculate"
        
        # Centrality measures for runtime graph
        if runtime_graph.number_of_nodes() > 0:
            algorithms["runtime_graph_centrality"] = {}
            
            # Degree centrality
            algorithms["runtime_graph_centrality"]["degree"] = dict(nx.degree_centrality(runtime_graph))
            
            # Betweenness centrality (for connected graphs)
            if nx.is_weakly_connected(runtime_graph) and runtime_graph.number_of_nodes() > 2:
                algorithms["runtime_graph_centrality"]["betweenness"] = dict(nx.betweenness_centrality(runtime_graph))
            
            # PageRank
            try:
                algorithms["runtime_graph_centrality"]["pagerank"] = dict(nx.pagerank(runtime_graph))
            except:
                algorithms["runtime_graph_centrality"]["pagerank"] = "Failed to calculate"
        
        return algorithms
        
    except Exception as e:
        logger.error(f"Graph algorithms failed: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    logger.info("Starting Graph Runtime Engine microservice on port 5004")
    app.run(host='0.0.0.0', port=5004, debug=False) 