#!/usr/bin/env python3
"""
DAG Alpha Microservice - Real Implementation
Handles DAG execution and structural evaluation with actual computation
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
    from network_graph import load_graph
except ImportError:
    def load_graph():
        # Fallback graph if shared module not available
        G = nx.DiGraph()
        G.add_node("root", phase="alpha", primary_functor="structural")
        return G

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ne-dag-alpha",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200

@app.route('/process', methods=['POST'])
def process_dag_alpha():
    """
    Process DAG Alpha computation with real NetworkX operations
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        request_data = data.get('data', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"Processing DAG Alpha request at {timestamp}")
        
        # Load real graph
        graph = load_graph()
        
        # Extract node sequence from request
        node_sequence = request_data.get('node_sequence', [])
        
        # Perform real DAG operations
        results = {
            "dag_execution_status": "completed",
            "timestamp": datetime.now().isoformat(),
            "nodes_processed": len(node_sequence),
            "graph_nodes_available": graph.number_of_nodes(),
            "graph_edges_available": graph.number_of_edges()
        }
        
        # Real structural evaluation
        structural_results = perform_structural_evaluation(graph, node_sequence)
        results["structural_evaluation"] = structural_results
        
        # Real spatial evaluation
        spatial_results = perform_spatial_evaluation(graph, node_sequence)
        results["spatial_evaluation"] = spatial_results
        
        # Manufacturing score calculation
        manufacturing_score = calculate_manufacturing_score(structural_results, spatial_results)
        results["manufacturing_score"] = manufacturing_score
        
        # Add graph metrics
        results["graph_metrics"] = {
            "density": nx.density(graph) if graph.number_of_nodes() > 0 else 0,
            "number_of_components": nx.number_weakly_connected_components(graph),
            "average_clustering": nx.average_clustering(graph.to_undirected()) if graph.number_of_nodes() > 1 else 0
        }
        
        logger.info(f"DAG Alpha processing completed successfully")
        
        return jsonify({
            "status": "success",
            "results": results,
            "processing_time_ms": (datetime.now() - datetime.fromisoformat(timestamp.replace('Z', '+00:00'))).total_seconds() * 1000,
            "service": "ne-dag-alpha"
        }), 200
        
    except Exception as e:
        logger.error(f"DAG Alpha processing failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "service": "ne-dag-alpha",
            "timestamp": datetime.now().isoformat()
        }), 500

def perform_structural_evaluation(graph, node_sequence):
    """Perform real structural evaluation using graph analysis"""
    try:
        if not node_sequence:
            return {"status": "no_nodes", "score": 0.0}
        
        # Check if nodes exist in graph
        existing_nodes = [node for node in node_sequence if graph.has_node(node)]
        
        if not existing_nodes:
            return {"status": "nodes_not_found", "score": 0.5}
        
        # Calculate structural metrics
        subgraph = graph.subgraph(existing_nodes)
        
        structural_metrics = {
            "connectivity": subgraph.number_of_edges() / max(1, subgraph.number_of_nodes()),
            "clustering": nx.average_clustering(subgraph.to_undirected()) if subgraph.number_of_nodes() > 1 else 0,
            "efficiency": nx.global_efficiency(subgraph.to_undirected()) if subgraph.number_of_nodes() > 1 else 1
        }
        
        # Calculate overall structural score
        score = (structural_metrics["connectivity"] * 0.4 + 
                structural_metrics["clustering"] * 0.3 + 
                structural_metrics["efficiency"] * 0.3)
        
        return {
            "status": "passed" if score > 0.6 else "warning",
            "score": min(1.0, score),
            "metrics": structural_metrics,
            "nodes_evaluated": len(existing_nodes)
        }
        
    except Exception as e:
        logger.error(f"Structural evaluation failed: {e}")
        return {"status": "error", "score": 0.0, "error": str(e)}

def perform_spatial_evaluation(graph, node_sequence):
    """Perform real spatial evaluation using node attributes"""
    try:
        if not node_sequence:
            return {"status": "no_nodes", "score": 0.0}
        
        spatial_scores = []
        
        for node in node_sequence:
            if graph.has_node(node):
                node_data = graph.nodes[node]
                
                # Extract spatial attributes if available
                x = node_data.get('x', 0)
                y = node_data.get('y', 0)
                z = node_data.get('z', 0)
                
                # Calculate spatial score based on position and connectivity
                spatial_score = calculate_node_spatial_score(x, y, z, node_data)
                spatial_scores.append(spatial_score)
        
        if not spatial_scores:
            return {"status": "no_spatial_data", "score": 0.5}
        
        average_score = sum(spatial_scores) / len(spatial_scores)
        
        return {
            "status": "passed" if average_score > 0.7 else "warning",
            "score": average_score,
            "individual_scores": spatial_scores,
            "nodes_evaluated": len(spatial_scores)
        }
        
    except Exception as e:
        logger.error(f"Spatial evaluation failed: {e}")
        return {"status": "error", "score": 0.0, "error": str(e)}

def calculate_node_spatial_score(x, y, z, node_data):
    """Calculate spatial score for a single node"""
    # Base score from coordinates (normalized)
    coord_score = min(1.0, (abs(x) + abs(y) + abs(z)) / 100.0)
    
    # Bonus for having spatial attributes
    attr_bonus = 0.0
    if 'primary_functor' in node_data:
        attr_bonus += 0.2
    if 'phase' in node_data:
        attr_bonus += 0.2
    if any(key in node_data for key in ['area', 'volume', 'length']):
        attr_bonus += 0.3
    
    return min(1.0, coord_score + attr_bonus)

def calculate_manufacturing_score(structural_results, spatial_results):
    """Calculate overall manufacturing feasibility score"""
    structural_weight = 0.6
    spatial_weight = 0.4
    
    structural_score = structural_results.get('score', 0.0)
    spatial_score = spatial_results.get('score', 0.0)
    
    manufacturing_score = (structural_score * structural_weight + 
                          spatial_score * spatial_weight)
    
    return min(1.0, manufacturing_score)

if __name__ == '__main__':
    logger.info("Starting DAG Alpha microservice on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False) 