# network_graph.py
# NetworkX graph loader for all containers - Real Implementation

import json
import networkx as nx
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class NetworkGraphLoader:
    """
    Real NetworkX graph loader for BEM system
    Loads node/edge data from JSON files and constructs operational graphs
    """
    
    def __init__(self, inputs_dir: str = "/inputs", shared_dir: str = "/shared"):
        self.inputs_dir = Path(inputs_dir)
        self.shared_dir = Path(shared_dir)
        self.graph = nx.DiGraph()  # Directed graph for BEM system
        
    def load_graph(self) -> nx.DiGraph:
        """
        Load complete graph from inputs and callback registry
        Returns NetworkX DiGraph ready for computation
        """
        try:
            # Load callback registry
            self._load_callback_registry()
            
            # Load node dictionary
            self._load_node_dictionary()
            
            # Load functor registry  
            self._load_functor_registry()
            
            # Load edges and relationships
            self._load_edges()
            
            # Validate graph structure
            self._validate_graph()
            
            logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return self.graph
            
        except Exception as e:
            logger.error(f"Graph loading failed: {e}")
            return self._create_fallback_graph()
    
    def _load_callback_registry(self):
        """Load callback registry for node relationships"""
        try:
            registry_path = self.shared_dir.parent / "callback_registry.json"
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                    
                # Add callback nodes to graph
                for callback_id, callback_data in registry.items():
                    self.graph.add_node(
                        callback_id,
                        node_type="callback",
                        **callback_data
                    )
                    
                logger.info(f"Loaded {len(registry)} callback nodes")
            else:
                logger.warning("Callback registry not found")
                
        except Exception as e:
            logger.error(f"Failed to load callback registry: {e}")
    
    def _load_node_dictionary(self):
        """Load node dictionary for main graph nodes"""
        try:
            node_dict_path = self.shared_dir.parent / "node_dictionarY.json"
            if node_dict_path.exists():
                with open(node_dict_path, 'r') as f:
                    nodes = json.load(f)
                    
                # Add nodes to graph
                for node_id, node_data in nodes.items():
                    self.graph.add_node(
                        node_id,
                        node_type="main",
                        **node_data
                    )
                    
                logger.info(f"Loaded {len(nodes)} main nodes")
            else:
                logger.warning("Node dictionary not found")
                
        except Exception as e:
            logger.error(f"Failed to load node dictionary: {e}")
    
    def _load_functor_registry(self):
        """Load functor registry for computational nodes"""
        try:
            functor_path = self.shared_dir.parent / "functor_registry.json"
            if functor_path.exists():
                with open(functor_path, 'r') as f:
                    functors = json.load(f)
                    
                # Add functor nodes
                for functor_id, functor_data in functors.items():
                    self.graph.add_node(
                        f"functor_{functor_id}",
                        node_type="functor",
                        **functor_data
                    )
                    
                logger.info(f"Loaded {len(functors)} functor nodes")
            else:
                logger.warning("Functor registry not found")
                
        except Exception as e:
            logger.error(f"Failed to load functor registry: {e}")
    
    def _load_edges(self):
        """Load edges from various edge definition files"""
        try:
            # Load functor edges
            edges_path = self.shared_dir.parent / "functor_edges_with_lookup.json"
            if edges_path.exists():
                with open(edges_path, 'r') as f:
                    edges_data = json.load(f)
                    
                for edge_def in edges_data:
                    source = edge_def.get('source')
                    target = edge_def.get('target')
                    edge_type = edge_def.get('type', 'default')
                    
                    if source and target:
                        self.graph.add_edge(
                            source,
                            target,
                            edge_type=edge_type,
                            **edge_def
                        )
                        
                logger.info(f"Loaded {len(edges_data)} edges")
            
            # Load unified edges
            unified_edges_path = self.shared_dir.parent / "unified_functor_variable_edges.json"
            if unified_edges_path.exists():
                with open(unified_edges_path, 'r') as f:
                    unified_edges = json.load(f)
                    
                for edge_def in unified_edges:
                    source = edge_def.get('source')
                    target = edge_def.get('target')
                    
                    if source and target and self.graph.has_node(source) and self.graph.has_node(target):
                        self.graph.add_edge(
                            source,
                            target,
                            edge_type="unified",
                            **edge_def
                        )
                        
                logger.info(f"Added {len(unified_edges)} unified edges")
                
        except Exception as e:
            logger.error(f"Failed to load edges: {e}")
    
    def _validate_graph(self):
        """Validate graph structure and requirements"""
        try:
            # Check for isolated nodes
            isolated = list(nx.isolates(self.graph))
            if isolated:
                logger.warning(f"Found {len(isolated)} isolated nodes")
            
            # Check connectivity
            if self.graph.number_of_nodes() > 0:
                # Convert to undirected for connectivity check
                undirected = self.graph.to_undirected()
                components = list(nx.connected_components(undirected))
                logger.info(f"Graph has {len(components)} connected components")
            
            # Validate required node attributes
            missing_phase = [n for n, d in self.graph.nodes(data=True) if "phase" not in d]
            missing_functor = [n for n, d in self.graph.nodes(data=True) if "primary_functor" not in d]
            
            if missing_phase:
                logger.warning(f"{len(missing_phase)} nodes missing phase attribute")
            if missing_functor:
                logger.warning(f"{len(missing_functor)} nodes missing primary_functor attribute")
                
        except Exception as e:
            logger.error(f"Graph validation failed: {e}")
    
    def _create_fallback_graph(self) -> nx.DiGraph:
        """Create a minimal fallback graph if loading fails"""
        fallback = nx.DiGraph()
        
        # Add basic nodes
        fallback.add_node("root", node_type="system", phase="alpha")
        fallback.add_node("agent_1", node_type="agent", phase="beta", primary_functor="bid")
        fallback.add_node("agent_2", node_type="agent", phase="gamma", primary_functor="investment")
        
        # Add basic edges
        fallback.add_edge("root", "agent_1", edge_type="system")
        fallback.add_edge("root", "agent_2", edge_type="system")
        
        logger.info("Created fallback graph with 3 nodes, 2 edges")
        return fallback
    
    def get_graph_metrics(self) -> Dict[str, Any]:
        """Get graph metrics for monitoring"""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types()
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by type"""
        type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts

# Global loader instance
_graph_loader = None

def load_graph() -> nx.DiGraph:
    """
    Main entry point for loading the system graph
    Returns NetworkX DiGraph ready for computation
    """
    global _graph_loader
    
    if _graph_loader is None:
        _graph_loader = NetworkGraphLoader()
    
    return _graph_loader.load_graph()

def get_graph_metrics() -> Dict[str, Any]:
    """Get current graph metrics"""
    global _graph_loader
    
    if _graph_loader is None:
        _graph_loader = NetworkGraphLoader()
    
    return _graph_loader.get_graph_metrics()

# Backwards compatibility
def get_graph() -> nx.DiGraph:
    """Alias for load_graph()"""
    return load_graph() 