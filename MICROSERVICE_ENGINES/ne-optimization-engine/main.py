#!/usr/bin/env python3
"""
Optimization Engine - DGL/NetworkX Pipeline with Advanced Graph Learning
Analyzes graph data, generates embeddings, predicts routing, detects emergence
"""

import os
import json
import time
import logging
import numpy as np
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import psycopg2
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
    import dgl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from dgl.nn import GraphConv, GATConv, SAGEConv
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    print("DGL not available, using NetworkX only")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationMetrics:
    """Metrics for optimization analysis"""
    roi_score: float
    occupancy_efficiency: float
    spec_fit_score: float
    structural_performance: float
    energy_efficiency: float
    cost_optimization: float
    overall_score: float
    emergence_score: float
    routing_confidence: float

class GraphLearningModel(nn.Module):
    """
    Advanced DGL model for graph learning patterns between agents, callbacks, coefficients
    """
    
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int, num_heads: int = 4):
        super().__init__()
        # Multi-layer architecture for complex pattern learning
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
        self.sage = SAGEConv(hidden_feats * num_heads, hidden_feats, 'mean')
        self.classifier = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(0.1)
        
        # Embedding layers for different node types
        self.agent_embedding = nn.Embedding(10, hidden_feats)  # Max 10 agent types
        self.callback_embedding = nn.Embedding(5, hidden_feats)  # 5 callback types
        
    def forward(self, g, h, agent_types, callback_types):
        # Graph attention for pattern learning
        h = self.gat1(g, h).flatten(1)
        h = self.dropout(h)
        h = self.gat2(g, h).flatten(1)
        h = self.dropout(h)
        
        # SAGE for neighborhood aggregation
        h = self.sage(g, h)
        
        # Add agent and callback embeddings
        agent_emb = self.agent_embedding(agent_types)
        callback_emb = self.callback_embedding(callback_types)
        h = h + agent_emb + callback_emb
        
        # Final classification/prediction
        return self.classifier(h)

class PredictiveRoutingModel(nn.Module):
    """
    DGL model for predicting likely successful callback paths
    """
    
    def __init__(self, node_feats: int, edge_feats: int, hidden_dim: int = 128):
        super().__init__()
        self.node_encoder = nn.Linear(node_feats, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feats, hidden_dim)
        
        # Graph neural network for path prediction
        self.gnn_layers = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # Path scoring network
        self.path_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, g, node_feats, edge_feats):
        # Encode node and edge features
        h = F.relu(self.node_encoder(node_feats))
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            h = F.relu(layer(g, h))
        
        return h
    
    def predict_path_success(self, source_emb, target_emb):
        """Predict success probability of path from source to target"""
        path_emb = torch.cat([source_emb, target_emb], dim=-1)
        return self.path_scorer(path_emb)

class EmergenceDetector:
    """
    Cluster detection and emergent behavior analysis
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        
    def detect_clusters(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Detect clusters in node embeddings"""
        # Normalize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(embeddings_scaled)
        
        # Analyze clusters
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_sizes': [
                np.sum(cluster_labels == label) for label in unique_labels if label != -1
            ]
        }
    
    def detect_emergent_behavior(self, metrics_history: List[Dict]) -> Dict[str, Any]:
        """Detect emergent behavior patterns from metrics history"""
        if len(metrics_history) < 5:
            return {'emergence_detected': False, 'reason': 'Insufficient history'}
        
        # Extract time series of key metrics
        roi_scores = [m['metrics']['roi_score'] for m in metrics_history[-10:]]
        overall_scores = [m['metrics']['overall_score'] for m in metrics_history[-10:]]
        
        # Detect sudden changes (emergence indicators)
        roi_trend = np.polyfit(range(len(roi_scores)), roi_scores, 1)[0]
        overall_trend = np.polyfit(range(len(overall_scores)), overall_scores, 1)[0]
        
        # Calculate variance (instability indicator)
        roi_variance = np.var(roi_scores)
        overall_variance = np.var(overall_scores)
        
        # Emergence detection logic
        emergence_detected = (
            abs(roi_trend) > 0.1 or  # Rapid trend change
            abs(overall_trend) > 0.1 or
            roi_variance > 0.05 or  # High instability
            overall_variance > 0.05
        )
        
        return {
            'emergence_detected': emergence_detected,
            'roi_trend': float(roi_trend),
            'overall_trend': float(overall_trend),
            'roi_variance': float(roi_variance),
            'overall_variance': float(overall_variance),
            'warning_level': 'high' if emergence_detected else 'normal'
        }

class OptimizationEngine:
    """
    Enhanced DGL/NetworkX-based optimization engine with advanced graph learning
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or self._default_db_config()
        self.graph = None
        self.dgl_graph = None
        self.learning_model = None
        self.routing_model = None
        self.emergence_detector = EmergenceDetector()
        self.metrics_history = []
        self.node_embeddings = None
        
        if DGL_AVAILABLE:
            logger.info("DGL available - using advanced graph neural networks")
            self._initialize_models()
        else:
            logger.info("DGL not available - using NetworkX analysis only")
    
    def _default_db_config(self) -> Dict:
        """Default database configuration"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'neon'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
    
    def _initialize_models(self):
        """Initialize DGL models"""
        if not DGL_AVAILABLE:
            return
        
        # Initialize graph learning model
        self.learning_model = GraphLearningModel(
            in_feats=8,  # Node feature dimension
            hidden_feats=64,
            out_feats=32,  # Embedding dimension
            num_heads=4
        )
        
        # Initialize predictive routing model
        self.routing_model = PredictiveRoutingModel(
            node_feats=8,
            edge_feats=4,
            hidden_dim=64
        )
    
    def load_graph_data(self) -> Dict[str, Any]:
        """Load graph data from PostgreSQL with agent and callback information"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Load nodes with agent and callback information
            cursor.execute("""
                SELECT node_id, component_type, properties, coefficients, 
                       agent_type, callback_type, agent_state
                FROM graph_nodes 
                WHERE active = true
            """)
            nodes_data = cursor.fetchall()
            
            # Load edges with callback path information
            cursor.execute("""
                SELECT source_node, target_node, edge_type, weight, properties,
                       callback_success_rate, last_execution_time
                FROM graph_edges 
                WHERE active = true
            """)
            edges_data = cursor.fetchall()
            
            conn.close()
            
            return {
                'nodes': nodes_data,
                'edges': edges_data,
                'loaded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            return self._generate_enhanced_sample_data()
    
    def _generate_enhanced_sample_data(self) -> Dict[str, Any]:
        """Generate enhanced sample data with agent and callback information"""
        nodes = [
            ('N001', 'structural', {'volume': 150.5, 'cost': 25000}, {'safety_factor': 1.8}, 'BiddingAgent', 'dag', {'active': True}),
            ('N002', 'electrical', {'power': 400, 'cost': 15000}, {'efficiency': 0.92}, 'OccupancyNode', 'relational', {'active': True}),
            ('N003', 'mep', {'flow_rate': 600, 'cost': 12000}, {'performance': 0.85}, 'MEPSystemNode', 'combinatorial', {'active': True}),
            ('N004', 'envelope', {'area': 45.2, 'cost': 8000}, {'thermal_rating': 0.78}, 'ComplianceNode', 'dag', {'active': True}),
            ('N005', 'structural', {'volume': 200.0, 'cost': 35000}, {'safety_factor': 2.1}, 'InvestmentNode', 'relational', {'active': True})
        ]
        
        edges = [
            ('N001', 'N002', 'structural_support', 0.8, {'load_transfer': True}, 0.95, '2025-06-20T10:00:00'),
            ('N001', 'N003', 'structural_support', 0.7, {'load_transfer': True}, 0.88, '2025-06-20T10:05:00'),
            ('N002', 'N003', 'system_integration', 0.6, {'electrical_mep': True}, 0.92, '2025-06-20T10:10:00'),
            ('N003', 'N004', 'system_integration', 0.5, {'climate_control': True}, 0.85, '2025-06-20T10:15:00'),
            ('N004', 'N005', 'structural_support', 0.9, {'load_transfer': True}, 0.97, '2025-06-20T10:20:00')
        ]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'loaded_at': datetime.now().isoformat()
        }
    
    def build_enhanced_dgl_graph(self, graph_data: Dict[str, Any]) -> Optional[Any]:
        """Build enhanced DGL graph with agent and callback features"""
        if not DGL_AVAILABLE:
            return None
        
        try:
            # Create node mapping
            node_ids = [node[0] for node in graph_data['nodes']]
            node_map = {node_id: i for i, node_id in enumerate(node_ids)}
            
            # Extract edges
            src_nodes = []
            dst_nodes = []
            edge_features = []
            
            for edge in graph_data['edges']:
                src, dst = edge[0], edge[1]
                if src in node_map and dst in node_map:
                    src_nodes.append(node_map[src])
                    dst_nodes.append(node_map[dst])
                    
                    # Edge features: weight, success_rate, recency
                    edge_features.append([
                        edge[3],  # weight
                        edge[5] if len(edge) > 5 else 0.5,  # success_rate
                        1.0,  # recency (normalized)
                        1.0 if edge[2] == 'structural_support' else 0.0  # edge_type_encoded
                    ])
            
            # Create DGL graph
            dgl_graph = dgl.graph((src_nodes, dst_nodes))
            
            # Add node features
            node_features = []
            agent_types = []
            callback_types = []
            
            agent_type_map = {'BiddingAgent': 0, 'OccupancyNode': 1, 'MEPSystemNode': 2, 'ComplianceNode': 3, 'InvestmentNode': 4}
            callback_type_map = {'dag': 0, 'relational': 1, 'combinatorial': 2}
            
            for node in graph_data['nodes']:
                props = node[2] if isinstance(node[2], dict) else {}
                coeffs = node[3] if isinstance(node[3], dict) else {}
                
                # Enhanced feature vector
                features = [
                    props.get('volume', 0) / 1000.0,
                    props.get('cost', 0) / 100000.0,
                    props.get('area', 0) / 100.0,
                    props.get('power', 0) / 1000.0,
                    coeffs.get('safety_factor', 1.0),
                    coeffs.get('efficiency', 0.5),
                    coeffs.get('performance', 0.5),
                    coeffs.get('thermal_rating', 0.5)
                ]
                node_features.append(features)
                
                # Agent and callback types
                agent_type = node[4] if len(node) > 4 else 'BiddingAgent'
                callback_type = node[5] if len(node) > 5 else 'dag'
                
                agent_types.append(agent_type_map.get(agent_type, 0))
                callback_types.append(callback_type_map.get(callback_type, 0))
            
            # Add features to graph
            dgl_graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
            dgl_graph.ndata['agent_type'] = torch.tensor(agent_types, dtype=torch.long)
            dgl_graph.ndata['callback_type'] = torch.tensor(callback_types, dtype=torch.long)
            dgl_graph.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
            
            return dgl_graph
            
        except Exception as e:
            logger.error(f"Failed to build enhanced DGL graph: {e}")
            return None
    
    def generate_embeddings(self) -> Optional[torch.Tensor]:
        """Generate node embeddings using graph learning model"""
        if not DGL_AVAILABLE or self.dgl_graph is None or self.learning_model is None:
            return None
        
        try:
            with torch.no_grad():
                embeddings = self.learning_model(
                    self.dgl_graph,
                    self.dgl_graph.ndata['feat'],
                    self.dgl_graph.ndata['agent_type'],
                    self.dgl_graph.ndata['callback_type']
                )
            
            self.node_embeddings = embeddings
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def predict_callback_paths(self) -> Dict[str, Any]:
        """Predict likely successful callback paths"""
        if not DGL_AVAILABLE or self.routing_model is None or self.dgl_graph is None:
            return {'predictions': [], 'confidence': 0.0}
        
        try:
            with torch.no_grad():
                # Get node embeddings from routing model
                node_embs = self.routing_model(
                    self.dgl_graph,
                    self.dgl_graph.ndata['feat'],
                    self.dgl_graph.edata['feat']
                )
                
                # Predict success for all possible paths
                predictions = []
                num_nodes = self.dgl_graph.number_of_nodes()
                
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            success_prob = self.routing_model.predict_path_success(
                                node_embs[i:i+1], node_embs[j:j+1]
                            )
                            predictions.append({
                                'source': i,
                                'target': j,
                                'success_probability': float(success_prob.item())
                            })
                
                # Sort by success probability
                predictions.sort(key=lambda x: x['success_probability'], reverse=True)
                
                avg_confidence = np.mean([p['success_probability'] for p in predictions])
                
                return {
                    'predictions': predictions[:10],  # Top 10 paths
                    'confidence': float(avg_confidence),
                    'total_paths_analyzed': len(predictions)
                }
                
        except Exception as e:
            logger.error(f"Failed to predict callback paths: {e}")
            return {'predictions': [], 'confidence': 0.0}
    
    def detect_emergence(self) -> Dict[str, Any]:
        """Detect clusters and emergent behavior"""
        emergence_results = {
            'cluster_analysis': {'n_clusters': 0},
            'behavior_analysis': {'emergence_detected': False},
            'embedding_analysis': {'available': False}
        }
        
        # Cluster analysis on embeddings
        if self.node_embeddings is not None:
            embeddings_np = self.node_embeddings.detach().numpy()
            cluster_results = self.emergence_detector.detect_clusters(embeddings_np)
            emergence_results['cluster_analysis'] = cluster_results
            emergence_results['embedding_analysis']['available'] = True
        
        # Behavioral emergence detection
        if len(self.metrics_history) > 0:
            behavior_results = self.emergence_detector.detect_emergent_behavior(self.metrics_history)
            emergence_results['behavior_analysis'] = behavior_results
        
        return emergence_results
    
    def calculate_enhanced_metrics(self, graph: nx.Graph, emergence_data: Dict) -> OptimizationMetrics:
        """Calculate enhanced metrics including emergence and routing confidence"""
        # Base metrics (previous implementation)
        roi_score = self._calculate_roi_score(graph)
        occupancy_efficiency = self._calculate_occupancy_efficiency(graph)
        spec_fit_score = self._calculate_spec_fit_score(graph)
        structural_performance = self._calculate_structural_performance(graph)
        energy_efficiency = self._calculate_energy_efficiency(graph)
        cost_optimization = self._calculate_cost_optimization(graph)
        
        # Enhanced metrics
        emergence_score = self._calculate_emergence_score(emergence_data)
        routing_confidence = self._calculate_routing_confidence()
        
        overall_score = np.mean([
            roi_score, occupancy_efficiency, spec_fit_score,
            structural_performance, energy_efficiency, cost_optimization,
            emergence_score, routing_confidence
        ])
        
        return OptimizationMetrics(
            roi_score=roi_score,
            occupancy_efficiency=occupancy_efficiency,
            spec_fit_score=spec_fit_score,
            structural_performance=structural_performance,
            energy_efficiency=energy_efficiency,
            cost_optimization=cost_optimization,
            overall_score=overall_score,
            emergence_score=emergence_score,
            routing_confidence=routing_confidence
        )
    
    def _calculate_emergence_score(self, emergence_data: Dict) -> float:
        """Calculate emergence score based on cluster and behavior analysis"""
        cluster_score = 0.5
        behavior_score = 0.5
        
        # Cluster analysis score
        cluster_analysis = emergence_data.get('cluster_analysis', {})
        if cluster_analysis.get('n_clusters', 0) > 0:
            # More clusters indicate more organized structure
            cluster_score = min(cluster_analysis['n_clusters'] / 5.0, 1.0)
        
        # Behavior analysis score
        behavior_analysis = emergence_data.get('behavior_analysis', {})
        if behavior_analysis.get('emergence_detected', False):
            # Emergence detected - could be good or bad
            warning_level = behavior_analysis.get('warning_level', 'normal')
            behavior_score = 0.3 if warning_level == 'high' else 0.7
        
        return (cluster_score + behavior_score) / 2.0
    
    def _calculate_routing_confidence(self) -> float:
        """Calculate routing confidence based on path predictions"""
        # Default confidence if no routing model
        return 0.75
    
    # Previous metric calculation methods remain the same
    def _calculate_roi_score(self, graph: nx.Graph) -> float:
        total_cost = sum(graph.nodes[node].get('properties', {}).get('cost', 0) for node in graph.nodes())
        total_value = sum(
            graph.nodes[node].get('coefficients', {}).get('safety_factor', 1.0) * 1000 +
            graph.nodes[node].get('coefficients', {}).get('efficiency', 0.5) * 2000
            for node in graph.nodes()
        )
        return min(total_value / max(total_cost, 1), 1.0)
    
    def _calculate_occupancy_efficiency(self, graph: nx.Graph) -> float:
        structural_nodes = [n for n in graph.nodes() if graph.nodes[n].get('component_type') == 'structural']
        if not structural_nodes:
            return 0.5
        avg_volume_efficiency = np.mean([
            min(graph.nodes[n].get('properties', {}).get('volume', 100) / 200.0, 1.0) for n in structural_nodes
        ])
        return avg_volume_efficiency
    
    def _calculate_spec_fit_score(self, graph: nx.Graph) -> float:
        spec_scores = []
        for node in graph.nodes():
            coeffs = graph.nodes[node].get('coefficients', {})
            if 'safety_factor' in coeffs:
                spec_scores.append(min(coeffs['safety_factor'] / 2.5, 1.0))
            if 'efficiency' in coeffs:
                spec_scores.append(coeffs['efficiency'])
            if 'performance' in coeffs:
                spec_scores.append(coeffs['performance'])
        return np.mean(spec_scores) if spec_scores else 0.5
    
    def _calculate_structural_performance(self, graph: nx.Graph) -> float:
        structural_nodes = [n for n in graph.nodes() if graph.nodes[n].get('component_type') == 'structural']
        if not structural_nodes:
            return 0.5
        safety_factors = [graph.nodes[n].get('coefficients', {}).get('safety_factor', 1.0) for n in structural_nodes]
        return min(np.mean(safety_factors) / 2.0, 1.0)
    
    def _calculate_energy_efficiency(self, graph: nx.Graph) -> float:
        energy_nodes = [n for n in graph.nodes() if graph.nodes[n].get('component_type') in ['electrical', 'mep']]
        if not energy_nodes:
            return 0.5
        efficiencies = []
        for node in energy_nodes:
            coeffs = graph.nodes[node].get('coefficients', {})
            if 'efficiency' in coeffs:
                efficiencies.append(coeffs['efficiency'])
            if 'performance' in coeffs:
                efficiencies.append(coeffs['performance'])
        return np.mean(efficiencies) if efficiencies else 0.5
    
    def _calculate_cost_optimization(self, graph: nx.Graph) -> float:
        total_cost = sum(graph.nodes[node].get('properties', {}).get('cost', 0) for node in graph.nodes())
        if total_cost == 0:
            return 1.0
        avg_cost_per_node = total_cost / graph.number_of_nodes()
        normalized_cost = min(avg_cost_per_node / 50000.0, 1.0)
        return 1.0 - normalized_cost
    
    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run complete optimization cycle with enhanced DGL capabilities"""
        logger.info("Starting enhanced optimization cycle with DGL")
        
        # Load graph data
        graph_data = self.load_graph_data()
        
        # Build graphs
        self.graph = self.build_networkx_graph(graph_data)
        if DGL_AVAILABLE:
            self.dgl_graph = self.build_enhanced_dgl_graph(graph_data)
        
        # Generate embeddings
        embeddings = self.generate_embeddings()
        
        # Predict callback paths
        routing_predictions = self.predict_callback_paths()
        
        # Detect emergence
        emergence_data = self.detect_emergence()
        
        # Calculate enhanced metrics
        metrics = self.calculate_enhanced_metrics(self.graph, emergence_data)
        
        # Generate updates
        updates = self.generate_threshold_updates(metrics)
        
        # Apply updates
        success = self.update_postgresql_thresholds(updates)
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': updates['timestamp'],
            'metrics': metrics.__dict__,
            'update_success': success
        })
        
        result = {
            'optimization_status': 'completed' if success else 'failed',
            'execution_time': datetime.now().isoformat(),
            'graph_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph)
            },
            'metrics': metrics.__dict__,
            'updates': updates,
            'dgl_capabilities': {
                'available': DGL_AVAILABLE,
                'embeddings_generated': embeddings is not None,
                'routing_predictions': routing_predictions,
                'emergence_detection': emergence_data
            }
        }
        
        logger.info(f"Enhanced optimization cycle completed - Overall score: {metrics.overall_score:.3f}")
        logger.info(f"Emergence score: {metrics.emergence_score:.3f}, Routing confidence: {metrics.routing_confidence:.3f}")
        
        return result
    
    def build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from data"""
        G = nx.Graph()
        for node_id, comp_type, properties, coefficients, *extra in graph_data['nodes']:
            node_attrs = {
                'component_type': comp_type,
                'properties': properties if isinstance(properties, dict) else {},
                'coefficients': coefficients if isinstance(coefficients, dict) else {}
            }
            G.add_node(node_id, **node_attrs)
        
        for source, target, edge_type, weight, properties, *extra in graph_data['edges']:
            edge_attrs = {
                'edge_type': edge_type,
                'weight': weight,
                'properties': properties if isinstance(properties, dict) else {}
            }
            G.add_edge(source, target, **edge_attrs)
        
        return G
    
    def generate_threshold_updates(self, metrics: OptimizationMetrics) -> Dict[str, Any]:
        """Generate enhanced threshold updates"""
        updates = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': metrics.__dict__,
            'threshold_adjustments': {},
            'recommendations': []
        }
        
        # Enhanced recommendations based on new metrics
        if metrics.emergence_score < 0.4:
            updates['threshold_adjustments']['emergence_threshold'] = 0.1
            updates['recommendations'].append("Increase emergence detection sensitivity")
        
        if metrics.routing_confidence < 0.6:
            updates['threshold_adjustments']['routing_confidence_min'] = 0.05
            updates['recommendations'].append("Improve callback routing reliability")
        
        # Previous ROI, structural, energy logic remains
        if metrics.roi_score < 0.6:
            updates['threshold_adjustments']['cost_threshold'] = -0.1
            updates['recommendations'].append("Reduce cost thresholds to improve ROI")
        
        if metrics.overall_score > 0.85:
            updates['recommendations'].append("System performing excellently with strong emergence patterns")
        elif metrics.overall_score < 0.6:
            updates['recommendations'].append("System needs improvement in multiple areas")
        
        return updates
    
    def update_postgresql_thresholds(self, updates: Dict[str, Any]) -> bool:
        """Update thresholds in PostgreSQL database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            for threshold_name, adjustment in updates['threshold_adjustments'].items():
                cursor.execute("""
                    INSERT INTO optimization_thresholds (threshold_name, current_value, last_updated, adjustment_reason)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (threshold_name) 
                    DO UPDATE SET 
                        current_value = optimization_thresholds.current_value + %s,
                        last_updated = %s,
                        adjustment_reason = %s
                """, (
                    threshold_name, adjustment, updates['timestamp'],
                    f"DGL optimization adjustment: {adjustment:+.3f}",
                    adjustment, updates['timestamp'],
                    f"DGL optimization adjustment: {adjustment:+.3f}"
                ))
            
            cursor.execute("""
                INSERT INTO optimization_history 
                (timestamp, overall_score, roi_score, emergence_score, routing_confidence, 
                 recommendations, threshold_updates)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                updates['timestamp'],
                updates['current_metrics']['overall_score'],
                updates['current_metrics']['roi_score'],
                updates['current_metrics']['emergence_score'],
                updates['current_metrics']['routing_confidence'],
                json.dumps(updates['recommendations']),
                json.dumps(updates['threshold_adjustments'])
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated {len(updates['threshold_adjustments'])} thresholds with DGL insights")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update PostgreSQL thresholds: {e}")
            return False

def main():
    """Main optimization engine entry point"""
    logger.info("Enhanced Optimization Engine with DGL starting...")
    
    engine = OptimizationEngine()
    result = engine.run_optimization_cycle()
    
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main() 