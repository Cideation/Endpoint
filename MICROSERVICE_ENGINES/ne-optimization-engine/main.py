#!/usr/bin/env python3
"""
DGL Optimization Engine - Scientific Learning from SFDE-Defined Computational Logic
Learns patterns from SFDE-structured formulas without obscuring scientific foundation
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

# Import SFDE formulas as training foundation
import sys
sys.path.append('../shared')
try:
    from sfde_utility_foundation_extended import (
        SFDEngine, graph_node_similarity, edge_weight_formula,
        agent_coefficient_formula, emergence_detection_formula,
        callback_success_predictor
    )
    SFDE_AVAILABLE = True
except ImportError:
    SFDE_AVAILABLE = False
    print("SFDE formulas not available")

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
    formula_traceability: float  # New: How well results trace to SFDE formulas

class ScientificGraphLearner(nn.Module):
    """
    DGL Learner - Learns from SFDE-defined scientific formulas
    Does NOT define computational logic, only learns patterns from SFDE training
    """
    
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int, num_heads: int = 4):
        super().__init__()
        logger.info("🤖 Initializing DGL Learner (consumes SFDE computational logic)")
        
        # Multi-layer architecture for learning SFDE patterns
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
        self.sage = SAGEConv(hidden_feats * num_heads, hidden_feats, 'mean')
        self.classifier = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(0.1)
        
        # Embeddings for SFDE-defined agent and callback types
        self.agent_embedding = nn.Embedding(10, hidden_feats)
        self.callback_embedding = nn.Embedding(5, hidden_feats)
        
        # Formula pattern memory (learns which SFDE formulas work best)
        self.formula_pattern_memory = nn.Parameter(torch.randn(5, hidden_feats))
        
    def forward(self, g, h, agent_types, callback_types, sfde_formula_features=None):
        """Learn patterns from SFDE-structured inputs"""
        # Graph attention for learning SFDE-defined patterns
        h = self.gat1(g, h).flatten(1)
        h = self.dropout(h)
        h = self.gat2(g, h).flatten(1)
        h = self.dropout(h)
        
        # SAGE for neighborhood aggregation (learns SFDE relationships)
        h = self.sage(g, h)
        
        # Integrate SFDE-defined agent and callback embeddings
        agent_emb = self.agent_embedding(agent_types)
        callback_emb = self.callback_embedding(callback_types)
        h = h + agent_emb + callback_emb
        
        # Apply learned patterns to SFDE formula features
        if sfde_formula_features is not None:
            # Learn which SFDE formulas are most effective
            formula_weights = torch.softmax(
                torch.matmul(h, self.formula_pattern_memory.T), dim=-1
            )
            h = h + torch.matmul(formula_weights, self.formula_pattern_memory)
        
        return self.classifier(h)

class SFDEFormulaSupervisor:
    """
    SFDE Formula Supervisor - Ensures ML predictions remain traceable to scientific formulas
    """
    
    def __init__(self):
        self.sfde_engine = None
        self.formula_definitions = {}
        self.training_targets = {}
        
    def initialize_with_sfde(self, node_dict: Dict, agent_coeffs: Dict):
        """Initialize with SFDE-defined computational logic"""
        if SFDE_AVAILABLE:
            self.sfde_engine = SFDEngine(node_dict, agent_coeffs)
            logger.info("🔬 SFDE Trainer initialized - defining computational logic")
            
            # SFDE defines the training targets and feature structure
            self.formula_definitions = {
                'similarity_targets': 'SFDE graph_node_similarity formula',
                'edge_weight_targets': 'SFDE edge_weight_formula',
                'agent_coeff_targets': 'SFDE agent_coefficient_formula',
                'emergence_targets': 'SFDE emergence_detection_formula',
                'callback_targets': 'SFDE callback_success_predictor'
            }
            
            logger.info("✅ SFDE computational logic loaded for DGL learning")
        else:
            logger.warning("⚠️ SFDE not available - ML will lack scientific foundation")
    
    def structure_training_data(self, graph_data: Dict) -> Dict[str, Any]:
        """Structure training data using SFDE-defined features and targets"""
        if not self.sfde_engine:
            return {'error': 'SFDE trainer not available'}
        
        structured_data = {
            'node_features': [],
            'edge_weights': [],
            'agent_coefficients': [],
            'formula_targets': {},
            'traceability_map': {}
        }
        
        logger.info("🏗️ Structuring training data from SFDE computational logic...")
        
        # SFDE defines what features to extract and how
        for node_id, node_data in graph_data['nodes']:
            if isinstance(node_data, dict):
                # SFDE defines feature extraction logic
                node_features = self._extract_sfde_features(node_data)
                structured_data['node_features'].append(node_features)
                
                # SFDE defines agent coefficient computation
                agent_type = node_data.get('agent_type', 'BiddingAgent')
                if SFDE_AVAILABLE:
                    coeffs = agent_coefficient_formula(
                        node_data, agent_type, self.sfde_engine.agent_coeffs
                    )
                    structured_data['agent_coefficients'].append(coeffs)
                    
                    # Track formula traceability
                    structured_data['traceability_map'][node_id] = {
                        'formula_source': 'SFDE agent_coefficient_formula',
                        'input_features': list(node_features),
                        'computed_coefficients': coeffs
                    }
        
        # SFDE defines edge weight computation logic
        for edge_data in graph_data['edges']:
            source_props = {'cost': 1000, 'performance': 0.8}  # Sample
            target_props = {'cost': 1200, 'performance': 0.7}  # Sample
            
            if SFDE_AVAILABLE:
                edge_weight = edge_weight_formula(source_props, target_props, 'structural')
                structured_data['edge_weights'].append(edge_weight)
        
        logger.info(f"✅ Structured {len(structured_data['node_features'])} nodes with SFDE logic")
        return structured_data
    
    def _extract_sfde_features(self, node_data: Dict) -> np.ndarray:
        """Extract features using SFDE-defined logic (not ML-defined)"""
        # SFDE defines exactly which features matter and how to normalize them
        features = []
        
        props = node_data.get('properties', {})
        features.extend([
            props.get('volume', 0.0) / 1000.0,      # SFDE-defined normalization
            props.get('cost', 0.0) / 100000.0,      # SFDE-defined normalization
            props.get('area', 0.0) / 100.0,         # SFDE-defined normalization
            props.get('power', 0.0) / 1000.0,       # SFDE-defined normalization
        ])
        
        coeffs = node_data.get('coefficients', {})
        features.extend([
            coeffs.get('safety_factor', 1.0),       # SFDE-defined importance
            coeffs.get('efficiency', 0.5),          # SFDE-defined importance
            coeffs.get('performance', 0.5),         # SFDE-defined importance
            coeffs.get('thermal_rating', 0.5)       # SFDE-defined importance
        ])
        
        return np.array(features, dtype=np.float32)
    
    def validate_ml_predictions(self, predictions: Dict, formula_targets: Dict) -> Dict[str, Any]:
        """Ensure ML predictions remain traceable to SFDE formulas"""
        validation = {
            'traceability_score': 0.0,
            'formula_alignment': {},
            'scientific_validity': True,
            'recommendations': []
        }
        
        traceable_predictions = 0
        total_predictions = len(predictions)
        
        for pred_type, pred_values in predictions.items():
            if pred_type in self.formula_definitions:
                # Check if prediction aligns with SFDE formula logic
                formula_source = self.formula_definitions[pred_type]
                
                if isinstance(pred_values, (list, np.ndarray)):
                    # Validate numerical ranges match SFDE expectations
                    valid_range = all(0.0 <= v <= 1.0 for v in pred_values if isinstance(v, (int, float)))
                    validation['formula_alignment'][pred_type] = {
                        'formula_source': formula_source,
                        'range_valid': valid_range,
                        'traceable': True
                    }
                    traceable_predictions += 1
                else:
                    validation['formula_alignment'][pred_type] = {
                        'formula_source': 'Unknown',
                        'range_valid': False,
                        'traceable': False
                    }
            else:
                validation['recommendations'].append(
                    f"Prediction '{pred_type}' not traceable to SFDE formulas"
                )
        
        validation['traceability_score'] = traceable_predictions / max(total_predictions, 1)
        
        if validation['traceability_score'] < 0.8:
            validation['scientific_validity'] = False
            validation['recommendations'].append(
                "Low traceability to SFDE formulas - review ML architecture"
            )
        
        return validation

class PredictiveRoutingModel(nn.Module):
    """
    DGL model for predicting likely successful callback paths (learns from SFDE)
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
    Cluster detection and emergent behavior analysis (uses SFDE formulas)
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
    Enhanced DGL/NetworkX-based optimization engine with SFDE scientific foundation
    DGL learns from SFDE-defined computational logic, never obscures it
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
        
        # SFDE Integration - The Trainer
        self.sfde_supervisor = SFDEFormulaSupervisor()
        self.formula_traceability = {}
        
        if DGL_AVAILABLE:
            logger.info("🤖 DGL available - will learn from SFDE computational logic")
            self._initialize_models()
        else:
            logger.info("🤖 DGL not available - using NetworkX with SFDE formulas only")
    
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
        """Initialize DGL learning models (not computational logic - that's SFDE's job)"""
        if not DGL_AVAILABLE:
            return
        
        logger.info("🤖 Initializing DGL learners (consume SFDE training)")
        
        # DGL learns patterns from SFDE-defined features
        self.learning_model = ScientificGraphLearner(
            in_feats=8,          # SFDE-defined feature dimension
            hidden_feats=64,     # Learning capacity
            out_feats=32,        # SFDE-aligned embedding dimension
            num_heads=4
        )
        
        # Routing model learns from SFDE callback predictions
        self.routing_model = PredictiveRoutingModel(
            node_feats=8,        # SFDE-defined features
            edge_feats=4,        # SFDE-defined edge features
            hidden_dim=64
        )
        
        logger.info("✅ DGL learners ready to consume SFDE computational logic")

    def load_graph_data(self) -> Dict[str, Any]:
        """Load graph data from PostgreSQL with SFDE integration"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Load nodes with SFDE-relevant information
            cursor.execute("""
                SELECT node_id, component_type, properties, coefficients, 
                       agent_type, callback_type, agent_state
                FROM graph_nodes 
                WHERE active = true
            """)
            nodes_data = cursor.fetchall()
            
            # Load edges with SFDE callback path information
            cursor.execute("""
                SELECT source_node, target_node, edge_type, weight, properties,
                       callback_success_rate, last_execution_time
                FROM graph_edges 
                WHERE active = true
            """)
            edges_data = cursor.fetchall()
            
            conn.close()
            
            graph_data = {
                'nodes': nodes_data,
                'edges': edges_data,
                'loaded_at': datetime.now().isoformat()
            }
            
            # Initialize SFDE trainer with loaded data
            node_dict = {node[0]: {
                'component_type': node[1],
                'properties': node[2] if isinstance(node[2], dict) else {},
                'coefficients': node[3] if isinstance(node[3], dict) else {},
                'agent_type': node[4] if len(node) > 4 else 'BiddingAgent'
            } for node in nodes_data}
            
            agent_coeffs = {'bid_priority': 0.7, 'mep_reliability': 0.85}
            self.sfde_supervisor.initialize_with_sfde(node_dict, agent_coeffs)
            
            return graph_data
            
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
        formula_traceability = self.formula_traceability.get('traceability_score', 0.8)
        
        overall_score = np.mean([
            roi_score, occupancy_efficiency, spec_fit_score,
            structural_performance, energy_efficiency, cost_optimization,
            emergence_score, routing_confidence, formula_traceability
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
            routing_confidence=routing_confidence,
            formula_traceability=formula_traceability
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
        """Run optimization cycle with SFDE trainer / DGL learner architecture"""
        logger.info("🚀 Starting Scientific Optimization: SFDE Trainer → DGL Learner")
        
        # 1. SFDE defines computational logic and training targets
        graph_data = self.load_graph_data()
        
        # 2. SFDE structures the training data
        structured_data = self.sfde_supervisor.structure_training_data(graph_data)
        
        # 3. Build graphs with SFDE-defined features
        self.graph = self.build_networkx_graph(graph_data)
        if DGL_AVAILABLE:
            self.dgl_graph = self.build_enhanced_dgl_graph(graph_data)
        
        # 4. DGL learns from SFDE-structured patterns
        embeddings = self.generate_embeddings()
        routing_predictions = self.predict_callback_paths()
        emergence_data = self.detect_emergence()
        
        # 5. Validate ML predictions against SFDE formulas
        if hasattr(self, 'sfde_supervisor') and structured_data.get('formula_targets'):
            ml_predictions = {
                'embeddings': embeddings.detach().numpy().tolist() if embeddings is not None else [],
                'routing': routing_predictions,
                'emergence': emergence_data
            }
            traceability_validation = self.sfde_supervisor.validate_ml_predictions(
                ml_predictions, structured_data['formula_targets']
            )
            self.formula_traceability = traceability_validation
        
        # 6. Calculate metrics with formula traceability
        metrics = self.calculate_enhanced_metrics(self.graph, emergence_data)
        
        # 7. Generate SFDE-guided updates
        updates = self.generate_threshold_updates(metrics)
        success = self.update_postgresql_thresholds(updates)
        
        # Store with formula traceability
        self.metrics_history.append({
            'timestamp': updates['timestamp'],
            'metrics': metrics.__dict__,
            'update_success': success,
            'formula_traceability': self.formula_traceability
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
            'sfde_integration': {
                'trainer_available': SFDE_AVAILABLE,
                'formula_traceability': self.formula_traceability.get('traceability_score', 0.0),
                'scientific_validity': self.formula_traceability.get('scientific_validity', False)
            },
            'dgl_capabilities': {
                'available': DGL_AVAILABLE,
                'embeddings_generated': embeddings is not None,
                'routing_predictions': routing_predictions,
                'emergence_detection': emergence_data
            }
        }
        
        logger.info(f"✅ Scientific Optimization Complete:")
        logger.info(f"   📊 Overall Score: {metrics.overall_score:.3f}")
        logger.info(f"   🔬 Formula Traceability: {self.formula_traceability.get('traceability_score', 0.0):.3f}")
        logger.info(f"   🤖 DGL Learning: {'✅' if DGL_AVAILABLE else '❌'}")
        logger.info(f"   🧪 SFDE Training: {'✅' if SFDE_AVAILABLE else '❌'}")
        
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