#!/usr/bin/env python3
"""
DGL Graph Builder for BEM System Training Loop
Builds DGL graphs from node_features.json and edge_features.json with user interaction support
"""

import dgl
import torch
import torch.nn.functional as F
import json
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DGLGraphBuilder:
    """Build DGL graphs from feature configurations with user interaction support"""
    
    def __init__(self, node_features_path: str = "Final_Phase/node_features.json",
                 edge_features_path: str = "Final_Phase/edge_features.json"):
        self.node_features_path = node_features_path
        self.edge_features_path = edge_features_path
        
        # Load feature configurations
        self.node_config = self._load_json_config(node_features_path)
        self.edge_config = self._load_json_config(edge_features_path)
        
        # Feature dimensions
        self.node_feature_dim = self.node_config.get("feature_dimensions", {}).get("total_dimensions", 18)
        self.edge_feature_dim = self.edge_config.get("edge_feature_dimensions", {}).get("total_dimensions", 15)
        
        logger.info(f"🏗️ DGL Graph Builder initialized")
        logger.info(f"   Node feature dimensions: {self.node_feature_dim}")
        logger.info(f"   Edge feature dimensions: {self.edge_feature_dim}")
    
    def _load_json_config(self, file_path: str) -> Dict[str, Any]:
        """Load JSON configuration file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return {}
    
    def build_graph_from_data(self, nodes_data: List[Dict[str, Any]], 
                             edges_data: List[Dict[str, Any]],
                             user_interactions: Dict[str, Any] = None) -> dgl.DGLGraph:
        """Build DGL graph from node and edge data with user interactions"""
        
        logger.info(f"🔄 Building DGL graph: {len(nodes_data)} nodes, {len(edges_data)} edges")
        
        # Create node mapping
        node_ids = [node.get("id", f"node_{i}") for i, node in enumerate(nodes_data)]
        node_map = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Extract edges
        src_nodes, dst_nodes = [], []
        edge_features = []
        
        for edge in edges_data:
            src_id = edge.get("source", edge.get("src"))
            dst_id = edge.get("target", edge.get("dst"))
            
            if src_id in node_map and dst_id in node_map:
                src_nodes.append(node_map[src_id])
                dst_nodes.append(node_map[dst_id])
                
                # Build edge features
                edge_feat = self._build_edge_features(edge, user_interactions)
                edge_features.append(edge_feat)
        
        # Create DGL graph
        if not src_nodes:
            logger.warning("No valid edges found, creating empty graph")
            g = dgl.graph(([], []))
            g.add_nodes(len(nodes_data))
        else:
            g = dgl.graph((src_nodes, dst_nodes))
        
        # Add node features
        node_features = []
        for node in nodes_data:
            node_feat = self._build_node_features(node, user_interactions)
            node_features.append(node_feat)
        
        # Add features to graph
        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        
        if edge_features:
            g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
        
        # Add metadata
        g.ndata['node_ids'] = torch.tensor([node_map[nid] for nid in node_ids], dtype=torch.long)
        
        if edge_features:
            edge_types = [self._get_edge_type_id(edge) for edge in edges_data if 
                         edge.get("source", edge.get("src")) in node_map and 
                         edge.get("target", edge.get("dst")) in node_map]
            g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)
        
        logger.info(f"✅ DGL graph built: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        return g
    
    def _build_node_features(self, node_data: Dict[str, Any], 
                            user_interactions: Dict[str, Any] = None) -> List[float]:
        """Build node feature vector from node data"""
        
        node_type = node_data.get("type", "V01_ProductComponent")
        node_id = node_data.get("id", "unknown")
        
        # Get feature template for node type
        feature_template = self.node_config.get("node_feature_templates", {}).get(node_type)
        if not feature_template:
            # Use first available template as fallback
            available_templates = list(self.node_config.get("node_feature_templates", {}).keys())
            if available_templates:
                feature_template = self.node_config["node_feature_templates"][available_templates[0]]
                logger.debug(f"Using fallback template for {node_type}")
            else:
                logger.warning(f"No feature template found for {node_type}")
                return [0.5] * self.node_feature_dim
        
        features = []
        
        # Build base features
        for feat_def in feature_template.get("base_features", []):
            value = node_data.get(feat_def["name"], 0.5)
            if feat_def.get("normalize", False):
                value = self._normalize_value(value, feat_def["range"])
            features.append(float(value))
        
        # Build agent features
        for feat_def in feature_template.get("agent_features", []):
            value = node_data.get(feat_def["name"], 0.5)
            if feat_def.get("normalize", False):
                value = self._normalize_value(value, feat_def["range"])
            features.append(float(value))
        
        # Build interaction features (with user interaction data)
        for feat_def in feature_template.get("interaction_features", []):
            if user_interactions and node_id in user_interactions:
                user_data = user_interactions[node_id]
                value = user_data.get(feat_def["name"], 0.0)
            else:
                value = node_data.get(feat_def["name"], 0.0)
            
            if feat_def.get("normalize", False):
                value = self._normalize_value(value, feat_def["range"])
            features.append(float(value))
        
        # Pad or truncate to expected dimension
        while len(features) < self.node_feature_dim:
            features.append(0.0)
        
        return features[:self.node_feature_dim]
    
    def _build_edge_features(self, edge_data: Dict[str, Any], 
                            user_interactions: Dict[str, Any] = None) -> List[float]:
        """Build edge feature vector from edge data"""
        
        edge_type = edge_data.get("type", "alpha_edges")
        edge_id = f"{edge_data.get('source', 'src')}_{edge_data.get('target', 'dst')}"
        
        # Get edge type configuration
        edge_type_config = self.edge_config.get("edge_types", {}).get(edge_type)
        if not edge_type_config:
            # Use first available edge type as fallback
            available_types = list(self.edge_config.get("edge_types", {}).keys())
            if available_types:
                edge_type_config = self.edge_config["edge_types"][available_types[0]]
                logger.debug(f"Using fallback edge type for {edge_type}")
            else:
                logger.warning(f"No edge type configuration found for {edge_type}")
                return [0.5] * self.edge_feature_dim
        
        features = []
        
        # Build structural features
        for feat_def in edge_type_config.get("structural_features", []):
            value = edge_data.get(feat_def["name"], 0.5)
            if feat_def.get("normalize", False):
                value = self._normalize_value(value, feat_def["range"])
            features.append(float(value))
        
        # Build interaction features (with user interaction data)
        for feat_def in edge_type_config.get("interaction_features", []):
            if user_interactions and edge_id in user_interactions:
                user_data = user_interactions[edge_id]
                value = user_data.get(feat_def["name"], 0.0)
            else:
                value = edge_data.get(feat_def["name"], 0.0)
            
            if feat_def.get("normalize", False):
                value = self._normalize_value(value, feat_def["range"])
            features.append(float(value))
        
        # Build learning features
        for feat_def in edge_type_config.get("learning_features", []):
            value = edge_data.get(feat_def["name"], 0.5)
            if feat_def.get("normalize", False):
                value = self._normalize_value(value, feat_def["range"])
            features.append(float(value))
        
        # Pad or truncate to expected dimension
        while len(features) < self.edge_feature_dim:
            features.append(0.0)
        
        return features[:self.edge_feature_dim]
    
    def _normalize_value(self, value: float, value_range: List[float]) -> float:
        """Normalize value to 0-1 range"""
        min_val, max_val = value_range
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def _get_edge_type_id(self, edge_data: Dict[str, Any]) -> int:
        """Get edge type ID from edge data"""
        edge_type = edge_data.get("type", "alpha_edges")
        edge_type_config = self.edge_config.get("edge_types", {}).get(edge_type, {})
        return edge_type_config.get("type_id", 0)
    
    def build_training_graph(self, graph_data: Dict[str, Any],
                           user_feedback: Dict[str, Any] = None) -> dgl.DGLGraph:
        """Build training graph with user feedback integration"""
        
        nodes_data = graph_data.get("nodes", [])
        edges_data = graph_data.get("edges", [])
        
        # Process user feedback into interaction format
        user_interactions = {}
        if user_feedback:
            user_interactions = self._process_user_feedback(user_feedback)
        
        # Build graph
        graph = self.build_graph_from_data(nodes_data, edges_data, user_interactions)
        
        # Add training labels
        labels = self._generate_training_labels(nodes_data, user_interactions)
        graph.ndata['label'] = torch.tensor(labels, dtype=torch.float32)
        
        # Add reward scores
        rewards = self._calculate_reward_scores(nodes_data, user_interactions)
        graph.ndata['reward'] = torch.tensor(rewards, dtype=torch.float32)
        
        return graph
    
    def _process_user_feedback(self, user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback into interaction features"""
        
        interactions = {}
        
        for item_id, feedback in user_feedback.items():
            interactions[item_id] = {
                "user_rating": feedback.get("rating", 0) / 5.0,  # Normalize to 0-1
                "user_interaction_count": min(feedback.get("interaction_count", 0), 100),
                "user_preference_score": feedback.get("preference", 0.5),
                "feedback_sentiment": feedback.get("sentiment", 0.0),  # -1 to 1
                "user_approval_rate": feedback.get("approval", 0.5),
                "user_modification_count": min(feedback.get("modifications", 0), 20)
            }
        
        return interactions
    
    def _generate_training_labels(self, nodes_data: List[Dict[str, Any]], 
                                 user_interactions: Dict[str, Any]) -> List[float]:
        """Generate training labels incorporating user feedback"""
        
        labels = []
        for node in nodes_data:
            node_id = node.get("id", "unknown")
            base_score = node.get("score", 0.5)
            
            # Incorporate user feedback if available
            if user_interactions and node_id in user_interactions:
                user_data = user_interactions[node_id]
                user_rating = user_data.get("user_rating", 0.5)
                user_preference = user_data.get("user_preference_score", 0.5)
                
                # Weighted combination of base score and user feedback
                label = (base_score * 0.6) + (user_rating * 0.25) + (user_preference * 0.15)
            else:
                label = base_score
            
            labels.append(float(label))
        
        return labels
    
    def _calculate_reward_scores(self, nodes_data: List[Dict[str, Any]], 
                               user_interactions: Dict[str, Any]) -> List[float]:
        """Calculate reward scores for reinforcement learning"""
        
        rewards = []
        for node in nodes_data:
            node_id = node.get("id", "unknown")
            base_reward = node.get("reward_score", 0.5)
            
            # Incorporate user feedback into reward
            if user_interactions and node_id in user_interactions:
                user_data = user_interactions[node_id]
                user_approval = user_data.get("user_approval_rate", 0.5)
                interaction_frequency = min(user_data.get("user_interaction_count", 0) / 100.0, 1.0)
                
                # Reward calculation with user engagement
                reward = base_reward + (user_approval * 0.3) + (interaction_frequency * 0.1)
                reward = max(0.0, min(1.0, reward))
            else:
                reward = base_reward
            
            rewards.append(float(reward))
        
        return rewards
    
    def create_sample_training_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create sample training data for testing"""
        
        # Sample nodes
        nodes_data = [
            {
                "id": "V01_ProductComponent_001",
                "type": "V01_ProductComponent",
                "volume": 150.0,
                "cost": 25000.0,
                "area": 45.0,
                "quality_factor": 0.85,
                "manufacturing_score": 0.78,
                "safety_factor": 2.1,
                "efficiency": 0.82,
                "performance": 0.79,
                "score": 0.8,
                "reward_score": 0.75
            },
            {
                "id": "V02_EconomicProfile_001", 
                "type": "V02_EconomicProfile",
                "cost_efficiency": 0.72,
                "market_demand": 0.68,
                "roi_potential": 1.45,
                "investment_risk": 0.35,
                "payback_period": 36,
                "profit_margin": 0.22,
                "economic_stability": 0.78,
                "market_position": 0.65,
                "score": 0.7,
                "reward_score": 0.68
            },
            {
                "id": "V05_ComplianceCheck_001",
                "type": "V05_ComplianceCheck", 
                "regulatory_score": 0.95,
                "safety_rating": 0.92,
                "code_compliance": 0.88,
                "certification_level": 4,
                "audit_score": 87,
                "risk_assessment": 0.15,
                "documentation_quality": 0.91,
                "compliance_confidence": 0.89,
                "score": 0.9,
                "reward_score": 0.88
            }
        ]
        
        # Sample edges
        edges_data = [
            {
                "source": "V01_ProductComponent_001",
                "target": "V02_EconomicProfile_001",
                "type": "alpha_edges",
                "edge_weight": 0.75,
                "flow_capacity": 85,
                "structural_strength": 0.8,
                "dependency_level": 3,
                "criticality_score": 0.7,
                "reliability_factor": 0.85,
                "success_rate": 0.78
            },
            {
                "source": "V02_EconomicProfile_001",
                "target": "V05_ComplianceCheck_001",
                "type": "beta_relationships",
                "relationship_strength": 1.2,
                "mutual_dependency": 0.65,
                "optimization_potential": 0.8,
                "conflict_resolution": 0.9,
                "synergy_score": 0.75,
                "resource_sharing": 0.7,
                "success_rate": 0.82
            }
        ]
        
        return nodes_data, edges_data
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions for model initialization"""
        return {
            "node_features": self.node_feature_dim,
            "edge_features": self.edge_feature_dim,
            "node_types": len(self.node_config.get("node_feature_templates", {})),
            "edge_types": len(self.edge_config.get("edge_types", {}))
        }

# Demo function
def demo_dgl_graph_builder():
    """Demonstrate DGL graph building with user interactions"""
    
    logger.info("🎭 Starting DGL Graph Builder Demo...")
    
    # Initialize builder
    builder = DGLGraphBuilder()
    
    # Create sample data
    nodes_data, edges_data = builder.create_sample_training_data()
    
    # Sample user feedback
    user_feedback = {
        "V01_ProductComponent_001": {
            "rating": 4,
            "interaction_count": 15,
            "preference": 0.8,
            "sentiment": 0.6,
            "approval": 0.85,
            "modifications": 3
        },
        "V02_EconomicProfile_001": {
            "rating": 3,
            "interaction_count": 8,
            "preference": 0.6,
            "sentiment": 0.2,
            "approval": 0.65,
            "modifications": 1
        }
    }
    
    # Build training graph
    graph_data = {"nodes": nodes_data, "edges": edges_data}
    training_graph = builder.build_training_graph(graph_data, user_feedback)
    
    # Display results
    logger.info(f"✅ Training graph built:")
    logger.info(f"   Nodes: {training_graph.number_of_nodes()}")
    logger.info(f"   Edges: {training_graph.number_of_edges()}")
    logger.info(f"   Node features shape: {training_graph.ndata['feat'].shape}")
    if 'feat' in training_graph.edata:
        logger.info(f"   Edge features shape: {training_graph.edata['feat'].shape}")
    logger.info(f"   Labels shape: {training_graph.ndata['label'].shape}")
    logger.info(f"   Rewards shape: {training_graph.ndata['reward'].shape}")
    
    # Feature dimensions
    dims = builder.get_feature_dimensions()
    logger.info(f"📊 Feature dimensions: {dims}")
    
    return training_graph

if __name__ == "__main__":
    demo_dgl_graph_builder() 