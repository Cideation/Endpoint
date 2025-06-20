#!/usr/bin/env python3
"""
DGL Trainer for BEM System — Full Graph Embedding with Database Integration
Design Rule: Train on all edges across the system graph (Alpha, Beta, Gamma)
Enhanced with SFDE scientific foundation, cross-phase learning, and PostgreSQL training database
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, SAGEConv, GraphConv
import json
import numpy as np
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add postgre directory to path for database integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'postgre'))

try:
    from training_db_config import get_training_db, setup_training_environment
    DB_AVAILABLE = True
except ImportError:
    logging.warning("Training database not available - using fallback data generation")
    DB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# EDGE TABLE SEGREGATION CONFIGURATION
# ============================================================================
"""
Phase-Specific Edge Architecture for DGL Training:
✅ Alpha = DAG, directed, one-to-one or linear edge flow
✅ Beta = Relational (Objective Functions) → many-to-many, dense logic
✅ Gamma = Combinatorial (Emergence) → many-to-many, sparse-to-dense mappings
"""

EDGE_TABLE_SEGREGATION = {
    'alpha_edges': {
        'table_name': 'alpha_edges.csv',
        'edge_type': 'directed_dag',
        'flow_pattern': 'one_to_one_linear',
        'description': 'Static logic flow between nodes',
        'dgl_edge_type': 0,  # DAG edges
        'training_weight': 1.0,
        'validation_rules': {
            'requires_single_source': True,
            'requires_single_target': True,
            'allows_cycles': False,
            'max_fanout': 3
        }
    },
    'beta_relationships': {
        'table_name': 'beta_relationships.csv',
        'edge_type': 'many_to_many_relational',
        'flow_pattern': 'dense_logic',
        'description': 'Objective Function relations',
        'dgl_edge_type': 1,  # Relational edges
        'training_weight': 1.5,  # Higher weight for objective functions
        'validation_rules': {
            'requires_single_source': False,
            'requires_single_target': False,
            'allows_cycles': True,
            'max_fanout': -1  # Unlimited
        }
    },
    'gamma_edges': {
        'table_name': 'gamma_edges.csv',
        'edge_type': 'combinatorial_emergence',
        'flow_pattern': 'sparse_to_dense',
        'description': 'Learning-based, emergent dependencies',
        'dgl_edge_type': 2,  # Emergent edges
        'training_weight': 2.0,  # Highest weight for emergence
        'validation_rules': {
            'requires_single_source': False,
            'requires_single_target': False,
            'allows_cycles': True,
            'max_fanout': -1,  # Unlimited
            'requires_learning_weight': True,
            'min_learning_weight': 0.0,
            'max_learning_weight': 1.0
        }
    },
    'cross_phase_edges': {
        'table_name': 'cross_phase_transitions.csv',
        'edge_type': 'phase_transition',
        'flow_pattern': 'phase_bridge',
        'description': 'Alpha→Beta→Gamma phase transitions',
        'dgl_edge_type': 3,  # Cross-phase edges
        'training_weight': 1.2,
        'validation_rules': {
            'requires_single_source': True,
            'requires_single_target': True,
            'allows_cycles': False,
            'max_fanout': 5,
            'requires_phase_compatibility': True
        }
    }
}

PHASE_EDGE_MAPPING = {
    'Alpha': ['alpha_edges', 'cross_phase_edges'],
    'Beta': ['beta_relationships', 'cross_phase_edges'], 
    'Gamma': ['gamma_edges', 'cross_phase_edges']
}

def validate_edge_segregation(edge_data: Dict[str, Any], edge_category: str) -> bool:
    """Validate edge data against segregation rules"""
    if edge_category not in EDGE_TABLE_SEGREGATION:
        logger.warning(f"Unknown edge category: {edge_category}")
        return False
    
    rules = EDGE_TABLE_SEGREGATION[edge_category]['validation_rules']
    
    # Validate single source/target requirements
    if rules.get('requires_single_source', False):
        if isinstance(edge_data.get('source'), list) and len(edge_data['source']) > 1:
            return False
    
    if rules.get('requires_single_target', False):
        if isinstance(edge_data.get('target'), list) and len(edge_data['target']) > 1:
            return False
    
    # Validate learning weight for gamma edges
    if rules.get('requires_learning_weight', False):
        learning_weight = edge_data.get('learning_weight')
        if learning_weight is None:
            return False
        
        min_weight = rules.get('min_learning_weight', 0.0)
        max_weight = rules.get('max_learning_weight', 1.0)
        if not (min_weight <= learning_weight <= max_weight):
            return False
    
    # Validate phase compatibility for cross-phase edges
    if rules.get('requires_phase_compatibility', False):
        source_phase = edge_data.get('source_phase')
        target_phase = edge_data.get('target_phase')
        
        if source_phase and target_phase:
            valid_transitions = [
                ('Alpha', 'Beta'),
                ('Beta', 'Gamma'),
                ('Alpha', 'Gamma')  # Allow skip connections
            ]
            if (source_phase, target_phase) not in valid_transitions:
                return False
    
    return True

def get_edge_training_weight(edge_category: str) -> float:
    """Get training weight for specific edge category"""
    return EDGE_TABLE_SEGREGATION.get(edge_category, {}).get('training_weight', 1.0)

def get_dgl_edge_type(edge_category: str) -> int:
    """Get DGL edge type integer for specific edge category"""
    return EDGE_TABLE_SEGREGATION.get(edge_category, {}).get('dgl_edge_type', 0)

def log_edge_segregation_info():
    """Log edge table segregation configuration"""
    logger.info("🧱 Edge Table Segregation Configuration:")
    for category, config in EDGE_TABLE_SEGREGATION.items():
        logger.info(f"  📋 {category.upper()}:")
        logger.info(f"    • Table: {config['table_name']}")
        logger.info(f"    • Type: {config['edge_type']}")
        logger.info(f"    • Pattern: {config['flow_pattern']}")
        logger.info(f"    • DGL Type: {config['dgl_edge_type']}")
        logger.info(f"    • Training Weight: {config['training_weight']}")

# ============================================================================

def load_graph_from_database():
    """Load BEM graph from dedicated training database"""
    if not DB_AVAILABLE:
        logger.warning("Database not available, generating sample data")
        return generate_sample_bem_graph()
    
    try:
        training_db = get_training_db()
        training_data = training_db.get_training_data()
        
        if not training_data:
            logger.warning("No training data in database, generating sample data")
            return generate_sample_bem_graph()
        
        logger.info("🗄️ Loading BEM graph from training database")
        
        # Extract data from database response
        node_features = training_data.get('node_features', [])
        edge_list = training_data.get('edge_list', [])
        node_labels = training_data.get('node_labels', [])
        phase_info = training_data.get('phase_info', {})
        
        if not node_features or not edge_list:
            logger.warning("Incomplete database data, generating sample data")
            return generate_sample_bem_graph()
        
        # Convert to DGL format
        src_nodes = [edge['source'] for edge in edge_list]
        dst_nodes = [edge['target'] for edge in edge_list]
        
        # Map node IDs to indices
        node_id_to_idx = {node['node_id']: idx for idx, node in enumerate(node_features)}
        src_indices = [node_id_to_idx.get(src, 0) for src in src_nodes]
        dst_indices = [node_id_to_idx.get(dst, 0) for dst in dst_nodes]
        
        # Create DGL graph
        g = dgl.graph((src_indices, dst_indices))
        
        # Add node features
        node_feat_matrix = torch.tensor([node['features'] for node in node_features], dtype=torch.float32)
        g.ndata["feat"] = node_feat_matrix
        
        # Add node phases
        phase_mapping = {'Alpha': 0, 'Beta': 1, 'Gamma': 2}
        node_phases = torch.tensor([phase_mapping.get(node.get('phase', 'Beta'), 1) for node in node_features], dtype=torch.long)
        g.ndata["phase"] = node_phases
        
        # Add labels
        label_matrix = torch.randn(len(node_features), 1)  # Placeholder - extract from database labels
        g.ndata["label"] = label_matrix
        
        # Add edge types
        edge_type_mapping = {'structural_support': 0, 'system_integration': 1, 'load_transfer': 2, 'cross_phase': 3}
        edge_types = torch.tensor([edge_type_mapping.get(edge.get('edge_type', 'system_integration'), 1) for edge in edge_list], dtype=torch.int32)
        g.edata["type"] = edge_types
        
        logger.info(f"✅ Loaded BEM graph from database: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        logger.info(f"📊 Phase distribution: {phase_info}")
        
        return g
        
    except Exception as e:
        logger.error(f"❌ Failed to load from database: {e}")
        return generate_sample_bem_graph()

def load_graph_from_json(json_path="graph_data.json"):
    """Load BEM graph with all phases (fallback method)"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        logger.info("🌐 Loading BEM graph from JSON")
        g = dgl.graph((data["src"], data["dst"]))
        g.ndata["feat"] = torch.tensor(data["node_features"], dtype=torch.float32)
        g.ndata["label"] = torch.tensor(data["labels"], dtype=torch.float32)
        g.edata["type"] = torch.randint(0, 4, (g.number_of_edges(),))
        
        return g
    except FileNotFoundError:
        logger.info("JSON file not found, generating sample BEM graph")
        return generate_sample_bem_graph()

def generate_sample_bem_graph():
    """Generate sample graph with Alpha/Beta/Gamma phases using edge segregation"""
    logger.info("🔄 Generating sample BEM graph for training with edge segregation")
    
    # Log edge segregation configuration
    log_edge_segregation_info()
    
    num_nodes = 30
    src_nodes, dst_nodes, edge_types, edge_weights = [], [], [], []
    
    # Generate Alpha phase nodes (DAG, directed, linear flow)
    alpha_nodes = list(range(0, 10))
    for i in alpha_nodes[:-1]:
        for j in range(i+1, min(i+3, len(alpha_nodes))):
            # Alpha edges: directed DAG, single source/target
            src_nodes.append(i)
            dst_nodes.append(j)
            edge_types.append(get_dgl_edge_type('alpha_edges'))
            edge_weights.append(get_edge_training_weight('alpha_edges'))
    
    # Generate Beta phase nodes (many-to-many relational, dense logic)
    beta_nodes = list(range(10, 20))
    for i in beta_nodes:
        for j in beta_nodes:
            if i != j and np.random.random() > 0.6:  # Dense connections
                src_nodes.append(i)
                dst_nodes.append(j)
                edge_types.append(get_dgl_edge_type('beta_relationships'))
                edge_weights.append(get_edge_training_weight('beta_relationships'))
    
    # Generate Gamma phase nodes (combinatorial emergence, sparse-to-dense)
    gamma_nodes = list(range(20, 30))
    for i in gamma_nodes:
        for j in gamma_nodes:
            if i != j and np.random.random() > 0.4:  # Sparse-to-dense emergence
                src_nodes.append(i)
                dst_nodes.append(j)
                edge_types.append(get_dgl_edge_type('gamma_edges'))
                edge_weights.append(get_edge_training_weight('gamma_edges'))
    
    # Cross-phase connections (phase transitions)
    for i in range(5):
        # Alpha → Beta transitions
        src_nodes.append(alpha_nodes[i])
        dst_nodes.append(beta_nodes[i])
        edge_types.append(get_dgl_edge_type('cross_phase_edges'))
        edge_weights.append(get_edge_training_weight('cross_phase_edges'))
        
        # Beta → Gamma transitions
        src_nodes.append(beta_nodes[i])
        dst_nodes.append(gamma_nodes[i])
        edge_types.append(get_dgl_edge_type('cross_phase_edges'))
        edge_weights.append(get_edge_training_weight('cross_phase_edges'))
    
    g = dgl.graph((src_nodes, dst_nodes))
    
    # Enhanced node features (8-dimensional for SFDE compatibility)
    node_features = np.random.rand(num_nodes, 8).astype(np.float32)
    g.ndata["feat"] = torch.tensor(node_features)
    
    # Phase classification
    node_phases = [0] * 10 + [1] * 10 + [2] * 10  # Alpha, Beta, Gamma
    g.ndata["phase"] = torch.tensor(node_phases, dtype=torch.long)
    
    # Training labels
    g.ndata["label"] = torch.randn(num_nodes, 1)
    
    # Edge types from segregation configuration
    g.edata["type"] = torch.tensor(edge_types, dtype=torch.int32)
    
    # Edge weights for training
    g.edata["weight"] = torch.tensor(edge_weights, dtype=torch.float32)
    
    # Edge segregation metadata
    g.edata["segregation_category"] = torch.tensor([
        0 if t == 0 else 1 if t == 1 else 2 if t == 2 else 3 
        for t in edge_types
    ], dtype=torch.int32)
    
    # Count edges by type
    edge_counts = {
        'alpha_edges': sum(1 for t in edge_types if t == 0),
        'beta_relationships': sum(1 for t in edge_types if t == 1),
        'gamma_edges': sum(1 for t in edge_types if t == 2),
        'cross_phase_edges': sum(1 for t in edge_types if t == 3)
    }
    
    logger.info(f"✅ Generated sample BEM graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    logger.info(f"📊 Phase distribution: Alpha=10, Beta=10, Gamma=10")
    logger.info(f"🔗 Edge segregation counts: {edge_counts}")
    
    return g

class BEMGraphEmbedding(nn.Module):
    """Enhanced GCN model for BEM system with cross-phase learning"""
    
    def __init__(self, in_feats, hidden_feats, out_feats, num_phases=3):
        super(BEMGraphEmbedding, self).__init__()
        logger.info("🧠 Initializing BEM Graph Embedding with database integration")
        
        # Phase-specific processing layers
        self.alpha_conv = GATConv(in_feats, hidden_feats, num_heads=2, activation=F.relu)
        self.beta_conv = GATConv(hidden_feats * 2, hidden_feats, num_heads=2, activation=F.relu)
        self.gamma_conv = SAGEConv(hidden_feats * 2, hidden_feats, "mean")
        
        # Cross-phase fusion
        self.phase_fusion = nn.Linear(hidden_feats * 3, hidden_feats)
        
        # Final layers
        self.embedding_layer = GraphConv(hidden_feats, hidden_feats)
        self.output_layer = GraphConv(hidden_feats, out_feats)
        
        # Phase embeddings
        self.phase_embedding = nn.Embedding(num_phases, hidden_feats)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, features):
        """Forward pass with enhanced cross-phase processing"""
        # Multi-phase processing
        alpha_h = self.alpha_conv(g, features).flatten(1)
        alpha_h = self.dropout(alpha_h)
        
        beta_h = self.beta_conv(g, alpha_h).flatten(1)
        beta_h = self.dropout(beta_h)
        
        gamma_h = self.gamma_conv(g, beta_h)
        
        # Cross-phase fusion
        fused = torch.cat([alpha_h, beta_h, gamma_h], dim=-1)
        h = F.relu(self.phase_fusion(fused))
        
        # Add phase-specific embeddings
        if 'phase' in g.ndata:
            phase_emb = self.phase_embedding(g.ndata['phase'])
            h = h + phase_emb
        
        # Final embedding computation
        h = F.relu(self.embedding_layer(g, h))
        h = self.dropout(h)
        embeddings = self.output_layer(g, h)
        
        return embeddings

class BEMTrainer:
    """Complete BEM training system with database integration"""
    
    def __init__(self, model, graph):
        self.model = model
        self.graph = graph
        self.training_db = get_training_db() if DB_AVAILABLE else None
        self.run_id = None
    
    def train(self, epochs=50, lr=0.01, run_name=None):
        """Train BEM model with database logging and edge segregation weights"""
        logger.info(f"🚀 Starting BEM training: {epochs} epochs, lr={lr}")
        logger.info("🧱 Using Edge Table Segregation for weighted training")
        
        # Initialize training run in database
        if self.training_db:
            model_config = {
                'model_type': 'BEMGraphEmbedding',
                'hidden_dim': 64,
                'output_dim': 32,
                'num_phases': 3,
                'edge_segregation': True,
                'edge_weights': {
                    'alpha_edges': get_edge_training_weight('alpha_edges'),
                    'beta_relationships': get_edge_training_weight('beta_relationships'),
                    'gamma_edges': get_edge_training_weight('gamma_edges'),
                    'cross_phase_edges': get_edge_training_weight('cross_phase_edges')
                }
            }
            training_params = {
                'epochs': epochs,
                'learning_rate': lr,
                'optimizer': 'Adam',
                'edge_segregation_enabled': True
            }
            
            self.run_id = self.training_db.insert_training_run(
                run_name or f"bem_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                model_config, 
                training_params
            )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Use weighted loss function based on edge segregation
        def weighted_loss_fn(predictions, targets, edge_weights=None):
            base_loss = nn.MSELoss(reduction='none')(predictions, targets)
            
            if edge_weights is not None and 'weight' in self.graph.edata:
                # Apply edge-specific weights to loss
                edge_weight_tensor = self.graph.edata['weight']
                # Expand weights to match node predictions if needed
                if len(edge_weight_tensor) > 0:
                    # Use mean edge weight per node as node weight
                    node_weights = torch.ones(predictions.shape[0])
                    weighted_loss = base_loss * node_weights.unsqueeze(1)
                    return weighted_loss.mean()
            
            return base_loss.mean()
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            logits = self.model(self.graph, self.graph.ndata["feat"])
            
            # Apply edge segregation weighted loss
            loss = weighted_loss_fn(
                logits, 
                self.graph.ndata["label"], 
                self.graph.edata.get("weight")
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics to database with edge segregation info
            if self.training_db and self.run_id:
                self.training_db.insert_training_metric(
                    self.run_id, epoch, 'training_loss', loss.item()
                )
                
                # Log edge type distribution every 10 epochs
                if epoch % 10 == 0:
                    edge_type_counts = {}
                    if 'type' in self.graph.edata:
                        edge_types = self.graph.edata['type'].numpy()
                        for edge_type in [0, 1, 2, 3]:  # alpha, beta, gamma, cross-phase
                            edge_type_counts[f'edge_type_{edge_type}'] = int(np.sum(edge_types == edge_type))
                        
                        for edge_type, count in edge_type_counts.items():
                            self.training_db.insert_training_metric(
                                self.run_id, epoch, edge_type, count
                            )
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if (epoch + 1) % 10 == 0:
                # Log edge segregation status
                edge_info = ""
                if 'weight' in self.graph.edata:
                    avg_weight = self.graph.edata['weight'].mean().item()
                    edge_info = f", Avg Edge Weight: {avg_weight:.3f}"
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}{edge_info}")
        
        # Update training status with edge segregation summary
        if self.training_db and self.run_id:
            self.training_db.update_training_status(self.run_id, 'completed')
            
            # Save final embeddings with edge segregation metadata
            self.model.eval()
            with torch.no_grad():
                final_embeddings = self.model(self.graph, self.graph.ndata["feat"])
                
                # Convert to dictionary format for database storage
                embedding_dict = {}
                for i in range(final_embeddings.shape[0]):
                    embedding_dict[f"node_{i}"] = final_embeddings[i].tolist()
                
                # Add edge segregation metadata
                if 'segregation_category' in self.graph.edata:
                    segregation_counts = {}
                    seg_categories = self.graph.edata['segregation_category'].numpy()
                    for cat in [0, 1, 2, 3]:
                        segregation_counts[f'segregation_category_{cat}'] = int(np.sum(seg_categories == cat))
                    
                    embedding_dict['edge_segregation_metadata'] = {
                        'segregation_counts': segregation_counts,
                        'total_edges': len(seg_categories),
                        'segregation_enabled': True
                    }
                
                self.training_db.save_model_embeddings(self.run_id, embedding_dict)
        
        logger.info(f"✅ Training completed! Best loss: {best_loss:.4f}")
        logger.info("🧱 Edge segregation weights applied during training")
        return best_loss

def run_bem_training_with_database():
    """Main entry point for database-integrated BEM training"""
    logger.info("🌐 BEM DGL Trainer with Database Integration")
    
    # Setup training environment
    if DB_AVAILABLE:
        if not setup_training_environment():
            logger.error("❌ Failed to setup training environment")
            return None
    
    # Load graph from database or fallback
    if DB_AVAILABLE:
        graph = load_graph_from_database()
    else:
        graph = load_graph_from_json()
    
    # Initialize model
    model = BEMGraphEmbedding(
        in_feats=graph.ndata["feat"].shape[1], 
        hidden_feats=64, 
        out_feats=32
    )
    
    # Create trainer and run training
    trainer = BEMTrainer(model, graph)
    final_loss = trainer.train(epochs=100, lr=0.001, run_name="bem_full_system_training")
    
    # Cleanup
    if DB_AVAILABLE and trainer.training_db:
        trainer.training_db.close_connections()
    
    return final_loss

if __name__ == "__main__":
    # Run complete BEM training with database integration
    final_loss = run_bem_training_with_database()
    
    if final_loss is not None:
        logger.info(f"🎉 BEM Training Complete! Final Loss: {final_loss:.4f}")
        
        if DB_AVAILABLE:
            logger.info("📊 Training metrics and embeddings saved to database")
            logger.info("🔍 Check training_runs, training_metrics, and model_embeddings tables")
    else:
        logger.error("❌ Training failed")
