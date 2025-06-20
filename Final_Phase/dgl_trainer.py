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
    """Generate sample graph with Alpha/Beta/Gamma phases"""
    logger.info("🔄 Generating sample BEM graph for training")
    
    num_nodes = 30
    src_nodes, dst_nodes = [], []
    
    # Generate Alpha phase nodes (hierarchical structure)
    alpha_nodes = list(range(0, 10))
    for i in alpha_nodes[:-1]:
        for j in range(i+1, min(i+3, len(alpha_nodes))):
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
    
    # Generate Beta phase nodes (relational)
    beta_nodes = list(range(10, 20))
    for i in beta_nodes:
        for j in beta_nodes:
            if i != j and np.random.random() > 0.7:
                src_nodes.append(i)
                dst_nodes.append(j)
    
    # Generate Gamma phase nodes (combinatorial)
    gamma_nodes = list(range(20, 30))
    for i in gamma_nodes:
        for j in gamma_nodes:
            if i != j and np.random.random() > 0.5:
                src_nodes.append(i)
                dst_nodes.append(j)
    
    # Cross-phase connections
    for i in range(5):
        src_nodes.append(alpha_nodes[i])
        dst_nodes.append(beta_nodes[i])
        src_nodes.append(beta_nodes[i])
        dst_nodes.append(gamma_nodes[i])
    
    g = dgl.graph((src_nodes, dst_nodes))
    
    # Enhanced node features (8-dimensional for SFDE compatibility)
    node_features = np.random.rand(num_nodes, 8).astype(np.float32)
    g.ndata["feat"] = torch.tensor(node_features)
    
    # Phase classification
    node_phases = [0] * 10 + [1] * 10 + [2] * 10  # Alpha, Beta, Gamma
    g.ndata["phase"] = torch.tensor(node_phases, dtype=torch.long)
    
    # Training labels
    g.ndata["label"] = torch.randn(num_nodes, 1)
    
    # Edge types
    g.edata["type"] = torch.randint(0, 4, (g.number_of_edges(),))
    
    logger.info(f"✅ Generated sample BEM graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    logger.info(f"📊 Phase distribution: Alpha=10, Beta=10, Gamma=10")
    
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
        """Train BEM model with database logging"""
        logger.info(f"🚀 Starting BEM training: {epochs} epochs, lr={lr}")
        
        # Initialize training run in database
        if self.training_db:
            model_config = {
                'model_type': 'BEMGraphEmbedding',
                'hidden_dim': 64,
                'output_dim': 32,
                'num_phases': 3
            }
            training_params = {
                'epochs': epochs,
                'learning_rate': lr,
                'optimizer': 'Adam'
            }
            
            self.run_id = self.training_db.insert_training_run(
                run_name or f"bem_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                model_config, 
                training_params
            )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            logits = self.model(self.graph, self.graph.ndata["feat"])
            loss = loss_fn(logits, self.graph.ndata["label"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics to database
            if self.training_db and self.run_id:
                self.training_db.insert_training_metric(
                    self.run_id, epoch, 'training_loss', loss.item()
                )
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Update training status
        if self.training_db and self.run_id:
            self.training_db.update_training_status(self.run_id, 'completed')
            
            # Save final embeddings
            self.model.eval()
            with torch.no_grad():
                final_embeddings = self.model(self.graph, self.graph.ndata["feat"])
                
                # Convert to dictionary format for database storage
                embedding_dict = {}
                for i in range(final_embeddings.shape[0]):
                    embedding_dict[f"node_{i}"] = final_embeddings[i].tolist()
                
                self.training_db.save_model_embeddings(self.run_id, embedding_dict)
        
        logger.info(f"✅ Training completed! Best loss: {best_loss:.4f}")
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
