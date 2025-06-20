#!/usr/bin/env python3
"""
DGL Trainer for BEM System — Full Graph Embedding
Design Rule: Train on all edges across the system graph (Alpha, Beta, Gamma)
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv, SAGEConv
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_graph_from_json(json_path="graph_data.json"):
    """Load BEM graph with all phases"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        logger.info("🌐 Loading BEM graph with ALL phases")
        g = dgl.graph((data["src"], data["dst"]))
        g.ndata["feat"] = torch.tensor(data["node_features"], dtype=torch.float32)
        g.ndata["label"] = torch.tensor(data["labels"], dtype=torch.float32)
        g.edata["type"] = torch.tensor(data["edge_types"], dtype=torch.int32)
        
        return g
    except FileNotFoundError:
        logger.info("Generating sample BEM graph")
        return generate_sample_bem_graph()

def generate_sample_bem_graph():
    """Generate sample graph with Alpha/Beta/Gamma phases"""
    num_nodes = 20
    src_nodes, dst_nodes = [], []
    
    # Generate edges across phases
    for i in range(num_nodes):
        for j in range(i+1, min(i+4, num_nodes)):
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
    
    g = dgl.graph((src_nodes, dst_nodes))
    g.ndata["feat"] = torch.randn(num_nodes, 8)
    g.ndata["label"] = torch.randn(num_nodes, 1)
    g.edata["type"] = torch.randint(0, 4, (g.number_of_edges(),))
    
    logger.info(f"Generated: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g

class BEMGraphEmbedding(nn.Module):
    """Cross-phase GCN for BEM system"""
    
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(BEMGraphEmbedding, self).__init__()
        logger.info("🧠 Initializing BEM Graph Embedding")
        
        self.alpha_conv = GATConv(in_feats, hidden_feats, num_heads=2)
        self.beta_conv = GATConv(hidden_feats * 2, hidden_feats, num_heads=2)
        self.gamma_conv = SAGEConv(hidden_feats * 2, hidden_feats, "mean")
        self.fusion = nn.Linear(hidden_feats * 3, hidden_feats)
        self.output = GraphConv(hidden_feats, out_feats)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, features):
        # Multi-phase processing
        alpha_h = self.alpha_conv(g, features).flatten(1)
        alpha_h = self.dropout(alpha_h)
        
        beta_h = self.beta_conv(g, alpha_h).flatten(1)
        beta_h = self.dropout(beta_h)
        
        gamma_h = self.gamma_conv(g, beta_h)
        
        # Cross-phase fusion
        fused = torch.cat([alpha_h, beta_h, gamma_h], dim=-1)
        h = F.relu(self.fusion(fused))
        
        return self.output(g, h)

def train(g, model, epochs=50, lr=0.01):
    """Train BEM model"""
    logger.info(f"🚀 Training BEM model: {epochs} epochs")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        logits = model(g, g.ndata["feat"])
        loss = loss_fn(logits, g.ndata["label"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    logger.info("🌐 BEM DGL Trainer - Full Graph Embedding")
    
    graph = load_graph_from_json()
    model = BEMGraphEmbedding(
        in_feats=graph.ndata["feat"].shape[1], 
        hidden_feats=64, 
        out_feats=32
    )
    
    train(graph, model, epochs=100, lr=0.001)
    logger.info("✅ BEM Training Complete!")
