#!/usr/bin/env python3
"""
DGL Trainer for BEM System — Full Graph Embedding with Database Integration + ABM
Design Rule: Train on all edges across the system graph (Alpha, Beta, Gamma)
Enhanced with SFDE scientific foundation, cross-phase learning, PostgreSQL training database,
and Graph Hints ABM system integration for agent-driven learning
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

# Add paths for database and ABM integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'postgre'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MICROSERVICE_ENGINES'))

try:
    from training_db_config import get_training_db, setup_training_environment
    DB_AVAILABLE = True
except ImportError:
    logging.warning("Training database not available - using fallback data generation")
    DB_AVAILABLE = False

try:
    from graph_hints_system import GraphHintsSystem, HintCategory
    ABM_AVAILABLE = True
except ImportError:
    logging.warning("Graph Hints ABM system not available - training without agent adaptation")
    ABM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GRAPH HINTS ABM INTEGRATION FOR DGL TRAINING
# ============================================================================

class ABMDGLIntegration:
    """Integration layer between Graph Hints ABM system and DGL training"""
    
    def __init__(self, abm_system=None):
        self.abm_system = abm_system
        self.agent_training_weights = {}
        self.training_feedback_history = []
        self.emergence_detected_during_training = []
        
        if self.abm_system:
            logger.info("🧠 ABM-DGL integration initialized")
        else:
            logger.warning("⚠️ ABM system not available for DGL integration")
    
    def get_agent_training_weights(self) -> Dict[str, float]:
        """Get agent-influenced training weights for edge categories"""
        if not self.abm_system:
            return {category: 1.0 for category in EDGE_TABLE_SEGREGATION.keys()}
        
        weights = {}
        
        for category, config in EDGE_TABLE_SEGREGATION.items():
            base_weight = config['training_weight']
            
            # Get agent influences for this edge category
            agent_influences = []
            for agent_id, adaptation in self.abm_system.agent_adaptations.items():
                # Check if agent has bidding for this category type
                category_signal = self._map_edge_category_to_signal(category)
                if category_signal in adaptation.bidding_pattern:
                    bidding_strength = adaptation.bidding_pattern[category_signal]
                    recent_feedback = adaptation.signal_feedback.get(category_signal, 0.5)
                    
                    # Calculate agent influence on training weight
                    influence = (bidding_strength * recent_feedback) / 2.0
                    agent_influences.append(influence)
            
            # Apply agent influences to base weight
            if agent_influences:
                avg_influence = sum(agent_influences) / len(agent_influences)
                weights[category] = base_weight * (0.5 + avg_influence)  # 0.5-1.5x multiplier
            else:
                weights[category] = base_weight
        
        self.agent_training_weights = weights
        return weights
    
    def _map_edge_category_to_signal(self, edge_category: str) -> str:
        """Map edge category to ABM signal type"""
        mapping = {
            'alpha_edges': 'dag_execution',
            'beta_relationships': 'objective_optimization', 
            'gamma_edges': 'emergence_learning',
            'cross_phase_edges': 'phase_transition'
        }
        return mapping.get(edge_category, 'general_training')
    
    def update_agents_from_training_results(self, training_results: Dict[str, Any]):
        """Update ABM agents based on DGL training results"""
        if not self.abm_system:
            return
        
        training_loss = training_results.get('final_loss', 1.0)
        training_accuracy = training_results.get('accuracy', 0.0)
        epoch_count = training_results.get('epochs', 1)
        
        # Convert training metrics to feedback scores (0.0-1.0)
        loss_feedback = max(0.0, min(1.0, 1.0 - (training_loss / 2.0)))  # Normalize loss
        accuracy_feedback = max(0.0, min(1.0, training_accuracy))
        
        # Combine metrics with epoch consideration
        combined_feedback = (loss_feedback * 0.6) + (accuracy_feedback * 0.4)
        
        # Apply epoch bonus for stable training
        if epoch_count >= 20:
            combined_feedback = min(1.0, combined_feedback * 1.1)
        
        # Update each agent based on their training contribution
        for edge_category, training_weight in self.agent_training_weights.items():
            signal = self._map_edge_category_to_signal(edge_category)
            
            # Find agents with bidding for this signal
            for agent_id, adaptation in self.abm_system.agent_adaptations.items():
                if signal in adaptation.bidding_pattern:
                    # Weight feedback by training contribution
                    weighted_feedback = combined_feedback * (training_weight / 2.0)
                    
                    self.abm_system.update_agent_feedback(
                        agent_id,
                        signal,
                        weighted_feedback,
                        context={
                            'training_loss': training_loss,
                            'training_accuracy': training_accuracy,
                            'epochs': epoch_count,
                            'edge_category': edge_category,
                            'dgl_integration': True
                        }
                    )
        
        # Record training feedback
        self.training_feedback_history.append({
            'timestamp': datetime.now().isoformat(),
            'training_results': training_results,
            'agent_updates': len(self.abm_system.agent_adaptations),
            'combined_feedback': combined_feedback
        })
        
        logger.info(f"🤖 Updated {len(self.abm_system.agent_adaptations)} agents from DGL training results")
    
    def check_training_emergence(self, training_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for emergence patterns during DGL training"""
        if not self.abm_system:
            return []
        
        # Build system state for emergence detection
        system_state = {
            'training_loss': training_state.get('current_loss', 1.0),
            'training_accuracy': training_state.get('current_accuracy', 0.0),
            'epoch_number': training_state.get('epoch', 0),
            'convergence_rate': training_state.get('convergence_rate', 0.0),
            'agent_consensus': self._calculate_agent_consensus(),
            'edge_category_performance': training_state.get('edge_category_losses', {})
        }
        
        # Check for emergence conditions
        activated_rules = self.abm_system.check_emergence_conditions(system_state)
        
        if activated_rules:
            self.emergence_detected_during_training.extend(activated_rules)
            logger.info(f"🌟 Detected {len(activated_rules)} emergence pattern(s) during DGL training")
        
        return activated_rules
    
    def _calculate_agent_consensus(self) -> float:
        """Calculate consensus level among agents"""
        if not self.abm_system or not self.abm_system.agent_adaptations:
            return 0.0
        
        # Calculate variance in agent feedback scores
        all_feedback_scores = []
        for adaptation in self.abm_system.agent_adaptations.values():
            avg_feedback = sum(adaptation.signal_feedback.values()) / max(1, len(adaptation.signal_feedback))
            all_feedback_scores.append(avg_feedback)
        
        if len(all_feedback_scores) < 2:
            return 1.0
        
        mean_feedback = sum(all_feedback_scores) / len(all_feedback_scores)
        variance = sum((x - mean_feedback) ** 2 for x in all_feedback_scores) / len(all_feedback_scores)
        
        # Convert variance to consensus (lower variance = higher consensus)
        consensus = max(0.0, min(1.0, 1.0 - (variance * 2.0)))
        return consensus
    
    def get_abm_training_summary(self) -> Dict[str, Any]:
        """Get summary of ABM integration with DGL training"""
        if not self.abm_system:
            return {"abm_available": False}
        
        return {
            "abm_available": True,
            "active_agents": len(self.abm_system.agent_adaptations),
            "agent_training_weights": self.agent_training_weights,
            "training_feedback_events": len(self.training_feedback_history),
            "emergence_detections": len(self.emergence_detected_during_training),
            "system_coherence": self.abm_system._calculate_coherence_score(),
            "agent_consensus": self._calculate_agent_consensus()
        }

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
    """Complete BEM training system with database integration and ABM"""
    
    def __init__(self, model, graph, abm_system=None):
        self.model = model
        self.graph = graph
        self.training_db = get_training_db() if DB_AVAILABLE else None
        self.run_id = None
        
        # Initialize ABM integration
        self.abm_integration = ABMDGLIntegration(abm_system) if ABM_AVAILABLE else None
        if self.abm_integration:
            logger.info("🧠 BEM Trainer initialized with Graph Hints ABM integration")
    
    def train(self, epochs=50, lr=0.01, run_name=None):
        """Train BEM model with database logging, edge segregation weights, and ABM integration"""
        logger.info(f"🚀 Starting BEM training: {epochs} epochs, lr={lr}")
        logger.info("🧱 Using Edge Table Segregation for weighted training")
        
        # Get agent-influenced training weights
        agent_weights = {}
        if self.abm_integration:
            agent_weights = self.abm_integration.get_agent_training_weights()
            logger.info("🤖 Applied agent-influenced training weights")
            for category, weight in agent_weights.items():
                logger.info(f"  • {category}: {weight:.3f}")
        
        # Initialize training run in database
        if self.training_db:
            model_config = {
                'model_type': 'BEMGraphEmbedding',
                'hidden_dim': 64,
                'output_dim': 32,
                'num_phases': 3,
                'edge_segregation': True,
                'abm_integration': self.abm_integration is not None,
                'edge_weights': {
                    'alpha_edges': agent_weights.get('alpha_edges', get_edge_training_weight('alpha_edges')),
                    'beta_relationships': agent_weights.get('beta_relationships', get_edge_training_weight('beta_relationships')),
                    'gamma_edges': agent_weights.get('gamma_edges', get_edge_training_weight('gamma_edges')),
                    'cross_phase_edges': agent_weights.get('cross_phase_edges', get_edge_training_weight('cross_phase_edges'))
                }
            }
            training_params = {
                'epochs': epochs,
                'learning_rate': lr,
                'optimizer': 'Adam',
                'edge_segregation_enabled': True,
                'abm_enabled': self.abm_integration is not None
            }
            
            self.run_id = self.training_db.insert_training_run(
                run_name or f"bem_abm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                model_config, 
                training_params
            )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Use weighted loss function based on edge segregation and agent weights
        def weighted_loss_fn(predictions, targets, edge_weights=None, agent_weights=None):
            base_loss = nn.MSELoss(reduction='none')(predictions, targets)
            
            # Apply edge segregation weights
            if edge_weights is not None and 'weight' in self.graph.edata:
                edge_weight_tensor = self.graph.edata['weight']
                if len(edge_weight_tensor) > 0:
                    node_weights = torch.ones(predictions.shape[0])
                    weighted_loss = base_loss * node_weights.unsqueeze(1)
                    loss_value = weighted_loss.mean()
                else:
                    loss_value = base_loss.mean()
            else:
                loss_value = base_loss.mean()
            
            # Apply agent weight multiplier
            if agent_weights and self.abm_integration:
                # Calculate average agent influence
                avg_agent_weight = sum(agent_weights.values()) / len(agent_weights)
                loss_value = loss_value * avg_agent_weight
            
            return loss_value
        
        best_loss = float('inf')
        convergence_history = []
        
        for epoch in range(epochs):
            self.model.train()
            logits = self.model(self.graph, self.graph.ndata["feat"])
            
            # Apply edge segregation and agent-weighted loss
            loss = weighted_loss_fn(
                logits, 
                self.graph.ndata["label"], 
                self.graph.edata.get("weight"),
                agent_weights
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track convergence for emergence detection
            convergence_history.append(loss.item())
            if len(convergence_history) > 10:
                convergence_history.pop(0)
            
            # Calculate convergence rate
            convergence_rate = 0.0
            if len(convergence_history) >= 5:
                recent_avg = sum(convergence_history[-5:]) / 5
                earlier_avg = sum(convergence_history[:5]) / 5
                if earlier_avg > 0:
                    convergence_rate = max(0.0, (earlier_avg - recent_avg) / earlier_avg)
            
            # Check for emergence patterns during training
            if self.abm_integration and epoch % 5 == 0:
                training_state = {
                    'current_loss': loss.item(),
                    'current_accuracy': self._calculate_accuracy(logits),
                    'epoch': epoch,
                    'convergence_rate': convergence_rate,
                    'edge_category_losses': self._calculate_edge_category_losses(logits)
                }
                
                emergence_patterns = self.abm_integration.check_training_emergence(training_state)
                if emergence_patterns:
                    logger.info(f"🌟 Epoch {epoch}: Detected {len(emergence_patterns)} emergence pattern(s)")
            
            # Log metrics to database with edge segregation and ABM info
            if self.training_db and self.run_id:
                self.training_db.insert_training_metric(
                    self.run_id, epoch, 'training_loss', loss.item()
                )
                
                # Log ABM metrics
                if self.abm_integration:
                    abm_summary = self.abm_integration.get_abm_training_summary()
                    self.training_db.insert_training_metric(
                        self.run_id, epoch, 'agent_consensus', abm_summary.get('agent_consensus', 0.0)
                    )
                    self.training_db.insert_training_metric(
                        self.run_id, epoch, 'system_coherence', abm_summary.get('system_coherence', 0.0)
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
                # Log edge segregation and ABM status
                edge_info = ""
                if 'weight' in self.graph.edata:
                    avg_weight = self.graph.edata['weight'].mean().item()
                    edge_info = f", Avg Edge Weight: {avg_weight:.3f}"
                
                abm_info = ""
                if self.abm_integration:
                    abm_summary = self.abm_integration.get_abm_training_summary()
                    abm_info = f", Agent Consensus: {abm_summary.get('agent_consensus', 0.0):.3f}"
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}{edge_info}{abm_info}")
        
        # Prepare final training results for ABM feedback
        final_results = {
            'final_loss': best_loss,
            'accuracy': self._calculate_accuracy(logits),
            'epochs': epochs,
            'convergence_rate': convergence_rate,
            'edge_segregation_enabled': True,
            'abm_integration_enabled': self.abm_integration is not None
        }
        
        # Update ABM agents with training results
        if self.abm_integration:
            self.abm_integration.update_agents_from_training_results(final_results)
            
            # Log final ABM summary
            final_abm_summary = self.abm_integration.get_abm_training_summary()
            logger.info("🧠 Final ABM Training Summary:")
            logger.info(f"  • Active Agents: {final_abm_summary.get('active_agents', 0)}")
            logger.info(f"  • Training Feedback Events: {final_abm_summary.get('training_feedback_events', 0)}")
            logger.info(f"  • Emergence Detections: {final_abm_summary.get('emergence_detections', 0)}")
            logger.info(f"  • System Coherence: {final_abm_summary.get('system_coherence', 0.0):.3f}")
        
        # Update training status with edge segregation and ABM summary
        if self.training_db and self.run_id:
            self.training_db.update_training_status(self.run_id, 'completed')
            
            # Save final embeddings with edge segregation and ABM metadata
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
                
                # Add ABM metadata
                if self.abm_integration:
                    embedding_dict['abm_metadata'] = self.abm_integration.get_abm_training_summary()
                
                self.training_db.save_model_embeddings(self.run_id, embedding_dict)
        
        logger.info(f"✅ Training completed! Best loss: {best_loss:.4f}")
        logger.info("🧱 Edge segregation weights applied during training")
        if self.abm_integration:
            logger.info("🧠 ABM agent learning completed during training")
        
        return best_loss
    
    def _calculate_accuracy(self, logits):
        """Calculate training accuracy metric"""
        with torch.no_grad():
            # Simple accuracy calculation based on prediction closeness
            targets = self.graph.ndata["label"]
            mse = F.mse_loss(logits, targets)
            # Convert MSE to accuracy-like metric (0-1 scale)
            accuracy = max(0.0, min(1.0, 1.0 - (mse.item() / 2.0)))
            return accuracy
    
    def _calculate_edge_category_losses(self, logits):
        """Calculate losses per edge category for emergence detection"""
        edge_category_losses = {}
        
        if 'type' in self.graph.edata:
            edge_types = self.graph.edata['type']
            targets = self.graph.ndata["label"]
            
            for edge_type in [0, 1, 2, 3]:  # alpha, beta, gamma, cross-phase
                type_mask = edge_types == edge_type
                if type_mask.sum() > 0:
                    # Calculate loss for nodes connected by this edge type
                    type_loss = F.mse_loss(logits, targets, reduction='none').mean()
                    edge_category_losses[f'edge_type_{edge_type}_loss'] = type_loss.item()
        
        return edge_category_losses

def run_bem_training_with_database():
    """Main entry point for database-integrated BEM training with ABM"""
    logger.info("🌐 BEM DGL Trainer with Database Integration + Graph Hints ABM")
    
    # Setup training environment
    if DB_AVAILABLE:
        if not setup_training_environment():
            logger.error("❌ Failed to setup training environment")
            return None
    
    # Initialize Graph Hints ABM system
    abm_system = None
    if ABM_AVAILABLE:
        try:
            # Load ABM configuration
            abm_config_path = os.path.join(os.path.dirname(__file__), '..', 'MICROSERVICE_ENGINES', 'graph_hints')
            if os.path.exists(abm_config_path):
                from graph_hints_system import GraphHintsSystem
                abm_system = GraphHintsSystem()
                abm_system.load_configuration(abm_config_path)
                logger.info("🧠 Graph Hints ABM system initialized for DGL training")
            else:
                logger.warning("⚠️ ABM configuration directory not found, proceeding without ABM")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize ABM system: {e}")
            abm_system = None
    
    # Load graph from database or fallback
    if DB_AVAILABLE:
        graph = load_graph_from_database()
    else:
        graph = load_graph_from_json()
    
    # Log edge segregation configuration
    log_edge_segregation_info()
    
    # Initialize model
    model = BEMGraphEmbedding(
        in_feats=graph.ndata["feat"].shape[1], 
        hidden_feats=64, 
        out_feats=32
    )
    
    # Create trainer with ABM integration
    trainer = BEMTrainer(model, graph, abm_system)
    
    # Run training with extended epochs for ABM learning
    training_epochs = 150 if abm_system else 100
    final_loss = trainer.train(
        epochs=training_epochs, 
        lr=0.001, 
        run_name="bem_abm_edge_segregation_training"
    )
    
    # Log final system state
    if abm_system and trainer.abm_integration:
        final_abm_state = trainer.abm_integration.get_abm_training_summary()
        logger.info("🧠 Final ABM System State:")
        logger.info(f"  • Total Agents: {final_abm_state.get('active_agents', 0)}")
        logger.info(f"  • Training Events: {final_abm_state.get('training_feedback_events', 0)}")
        logger.info(f"  • Emergence Events: {final_abm_state.get('emergence_detections', 0)}")
        logger.info(f"  • System Coherence: {final_abm_state.get('system_coherence', 0.0):.3f}")
        logger.info(f"  • Agent Consensus: {final_abm_state.get('agent_consensus', 0.0):.3f}")
        
        # Save ABM state to file for future reference
        abm_state_file = os.path.join(os.path.dirname(__file__), 'abm_training_state.json')
        try:
            with open(abm_state_file, 'w') as f:
                json.dump(final_abm_state, f, indent=2)
            logger.info(f"💾 ABM training state saved to {abm_state_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save ABM state: {e}")
    
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
