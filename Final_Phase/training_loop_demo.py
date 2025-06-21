#!/usr/bin/env python3
"""
Training Loop Demo - Complete Integration
Shows how DGL Training + Agent Memory + Reward Scoring work together with user interactions
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add paths
sys.path.append("Final_Phase")

try:
    from dgl_graph_builder import DGLGraphBuilder
    from node_engine_integration import NodeEngineWithInteractionLanguage
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingLoopDemo:
    """Complete training loop demonstration with user interactions"""
    
    def __init__(self):
        self.graph_builder = DGLGraphBuilder() if DGL_AVAILABLE else None
        self.node_engine = NodeEngineWithInteractionLanguage()
        self.training_history = []
        self.user_sessions = {}
        
        logger.info("🚀 Training Loop Demo initialized")
    
    def simulate_user_session(self, user_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a user interaction session"""
        
        logger.info(f"👤 Starting user session for {user_id}")
        
        session_results = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "interactions": [],
            "feedback": {},
            "rewards_generated": []
        }
        
        # Simulate user interactions with nodes
        nodes_to_interact = session_data.get("nodes", [])
        
        for node_interaction in nodes_to_interact:
            node_id = node_interaction["node_id"]
            user_action = node_interaction["action"]  # "rate", "modify", "approve"
            
            # Process the interaction
            interaction_result = self._process_user_interaction(
                user_id, node_id, user_action, node_interaction
            )
            
            session_results["interactions"].append(interaction_result)
            
            # Update user feedback for this node
            if node_id not in session_results["feedback"]:
                session_results["feedback"][node_id] = {
                    "rating": 0,
                    "interaction_count": 0,
                    "approval": 0.5,
                    "modifications": 0
                }
            
            feedback = session_results["feedback"][node_id]
            
            if user_action == "rate":
                feedback["rating"] = node_interaction.get("rating", 3)
                feedback["interaction_count"] += 1
            elif user_action == "approve":
                feedback["approval"] = node_interaction.get("approval", 0.8)
                feedback["interaction_count"] += 1
            elif user_action == "modify":
                feedback["modifications"] += 1
                feedback["interaction_count"] += 1
        
        # Store session
        self.user_sessions[user_id] = session_results
        
        logger.info(f"✅ User session completed: {len(session_results['interactions'])} interactions")
        return session_results
    
    def _process_user_interaction(self, user_id: str, node_id: str, 
                                 action: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual user interaction"""
        
        # Create node data for processing
        node_data = {
            "id": node_id,
            "type": interaction_data.get("node_type", "V01_ProductComponent"),
            "score": interaction_data.get("current_score", 0.5),
            "quality_factor": interaction_data.get("quality", 0.7),
            "efficiency": interaction_data.get("efficiency", 0.6),
            "performance": interaction_data.get("performance", 0.7)
        }
        
        # Process through node engine
        result = self.node_engine.process_node_with_interaction(node_id, node_data)
        
        # Calculate reward with user feedback
        user_feedback = {
            node_id: {
                "rating": interaction_data.get("rating", 3),
                "approval": interaction_data.get("approval", 0.7),
                "interaction_count": 1
            }
        }
        
        reward = self.node_engine.reward_score(node_id, result, user_feedback)
        
        return {
            "user_id": user_id,
            "node_id": node_id,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "node_score": result["score"],
            "reward_score": reward,
            "interaction_mode": result["interaction_mode"],
            "design_signal": result["interaction_language"]["design_signal"]
        }
    
    def run_dgl_training_cycle(self, user_feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run DGL training cycle with user feedback"""
        
        if not DGL_AVAILABLE:
            logger.warning("DGL not available, skipping training")
            return {"status": "skipped", "reason": "DGL not available"}
        
        logger.info("🧠 Starting DGL training cycle with user feedback")
        
        # Create training data
        nodes_data, edges_data = self.graph_builder.create_sample_training_data()
        
        # Add user feedback from sessions
        aggregated_feedback = self._aggregate_user_feedback()
        
        # Build training graph
        graph_data = {"nodes": nodes_data, "edges": edges_data}
        training_graph = self.graph_builder.build_training_graph(graph_data, aggregated_feedback)
        
        # Simulate training (in real system, this would use the DGL trainer)
        training_results = {
            "graph_nodes": training_graph.number_of_nodes(),
            "graph_edges": training_graph.number_of_edges(),
            "node_features_shape": list(training_graph.ndata['feat'].shape),
            "labels_shape": list(training_graph.ndata['label'].shape),
            "rewards_shape": list(training_graph.ndata['reward'].shape),
            "user_feedback_integrated": len(aggregated_feedback),
            "training_timestamp": datetime.now().isoformat()
        }
        
        # Store training history
        self.training_history.append(training_results)
        
        logger.info(f"✅ DGL training completed: {training_results['graph_nodes']} nodes trained")
        return training_results
    
    def _aggregate_user_feedback(self) -> Dict[str, Any]:
        """Aggregate feedback from all user sessions"""
        
        aggregated = {}
        
        for user_id, session in self.user_sessions.items():
            for node_id, feedback in session["feedback"].items():
                if node_id not in aggregated:
                    aggregated[node_id] = {
                        "rating": 0,
                        "interaction_count": 0,
                        "approval": 0.5,
                        "modifications": 0,
                        "user_count": 0
                    }
                
                agg = aggregated[node_id]
                agg["rating"] = (agg["rating"] * agg["user_count"] + feedback["rating"]) / (agg["user_count"] + 1)
                agg["interaction_count"] += feedback["interaction_count"]
                agg["approval"] = (agg["approval"] * agg["user_count"] + feedback["approval"]) / (agg["user_count"] + 1)
                agg["modifications"] += feedback["modifications"]
                agg["user_count"] += 1
        
        return aggregated
    
    def update_agent_memory(self, training_results: Dict[str, Any]):
        """Update agent memory based on training results"""
        
        logger.info("💭 Updating agent memory with training results")
        
        # Update memory for each active node
        for user_id, session in self.user_sessions.items():
            for interaction in session["interactions"]:
                node_id = interaction["node_id"]
                
                # Update short-term memory
                memory_data = {
                    "training_cycle": len(self.training_history),
                    "user_interaction": interaction,
                    "training_results": training_results,
                    "reward_score": interaction["reward_score"]
                }
                
                self.node_engine.update_short_term_memory(
                    node_id, memory_data, "training_cycle"
                )
        
        logger.info("✅ Agent memory updated")
    
    def demonstrate_complete_cycle(self):
        """Demonstrate complete training loop cycle"""
        
        logger.info("🎭 Starting Complete Training Loop Demonstration")
        
        # Step 1: Simulate multiple user sessions
        user_sessions = [
            {
                "user_id": "user_001",
                "nodes": [
                    {
                        "node_id": "V01_ProductComponent_001",
                        "node_type": "V01_ProductComponent",
                        "action": "rate",
                        "rating": 4,
                        "current_score": 0.8,
                        "quality": 0.85,
                        "efficiency": 0.82
                    },
                    {
                        "node_id": "V02_EconomicProfile_001",
                        "node_type": "V02_EconomicProfile",
                        "action": "approve",
                        "approval": 0.7,
                        "current_score": 0.6
                    }
                ]
            },
            {
                "user_id": "user_002", 
                "nodes": [
                    {
                        "node_id": "V01_ProductComponent_001",
                        "node_type": "V01_ProductComponent",
                        "action": "modify",
                        "rating": 3,
                        "current_score": 0.75,
                        "quality": 0.8
                    },
                    {
                        "node_id": "V05_ComplianceCheck_001",
                        "node_type": "V05_ComplianceCheck",
                        "action": "rate",
                        "rating": 5,
                        "approval": 0.95,
                        "current_score": 0.9
                    }
                ]
            }
        ]
        
        # Process user sessions
        session_results = []
        for session_config in user_sessions:
            result = self.simulate_user_session(session_config["user_id"], session_config)
            session_results.append(result)
        
        # Step 2: Run DGL training with user feedback
        training_results = self.run_dgl_training_cycle(self._aggregate_user_feedback())
        
        # Step 3: Update agent memory
        self.update_agent_memory(training_results)
        
        # Step 4: Generate summary
        summary = self._generate_cycle_summary(session_results, training_results)
        
        logger.info("🎉 Complete training loop cycle demonstrated!")
        return summary
    
    def _generate_cycle_summary(self, session_results: List[Dict[str, Any]], 
                               training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of the complete cycle"""
        
        total_interactions = sum(len(session["interactions"]) for session in session_results)
        total_rewards = sum(
            len(session["rewards_generated"]) for session in session_results
        )
        
        # Check agent memory
        memory_summary = {}
        for session in session_results:
            for interaction in session["interactions"]:
                node_id = interaction["node_id"]
                memory = self.node_engine.short_term_memory(node_id, "training_cycle")
                memory_summary[node_id] = len(memory)
        
        summary = {
            "cycle_completed": True,
            "timestamp": datetime.now().isoformat(),
            "user_sessions": len(session_results),
            "total_interactions": total_interactions,
            "nodes_affected": len(set(
                interaction["node_id"] 
                for session in session_results 
                for interaction in session["interactions"]
            )),
            "dgl_training": training_results,
            "agent_memory_entries": memory_summary,
            "system_status": "operational",
            "next_cycle_ready": True
        }
        
        return summary

def main():
    """Run the training loop demo"""
    
    print("🚀 Training Loop Demo - Complete Integration")
    print("=" * 60)
    
    # Initialize demo
    demo = TrainingLoopDemo()
    
    # Run complete cycle
    summary = demo.demonstrate_complete_cycle()
    
    # Display results
    print(f"\n📊 Training Loop Cycle Summary:")
    print(f"   User Sessions: {summary['user_sessions']}")
    print(f"   Total Interactions: {summary['total_interactions']}")
    print(f"   Nodes Affected: {summary['nodes_affected']}")
    print(f"   DGL Graph Nodes: {summary['dgl_training']['graph_nodes']}")
    print(f"   User Feedback Integrated: {summary['dgl_training']['user_feedback_integrated']}")
    print(f"   Agent Memory Entries: {sum(summary['agent_memory_entries'].values())}")
    print(f"   System Status: {summary['system_status']}")
    
    print(f"\n✅ Complete training loop demonstrated successfully!")
    print(f"🔄 Ready for next cycle: {summary['next_cycle_ready']}")
    
    return summary

if __name__ == "__main__":
    main() 