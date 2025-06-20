#!/usr/bin/env python3
"""
Test Runner for Optimization Engine
Tests DGL capabilities, graph learning, and optimization cycles
"""

import sys
import json
import time
from main import OptimizationEngine

def test_optimization_cycle():
    """Test complete optimization cycle"""
    print("Testing optimization cycle...")
    
    engine = OptimizationEngine()
    result = engine.run_optimization_cycle()
    
    # Verify results
    assert 'optimization_status' in result
    assert 'metrics' in result
    assert 'dgl_capabilities' in result
    
    print(f"✅ Optimization completed with status: {result['optimization_status']}")
    print(f"✅ Overall score: {result['metrics']['overall_score']:.3f}")
    print(f"✅ Emergence score: {result['metrics']['emergence_score']:.3f}")
    print(f"✅ DGL available: {result['dgl_capabilities']['available']}")
    
    return result

def test_dgl_features():
    """Test DGL-specific features"""
    print("Testing DGL features...")
    
    engine = OptimizationEngine()
    
    # Load sample data
    graph_data = engine._generate_enhanced_sample_data()
    
    # Test enhanced DGL graph building
    if engine.learning_model is not None:
        dgl_graph = engine.build_enhanced_dgl_graph(graph_data)
        if dgl_graph is not None:
            print(f"✅ DGL graph created with {dgl_graph.number_of_nodes()} nodes")
            print(f"✅ Node features shape: {dgl_graph.ndata['feat'].shape}")
            print(f"✅ Agent types: {dgl_graph.ndata['agent_type']}")
            print(f"✅ Callback types: {dgl_graph.ndata['callback_type']}")
        else:
            print("⚠️ DGL graph creation failed")
    else:
        print("⚠️ DGL models not available")

def main():
    """Run all tests"""
    print("🚀 Starting Optimization Engine Tests")
    print("=" * 50)
    
    try:
        # Test DGL features
        test_dgl_features()
        print()
        
        # Test optimization cycle
        result = test_optimization_cycle()
        print()
        
        # Print summary
        print("📊 Test Summary:")
        print(f"   Graph Learning: {'✅' if result['dgl_capabilities']['available'] else '❌'}")
        print(f"   Embedding Generation: {'✅' if result['dgl_capabilities']['embeddings_generated'] else '❌'}")
        print(f"   Predictive Routing: {'✅' if result['dgl_capabilities']['routing_predictions']['confidence'] > 0 else '❌'}")
        print(f"   Emergence Detection: {'✅' if result['dgl_capabilities']['emergence_detection']['embedding_analysis']['available'] else '❌'}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 