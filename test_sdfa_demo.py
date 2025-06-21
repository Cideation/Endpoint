#!/usr/bin/env python3
"""
SDFA Visual System - Complete Demo
🎨 Demonstrates color as structured, interpretable output from Node Engine
Visual feedback emerges from real system data, not hardcoded UI rules
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from MICROSERVICE_ENGINES.sdfa_visual_engine import create_sdfa_engine, NodeState
from MICROSERVICE_ENGINES.sdfa_node_integration import create_sdfa_integration, sdfa_visual_tracking

def demo_sdfa_system():
    """Demonstrate complete SDFA visual system"""
    
    print("🎨 Scientific Design Formula Assignment (SDFA) - Complete Demo")
    print("=" * 70)
    print("🎯 Color as structured, interpretable output from Node Engine")
    print("📊 Visual feedback emerges from real system data")
    print("=" * 70)
    
    # Create SDFA components
    sdfa_engine = create_sdfa_engine()
    integration = create_sdfa_integration(sdfa_engine)
    
    # Demo 1: Performance-Based Color Assignment
    print("\n🔍 Demo 1: Performance-Based Color Assignment")
    print("-" * 50)
    
    performance_scenarios = [
        {
            "name": "Evolutionary Peak Node",
            "node_id": "V01_PEAK",
            "data": {
                "score": 0.95, "quality": 0.92, "stability": 0.88, "convergence": 0.94,
                "iterations": 15, "delta": 0.02, "variance": 0.05
            }
        },
        {
            "name": "High Performance Node", 
            "node_id": "V02_HIGH",
            "data": {
                "score": 0.82, "quality": 0.78, "stability": 0.85, "convergence": 0.80,
                "iterations": 12, "delta": 0.05, "variance": 0.08
            }
        },
        {
            "name": "Neutral State Node",
            "node_id": "V03_NEUTRAL", 
            "data": {
                "score": 0.65, "quality": 0.58, "stability": 0.62, "convergence": 0.55,
                "iterations": 8, "delta": 0.12, "variance": 0.15
            }
        },
        {
            "name": "Low Precision Node",
            "node_id": "V04_LOW",
            "data": {
                "score": 0.35, "quality": 0.28, "stability": 0.32, "convergence": 0.25,
                "iterations": 5, "delta": 0.25, "variance": 0.30
            }
        },
        {
            "name": "Critical State Node",
            "node_id": "V05_CRITICAL",
            "data": {
                "score": 0.08, "quality": 0.12, "stability": 0.05, "convergence": 0.15,
                "iterations": 3, "delta": 0.45, "variance": 0.50
            }
        }
    ]
    
    visual_results = []
    
    for scenario in performance_scenarios:
        print(f"\n�� {scenario['name']} ({scenario['node_id']}):")
        
        # Generate visual dictionary
        visual_dict = integration.register_node_execution(scenario['node_id'], scenario['data'])
        visual_results.append(visual_dict)
        
        print(f"  Performance Score: {visual_dict.performance_score:.3f}")
        print(f"  Node State: {visual_dict.node_state.value}")
        print(f"  Primary Color: {visual_dict.color_mapping.primary_color}")
        print(f"  Animation: {visual_dict.color_mapping.animation_type}")
        print(f"  Design Signals: {len(visual_dict.design_signals)}")
        
        # Show key design signals
        for signal in visual_dict.design_signals[:2]:  # Show first 2 signals
            print(f"    • {signal.signal_type.value}: {signal.value:.3f} (confidence: {signal.confidence:.2f})")
    
    # Demo 2: Platform-Specific Rendering
    print("\n\n🎮 Demo 2: Platform-Specific Rendering")
    print("-" * 50)
    
    test_node = "V01_PEAK"
    
    # Web/CSS rendering
    web_data = integration.get_node_visual_data(test_node, "web")
    print(f"\n🌐 Web/CSS Rendering for {test_node}:")
    print(f"  CSS Color: {web_data['css_color']}")
    print(f"  State Class: {web_data['state_class']}")
    print(f"  Animation Class: {web_data['animation_class']}")
    print(f"  Performance: {web_data['performance_score']:.3f}")
    
    # Unreal Engine rendering
    unreal_data = integration.get_node_visual_data(test_node, "unreal")
    print(f"\n🎮 Unreal Engine Rendering for {test_node}:")
    print(f"  RGBA Color: {unreal_data['color']}")
    print(f"  State: {unreal_data['state']}")
    print(f"  Animation Type: {unreal_data['animation']['type']}")
    print(f"  Animation Frequency: {unreal_data['animation']['frequency']:.2f}")
    
    # Cytoscape rendering
    cytoscape_data = integration.get_node_visual_data(test_node, "cytoscape")
    print(f"\n🕸️ Cytoscape Rendering for {test_node}:")
    print(f"  Background Color: {cytoscape_data['style']['background-color']}")
    print(f"  Border Width: {cytoscape_data['style']['border-width']}")
    print(f"  CSS Classes: {cytoscape_data['classes']}")
    
    # Demo 3: Real-Time Performance Tracking
    print("\n\n📈 Demo 3: Real-Time Performance Tracking")
    print("-" * 50)
    
    tracking_node = "V06_TRACKING"
    
    # Simulate multiple executions with changing performance
    performance_sequence = [
        {"performance_score": 0.3, "quality": 0.25, "stability": 0.35},
        {"performance_score": 0.5, "quality": 0.45, "stability": 0.55},
        {"performance_score": 0.7, "quality": 0.68, "stability": 0.72},
        {"performance_score": 0.85, "quality": 0.82, "stability": 0.88},
        {"performance_score": 0.92, "quality": 0.90, "stability": 0.94}
    ]
    
    print(f"\n🔄 Tracking performance evolution for {tracking_node}:")
    
    for i, perf_data in enumerate(performance_sequence):
        visual_dict = integration.update_node_performance(tracking_node, perf_data)
        
        print(f"  Execution {i+1}: {perf_data['performance_score']:.2f} -> {visual_dict.node_state.value} -> {visual_dict.color_mapping.primary_color}")
    
    # Demo 4: System Visual Overview
    print("\n\n📊 Demo 4: System Visual Overview")
    print("-" * 50)
    
    overview = integration.get_system_visual_overview()
    
    print(f"\n📋 System-Wide Visual State:")
    print(f"  Total Nodes: {overview['total_nodes']}")
    print(f"  Average Performance: {overview['performance_summary']['average_performance']:.3f}")
    print(f"  Total Executions: {overview['performance_summary']['total_executions']}")
    
    print(f"\n🎯 State Distribution:")
    for state, count in overview['state_distribution'].items():
        percentage = (count / overview['total_nodes']) * 100
        print(f"  • {state}: {count} nodes ({percentage:.1f}%)")
    
    if overview['performance_summary']['peak_performers']:
        print(f"\n🏆 Peak Performers:")
        for performer in overview['performance_summary']['peak_performers']:
            print(f"  • {performer['node_id']}: {performer['performance']:.3f}")
    
    if overview['performance_summary']['critical_nodes']:
        print(f"\n⚠️ Critical Nodes:")
        for critical in overview['performance_summary']['critical_nodes']:
            print(f"  • {critical['node_id']}: {critical['performance']:.3f}")
    
    # Demo 5: Emergence Detection
    print("\n\n🌟 Demo 5: Emergence Detection")
    print("-" * 50)
    
    # Create node with emergence factors
    emergence_node = "V07_EMERGENCE"
    emergence_data = {
        "score": 0.75,
        "quality": 0.70,
        "stability": 0.65,
        "convergence": 0.80,
        "emergence": {
            "interaction_strength": 0.85,
            "complexity": 0.78,
            "novelty": 0.92
        }
    }
    
    emergence_visual = integration.register_node_execution(emergence_node, emergence_data)
    
    print(f"\n🌟 Emergence Node Analysis:")
    print(f"  Node ID: {emergence_node}")
    print(f"  Base Performance: {emergence_visual.performance_score:.3f}")
    print(f"  State: {emergence_visual.node_state.value}")
    print(f"  Color: {emergence_visual.color_mapping.primary_color}")
    print(f"  Animation: {emergence_visual.color_mapping.animation_type}")
    
    # Find emergence signal
    emergence_signals = [s for s in emergence_visual.design_signals if s.signal_type.value == "emergence_factor"]
    if emergence_signals:
        signal = emergence_signals[0]
        print(f"  Emergence Factor: {signal.value:.3f}")
        print(f"  Contributing Factors: {signal.contributing_factors}")
    
    # Demo 6: Export Rendering Packages
    print("\n\n📦 Demo 6: Export Rendering Packages")
    print("-" * 50)
    
    # Export for different platforms
    web_package = integration.export_rendering_package("web")
    unreal_package = integration.export_rendering_package("unreal")
    cytoscape_package = integration.export_rendering_package("cytoscape")
    
    print(f"\n📦 Rendering Package Export:")
    print(f"  Web Package: {len(web_package['nodes'])} nodes")
    print(f"  Unreal Package: {len(unreal_package['nodes'])} nodes")
    print(f"  Cytoscape Package: {len(cytoscape_package['nodes'])} nodes")
    
    # Show sample web package structure
    if web_package['nodes']:
        sample_node = list(web_package['nodes'].keys())[0]
        sample_data = web_package['nodes'][sample_node]
        print(f"\n🌐 Sample Web Package Data ({sample_node}):")
        print(f"  CSS Color: {sample_data.get('css_color', 'N/A')}")
        print(f"  State Class: {sample_data.get('state_class', 'N/A')}")
        print(f"  Performance: {sample_data.get('performance_score', 'N/A')}")
    
    # Demo 7: Decorator Usage
    print("\n\n🔧 Demo 7: Automatic Visual Tracking with Decorators")
    print("-" * 50)
    
    @sdfa_visual_tracking(integration)
    def sample_functor(node_id: str, input_data: dict):
        """Sample functor with automatic visual tracking"""
        # Simulate some processing
        base_score = input_data.get("base_value", 0.5)
        quality_factor = input_data.get("quality_factor", 1.0)
        
        # Calculate result
        result_score = min(1.0, base_score * quality_factor)
        
        return {
            "score": result_score,
            "quality": quality_factor * 0.8,
            "stability": 0.7,
            "status": "processed"
        }
    
    # Execute decorated function
    print(f"\n🔧 Executing decorated functor:")
    
    decorator_test_cases = [
        {"node_id": "V08_AUTO_HIGH", "input_data": {"base_value": 0.8, "quality_factor": 1.1}},
        {"node_id": "V09_AUTO_LOW", "input_data": {"base_value": 0.3, "quality_factor": 0.9}}
    ]
    
    for test_case in decorator_test_cases:
        result = sample_functor(**test_case)
        visual_data = integration.get_node_visual_data(test_case["node_id"], "web")
        
        print(f"  {test_case['node_id']}: {result['score']:.3f} -> {visual_data['state_class']} -> {visual_data['css_color']}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("🎉 SDFA Visual System Demo Complete!")
    print("=" * 70)
    
    final_overview = integration.get_system_visual_overview()
    print(f"�� Final System State:")
    print(f"  Total Nodes Processed: {final_overview['total_nodes']}")
    print(f"  Total Executions: {final_overview['performance_summary']['total_executions']}")
    print(f"  System Average Performance: {final_overview['performance_summary']['average_performance']:.3f}")
    
    print(f"\n🎨 Key Achievements:")
    print(f"  ✅ Color determined by real system performance data")
    print(f"  ✅ Node states computed from actual metrics")
    print(f"  ✅ Visual feedback emerges from Node Engine coordination")
    print(f"  ✅ Platform-agnostic rendering data generated")
    print(f"  ✅ Real-time performance tracking with visual updates")
    print(f"  ✅ Emergence detection with specialized visual cues")
    print(f"  ✅ Automatic integration with functor execution")
    
    print(f"\n🔍 SDFA Principle Demonstrated:")
    print(f"  'Visual feedback emerges from real system data — not hardcoded UI rules'")
    print(f"  Every color, animation, and visual cue is computed from actual node performance!")

if __name__ == "__main__":
    demo_sdfa_system()
