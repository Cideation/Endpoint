#!/usr/bin/env python3
"""
Engineering Traceability System - Simple Demo
🔍 Demonstrates complete audit trail functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from MICROSERVICE_ENGINES.traceability_engine import create_traceability_engine

def demo_complete_traceability():
    """Demonstrate complete traceability workflow"""
    
    print("🔍 Engineering Traceability System - Complete Demo")
    print("=" * 60)
    
    # Create traceability engine
    engine = create_traceability_engine()
    
    # Demo 1: Basic Trace Lifecycle
    print("\n📋 Demo 1: Basic Trace Lifecycle")
    
    trace_id = engine.start_trace(
        node_id="V01_DEMO",
        functor="evaluate_manufacturing",
        agent_triggered="DemoAgent",
        input_data={"component": "beam", "material": "steel", "dimensions": {"length": 100, "width": 50}},
        dependencies=["V02", "V05"]
    )
    
    # Log decisions with reasoning
    engine.log_decision(trace_id, "spec_check_passed", "material_grade >= min_grade", "Steel grade meets requirements")
    engine.log_decision(trace_id, "material_ok", "availability == true", "Material available in inventory")
    engine.log_decision(trace_id, "geometry_valid", "dimensions_within_tolerance", "All dimensions within manufacturing tolerance")
    engine.log_decision(trace_id, "cost_approved", "cost <= budget", "Component cost within project budget")
    
    # Log global parameter usage
    engine.log_global_param_usage(trace_id, "form_strategy")
    engine.log_global_param_usage(trace_id, "precision_tolerance")
    engine.log_global_param_usage(trace_id, "material_expression")
    
    # Complete trace
    output_data = {
        "manufacturing_score": 0.86,
        "status": "approved",
        "estimated_cost": 1250.00,
        "lead_time_days": 5
    }
    
    completed_trace = engine.end_trace(trace_id, output_data, 45.2)
    
    print(f"✅ Trace completed: {completed_trace.trace_id}")
    print(f"📊 Decisions logged: {len(completed_trace.decision_path)}")
    print(f"🎛️ Parameters used: {len(completed_trace.global_params_used)}")
    print(f"⏱️ Execution time: {completed_trace.execution_time_ms}ms")
    
    # Demo 2: Component Score with Origin Tracking
    print("\n📊 Demo 2: Component Score Origin Tracking")
    
    component_score = engine.create_component_score(
        value=0.86,
        source_functor="evaluate_component_vector",
        input_nodes=["V01_DEMO", "V02"],
        agent="DemoAgent",
        design_param="evolutionary_potential",
        component_id="COMP_DEMO_001"
    )
    
    print(f"✅ Component score: {component_score.value}")
    print(f"🔍 Source functor: {component_score.origin_tag.source_functor}")
    print(f"👤 Agent: {component_score.origin_tag.agent}")
    print(f"🎯 Design parameter: {component_score.origin_tag.design_param}")
    print(f"✅ Validation status: {component_score.validation_status}")
    
    # Demo 3: Multiple Traces for Chain Analysis
    print("\n🔗 Demo 3: Graph Chain Analysis")
    
    # Create related traces
    dependencies = ["V02", "V05"]
    
    trace_ids = []
    for i in range(3):
        tid = engine.start_trace(
            node_id=f"V0{i+1}_CHAIN",
            functor=f"process_step_{i+1}",
            agent_triggered=f"Agent{i+1}",
            input_data={"step": i+1, "input_value": 0.5 + (i * 0.1)},
            dependencies=dependencies
        )
        
        engine.log_decision(tid, f"step_{i+1}_validated", f"input_check_passed", f"Step {i+1} input validation successful")
        engine.log_global_param_usage(tid, f"param_{i+1}")
        
        engine.end_trace(tid, {"step_result": 0.7 + (i * 0.05), "next_step": i+2}, 20.0 + (i * 5))
        trace_ids.append(tid)
    
    # Get chain information
    chain_id = None
    for tid in trace_ids:
        # Find the chain_id from completed traces (check log files)
        log_files = list(Path("MICROSERVICE_ENGINES/traceability_logs").glob("trace_*.json"))
        if log_files:
            import json
            with open(log_files[0], 'r') as f:
                trace_data = json.load(f)
                chain_id = trace_data.get("chain_id")
                break
    
    if chain_id:
        trace_path = engine.get_trace_path(chain_id)
        print(f"🔗 Chain ID: {chain_id}")
        print(f"📍 Trace path: {' → '.join(trace_path)}")
    
    # Demo 4: Decision Lineage Analysis
    print("\n🕵️ Demo 4: Decision Lineage Analysis")
    
    lineage = engine.analyze_decision_lineage("V01_DEMO")
    
    print(f"🎯 Target node: {lineage['target_node']}")
    print(f"📊 Decision chains found: {len(lineage['decision_chain'])}")
    print(f"👥 Agents involved: {lineage['agents_involved']}")
    print(f"🎛️ Parameters used: {lineage['parameters_used']}")
    print(f"🔢 Total decisions: {lineage['total_decisions']}")
    
    # Demo 5: Comprehensive Audit Report
    print("\n📋 Demo 5: Comprehensive Audit Report")
    
    audit_report = engine.generate_audit_report()
    
    print(f"📄 Report ID: {audit_report['report_id']}")
    print(f"📊 Total traces: {audit_report['summary']['total_traces']}")
    print(f"🔢 Total decisions: {audit_report['summary']['total_decisions']}")
    print(f"🔗 Active chains: {audit_report['summary']['active_chains']}")
    print(f"📊 Component scores: {audit_report['summary']['component_scores']}")
    
    # Show decision patterns
    if audit_report['decision_patterns']:
        print("\n🎯 Top Decision Patterns:")
        for decision, count in list(audit_report['decision_patterns'].items())[:5]:
            print(f"  • {decision}: {count} occurrences")
    
    # Show parameter usage
    if audit_report['parameter_usage']:
        print("\n🎛️ Parameter Usage:")
        for param, count in list(audit_report['parameter_usage'].items())[:5]:
            print(f"  • {param}: {count} uses")
    
    # Show agent activity
    if audit_report['agent_activity']:
        print("\n👥 Agent Activity:")
        for agent, stats in audit_report['agent_activity'].items():
            print(f"  • {agent}: {stats['traces']} traces, {stats['total_decisions']} decisions, {stats['avg_execution_time']:.1f}ms avg")
    
    # Demo 6: Performance Statistics
    print("\n⚡ Demo 6: Performance Statistics")
    
    stats = engine.execution_stats
    print(f"📊 Total traces processed: {stats['total_traces']}")
    print(f"⏱️ Average execution time: {stats['average_execution_time']:.2f}ms")
    print(f"🔢 Total decisions logged: {stats['total_decisions']}")
    print(f"👥 Active agents: {len(stats['agent_activity'])}")
    
    if stats['total_traces'] > 0:
        decisions_per_trace = stats['total_decisions'] / stats['total_traces']
        print(f"📈 Average decisions per trace: {decisions_per_trace:.1f}")
    
    print("\n" + "=" * 60)
    print("🎉 Engineering Traceability System Demo Complete!")
    print("🔍 Every design decision is now fully explainable and backtrackable!")
    print("📊 From agent intent to final output - complete audit trail established!")

if __name__ == "__main__":
    demo_complete_traceability()
