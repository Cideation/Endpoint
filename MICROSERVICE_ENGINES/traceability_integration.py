#!/usr/bin/env python3
"""
Traceability Integration Module v1.0
🔗 Integration layer for connecting traceability engine with existing BEM components
Provides seamless integration with pulse_router, functor execution, and node state management
"""

import time
import json
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from datetime import datetime, timezone

from .traceability_engine import TraceabilityEngine, create_traceability_engine

class TracedFunctorRegistry:
    """Registry for traced functor execution"""
    
    def __init__(self, traceability_engine: TraceabilityEngine = None):
        self.engine = traceability_engine or create_traceability_engine()
        self.registered_functors = {}
        self.active_traces = {}
    
    def register_functor(self, functor_name: str, node_type: str = None):
        """Register a functor for tracing"""
        
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract node information from arguments
                node_id = kwargs.get('node_id') or (args[0] if args else 'unknown')
                agent = kwargs.get('agent') or 'system'
                input_data = kwargs.get('input_data') or {}
                dependencies = kwargs.get('dependencies') or []
                
                # Start trace
                start_time = time.time()
                trace_id = self.engine.start_trace(
                    node_id=str(node_id),
                    functor=functor_name,
                    agent_triggered=agent,
                    input_data=input_data,
                    dependencies=dependencies
                )
                
                # Store active trace
                self.active_traces[trace_id] = {
                    'functor_name': functor_name,
                    'node_id': node_id,
                    'start_time': start_time
                }
                
                try:
                    # Execute original function
                    result = func(*args, **kwargs)
                    
                    # End trace with success
                    execution_time = (time.time() - start_time) * 1000
                    output_data = result if isinstance(result, dict) else {"result": result}
                    
                    self.engine.end_trace(trace_id, output_data, execution_time)
                    
                    # Clean up
                    if trace_id in self.active_traces:
                        del self.active_traces[trace_id]
                    
                    return result
                
                except Exception as e:
                    # End trace with error
                    execution_time = (time.time() - start_time) * 1000
                    error_data = {"error": str(e), "error_type": type(e).__name__}
                    
                    self.engine.end_trace(trace_id, error_data, execution_time)
                    
                    # Clean up
                    if trace_id in self.active_traces:
                        del self.active_traces[trace_id]
                    
                    raise
            
            # Store functor metadata
            self.registered_functors[functor_name] = {
                'function': wrapper,
                'node_type': node_type,
                'original_function': func
            }
            
            return wrapper
        
        return decorator
    
    def log_decision(self, node_id: str, decision: str, condition: str = "", reasoning: str = ""):
        """Log a decision for the currently active trace"""
        
        # Find active trace for this node
        active_trace_id = None
        for trace_id, trace_info in self.active_traces.items():
            if str(trace_info['node_id']) == str(node_id):
                active_trace_id = trace_id
                break
        
        if active_trace_id:
            self.engine.log_decision(active_trace_id, decision, condition, reasoning)
    
    def log_param_usage(self, node_id: str, param_name: str):
        """Log global parameter usage for the currently active trace"""
        
        # Find active trace for this node
        active_trace_id = None
        for trace_id, trace_info in self.active_traces.items():
            if str(trace_info['node_id']) == str(node_id):
                active_trace_id = trace_id
                break
        
        if active_trace_id:
            self.engine.log_global_param_usage(active_trace_id, param_name)

class PulseTraceabilityMixin:
    """Mixin class for adding traceability to pulse system components"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traceability_engine = create_traceability_engine()
        self.pulse_traces = {}
    
    def trace_pulse_execution(self, pulse_type: str, source_node: str, target_node: str, 
                            pulse_data: Dict[str, Any], agent: str = "pulse_system"):
        """Trace pulse execution between nodes"""
        
        trace_id = self.traceability_engine.start_trace(
            node_id=target_node,
            functor=f"pulse_{pulse_type}",
            agent_triggered=agent,
            input_data={
                "pulse_type": pulse_type,
                "source_node": source_node,
                "pulse_data": pulse_data
            },
            dependencies=[source_node]
        )
        
        self.pulse_traces[f"{source_node}→{target_node}"] = trace_id
        return trace_id
    
    def complete_pulse_trace(self, source_node: str, target_node: str, 
                           result_data: Dict[str, Any], execution_time_ms: float = 0.0):
        """Complete a pulse trace"""
        
        trace_key = f"{source_node}→{target_node}"
        if trace_key in self.pulse_traces:
            trace_id = self.pulse_traces[trace_key]
            self.traceability_engine.end_trace(trace_id, result_data, execution_time_ms)
            del self.pulse_traces[trace_key]

class NodeStateTracker:
    """Tracks node state changes with full traceability"""
    
    def __init__(self, traceability_engine: TraceabilityEngine = None):
        self.engine = traceability_engine or create_traceability_engine()
        self.node_states = {}
        self.state_history = {}
    
    def update_node_state(self, node_id: str, new_state: Dict[str, Any], 
                         agent: str = "system", functor: str = "state_update"):
        """Update node state with traceability"""
        
        # Get previous state
        previous_state = self.node_states.get(node_id, {})
        
        # Start trace for state change
        trace_id = self.engine.start_trace(
            node_id=node_id,
            functor=functor,
            agent_triggered=agent,
            input_data={"previous_state": previous_state, "new_state": new_state}
        )
        
        # Calculate state differences
        changes = self._calculate_state_changes(previous_state, new_state)
        
        # Log decisions based on changes
        for change_type, change_data in changes.items():
            self.engine.log_decision(trace_id, f"state_change_{change_type}", 
                                   str(change_data), f"State updated: {change_type}")
        
        # Update state
        self.node_states[node_id] = new_state.copy()
        
        # Store in history
        if node_id not in self.state_history:
            self.state_history[node_id] = []
        
        self.state_history[node_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": new_state.copy(),
            "trace_id": trace_id,
            "agent": agent,
            "changes": changes
        })
        
        # End trace
        self.engine.end_trace(trace_id, {
            "state_updated": True,
            "changes_count": len(changes),
            "new_state": new_state
        })
        
        return trace_id
    
    def get_node_lineage(self, node_id: str) -> List[Dict[str, Any]]:
        """Get complete state change lineage for a node"""
        
        return self.state_history.get(node_id, [])
    
    def _calculate_state_changes(self, old_state: Dict[str, Any], 
                               new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between states"""
        
        changes = {}
        
        # Find added keys
        added_keys = set(new_state.keys()) - set(old_state.keys())
        if added_keys:
            changes["added"] = {key: new_state[key] for key in added_keys}
        
        # Find removed keys
        removed_keys = set(old_state.keys()) - set(new_state.keys())
        if removed_keys:
            changes["removed"] = {key: old_state[key] for key in removed_keys}
        
        # Find modified keys
        modified_keys = []
        for key in set(old_state.keys()) & set(new_state.keys()):
            if old_state[key] != new_state[key]:
                modified_keys.append(key)
        
        if modified_keys:
            changes["modified"] = {
                key: {"old": old_state[key], "new": new_state[key]} 
                for key in modified_keys
            }
        
        return changes

# Global instances for easy access
_global_functor_registry = None
_global_node_tracker = None

def get_traced_functor_registry() -> TracedFunctorRegistry:
    """Get global traced functor registry"""
    global _global_functor_registry
    if _global_functor_registry is None:
        _global_functor_registry = TracedFunctorRegistry()
    return _global_functor_registry

def get_node_state_tracker() -> NodeStateTracker:
    """Get global node state tracker"""
    global _global_node_tracker
    if _global_node_tracker is None:
        _global_node_tracker = NodeStateTracker()
    return _global_node_tracker

# Convenience decorators
def traced_functor(functor_name: str, node_type: str = None):
    """Decorator for easy functor tracing"""
    registry = get_traced_functor_registry()
    return registry.register_functor(functor_name, node_type)

def log_decision(node_id: str, decision: str, condition: str = "", reasoning: str = ""):
    """Convenience function for logging decisions"""
    registry = get_traced_functor_registry()
    registry.log_decision(node_id, decision, condition, reasoning)

def log_param_usage(node_id: str, param_name: str):
    """Convenience function for logging parameter usage"""
    registry = get_traced_functor_registry()
    registry.log_param_usage(node_id, param_name)

def update_node_state(node_id: str, new_state: Dict[str, Any], 
                     agent: str = "system", functor: str = "state_update"):
    """Convenience function for updating node state with traceability"""
    tracker = get_node_state_tracker()
    return tracker.update_node_state(node_id, new_state, agent, functor)

if __name__ == "__main__":
    print("🔗 Traceability Integration Module v1.0 - Demo")
    
    # Demo traced functor
    registry = get_traced_functor_registry()
    
    @registry.register_functor("demo_calculate", "V01_ProductComponent")
    def calculate_component_score(node_id: str, input_data: Dict[str, Any], 
                                agent: str = "demo_agent"):
        # Simulate some calculations
        registry.log_decision(node_id, "validation_passed", "input_valid == True", "Input validation successful")
        registry.log_param_usage(node_id, "form_strategy")
        
        score = input_data.get("base_score", 0.5) * 1.5
        return {"component_score": min(score, 1.0), "status": "calculated"}
    
    # Demo execution
    result = calculate_component_score(
        node_id="V01_001",
        input_data={"base_score": 0.6, "material": "steel"},
        agent="demo_agent"
    )
    
    print(f"✅ Demo calculation result: {result}")
    
    # Demo node state tracking
    tracker = get_node_state_tracker()
    
    tracker.update_node_state(
        node_id="V01_001",
        new_state={"score": 0.9, "status": "validated", "material": "steel"},
        agent="demo_agent",
        functor="validation_update"
    )
    
    lineage = tracker.get_node_lineage("V01_001")
    print(f"📋 Node lineage entries: {len(lineage)}")
    
    print("🎉 Integration demo completed successfully!")
