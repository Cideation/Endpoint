#!/usr/bin/env python3
"""
FSM Runtime - Node State Management (Optional)

✅ MUTABLE COMPUTATION LAYER  
Manages Finite State Machine states for graph nodes.
Provides structured state transitions and state-based behavior.

This is an optional component that can be used when nodes need
explicit state management beyond simple functor execution.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Set, Optional, Callable
from enum import Enum
import uuid

# Configure FSM runtime logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - FSM-RUNTIME - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fsm_runtime.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FSM_RUNTIME")

class NodeState(str, Enum):
    """Standard node states for FSM"""
    IDLE = "idle"
    ACTIVE = "active"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    DISABLED = "disabled"
    TRANSITIONING = "transitioning"

class TransitionTrigger(str, Enum):
    """Triggers that can cause state transitions"""
    PULSE_RECEIVED = "pulse_received"
    COMPUTATION_COMPLETE = "computation_complete"
    ERROR_OCCURRED = "error_occurred"
    TIMEOUT = "timeout"
    MANUAL_TRIGGER = "manual_trigger"
    DEPENDENCY_SATISFIED = "dependency_satisfied"
    EXTERNAL_EVENT = "external_event"

class FSMNode:
    """Individual FSM node with state and transition logic"""
    
    def __init__(self, node_id: str, initial_state: NodeState = NodeState.IDLE):
        self.node_id = node_id
        self.current_state = initial_state
        self.previous_state = None
        self.state_history = [initial_state]
        self.transition_rules = {}
        self.state_handlers = {}
        self.state_data = {}
        self.created_at = datetime.now()
        self.last_transition = None
        
        logger.info(f"FSM Node created: {node_id} | Initial state: {initial_state}")
    
    def add_transition_rule(self, from_state: NodeState, trigger: TransitionTrigger, to_state: NodeState, condition: Optional[Callable] = None):
        """Add a state transition rule"""
        if from_state not in self.transition_rules:
            self.transition_rules[from_state] = {}
        
        self.transition_rules[from_state][trigger] = {
            'to_state': to_state,
            'condition': condition
        }
        
        logger.info(f"Node {self.node_id}: Transition rule added {from_state} --{trigger}--> {to_state}")
    
    def add_state_handler(self, state: NodeState, handler: Callable):
        """Add a handler for when node enters a specific state"""
        self.state_handlers[state] = handler
        logger.info(f"Node {self.node_id}: State handler added for {state}")
    
    async def trigger_transition(self, trigger: TransitionTrigger, context: Dict[str, Any] = None) -> bool:
        """Attempt to trigger a state transition"""
        if self.current_state not in self.transition_rules:
            logger.warning(f"Node {self.node_id}: No transition rules for state {self.current_state}")
            return False
        
        if trigger not in self.transition_rules[self.current_state]:
            logger.warning(f"Node {self.node_id}: No transition rule for trigger {trigger} in state {self.current_state}")
            return False
        
        transition_rule = self.transition_rules[self.current_state][trigger]
        to_state = transition_rule['to_state']
        condition = transition_rule['condition']
        
        # Check condition if present
        if condition and not condition(context or {}):
            logger.info(f"Node {self.node_id}: Transition condition not met for {trigger}")
            return False
        
        # Execute transition
        await self._execute_transition(to_state, trigger, context)
        return True
    
    async def _execute_transition(self, to_state: NodeState, trigger: TransitionTrigger, context: Dict[str, Any]):
        """Execute state transition"""
        self.previous_state = self.current_state
        self.current_state = to_state
        self.state_history.append(to_state)
        self.last_transition = {
            'from_state': self.previous_state,
            'to_state': to_state,
            'trigger': trigger,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Node {self.node_id}: Transition {self.previous_state} --{trigger}--> {to_state}")
        
        # Execute state handler if present
        if to_state in self.state_handlers:
            try:
                await self.state_handlers[to_state](context or {})
            except Exception as e:
                logger.error(f"Node {self.node_id}: State handler error for {to_state}: {str(e)}")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            'node_id': self.node_id,
            'current_state': self.current_state,
            'previous_state': self.previous_state,
            'state_history': self.state_history,
            'last_transition': self.last_transition,
            'state_data': self.state_data,
            'created_at': self.created_at.isoformat(),
            'uptime': str(datetime.now() - self.created_at)
        }

class FSMRuntime:
    """
    FSM Runtime Manager
    
    Manages multiple FSM nodes and their state transitions.
    Provides centralized state management for the graph system.
    """
    
    def __init__(self):
        self.nodes: Dict[str, FSMNode] = {}
        self.global_handlers = {}
        self.runtime_start_time = datetime.now()
        self.transition_count = 0
        
        logger.info("FSM Runtime initialized")
    
    def create_node(self, node_id: str, initial_state: NodeState = NodeState.IDLE) -> FSMNode:
        """Create a new FSM node"""
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already exists, returning existing node")
            return self.nodes[node_id]
        
        node = FSMNode(node_id, initial_state)
        self.nodes[node_id] = node
        
        logger.info(f"FSM Node created: {node_id}")
        return node
    
    def get_node(self, node_id: str) -> Optional[FSMNode]:
        """Get FSM node by ID"""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove FSM node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"FSM Node removed: {node_id}")
            return True
        return False
    
    async def trigger_node_transition(self, node_id: str, trigger: TransitionTrigger, context: Dict[str, Any] = None) -> bool:
        """Trigger transition for specific node"""
        node = self.get_node(node_id)
        if not node:
            logger.error(f"Node not found: {node_id}")
            return False
        
        success = await node.trigger_transition(trigger, context)
        if success:
            self.transition_count += 1
        
        return success
    
    async def broadcast_trigger(self, trigger: TransitionTrigger, context: Dict[str, Any] = None, filter_states: Set[NodeState] = None) -> Dict[str, bool]:
        """Broadcast trigger to multiple nodes"""
        results = {}
        
        for node_id, node in self.nodes.items():
            # Filter by current state if specified
            if filter_states and node.current_state not in filter_states:
                continue
            
            results[node_id] = await node.trigger_transition(trigger, context)
            if results[node_id]:
                self.transition_count += 1
        
        logger.info(f"Broadcast trigger {trigger}: {sum(results.values())}/{len(results)} nodes transitioned")
        return results
    
    def get_nodes_by_state(self, state: NodeState) -> List[FSMNode]:
        """Get all nodes in a specific state"""
        return [node for node in self.nodes.values() if node.current_state == state]
    
    def get_state_summary(self) -> Dict[NodeState, int]:
        """Get summary of node states"""
        summary = {}
        for node in self.nodes.values():
            state = node.current_state
            summary[state] = summary.get(state, 0) + 1
        return summary
    
    def setup_standard_node(self, node_id: str) -> FSMNode:
        """Setup a node with standard transition rules"""
        node = self.create_node(node_id)
        
        # Standard transitions
        node.add_transition_rule(NodeState.IDLE, TransitionTrigger.PULSE_RECEIVED, NodeState.ACTIVE)
        node.add_transition_rule(NodeState.ACTIVE, TransitionTrigger.COMPUTATION_COMPLETE, NodeState.IDLE)
        node.add_transition_rule(NodeState.ACTIVE, TransitionTrigger.ERROR_OCCURRED, NodeState.ERROR)
        node.add_transition_rule(NodeState.ERROR, TransitionTrigger.MANUAL_TRIGGER, NodeState.IDLE)
        node.add_transition_rule(NodeState.IDLE, TransitionTrigger.MANUAL_TRIGGER, NodeState.DISABLED)
        node.add_transition_rule(NodeState.DISABLED, TransitionTrigger.MANUAL_TRIGGER, NodeState.IDLE)
        
        logger.info(f"Standard FSM node setup complete: {node_id}")
        return node
    
    async def handle_pulse_routing_result(self, routing_result: Dict[str, Any]):
        """Handle results from pulse router to trigger FSM transitions"""
        interaction_type = routing_result.get('interaction_type')
        results = routing_result.get('results', [])
        
        # Trigger transitions based on routing results
        if interaction_type == 'pulse_trigger':
            # Trigger nodes to active state
            await self.broadcast_trigger(
                TransitionTrigger.PULSE_RECEIVED,
                context={'routing_result': routing_result},
                filter_states={NodeState.IDLE, NodeState.WAITING}
            )
        
        # Check for computation completion
        success_count = sum(1 for result in results if result.get('status') == 'success')
        if success_count > 0:
            await self.broadcast_trigger(
                TransitionTrigger.COMPUTATION_COMPLETE,
                context={'success_count': success_count},
                filter_states={NodeState.ACTIVE, NodeState.PROCESSING}
            )
        
        # Check for errors
        error_count = sum(1 for result in results if result.get('status') == 'error')
        if error_count > 0:
            await self.broadcast_trigger(
                TransitionTrigger.ERROR_OCCURRED,
                context={'error_count': error_count},
                filter_states={NodeState.ACTIVE, NodeState.PROCESSING}
            )
    
    def get_runtime_status(self) -> Dict[str, Any]:
        """Get FSM runtime status"""
        return {
            'type': 'fsm_runtime_status',
            'uptime': str(datetime.now() - self.runtime_start_time),
            'total_nodes': len(self.nodes),
            'total_transitions': self.transition_count,
            'state_summary': self.get_state_summary(),
            'runtime_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
    
    def export_state_snapshot(self) -> Dict[str, Any]:
        """Export complete state snapshot for persistence"""
        return {
            'snapshot_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'runtime_status': self.get_runtime_status(),
            'nodes': {
                node_id: node.get_state_info() 
                for node_id, node in self.nodes.items()
            }
        }

# Global FSM runtime instance
fsm_runtime = FSMRuntime()

# Convenience functions
def create_standard_node(node_id: str) -> FSMNode:
    """Create a node with standard FSM setup"""
    return fsm_runtime.setup_standard_node(node_id)

async def trigger_node(node_id: str, trigger: TransitionTrigger, context: Dict[str, Any] = None) -> bool:
    """Trigger transition for a specific node"""
    return await fsm_runtime.trigger_node_transition(node_id, trigger, context)

def get_node_state(node_id: str) -> Optional[NodeState]:
    """Get current state of a node"""
    node = fsm_runtime.get_node(node_id)
    return node.current_state if node else None 