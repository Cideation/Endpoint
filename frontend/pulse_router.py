#!/usr/bin/env python3
"""
Pulse Router - Interaction Signal Dispatcher

✅ MUTABLE COMPUTATION LAYER
Receives messages from ECM Gateway and dispatches interaction signals to functors.
This is where the emergent computation happens - interpreting, routing, and triggering.

Responsibilities:
- Interpret messages from ECM (post-delivery)
- Route interaction signals to appropriate functors
- Manage event type handling and dispatch logic
- Interface with Node Engine for graph operations
- Evolve routing patterns as system needs change
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
from enum import Enum

# Configure pulse router logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PULSE-ROUTER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pulse_router.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PULSE_ROUTER")

class InteractionType(str, Enum):
    """Types of interactions that can be routed"""
    PULSE_TRIGGER = "pulse_trigger"
    SPATIAL_EVENT = "spatial_event" 
    UI_INTERACTION = "ui_interaction"
    GRAPH_QUERY = "graph_query"
    FUNCTOR_EXECUTE = "functor_execute"
    STATE_UPDATE = "state_update"
    SYSTEM_COMMAND = "system_command"

class SemanticPulseType(str, Enum):
    """Semantic pulse types with specific business meaning"""
    BID_PULSE = "bid_pulse"                   # Competitive intent, pricing, offers
    OCCUPANCY_PULSE = "occupancy_pulse"       # Spatial request, usage claim
    COMPLIANCY_PULSE = "compliancy_pulse"     # Enforcement, constraints, violations
    FIT_PULSE = "fit_pulse"                   # Geometric, contextual match
    INVESTMENT_PULSE = "investment_pulse"     # Capital signal, resource allocation
    DECAY_PULSE = "decay_pulse"               # Decline, obsolescence, reset readiness

# Modern UX color coding for pulse types
PULSE_COLORS = {
    SemanticPulseType.BID_PULSE: "#FFC107",        # ⚠️ Amber - Attention-grabbing, action-oriented
    SemanticPulseType.OCCUPANCY_PULSE: "#2196F3",  # 🌊 Sky Blue - Movement, presence
    SemanticPulseType.COMPLIANCY_PULSE: "#1E3A8A", # 🛡️ Indigo - Trustworthy, serious, institutional
    SemanticPulseType.FIT_PULSE: "#4CAF50",        # ✅ Green - Success, confirmation
    SemanticPulseType.INVESTMENT_PULSE: "#FF9800", # 💰 Deep Orange - Warm, motivating, urgency
    SemanticPulseType.DECAY_PULSE: "#9E9E9E"       # ⚫ Neutral Gray - Low visibility, subtle decay
}

class PulseRouter:
    """
    Mutable Computation Layer - Interaction Signal Dispatcher
    
    Routes messages from ECM Gateway to appropriate functors and graph operations.
    This layer can evolve and adapt routing logic as system needs change.
    """
    
    def __init__(self):
        # Mutable routing registry
        self.interaction_handlers: Dict[InteractionType, List[Callable]] = {
            InteractionType.PULSE_TRIGGER: [],
            InteractionType.SPATIAL_EVENT: [],
            InteractionType.UI_INTERACTION: [],
            InteractionType.GRAPH_QUERY: [],
            InteractionType.FUNCTOR_EXECUTE: [],
            InteractionType.STATE_UPDATE: [],
            InteractionType.SYSTEM_COMMAND: []
        }
        
        # Router state (mutable)
        self.message_count = 0
        self.routing_stats = {}
        self.active_functors = {}
        self.router_start_time = datetime.now()
        
        # Initialize default handlers
        self._register_default_handlers()
        
        logger.info("Pulse Router initialized - Mutable computation layer ready")
    
    def _register_default_handlers(self):
        """Register default interaction handlers (can be evolved)"""
        
        # Pulse trigger handler
        self.register_handler(InteractionType.PULSE_TRIGGER, self._handle_pulse_trigger)
        
        # Spatial event handler
        self.register_handler(InteractionType.SPATIAL_EVENT, self._handle_spatial_event)
        
        # UI interaction handler
        self.register_handler(InteractionType.UI_INTERACTION, self._handle_ui_interaction)
        
        # Graph query handler
        self.register_handler(InteractionType.GRAPH_QUERY, self._handle_graph_query)
        
        # Functor execution handler
        self.register_handler(InteractionType.FUNCTOR_EXECUTE, self._handle_functor_execute)
        
        # State update handler
        self.register_handler(InteractionType.STATE_UPDATE, self._handle_state_update)
        
        logger.info("Default routing handlers registered")
    
    def register_handler(self, interaction_type: InteractionType, handler: Callable):
        """Register a handler for an interaction type (mutable registration)"""
        self.interaction_handlers[interaction_type].append(handler)
        logger.info(f"Handler registered for {interaction_type}: {handler.__name__}")
    
    def unregister_handler(self, interaction_type: InteractionType, handler: Callable):
        """Unregister a handler (supports evolution)"""
        if handler in self.interaction_handlers[interaction_type]:
            self.interaction_handlers[interaction_type].remove(handler)
            logger.info(f"Handler unregistered for {interaction_type}: {handler.__name__}")
    
    async def route_message(self, ecm_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route message from ECM to appropriate handlers
        
        This is the main dispatch logic - can evolve and adapt
        """
        self.message_count += 1
        start_time = time.time()
        
        try:
            # Extract interaction type from ECM message
            interaction_type = self._determine_interaction_type(ecm_message)
            
            logger.info(f"ROUTE-START: Message {self.message_count} | Type: {interaction_type}")
            
            # Route to appropriate handlers
            results = await self._dispatch_to_handlers(interaction_type, ecm_message)
            
            # Compile routing response
            routing_response = {
                'type': 'pulse_routing_response',
                'message_id': self.message_count,
                'interaction_type': interaction_type,
                'handlers_executed': len(results),
                'results': results,
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            # Update routing stats (mutable state)
            self._update_routing_stats(interaction_type, len(results))
            
            logger.info(f"ROUTE-COMPLETE: Message {self.message_count} | "
                       f"Handlers: {len(results)} | "
                       f"Time: {routing_response['processing_time_ms']}ms")
            
            return routing_response
            
        except Exception as e:
            logger.error(f"ROUTE-ERROR: Message {self.message_count} | Error: {str(e)}")
            return {
                'type': 'pulse_routing_error',
                'message_id': self.message_count,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _determine_interaction_type(self, message: Dict[str, Any]) -> InteractionType:
        """
        Determine interaction type from ECM message
        
        This logic can evolve to handle new message patterns
        """
        msg_type = message.get('type', 'unknown')
        
        # Check for semantic pulse types first
        if msg_type in [pulse_type.value for pulse_type in SemanticPulseType]:
            return InteractionType.PULSE_TRIGGER
        
        # Routing logic (mutable - can be updated)
        if msg_type in ['pulse_trigger', 'trigger']:
            return InteractionType.PULSE_TRIGGER
        elif msg_type in ['spatial_event', 'spatial', 'movement', 'interaction']:
            return InteractionType.SPATIAL_EVENT
        elif msg_type in ['ui_interaction', 'ui', 'click', 'input']:
            return InteractionType.UI_INTERACTION
        elif msg_type in ['graph_query', 'query', 'search']:
            return InteractionType.GRAPH_QUERY
        elif msg_type in ['functor_execute', 'execute', 'run']:
            return InteractionType.FUNCTOR_EXECUTE
        elif msg_type in ['state_update', 'update', 'set']:
            return InteractionType.STATE_UPDATE
        elif msg_type in ['system_command', 'command', 'system']:
            return InteractionType.SYSTEM_COMMAND
        else:
            # Default to pulse trigger for unknown types
            return InteractionType.PULSE_TRIGGER
    
    def _determine_semantic_pulse_type(self, message: Dict[str, Any]) -> Optional[SemanticPulseType]:
        """Determine semantic pulse type from message"""
        msg_type = message.get('type', 'unknown')
        
        # Direct semantic pulse type mapping
        for pulse_type in SemanticPulseType:
            if msg_type == pulse_type.value:
                return pulse_type
        
        # Intelligent inference based on message content (can evolve)
        payload = message.get('payload', {})
        
        if any(key in payload for key in ['price', 'bid', 'offer', 'quote']):
            return SemanticPulseType.BID_PULSE
        elif any(key in payload for key in ['spatial', 'occupancy', 'claim', 'usage']):
            return SemanticPulseType.OCCUPANCY_PULSE
        elif any(key in payload for key in ['compliance', 'violation', 'constraint', 'rule']):
            return SemanticPulseType.COMPLIANCY_PULSE
        elif any(key in payload for key in ['fit', 'match', 'geometry', 'context']):
            return SemanticPulseType.FIT_PULSE
        elif any(key in payload for key in ['investment', 'capital', 'resource', 'allocation']):
            return SemanticPulseType.INVESTMENT_PULSE
        elif any(key in payload for key in ['decay', 'decline', 'obsolete', 'reset']):
            return SemanticPulseType.DECAY_PULSE
        
        return None
    
    async def _dispatch_to_handlers(self, interaction_type: InteractionType, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Dispatch message to all registered handlers for the interaction type"""
        handlers = self.interaction_handlers.get(interaction_type, [])
        results = []
        
        for handler in handlers:
            try:
                result = await handler(message)
                results.append({
                    'handler': handler.__name__,
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                logger.error(f"Handler error: {handler.__name__} | {str(e)}")
                results.append({
                    'handler': handler.__name__,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _update_routing_stats(self, interaction_type: InteractionType, handler_count: int):
        """Update routing statistics (mutable state)"""
        if interaction_type not in self.routing_stats:
            self.routing_stats[interaction_type] = {
                'count': 0,
                'total_handlers': 0,
                'avg_handlers': 0
            }
        
        stats = self.routing_stats[interaction_type]
        stats['count'] += 1
        stats['total_handlers'] += handler_count
        stats['avg_handlers'] = round(stats['total_handlers'] / stats['count'], 2)
    
    # Default Handlers (can be evolved/replaced)
    
    async def _handle_pulse_trigger(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pulse trigger interactions with semantic pulse type support"""
        payload = message.get('payload', {})
        semantic_pulse = self._determine_semantic_pulse_type(message)
        
        logger.info(f"PULSE-TRIGGER: Type: {semantic_pulse} | Payload: {payload}")
        
        # Trigger specific graph node or functor
        target = message.get('target', 'default')
        
        # Enhanced pulse processing with semantic meaning
        pulse_response = {
            'action': 'pulse_triggered',
            'target': target,
            'payload': payload,
            'semantic_pulse_type': semantic_pulse.value if semantic_pulse else None,
            'pulse_color': PULSE_COLORS.get(semantic_pulse) if semantic_pulse else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Semantic-specific processing (can evolve)
        if semantic_pulse:
            pulse_response.update(await self._process_semantic_pulse(semantic_pulse, message))
        
        # TODO: Interface with Node Engine to trigger specific functors
        # TODO: Send visual pulse to Unreal Engine with color coding
        
        return pulse_response
    
    async def _process_semantic_pulse(self, pulse_type: SemanticPulseType, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process semantic pulse with type-specific logic"""
        payload = message.get('payload', {})
        
        if pulse_type == SemanticPulseType.BID_PULSE:
            # Competitive intent, pricing, offers
            return {
                'semantic_action': 'competitive_analysis',
                'urgency_level': 'high',
                'visual_effect': 'pulsing_amber',
                'expected_response': 'bid_evaluation'
            }
        
        elif pulse_type == SemanticPulseType.OCCUPANCY_PULSE:
            # Spatial request, usage claim
            return {
                'semantic_action': 'spatial_allocation',
                'urgency_level': 'medium',
                'visual_effect': 'flowing_blue',
                'expected_response': 'occupancy_update'
            }
        
        elif pulse_type == SemanticPulseType.COMPLIANCY_PULSE:
            # Enforcement, constraints, violations
            return {
                'semantic_action': 'compliance_check',
                'urgency_level': 'critical',
                'visual_effect': 'steady_indigo',
                'expected_response': 'compliance_report'
            }
        
        elif pulse_type == SemanticPulseType.FIT_PULSE:
            # Geometric, contextual match
            return {
                'semantic_action': 'fit_analysis',
                'urgency_level': 'low',
                'visual_effect': 'confirming_green',
                'expected_response': 'fit_validation'
            }
        
        elif pulse_type == SemanticPulseType.INVESTMENT_PULSE:
            # Capital signal, resource allocation
            return {
                'semantic_action': 'investment_evaluation',
                'urgency_level': 'high',
                'visual_effect': 'warming_orange',
                'expected_response': 'investment_decision'
            }
        
        elif pulse_type == SemanticPulseType.DECAY_PULSE:
            # Decline, obsolescence, reset readiness
            return {
                'semantic_action': 'decay_processing',
                'urgency_level': 'low',
                'visual_effect': 'fading_gray',
                'expected_response': 'cleanup_initiated'
            }
        
        return {}
    
    async def _handle_spatial_event(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle spatial events from Unreal"""
        logger.info(f"SPATIAL-EVENT: {message.get('spatial_data', {})}")
        
        spatial_data = message.get('spatial_data', {})
        event_type = spatial_data.get('event_type', 'movement')
        
        # TODO: Update graph state based on spatial event
        # TODO: Trigger spatial functors
        
        return {
            'action': 'spatial_processed',
            'event_type': event_type,
            'spatial_data': spatial_data,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_ui_interaction(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle UI interactions"""
        logger.info(f"UI-INTERACTION: {message.get('ui_data', {})}")
        
        ui_data = message.get('ui_data', {})
        action = ui_data.get('action', 'click')
        
        # TODO: Process UI interaction through graph
        # TODO: Update agent coefficients if needed
        
        return {
            'action': 'ui_processed',
            'ui_action': action,
            'ui_data': ui_data,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_graph_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph queries"""
        logger.info(f"GRAPH-QUERY: {message.get('query', {})}")
        
        query = message.get('query', {})
        
        # TODO: Execute graph query through Node Engine
        # TODO: Return graph state or search results
        
        return {
            'action': 'query_executed',
            'query': query,
            'results': 'placeholder_results',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_functor_execute(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle direct functor execution"""
        logger.info(f"FUNCTOR-EXECUTE: {message.get('functor', {})}")
        
        functor_data = message.get('functor', {})
        functor_name = functor_data.get('name', 'unknown')
        
        # TODO: Execute specific functor through Node Engine
        # TODO: Return execution results
        
        return {
            'action': 'functor_executed',
            'functor_name': functor_name,
            'functor_data': functor_data,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_state_update(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle state updates"""
        logger.info(f"STATE-UPDATE: {message.get('state_data', {})}")
        
        state_data = message.get('state_data', {})
        
        # TODO: Update graph state through Node Engine
        # TODO: Persist state changes
        
        return {
            'action': 'state_updated',
            'state_data': state_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get pulse router status (mutable state reporting)"""
        return {
            'type': 'pulse_router_status',
            'uptime': str(datetime.now() - self.router_start_time),
            'messages_processed': self.message_count,
            'registered_handlers': {
                str(interaction_type): len(handlers) 
                for interaction_type, handlers in self.interaction_handlers.items()
            },
            'routing_stats': self.routing_stats,
            'active_functors': len(self.active_functors),
            'router_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }

# Router instance for import
pulse_router = PulseRouter()

async def route_ecm_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for routing ECM messages"""
    return await pulse_router.route_message(message) 