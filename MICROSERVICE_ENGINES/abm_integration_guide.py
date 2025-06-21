#!/usr/bin/env python3
"""
ABM Integration Guide v1.0
🧠 Complete integration of Graph Hints System with BEM components
Transforms BEM into Agent-Based Model with shared interpretation maps

Integration Points:
1. ECM Gateway → Graph Hints Consumer
2. Pulse Router → ABM Signal Dispatcher  
3. Node Engine → Agent-Aware Execution
4. Frontend → ABM-Driven Visualization
5. DGL Trainer → Agent Learning Integration
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
import logging

from graph_hints_system import GraphHintsSystem, HintCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABMIntegrationLayer:
    """Integration layer for ABM transformation"""
    
    def __init__(self, hints_system: GraphHintsSystem = None):
        self.hints_system = hints_system or GraphHintsSystem()
        self.integration_points = {}
        self.active_agents = {}
        self.abm_callbacks = {}
        
        logger.info("🧠 ABM Integration Layer initialized")
    
    def integrate_ecm_gateway(self, ecm_gateway_instance) -> None:
        """Integrate ECM Gateway with Graph Hints consumption"""
        
        # Add hints consumption to ECM message processing
        original_process_message = ecm_gateway_instance.process_message
        
        async def abm_aware_process_message(message_data):
            """ABM-aware message processing"""
            
            # Get message type interpretation
            message_type = message_data.get("type", "unknown")
            type_hints = self.hints_system.get_type_role(message_type)
            
            # Enrich message with ABM context
            abm_context = {
                "abm_enabled": True,
                "type_hints": type_hints,
                "agent_adaptations": self._get_relevant_agents(message_type),
                "interpretation_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            message_data["abm_context"] = abm_context
            
            # Process with original logic
            result = await original_process_message(message_data)
            
            # Update agent feedback based on processing result
            if "success" in result:
                await self._update_agent_feedback_from_result(message_type, result)
            
            return result
        
        # Replace ECM method with ABM-aware version
        ecm_gateway_instance.process_message = abm_aware_process_message
        
        self.integration_points["ecm_gateway"] = {
            "status": "integrated",
            "features": ["hints_consumption", "agent_feedback", "message_enrichment"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("✅ ECM Gateway integrated with ABM")
    
    def integrate_pulse_router(self, pulse_router_instance) -> None:
        """Integrate Pulse Router with ABM signal dispatching"""
        
        # Add ABM signal interpretation to pulse routing
        original_route_pulse = pulse_router_instance.route_pulse
        
        async def abm_aware_route_pulse(pulse_data):
            """ABM-aware pulse routing"""
            
            pulse_type = pulse_data.get("pulse_type", "unknown")
            
            # Get signal mapping from hints system
            signal_mapping = self.hints_system.get_signal_mapping(pulse_type)
            
            # Determine routing based on agent bidding
            target_agents = self._calculate_agent_routing(pulse_type, pulse_data)
            
            # Enrich pulse with ABM routing information
            pulse_data["abm_routing"] = {
                "signal_mapping": signal_mapping,
                "target_agents": target_agents,
                "routing_confidence": self._calculate_routing_confidence(target_agents),
                "emergence_potential": self._assess_emergence_potential(pulse_data)
            }
            
            # Route with original logic
            result = await original_route_pulse(pulse_data)
            
            # Check for emergence conditions after routing
            await self._check_post_routing_emergence(pulse_data, result)
            
            return result
        
        # Replace pulse router method
        pulse_router_instance.route_pulse = abm_aware_route_pulse
        
        self.integration_points["pulse_router"] = {
            "status": "integrated",
            "features": ["signal_mapping", "agent_routing", "emergence_detection"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("✅ Pulse Router integrated with ABM")
    
    def integrate_node_engine(self, node_engine_instance) -> None:
        """Integrate Node Engine with agent-aware execution"""
        
        # Add agent influence to node execution
        original_execute_functor = node_engine_instance.execute_functor
        
        async def agent_aware_execute_functor(functor_type, node_data, context):
            """Agent-aware functor execution"""
            
            # Get agent influences for this functor type
            agent_influences = self._get_agent_influences(functor_type, context)
            
            # Apply agent coefficients to execution parameters
            modified_context = self._apply_agent_coefficients(context, agent_influences)
            
            # Get execution hints from graph hints system
            execution_hints = self.hints_system.get_type_role(functor_type)
            
            # Add ABM execution context
            modified_context["abm_execution"] = {
                "agent_influences": agent_influences,
                "execution_hints": execution_hints,
                "adaptation_weights": self._calculate_adaptation_weights(functor_type)
            }
            
            # Execute with original logic
            result = await original_execute_functor(functor_type, node_data, modified_context)
            
            # Update agent adaptations based on execution result
            await self._update_agents_from_execution(functor_type, result, agent_influences)
            
            return result
        
        # Replace node engine method
        node_engine_instance.execute_functor = agent_aware_execute_functor
        
        self.integration_points["node_engine"] = {
            "status": "integrated",
            "features": ["agent_influence", "execution_hints", "adaptation_feedback"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("✅ Node Engine integrated with ABM")
    
    def integrate_frontend_visualization(self, frontend_instance) -> None:
        """Integrate Frontend with ABM-driven visualization"""
        
        # Add ABM visualization data generation
        original_get_graph_data = frontend_instance.get_graph_data
        
        async def abm_aware_get_graph_data(request_params):
            """ABM-aware graph data generation"""
            
            # Get base graph data
            graph_data = await original_get_graph_data(request_params)
            
            # Enrich with ABM visualization data
            abm_visualization = await self._generate_abm_visualization_data(graph_data)
            
            # Add ABM layers to graph data
            graph_data["abm_layers"] = {
                "agent_influences": abm_visualization["agent_influences"],
                "signal_mappings": abm_visualization["signal_mappings"],
                "emergence_indicators": abm_visualization["emergence_indicators"],
                "adaptation_trails": abm_visualization["adaptation_trails"]
            }
            
            # Apply visual schema guarantees
            graph_data = self._apply_visual_schema_guarantees(graph_data)
            
            return graph_data
        
        # Replace frontend method
        frontend_instance.get_graph_data = abm_aware_get_graph_data
        
        self.integration_points["frontend"] = {
            "status": "integrated",
            "features": ["abm_visualization", "visual_schema", "agent_trails"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("✅ Frontend integrated with ABM visualization")
    
    def integrate_dgl_trainer(self, dgl_trainer_instance) -> None:
        """Integrate DGL Trainer with agent learning"""
        
        # Add agent learning to DGL training process
        original_train_epoch = dgl_trainer_instance.train_epoch
        
        async def abm_aware_train_epoch(epoch_data):
            """ABM-aware training epoch"""
            
            # Pre-training: Get agent learning parameters
            agent_learning_params = self._get_agent_learning_parameters()
            
            # Apply agent adaptations to training weights
            modified_epoch_data = self._apply_agent_adaptations_to_training(
                epoch_data, agent_learning_params
            )
            
            # Train with original logic
            training_result = await original_train_epoch(modified_epoch_data)
            
            # Post-training: Update agent learning based on training results
            await self._update_agent_learning_from_training(training_result, agent_learning_params)
            
            # Check for emergent learning patterns
            emergence_patterns = await self._detect_learning_emergence(training_result)
            
            training_result["abm_learning"] = {
                "agent_contributions": agent_learning_params,
                "emergence_patterns": emergence_patterns,
                "adaptation_updates": len(self.active_agents)
            }
            
            return training_result
        
        # Replace DGL trainer method
        dgl_trainer_instance.train_epoch = abm_aware_train_epoch
        
        self.integration_points["dgl_trainer"] = {
            "status": "integrated",
            "features": ["agent_learning", "adaptation_weights", "emergence_patterns"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("✅ DGL Trainer integrated with ABM learning")
    
    def register_abm_callback(self, event_type: str, callback_func: Callable) -> None:
        """Register callback for ABM events"""
        
        if event_type not in self.abm_callbacks:
            self.abm_callbacks[event_type] = []
        
        self.abm_callbacks[event_type].append(callback_func)
        
        logger.info(f"📞 Registered ABM callback for {event_type}")
    
    def trigger_abm_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger ABM event and execute callbacks"""
        
        if event_type in self.abm_callbacks:
            for callback in self.abm_callbacks[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"❌ ABM callback error for {event_type}: {e}")
        
        logger.debug(f"🎯 Triggered ABM event: {event_type}")
    
    async def _get_relevant_agents(self, message_type: str) -> List[Dict[str, Any]]:
        """Get agents relevant to message type"""
        
        relevant_agents = []
        
        for agent_id, adaptation in self.hints_system.agent_adaptations.items():
            # Check if agent has bidding pattern for this message type
            if message_type in adaptation.bidding_pattern:
                bidding_strength = adaptation.bidding_pattern[message_type]
                
                relevant_agents.append({
                    "agent_id": agent_id,
                    "bidding_strength": bidding_strength,
                    "learning_rate": adaptation.learning_rate,
                    "recent_feedback": adaptation.signal_feedback.get(message_type, 0.5)
                })
        
        # Sort by bidding strength
        relevant_agents.sort(key=lambda x: x["bidding_strength"], reverse=True)
        
        return relevant_agents
    
    async def _update_agent_feedback_from_result(self, message_type: str, result: Dict[str, Any]) -> None:
        """Update agent feedback based on processing result"""
        
        success_score = 1.0 if result.get("success", False) else 0.0
        
        # Update feedback for agents that participated
        relevant_agents = await self._get_relevant_agents(message_type)
        
        for agent_info in relevant_agents[:3]:  # Top 3 agents
            agent_id = agent_info["agent_id"]
            
            # Weight feedback by bidding strength
            weighted_feedback = success_score * agent_info["bidding_strength"]
            
            self.hints_system.update_agent_feedback(
                agent_id,
                message_type,
                weighted_feedback,
                context={"result": result, "integration": "ecm_gateway"}
            )
    
    def _calculate_agent_routing(self, pulse_type: str, pulse_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate agent-based routing for pulse"""
        
        routing_candidates = []
        
        for agent_id, adaptation in self.hints_system.agent_adaptations.items():
            # Calculate routing score based on agent capabilities
            routing_score = self._calculate_agent_routing_score(
                agent_id, pulse_type, pulse_data, adaptation
            )
            
            if routing_score > 0.3:  # Threshold for routing consideration
                routing_candidates.append({
                    "agent_id": agent_id,
                    "routing_score": routing_score,
                    "specialization": self._get_agent_specialization(agent_id),
                    "load_factor": self._get_agent_load_factor(agent_id)
                })
        
        # Sort by routing score and load balancing
        routing_candidates.sort(
            key=lambda x: (x["routing_score"], -x["load_factor"]),
            reverse=True
        )
        
        return routing_candidates[:5]  # Top 5 routing candidates
    
    def _calculate_agent_routing_score(self, agent_id: str, pulse_type: str, 
                                     pulse_data: Dict[str, Any], adaptation) -> float:
        """Calculate routing score for agent"""
        
        # Base score from bidding pattern
        base_score = adaptation.bidding_pattern.get(pulse_type, 0.5)
        
        # Adjust based on recent feedback
        feedback_adjustment = adaptation.signal_feedback.get(pulse_type, 0.5)
        
        # Consider pulse data complexity
        complexity_factor = self._assess_pulse_complexity(pulse_data)
        
        # Calculate final routing score
        routing_score = (base_score * 0.4) + (feedback_adjustment * 0.4) + (complexity_factor * 0.2)
        
        return min(1.0, max(0.0, routing_score))
    
    def _assess_pulse_complexity(self, pulse_data: Dict[str, Any]) -> float:
        """Assess pulse data complexity"""
        
        complexity_indicators = [
            len(pulse_data.get("nodes", [])) / 10.0,  # Node count factor
            len(pulse_data.get("edges", [])) / 20.0,  # Edge count factor
            len(str(pulse_data)) / 1000.0,  # Data size factor
            pulse_data.get("priority", 0.5)  # Priority factor
        ]
        
        return min(1.0, sum(complexity_indicators) / len(complexity_indicators))
    
    def _calculate_routing_confidence(self, target_agents: List[Dict[str, Any]]) -> float:
        """Calculate confidence in routing decision"""
        
        if not target_agents:
            return 0.0
        
        # Calculate confidence based on agent scores and diversity
        top_scores = [agent["routing_score"] for agent in target_agents[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Bonus for having multiple good candidates
        diversity_bonus = min(0.2, len(target_agents) * 0.05)
        
        return min(1.0, avg_score + diversity_bonus)
    
    def _assess_emergence_potential(self, pulse_data: Dict[str, Any]) -> float:
        """Assess potential for emergent behavior"""
        
        emergence_factors = [
            len(pulse_data.get("nodes", [])) > 5,  # Sufficient nodes
            pulse_data.get("complexity", 0) > 0.7,  # High complexity
            pulse_data.get("interconnectedness", 0) > 0.6,  # High connectivity
            len(self.active_agents) > 2  # Multiple agents available
        ]
        
        return sum(emergence_factors) / len(emergence_factors)
    
    async def _check_post_routing_emergence(self, pulse_data: Dict[str, Any], 
                                          routing_result: Dict[str, Any]) -> None:
        """Check for emergence conditions after pulse routing"""
        
        system_state = {
            "routing_success": routing_result.get("success", False),
            "agent_participation": len(routing_result.get("participating_agents", [])),
            "pulse_complexity": self._assess_pulse_complexity(pulse_data),
            "system_load": self._calculate_system_load()
        }
        
        # Check emergence conditions
        activated_rules = self.hints_system.check_emergence_conditions(system_state)
        
        if activated_rules:
            for rule in activated_rules:
                self.trigger_abm_event("emergence_detected", {
                    "rule": rule,
                    "pulse_data": pulse_data,
                    "routing_result": routing_result
                })
    
    def _get_agent_influences(self, functor_type: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Get agent influences for functor execution"""
        
        influences = {}
        
        for agent_id, adaptation in self.hints_system.agent_adaptations.items():
            # Calculate influence based on agent specialization and feedback
            influence_score = self._calculate_agent_influence(
                agent_id, functor_type, context, adaptation
            )
            
            if influence_score > 0.1:  # Minimum influence threshold
                influences[agent_id] = influence_score
        
        return influences
    
    def _calculate_agent_influence(self, agent_id: str, functor_type: str,
                                 context: Dict[str, Any], adaptation) -> float:
        """Calculate agent influence on functor execution"""
        
        # Base influence from bidding pattern
        base_influence = adaptation.bidding_pattern.get(functor_type, 0.0)
        
        # Adjust based on recent performance
        performance_factor = adaptation.signal_feedback.get(functor_type, 0.5)
        
        # Consider context relevance
        context_relevance = self._assess_context_relevance(agent_id, context)
        
        # Calculate final influence
        influence = (base_influence * 0.5) + (performance_factor * 0.3) + (context_relevance * 0.2)
        
        return min(1.0, max(0.0, influence))
    
    def _apply_agent_coefficients(self, context: Dict[str, Any], 
                                agent_influences: Dict[str, float]) -> Dict[str, Any]:
        """Apply agent coefficients to execution context"""
        
        modified_context = context.copy()
        
        # Apply agent influences to execution parameters
        if agent_influences:
            # Calculate weighted coefficients
            total_influence = sum(agent_influences.values())
            
            if total_influence > 0:
                agent_coefficients = {
                    agent_id: influence / total_influence
                    for agent_id, influence in agent_influences.items()
                }
                
                modified_context["agent_coefficients"] = agent_coefficients
                modified_context["total_agent_influence"] = total_influence
        
        return modified_context
    
    def _calculate_adaptation_weights(self, functor_type: str) -> Dict[str, float]:
        """Calculate adaptation weights for functor type"""
        
        weights = {}
        
        for agent_id, adaptation in self.hints_system.agent_adaptations.items():
            # Weight based on learning rate and recent activity
            learning_weight = adaptation.learning_rate
            activity_weight = len(adaptation.adaptation_history) / 100.0  # Normalize activity
            
            weights[agent_id] = min(1.0, learning_weight + activity_weight)
        
        return weights
    
    async def _update_agents_from_execution(self, functor_type: str, result: Dict[str, Any],
                                          agent_influences: Dict[str, float]) -> None:
        """Update agent adaptations from execution results"""
        
        execution_success = result.get("success", False)
        execution_quality = result.get("quality_score", 0.5)
        
        # Update each participating agent
        for agent_id, influence in agent_influences.items():
            # Calculate feedback based on execution result and agent influence
            feedback_score = execution_quality if execution_success else (execution_quality * 0.5)
            
            # Weight feedback by agent influence
            weighted_feedback = feedback_score * influence
            
            self.hints_system.update_agent_feedback(
                agent_id,
                functor_type,
                weighted_feedback,
                context={
                    "execution_result": result,
                    "agent_influence": influence,
                    "integration": "node_engine"
                }
            )
    
    async def _generate_abm_visualization_data(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ABM visualization data"""
        
        # Agent influence visualization
        agent_influences = {}
        for node in graph_data.get("nodes", []):
            node_type = node.get("type", "unknown")
            influences = self._get_agent_influences(node_type, {})
            
            if influences:
                agent_influences[node["id"]] = influences
        
        # Signal mapping visualization
        signal_mappings = {}
        for edge in graph_data.get("edges", []):
            edge_type = edge.get("type", "unknown")
            signal_data = self.hints_system.get_signal_mapping(edge_type)
            
            signal_mappings[edge["id"]] = {
                "color": signal_data.get("color_mapping", "#808080"),
                "animation": signal_data.get("animation", "none"),
                "interpretation": signal_data.get("interpretation", "unknown")
            }
        
        # Emergence indicators
        emergence_indicators = []
        system_state = self._build_system_state_from_graph(graph_data)
        activated_rules = self.hints_system.check_emergence_conditions(system_state)
        
        for rule in activated_rules:
            emergence_indicators.append({
                "rule_name": rule["rule_name"],
                "activation_timestamp": rule["activation_timestamp"],
                "visual_cue": "emergence_highlight"
            })
        
        # Adaptation trails (agent learning history visualization)
        adaptation_trails = {}
        for agent_id, adaptation in self.hints_system.agent_adaptations.items():
            if adaptation.adaptation_history:
                trails = []
                for event in adaptation.adaptation_history[-10:]:  # Last 10 events
                    trails.append({
                        "timestamp": event["timestamp"],
                        "signal": event["signal"],
                        "feedback": event["new_feedback"],
                        "context": event.get("context", {})
                    })
                
                adaptation_trails[agent_id] = trails
        
        return {
            "agent_influences": agent_influences,
            "signal_mappings": signal_mappings,
            "emergence_indicators": emergence_indicators,
            "adaptation_trails": adaptation_trails
        }
    
    def _apply_visual_schema_guarantees(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply visual schema guarantees to graph data"""
        
        # Ensure all nodes have required visual keys
        for node in graph_data.get("nodes", []):
            node_schema = self.hints_system.get_visual_schema("node")
            
            for required_key in node_schema.get("required_keys", []):
                if required_key not in node:
                    # Provide default value based on schema
                    node[required_key] = self._get_default_visual_value(required_key, node)
        
        # Ensure all edges have required visual keys
        for edge in graph_data.get("edges", []):
            edge_schema = self.hints_system.get_visual_schema("edge")
            
            for required_key in edge_schema.get("required_keys", []):
                if required_key not in edge:
                    edge[required_key] = self._get_default_visual_value(required_key, edge)
        
        return graph_data
    
    def _get_default_visual_value(self, key: str, element: Dict[str, Any]) -> Any:
        """Get default visual value for missing key"""
        
        defaults = {
            "id": element.get("id", f"element_{hash(str(element)) % 10000}"),
            "type": element.get("type", "unknown"),
            "state": "active",
            "color": "#4A90E2",
            "animation": "none",
            "metadata": {}
        }
        
        return defaults.get(key, "unknown")
    
    def _get_agent_learning_parameters(self) -> Dict[str, Any]:
        """Get agent learning parameters for DGL training"""
        
        learning_params = {}
        
        for agent_id, adaptation in self.hints_system.agent_adaptations.items():
            learning_params[agent_id] = {
                "learning_rate": adaptation.learning_rate,
                "bidding_weights": adaptation.bidding_pattern,
                "feedback_history": adaptation.signal_feedback,
                "adaptation_momentum": len(adaptation.adaptation_history) / 100.0
            }
        
        return learning_params
    
    def _apply_agent_adaptations_to_training(self, epoch_data: Dict[str, Any],
                                           agent_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply agent adaptations to training data"""
        
        modified_data = epoch_data.copy()
        
        # Calculate combined agent influence on training weights
        if agent_params:
            agent_weights = {}
            
            for agent_id, params in agent_params.items():
                # Combine learning rate and adaptation momentum
                weight = params["learning_rate"] * (1.0 + params["adaptation_momentum"])
                agent_weights[agent_id] = weight
            
            # Normalize weights
            total_weight = sum(agent_weights.values())
            if total_weight > 0:
                normalized_weights = {
                    agent_id: weight / total_weight
                    for agent_id, weight in agent_weights.items()
                }
                
                modified_data["agent_training_weights"] = normalized_weights
        
        return modified_data
    
    async def _update_agent_learning_from_training(self, training_result: Dict[str, Any],
                                                 agent_params: Dict[str, Any]) -> None:
        """Update agent learning based on training results"""
        
        training_loss = training_result.get("loss", 1.0)
        training_accuracy = training_result.get("accuracy", 0.0)
        
        # Convert training metrics to feedback scores
        feedback_score = max(0.0, min(1.0, training_accuracy - (training_loss * 0.1)))
        
        # Update each agent based on their contribution to training
        for agent_id, params in agent_params.items():
            # Weight feedback by agent's training contribution
            agent_weight = training_result.get("agent_training_weights", {}).get(agent_id, 0.1)
            weighted_feedback = feedback_score * agent_weight
            
            # Update agent feedback for "training" signal
            self.hints_system.update_agent_feedback(
                agent_id,
                "training",
                weighted_feedback,
                context={
                    "training_result": training_result,
                    "agent_contribution": agent_weight,
                    "integration": "dgl_trainer"
                }
            )
    
    async def _detect_learning_emergence(self, training_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect emergent learning patterns"""
        
        emergence_patterns = []
        
        # Check for convergence emergence
        if training_result.get("convergence_rate", 0) > 0.8:
            emergence_patterns.append({
                "pattern_type": "rapid_convergence",
                "confidence": training_result["convergence_rate"],
                "description": "Training converged faster than expected"
            })
        
        # Check for agent collaboration emergence
        agent_weights = training_result.get("agent_training_weights", {})
        if len(agent_weights) > 1:
            weight_variance = self._calculate_variance(list(agent_weights.values()))
            
            if weight_variance < 0.1:  # Low variance indicates collaboration
                emergence_patterns.append({
                    "pattern_type": "agent_collaboration",
                    "confidence": 1.0 - weight_variance,
                    "description": "Agents showing collaborative learning behavior"
                })
        
        return emergence_patterns
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        return variance
    
    def _build_system_state_from_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build system state dictionary from graph data"""
        
        return {
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get("edges", [])),
            "average_connectivity": self._calculate_average_connectivity(graph_data),
            "system_complexity": self._calculate_system_complexity(graph_data),
            "active_agents": len(self.active_agents)
        }
    
    def _calculate_average_connectivity(self, graph_data: Dict[str, Any]) -> float:
        """Calculate average node connectivity"""
        
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        if not nodes:
            return 0.0
        
        # Count connections per node
        node_connections = {}
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            
            if source:
                node_connections[source] = node_connections.get(source, 0) + 1
            if target:
                node_connections[target] = node_connections.get(target, 0) + 1
        
        # Calculate average
        total_connections = sum(node_connections.values())
        return total_connections / len(nodes) if nodes else 0.0
    
    def _calculate_system_complexity(self, graph_data: Dict[str, Any]) -> float:
        """Calculate overall system complexity"""
        
        node_count = len(graph_data.get("nodes", []))
        edge_count = len(graph_data.get("edges", []))
        
        # Complexity based on nodes, edges, and their relationships
        if node_count == 0:
            return 0.0
        
        edge_density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0
        size_factor = min(1.0, node_count / 20.0)  # Normalize to 20 nodes
        
        return (edge_density + size_factor) / 2.0
    
    def _assess_context_relevance(self, agent_id: str, context: Dict[str, Any]) -> float:
        """Assess how relevant context is to agent"""
        
        # Simple relevance based on context keys matching agent specializations
        agent_specializations = self._get_agent_specialization(agent_id)
        context_keys = set(context.keys())
        specialization_keys = set(agent_specializations.keys())
        
        # Calculate overlap
        overlap = len(context_keys.intersection(specialization_keys))
        total_keys = len(context_keys.union(specialization_keys))
        
        return overlap / total_keys if total_keys > 0 else 0.0
    
    def _get_agent_specialization(self, agent_id: str) -> Dict[str, float]:
        """Get agent specialization profile"""
        
        if agent_id not in self.hints_system.agent_adaptations:
            return {}
        
        adaptation = self.hints_system.agent_adaptations[agent_id]
        
        # Specialization based on bidding pattern strengths
        specializations = {}
        for signal, strength in adaptation.bidding_pattern.items():
            if strength > 1.0:  # Above average bidding indicates specialization
                specializations[signal] = strength
        
        return specializations
    
    def _get_agent_load_factor(self, agent_id: str) -> float:
        """Get current load factor for agent"""
        
        # Simple load factor based on recent activity
        if agent_id not in self.hints_system.agent_adaptations:
            return 0.0
        
        adaptation = self.hints_system.agent_adaptations[agent_id]
        recent_activity = len([
            event for event in adaptation.adaptation_history
            if self._is_recent_event(event["timestamp"])
        ])
        
        # Normalize to 0-1 scale (assume max 10 recent events = full load)
        return min(1.0, recent_activity / 10.0)
    
    def _is_recent_event(self, timestamp_str: str) -> bool:
        """Check if event timestamp is recent (within last hour)"""
        
        try:
            event_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            time_diff = current_time - event_time
            
            return time_diff.total_seconds() < 3600  # 1 hour
        except:
            return False
    
    def _calculate_system_load(self) -> float:
        """Calculate overall system load"""
        
        if not self.active_agents:
            return 0.0
        
        total_load = sum(
            self._get_agent_load_factor(agent_id)
            for agent_id in self.active_agents
        )
        
        return total_load / len(self.active_agents)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get complete integration status"""
        
        return {
            "integration_points": self.integration_points,
            "active_agents": len(self.active_agents),
            "abm_callbacks": {
                event_type: len(callbacks)
                for event_type, callbacks in self.abm_callbacks.items()
            },
            "system_coherence": self.hints_system._calculate_coherence_score(),
            "total_hints": sum(len(hints) for hints in self.hints_system.hints.values()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Convenience function for full BEM → ABM transformation
async def transform_bem_to_abm(bem_components: Dict[str, Any],
                              hints_directory: str = None) -> ABMIntegrationLayer:
    """Transform complete BEM system to ABM"""
    
    # Create ABM integration layer
    abm_layer = ABMIntegrationLayer(
        GraphHintsSystem(hints_directory) if hints_directory else None
    )
    
    # Integrate all BEM components
    integration_tasks = []
    
    if "ecm_gateway" in bem_components:
        abm_layer.integrate_ecm_gateway(bem_components["ecm_gateway"])
    
    if "pulse_router" in bem_components:
        abm_layer.integrate_pulse_router(bem_components["pulse_router"])
    
    if "node_engine" in bem_components:
        abm_layer.integrate_node_engine(bem_components["node_engine"])
    
    if "frontend" in bem_components:
        abm_layer.integrate_frontend_visualization(bem_components["frontend"])
    
    if "dgl_trainer" in bem_components:
        abm_layer.integrate_dgl_trainer(bem_components["dgl_trainer"])
    
    logger.info("🎉 BEM → ABM transformation complete!")
    
    return abm_layer

if __name__ == "__main__":
    print("🧠 ABM Integration Guide v1.0")
    print("🔄 Transforming BEM into Agent-Based Model...")
    
    # Demo integration layer
    abm_layer = ABMIntegrationLayer()
    
    # Register sample ABM callback
    def emergence_callback(event_data):
        print(f"🌟 Emergence detected: {event_data['rule']['rule_name']}")
    
    abm_layer.register_abm_callback("emergence_detected", emergence_callback)
    
    # Show integration status
    status = abm_layer.get_integration_status()
    print(f"\n📊 Integration Status:")
    print(f"  Integration Points: {len(status['integration_points'])}")
    print(f"  Active Agents: {status['active_agents']}")
    print(f"  ABM Callbacks: {sum(status['abm_callbacks'].values())}")
    print(f"  System Coherence: {status['system_coherence']:.3f}")
    
    print("\n🎯 ABM Integration Ready!")
    print("🧠 BEM transformed into Agent-Based Model with shared interpretation maps!")
