#!/usr/bin/env python3
"""
SDFA Node Engine Integration v1.0
🔗 Integration layer connecting SDFA Visual Engine with BEM Node Engine
Ensures visual feedback emerges from real system data, not hardcoded UI rules

Features:
- Automatic visual dictionary generation during node execution
- Real-time performance monitoring and color assignment
- Node state tracking with visual feedback
- Platform-agnostic rendering data export
- Seamless integration with existing functor execution
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path

# Import SDFA components
from .sdfa_visual_engine import (
    SDFAVisualEngine, create_sdfa_engine, NodeState, DesignSignal, 
    VisualDictionary, ColorMapping
)

# Import traceability for integration
try:
    from .traceability_engine import TraceabilityEngine
except ImportError:
    TraceabilityEngine = None

class SDFANodeIntegration:
    """Integration layer for SDFA with Node Engine"""
    
    def __init__(self, sdfa_engine: SDFAVisualEngine = None, 
                 traceability_engine: TraceabilityEngine = None):
        self.sdfa_engine = sdfa_engine or create_sdfa_engine()
        self.traceability_engine = traceability_engine
        
        # Node performance tracking
        self.node_performance_history: Dict[str, List[float]] = {}
        self.node_execution_count: Dict[str, int] = {}
        
        # Visual state cache
        self.visual_cache: Dict[str, VisualDictionary] = {}
        self.cache_expiry: Dict[str, float] = {}
        self.cache_ttl = 300  # 5 minutes
        
        print("🔗 SDFA Node Integration v1.0 initialized")
    
    def register_node_execution(self, node_id: str, performance_data: Dict[str, Any],
                               execution_context: Dict[str, Any] = None) -> VisualDictionary:
        """Register node execution and generate visual dictionary"""
        
        # Update execution tracking
        if node_id not in self.node_execution_count:
            self.node_execution_count[node_id] = 0
        self.node_execution_count[node_id] += 1
        
        # Track performance history
        performance_score = performance_data.get('performance_score', 0.5)
        if node_id not in self.node_performance_history:
            self.node_performance_history[node_id] = []
        
        self.node_performance_history[node_id].append(performance_score)
        
        # Keep only last 10 performance scores
        if len(self.node_performance_history[node_id]) > 10:
            self.node_performance_history[node_id] = self.node_performance_history[node_id][-10:]
        
        # Calculate additional metrics based on history
        enhanced_data = self._enhance_performance_data(node_id, performance_data)
        
        # Generate visual dictionary
        visual_dict = self.sdfa_engine.generate_visual_dictionary(node_id, enhanced_data)
        
        # Cache the result
        self.visual_cache[node_id] = visual_dict
        self.cache_expiry[node_id] = time.time() + self.cache_ttl
        
        # Log to traceability if available
        if self.traceability_engine:
            self._log_visual_generation(node_id, visual_dict, execution_context)
        
        return visual_dict
    
    def get_node_visual_data(self, node_id: str, target_platform: str = "web",
                           force_refresh: bool = False) -> Dict[str, Any]:
        """Get visual rendering data for a node"""
        
        # Check cache first
        if not force_refresh and self._is_cache_valid(node_id):
            visual_dict = self.visual_cache[node_id]
        else:
            # Cache miss or expired - use default data
            if node_id not in self.visual_cache:
                # Generate default visual dictionary
                default_data = {"score": 0.5, "quality": 0.5, "stability": 0.5}
                visual_dict = self.sdfa_engine.generate_visual_dictionary(node_id, default_data)
                self.visual_cache[node_id] = visual_dict
                self.cache_expiry[node_id] = time.time() + self.cache_ttl
            else:
                visual_dict = self.visual_cache[node_id]
        
        # Get platform-specific rendering data
        return self.sdfa_engine.get_rendering_data(node_id, target_platform)
    
    def update_node_performance(self, node_id: str, new_performance: Dict[str, float]) -> VisualDictionary:
        """Update node performance and regenerate visuals"""
        
        # Get existing data or create new
        if node_id in self.visual_cache:
            existing_dict = self.visual_cache[node_id]
            # Merge with existing visual metadata
            current_data = {
                "score": existing_dict.performance_score,
                "quality": new_performance.get("quality", 0.5),
                "stability": new_performance.get("stability", 0.5),
                "convergence": new_performance.get("convergence", 0.5)
            }
            current_data.update(new_performance)
        else:
            current_data = new_performance
        
        # Register execution with updated data
        return self.register_node_execution(node_id, current_data)
    
    def get_system_visual_overview(self) -> Dict[str, Any]:
        """Get overview of visual state across all nodes"""
        
        overview = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_nodes": len(self.visual_cache),
            "state_distribution": {},
            "performance_summary": {
                "average_performance": 0.0,
                "peak_performers": [],
                "critical_nodes": [],
                "total_executions": sum(self.node_execution_count.values())
            },
            "visual_health": {
                "cache_hit_rate": 0.0,
                "expired_entries": 0,
                "last_update_times": {}
            }
        }
        
        # Calculate state distribution and performance
        total_performance = 0.0
        valid_nodes = 0
        
        for node_id, visual_dict in self.visual_cache.items():
            # State distribution
            state = visual_dict.node_state.value
            overview["state_distribution"][state] = overview["state_distribution"].get(state, 0) + 1
            
            # Performance tracking
            performance = visual_dict.performance_score
            total_performance += performance
            valid_nodes += 1
            
            # Identify peak performers and critical nodes
            if performance >= 0.9:
                overview["performance_summary"]["peak_performers"].append({
                    "node_id": node_id,
                    "performance": performance,
                    "state": state
                })
            elif performance <= 0.2:
                overview["performance_summary"]["critical_nodes"].append({
                    "node_id": node_id,
                    "performance": performance,
                    "state": state
                })
            
            # Visual health tracking
            overview["visual_health"]["last_update_times"][node_id] = visual_dict.last_updated
        
        # Calculate averages
        if valid_nodes > 0:
            overview["performance_summary"]["average_performance"] = total_performance / valid_nodes
        
        # Cache health
        current_time = time.time()
        expired_count = sum(1 for expiry in self.cache_expiry.values() if expiry < current_time)
        overview["visual_health"]["expired_entries"] = expired_count
        
        if len(self.cache_expiry) > 0:
            overview["visual_health"]["cache_hit_rate"] = (len(self.cache_expiry) - expired_count) / len(self.cache_expiry)
        
        return overview
    
    def export_rendering_package(self, target_platform: str, 
                               output_path: str = None) -> Dict[str, Any]:
        """Export complete rendering package for target platform"""
        
        package = {
            "platform": target_platform,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "nodes": {},
            "global_styles": self._get_platform_styles(target_platform),
            "animation_definitions": self._get_animation_definitions(target_platform),
            "metadata": {
                "total_nodes": len(self.visual_cache),
                "sdfa_version": "1.0",
                "cache_status": "current"
            }
        }
        
        # Export all node rendering data
        for node_id in self.visual_cache.keys():
            package["nodes"][node_id] = self.get_node_visual_data(node_id, target_platform)
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(package, f, indent=2, default=str)
            print(f"📦 Rendering package exported to {output_path}")
        
        return package
    
    def _enhance_performance_data(self, node_id: str, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance performance data with historical context"""
        
        enhanced = base_data.copy()
        
        # Add execution context
        enhanced["execution_count"] = self.node_execution_count.get(node_id, 0)
        
        # Add performance trends
        if node_id in self.node_performance_history:
            history = self.node_performance_history[node_id]
            
            if len(history) > 1:
                # Calculate trend
                recent_avg = sum(history[-3:]) / min(3, len(history))
                overall_avg = sum(history) / len(history)
                trend = recent_avg - overall_avg
                
                enhanced["performance_trend"] = trend
                enhanced["stability"] = 1.0 - (max(history) - min(history))
                enhanced["convergence"] = 1.0 - abs(trend)
            else:
                enhanced["performance_trend"] = 0.0
                enhanced["stability"] = base_data.get("stability", 0.5)
                enhanced["convergence"] = base_data.get("convergence", 0.5)
        
        # Add emergence factors for complex behaviors
        if enhanced.get("execution_count", 0) > 5:
            interaction_strength = min(1.0, enhanced.get("execution_count", 0) / 20.0)
            complexity = enhanced.get("stability", 0.5) * enhanced.get("convergence", 0.5)
            novelty = abs(enhanced.get("performance_trend", 0.0))
            
            enhanced["emergence"] = {
                "interaction_strength": interaction_strength,
                "complexity": complexity,
                "novelty": novelty
            }
        
        return enhanced
    
    def _is_cache_valid(self, node_id: str) -> bool:
        """Check if cached visual data is still valid"""
        
        if node_id not in self.cache_expiry:
            return False
        
        return time.time() < self.cache_expiry[node_id]
    
    def _log_visual_generation(self, node_id: str, visual_dict: VisualDictionary,
                             execution_context: Dict[str, Any] = None):
        """Log visual generation to traceability system"""
        
        if not self.traceability_engine:
            return
        
        try:
            # Create trace for visual generation
            trace_id = self.traceability_engine.start_trace(
                node_id=node_id,
                functor="sdfa_visual_generation",
                agent_triggered="SDFA_Engine",
                input_data={
                    "performance_score": visual_dict.performance_score,
                    "node_state": visual_dict.node_state.value,
                    "signal_count": len(visual_dict.design_signals)
                }
            )
            
            # Log decisions made during visual generation
            self.traceability_engine.log_decision(
                trace_id, 
                f"state_determined_{visual_dict.node_state.value}",
                f"performance={visual_dict.performance_score:.3f}",
                f"Node state determined based on performance evaluation"
            )
            
            self.traceability_engine.log_decision(
                trace_id,
                f"color_assigned_{visual_dict.color_mapping.primary_color}",
                f"state={visual_dict.node_state.value}",
                f"Color assigned based on node state and design signals"
            )
            
            # End trace
            self.traceability_engine.end_trace(trace_id, {
                "visual_dictionary_generated": True,
                "color": visual_dict.color_mapping.primary_color,
                "animation": visual_dict.color_mapping.animation_type
            })
            
        except Exception as e:
            print(f"⚠️ Failed to log visual generation: {e}")
    
    def _get_platform_styles(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific global styles"""
        
        if platform == "web":
            return {
                "css_variables": {
                    "--sdfa-evolutionary-peak": "#00FF41",
                    "--sdfa-high-performance": "#41FF00",
                    "--sdfa-neutral-state": "#FFD700",
                    "--sdfa-low-precision": "#FF8C00",
                    "--sdfa-critical-state": "#FF0000",
                    "--sdfa-undefined": "#808080"
                },
                "base_classes": [
                    ".sdfa-node { transition: all 0.3s ease; }",
                    ".sdfa-evolutionary-peak { border: 2px solid var(--sdfa-evolutionary-peak); }",
                    ".sdfa-critical-state { box-shadow: 0 0 10px var(--sdfa-critical-state); }"
                ]
            }
        
        elif platform == "unreal":
            return {
                "material_parameters": {
                    "EmissiveStrength": 1.0,
                    "Roughness": 0.3,
                    "Metallic": 0.1
                },
                "lighting_setup": {
                    "AmbientOcclusion": True,
                    "GlobalIllumination": True
                }
            }
        
        elif platform == "cytoscape":
            return {
                "default_style": {
                    "node": {
                        "width": 30,
                        "height": 30,
                        "border-width": 2,
                        "border-opacity": 1,
                        "font-size": 12
                    },
                    "edge": {
                        "width": 2,
                        "line-color": "#cccccc",
                        "target-arrow-color": "#cccccc",
                        "target-arrow-shape": "triangle"
                    }
                }
            }
        
        return {}
    
    def _get_animation_definitions(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific animation definitions"""
        
        if platform == "web":
            return {
                "keyframes": {
                    "pulse": "@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }",
                    "glow": "@keyframes glow { 0%, 100% { box-shadow: 0 0 5px currentColor; } 50% { box-shadow: 0 0 20px currentColor; } }",
                    "urgent_pulse": "@keyframes urgent_pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }"
                },
                "classes": {
                    ".pulse": "animation: pulse 1s ease-in-out infinite;",
                    ".glow": "animation: glow 2s linear infinite;",
                    ".urgent_pulse": "animation: urgent_pulse 0.5s linear infinite;"
                }
            }
        
        elif platform == "unreal":
            return {
                "timeline_curves": {
                    "pulse": "Linear interpolation from 0 to 1 over 1 second",
                    "glow": "Sine wave interpolation over 2 seconds",
                    "urgent_pulse": "Rapid scale animation over 0.5 seconds"
                }
            }
        
        return {}

# Decorator for automatic SDFA integration
def sdfa_visual_tracking(sdfa_integration: SDFANodeIntegration = None):
    """Decorator for automatic visual tracking of functor execution"""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create SDFA integration
            integration = sdfa_integration or SDFANodeIntegration()
            
            # Extract node information
            node_id = kwargs.get('node_id') or (args[0] if args else 'unknown')
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Extract performance data from result
                if isinstance(result, dict):
                    performance_data = {
                        "performance_score": result.get("score", 0.5),
                        "quality": result.get("quality", 0.5),
                        "execution_time": execution_time
                    }
                else:
                    performance_data = {
                        "performance_score": 0.5,
                        "execution_time": execution_time
                    }
                
                # Register execution and generate visuals
                integration.register_node_execution(
                    str(node_id), 
                    performance_data,
                    {"function_name": func.__name__, "execution_time": execution_time}
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Register failed execution
                integration.register_node_execution(
                    str(node_id),
                    {"performance_score": 0.1, "execution_time": execution_time, "error": str(e)},
                    {"function_name": func.__name__, "error": True}
                )
                
                raise
        
        return wrapper
    return decorator

# Convenience functions
def create_sdfa_integration(sdfa_engine: SDFAVisualEngine = None) -> SDFANodeIntegration:
    """Create SDFA node integration instance"""
    return SDFANodeIntegration(sdfa_engine)

def get_node_colors(node_id: str, integration: SDFANodeIntegration = None) -> Dict[str, str]:
    """Get color information for a node"""
    if integration is None:
        integration = create_sdfa_integration()
    
    visual_data = integration.get_node_visual_data(node_id, "web")
    return {
        "primary": visual_data.get("css_color", "#808080"),
        "state": visual_data.get("state_class", "undefined"),
        "animation": visual_data.get("animation_class", "none")
    }

if __name__ == "__main__":
    print("🔗 SDFA Node Integration v1.0 - Demo")
    
    # Create integration
    integration = create_sdfa_integration()
    
    # Demo node executions with different performance levels
    demo_executions = [
        {"node_id": "V01_DEMO", "performance": {"performance_score": 0.95, "quality": 0.92}},
        {"node_id": "V02_DEMO", "performance": {"performance_score": 0.78, "quality": 0.85}},
        {"node_id": "V03_DEMO", "performance": {"performance_score": 0.45, "quality": 0.50}},
        {"node_id": "V04_DEMO", "performance": {"performance_score": 0.15, "quality": 0.20}}
    ]
    
    print("\n🎯 Registering Node Executions:")
    for execution in demo_executions:
        visual_dict = integration.register_node_execution(
            execution["node_id"], 
            execution["performance"]
        )
        
        print(f"📊 {execution['node_id']}: {visual_dict.node_state.value} -> {visual_dict.color_mapping.primary_color}")
    
    # Get system overview
    overview = integration.get_system_visual_overview()
    print(f"\n📋 System Overview:")
    print(f"  Total Nodes: {overview['total_nodes']}")
    print(f"  Average Performance: {overview['performance_summary']['average_performance']:.3f}")
    print(f"  State Distribution: {overview['state_distribution']}")
    
    # Export rendering packages
    web_package = integration.export_rendering_package("web")
    print(f"\n🌐 Web Package: {len(web_package['nodes'])} nodes exported")
    
    unreal_package = integration.export_rendering_package("unreal")
    print(f"🎮 Unreal Package: {len(unreal_package['nodes'])} nodes exported")
    
    print("\n🎉 SDFA Node Integration Demo Complete!")
    print("🎨 Visual feedback emerging from real system data!")
