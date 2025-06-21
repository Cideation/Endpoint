#!/usr/bin/env python3
"""
Scientific Design Formula Assignment (SDFA) Visual Engine v1.0
🎨 Structured visual output system for BEM Node Engine coordination
Color as interpretable system data, not cosmetic UI elements

Features:
- Performance-based design signal computation
- Node state evaluation (evolutionary_peak, neutral_state, low_precision)
- Automatic color mapping from system logic
- Traceable visual cues reflecting system health
- Real-time emergent behavior visualization
- Modular SDFA helper decoupled from core engine
"""

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeState(Enum):
    """Node states based on performance evaluation"""
    EVOLUTIONARY_PEAK = "evolutionary_peak"
    HIGH_PERFORMANCE = "high_performance"
    NEUTRAL_STATE = "neutral_state"
    LOW_PRECISION = "low_precision"
    CRITICAL_STATE = "critical_state"
    UNDEFINED = "undefined"

class DesignSignalType(Enum):
    """Types of design signals computed by SDFA"""
    PERFORMANCE_GRADIENT = "performance_gradient"
    CONVERGENCE_INDICATOR = "convergence_indicator"
    STABILITY_MEASURE = "stability_measure"
    EMERGENCE_FACTOR = "emergence_factor"
    QUALITY_ASSESSMENT = "quality_assessment"

@dataclass
class ColorMapping:
    """Color mapping with gradient information"""
    primary_color: str
    secondary_color: str = None
    gradient_direction: str = "linear"
    opacity: float = 1.0
    animation_type: str = "none"
    pulse_frequency: float = 0.0
    
    def to_css(self) -> str:
        """Convert to CSS color representation"""
        if self.secondary_color:
            return f"linear-gradient({self.gradient_direction}, {self.primary_color}, {self.secondary_color})"
        return self.primary_color
    
    def to_unreal_color(self) -> Dict[str, float]:
        """Convert to Unreal Engine color format (RGBA 0-1)"""
        # Parse hex color to RGBA
        color = self.primary_color.lstrip('#')
        if len(color) == 6:
            r, g, b = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            return {"r": r, "g": g, "b": b, "a": self.opacity}
        return {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0}  # Fallback gray

@dataclass
class DesignSignal:
    """Design signal computed by SDFA helper"""
    signal_type: DesignSignalType
    value: float
    confidence: float
    timestamp: str
    contributing_factors: Dict[str, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class VisualDictionary:
    """Complete visual dictionary for a node"""
    node_id: str
    node_state: NodeState
    design_signals: List[DesignSignal]
    color_mapping: ColorMapping
    performance_score: float
    visual_metadata: Dict[str, Any]
    last_updated: str
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()

class SDFAVisualEngine:
    """Scientific Design Formula Assignment Visual Engine"""
    
    def __init__(self, config_path: str = "MICROSERVICE_ENGINES/sdfa_config.json"):
        self.config_path = Path(config_path)
        self.visual_dictionaries: Dict[str, VisualDictionary] = {}
        
        # Load SDFA configuration
        self.config = self._load_config()
        
        # Performance thresholds for state determination
        self.state_thresholds = self.config.get("state_thresholds", {
            "evolutionary_peak": 0.9,
            "high_performance": 0.75,
            "neutral_state": 0.5,
            "low_precision": 0.25,
            "critical_state": 0.1
        })
        
        # Color mappings for each state
        self.color_mappings = self.config.get("color_mappings", {
            "evolutionary_peak": {"primary": "#00FF41", "secondary": "#00CC33", "animation": "pulse"},
            "high_performance": {"primary": "#41FF00", "secondary": "#33CC00", "animation": "glow"},
            "neutral_state": {"primary": "#FFD700", "secondary": "#FFA500", "animation": "none"},
            "low_precision": {"primary": "#FF8C00", "secondary": "#FF4500", "animation": "fade"},
            "critical_state": {"primary": "#FF0000", "secondary": "#CC0000", "animation": "urgent_pulse"},
            "undefined": {"primary": "#808080", "secondary": "#606060", "animation": "none"}
        })
        
        logger.info("🎨 SDFA Visual Engine v1.0 initialized")
    
    def evaluate_node_performance(self, node_data: Dict[str, Any], 
                                 performance_metrics: Dict[str, float] = None) -> float:
        """Evaluate node performance based on multiple metrics"""
        
        if performance_metrics is None:
            performance_metrics = {}
        
        # Extract key performance variables
        base_score = node_data.get("score", 0.5)
        quality_score = node_data.get("quality", 0.5)
        stability_score = node_data.get("stability", 0.5)
        convergence_score = node_data.get("convergence", 0.5)
        
        # Apply performance metrics if provided
        if performance_metrics:
            base_score = performance_metrics.get("performance_score", base_score)
            quality_score = performance_metrics.get("quality_score", quality_score)
            stability_score = performance_metrics.get("stability_score", stability_score)
            convergence_score = performance_metrics.get("convergence_score", convergence_score)
        
        # Weighted performance calculation
        weights = self.config.get("performance_weights", {
            "base": 0.4,
            "quality": 0.3,
            "stability": 0.2,
            "convergence": 0.1
        })
        
        performance_score = (
            base_score * weights["base"] +
            quality_score * weights["quality"] +
            stability_score * weights["stability"] +
            convergence_score * weights["convergence"]
        )
        
        # Normalize to 0-1 range
        performance_score = max(0.0, min(1.0, performance_score))
        
        logger.debug(f"📊 Node performance evaluated: {performance_score:.3f}")
        return performance_score
    
    def compute_design_signals(self, node_data: Dict[str, Any], 
                             performance_score: float) -> List[DesignSignal]:
        """Compute design signals based on node data and performance"""
        
        signals = []
        
        # Performance Gradient Signal
        gradient_signal = DesignSignal(
            signal_type=DesignSignalType.PERFORMANCE_GRADIENT,
            value=performance_score,
            confidence=0.95,
            timestamp=datetime.now(timezone.utc).isoformat(),
            contributing_factors={
                "base_score": node_data.get("score", 0.5),
                "quality": node_data.get("quality", 0.5),
                "stability": node_data.get("stability", 0.5)
            }
        )
        signals.append(gradient_signal)
        
        # Convergence Indicator Signal
        convergence_value = node_data.get("convergence", 0.5)
        convergence_confidence = min(1.0, abs(convergence_value - 0.5) * 2)
        
        convergence_signal = DesignSignal(
            signal_type=DesignSignalType.CONVERGENCE_INDICATOR,
            value=convergence_value,
            confidence=convergence_confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            contributing_factors={
                "iteration_count": node_data.get("iterations", 1),
                "delta_change": node_data.get("delta", 0.1)
            }
        )
        signals.append(convergence_signal)
        
        # Stability Measure Signal
        stability_value = node_data.get("stability", 0.5)
        stability_variance = node_data.get("variance", 0.1)
        
        stability_signal = DesignSignal(
            signal_type=DesignSignalType.STABILITY_MEASURE,
            value=stability_value,
            confidence=1.0 - stability_variance,
            timestamp=datetime.now(timezone.utc).isoformat(),
            contributing_factors={
                "variance": stability_variance,
                "oscillation": node_data.get("oscillation", 0.0)
            }
        )
        signals.append(stability_signal)
        
        # Emergence Factor Signal (for complex behaviors)
        emergence_factors = node_data.get("emergence", {})
        emergence_value = 0.0
        
        if emergence_factors:
            # Calculate emergence based on non-linear interactions
            interaction_strength = emergence_factors.get("interaction_strength", 0.0)
            complexity_measure = emergence_factors.get("complexity", 0.0)
            novelty_factor = emergence_factors.get("novelty", 0.0)
            
            emergence_value = (interaction_strength * complexity_measure * novelty_factor) ** (1/3)
        
        emergence_signal = DesignSignal(
            signal_type=DesignSignalType.EMERGENCE_FACTOR,
            value=emergence_value,
            confidence=0.8 if emergence_factors else 0.3,
            timestamp=datetime.now(timezone.utc).isoformat(),
            contributing_factors=emergence_factors
        )
        signals.append(emergence_signal)
        
        # Quality Assessment Signal
        quality_metrics = node_data.get("quality_metrics", {})
        quality_value = node_data.get("quality", 0.5)
        
        quality_signal = DesignSignal(
            signal_type=DesignSignalType.QUALITY_ASSESSMENT,
            value=quality_value,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc).isoformat(),
            contributing_factors=quality_metrics
        )
        signals.append(quality_signal)
        
        logger.debug(f"🎯 Computed {len(signals)} design signals")
        return signals
    
    def determine_node_state(self, performance_score: float, 
                           design_signals: List[DesignSignal]) -> NodeState:
        """Determine node state based on performance and design signals"""
        
        # Primary state determination based on performance score
        if performance_score >= self.state_thresholds["evolutionary_peak"]:
            base_state = NodeState.EVOLUTIONARY_PEAK
        elif performance_score >= self.state_thresholds["high_performance"]:
            base_state = NodeState.HIGH_PERFORMANCE
        elif performance_score >= self.state_thresholds["neutral_state"]:
            base_state = NodeState.NEUTRAL_STATE
        elif performance_score >= self.state_thresholds["low_precision"]:
            base_state = NodeState.LOW_PRECISION
        elif performance_score >= self.state_thresholds["critical_state"]:
            base_state = NodeState.CRITICAL_STATE
        else:
            base_state = NodeState.UNDEFINED
        
        # Refine state based on design signals
        stability_signals = [s for s in design_signals if s.signal_type == DesignSignalType.STABILITY_MEASURE]
        convergence_signals = [s for s in design_signals if s.signal_type == DesignSignalType.CONVERGENCE_INDICATOR]
        
        # Check for instability that might downgrade state
        if stability_signals and stability_signals[0].value < 0.3:
            if base_state == NodeState.EVOLUTIONARY_PEAK:
                base_state = NodeState.HIGH_PERFORMANCE
            elif base_state == NodeState.HIGH_PERFORMANCE:
                base_state = NodeState.NEUTRAL_STATE
        
        # Check for poor convergence
        if convergence_signals and convergence_signals[0].value < 0.2:
            if base_state in [NodeState.EVOLUTIONARY_PEAK, NodeState.HIGH_PERFORMANCE]:
                base_state = NodeState.LOW_PRECISION
        
        logger.debug(f"🎯 Node state determined: {base_state.value}")
        return base_state
    
    def create_color_mapping(self, node_state: NodeState, 
                           design_signals: List[DesignSignal]) -> ColorMapping:
        """Create color mapping based on node state and design signals"""
        
        # Get base color mapping for state
        state_colors = self.color_mappings.get(node_state.value, self.color_mappings["undefined"])
        
        # Create base color mapping
        color_mapping = ColorMapping(
            primary_color=state_colors["primary"],
            secondary_color=state_colors.get("secondary"),
            gradient_direction="45deg",
            opacity=1.0,
            animation_type=state_colors.get("animation", "none")
        )
        
        # Modify based on design signals
        for signal in design_signals:
            if signal.signal_type == DesignSignalType.PERFORMANCE_GRADIENT:
                # Adjust opacity based on performance
                color_mapping.opacity = max(0.3, min(1.0, signal.value))
            
            elif signal.signal_type == DesignSignalType.STABILITY_MEASURE:
                # Adjust pulse frequency based on stability
                if signal.value < 0.5:
                    color_mapping.pulse_frequency = (1.0 - signal.value) * 2.0
                    color_mapping.animation_type = "instability_pulse"
            
            elif signal.signal_type == DesignSignalType.EMERGENCE_FACTOR:
                # Special handling for emergence
                if signal.value > 0.7:
                    color_mapping.animation_type = "emergence_glow"
                    color_mapping.pulse_frequency = signal.value
        
        logger.debug(f"🎨 Color mapping created: {color_mapping.primary_color}")
        return color_mapping
    
    def generate_visual_dictionary(self, node_id: str, node_data: Dict[str, Any],
                                 performance_metrics: Dict[str, float] = None) -> VisualDictionary:
        """Generate complete visual dictionary for a node"""
        
        # Evaluate performance
        performance_score = self.evaluate_node_performance(node_data, performance_metrics)
        
        # Compute design signals
        design_signals = self.compute_design_signals(node_data, performance_score)
        
        # Determine node state
        node_state = self.determine_node_state(performance_score, design_signals)
        
        # Create color mapping
        color_mapping = self.create_color_mapping(node_state, design_signals)
        
        # Generate visual metadata
        visual_metadata = {
            "rendering_hints": {
                "priority": "high" if node_state == NodeState.EVOLUTIONARY_PEAK else "normal",
                "highlight": node_state in [NodeState.EVOLUTIONARY_PEAK, NodeState.CRITICAL_STATE],
                "interactive": True,
                "tooltip_data": {
                    "performance": f"{performance_score:.3f}",
                    "state": node_state.value,
                    "signal_count": len(design_signals)
                }
            },
            "animation_config": {
                "type": color_mapping.animation_type,
                "frequency": color_mapping.pulse_frequency,
                "duration": 1.0 if color_mapping.animation_type != "none" else 0.0
            },
            "accessibility": {
                "high_contrast_mode": node_state == NodeState.CRITICAL_STATE,
                "screen_reader_text": f"Node {node_id} in {node_state.value} state with {performance_score:.1%} performance"
            }
        }
        
        # Create visual dictionary
        visual_dict = VisualDictionary(
            node_id=node_id,
            node_state=node_state,
            design_signals=design_signals,
            color_mapping=color_mapping,
            performance_score=performance_score,
            visual_metadata=visual_metadata,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        # Store in engine
        self.visual_dictionaries[node_id] = visual_dict
        
        logger.info(f"🎨 Visual dictionary generated for {node_id}: {node_state.value}")
        return visual_dict
    
    def update_visual_dictionary(self, node_id: str, updated_data: Dict[str, Any]) -> VisualDictionary:
        """Update existing visual dictionary with new data"""
        
        if node_id in self.visual_dictionaries:
            # Get current dictionary
            current_dict = self.visual_dictionaries[node_id]
            
            # Merge with updated data
            merged_data = {**updated_data}
            
            # Regenerate visual dictionary
            return self.generate_visual_dictionary(node_id, merged_data)
        else:
            # Create new dictionary
            return self.generate_visual_dictionary(node_id, updated_data)
    
    def get_rendering_data(self, node_id: str, target_platform: str = "web") -> Dict[str, Any]:
        """Get platform-specific rendering data"""
        
        if node_id not in self.visual_dictionaries:
            logger.warning(f"⚠️ No visual dictionary found for node {node_id}")
            return self._get_default_rendering_data(target_platform)
        
        visual_dict = self.visual_dictionaries[node_id]
        
        if target_platform == "unreal":
            return {
                "node_id": node_id,
                "color": visual_dict.color_mapping.to_unreal_color(),
                "state": visual_dict.node_state.value,
                "performance": visual_dict.performance_score,
                "animation": {
                    "type": visual_dict.color_mapping.animation_type,
                    "frequency": visual_dict.color_mapping.pulse_frequency
                },
                "metadata": visual_dict.visual_metadata
            }
        
        elif target_platform == "cytoscape":
            return {
                "data": {"id": node_id},
                "style": {
                    "background-color": visual_dict.color_mapping.primary_color,
                    "background-gradient-stop-colors": visual_dict.color_mapping.secondary_color,
                    "opacity": visual_dict.color_mapping.opacity,
                    "border-width": 3 if visual_dict.node_state == NodeState.EVOLUTIONARY_PEAK else 1,
                    "border-color": "#FFFFFF" if visual_dict.node_state == NodeState.CRITICAL_STATE else "#000000"
                },
                "classes": [visual_dict.node_state.value, visual_dict.color_mapping.animation_type]
            }
        
        else:  # Default web/CSS
            return {
                "node_id": node_id,
                "css_color": visual_dict.color_mapping.to_css(),
                "state_class": visual_dict.node_state.value,
                "animation_class": visual_dict.color_mapping.animation_type,
                "performance_score": visual_dict.performance_score,
                "tooltip": visual_dict.visual_metadata["rendering_hints"]["tooltip_data"],
                "accessibility": visual_dict.visual_metadata["accessibility"]
            }
    
    def export_visual_state(self, output_path: str = None) -> Dict[str, Any]:
        """Export complete visual state for all nodes"""
        
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_nodes": len(self.visual_dictionaries),
            "state_distribution": {},
            "visual_dictionaries": {}
        }
        
        # Calculate state distribution
        for visual_dict in self.visual_dictionaries.values():
            state = visual_dict.node_state.value
            export_data["state_distribution"][state] = export_data["state_distribution"].get(state, 0) + 1
        
        # Export visual dictionaries
        for node_id, visual_dict in self.visual_dictionaries.items():
            export_data["visual_dictionaries"][node_id] = asdict(visual_dict)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"📄 Visual state exported to {output_path}")
        
        return export_data
    
    def _load_config(self) -> Dict[str, Any]:
        """Load SDFA configuration"""
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"✅ SDFA config loaded from {self.config_path}")
                return config
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config: {e}")
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default SDFA configuration"""
        
        return {
            "state_thresholds": {
                "evolutionary_peak": 0.9,
                "high_performance": 0.75,
                "neutral_state": 0.5,
                "low_precision": 0.25,
                "critical_state": 0.1
            },
            "performance_weights": {
                "base": 0.4,
                "quality": 0.3,
                "stability": 0.2,
                "convergence": 0.1
            },
            "color_mappings": {
                "evolutionary_peak": {"primary": "#00FF41", "secondary": "#00CC33", "animation": "pulse"},
                "high_performance": {"primary": "#41FF00", "secondary": "#33CC00", "animation": "glow"},
                "neutral_state": {"primary": "#FFD700", "secondary": "#FFA500", "animation": "none"},
                "low_precision": {"primary": "#FF8C00", "secondary": "#FF4500", "animation": "fade"},
                "critical_state": {"primary": "#FF0000", "secondary": "#CC0000", "animation": "urgent_pulse"},
                "undefined": {"primary": "#808080", "secondary": "#606060", "animation": "none"}
            }
        }
    
    def _get_default_rendering_data(self, target_platform: str) -> Dict[str, Any]:
        """Get default rendering data for unknown nodes"""
        
        if target_platform == "unreal":
            return {
                "color": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},
                "state": "undefined",
                "performance": 0.5,
                "animation": {"type": "none", "frequency": 0.0}
            }
        elif target_platform == "cytoscape":
            return {
                "style": {
                    "background-color": "#808080",
                    "opacity": 0.5,
                    "border-width": 1
                },
                "classes": ["undefined"]
            }
        else:
            return {
                "css_color": "#808080",
                "state_class": "undefined",
                "animation_class": "none",
                "performance_score": 0.5
            }

# Convenience functions for easy integration
def create_sdfa_engine(config_path: str = None) -> SDFAVisualEngine:
    """Create SDFA visual engine instance"""
    return SDFAVisualEngine(config_path) if config_path else SDFAVisualEngine()

def generate_node_visuals(node_id: str, node_data: Dict[str, Any], 
                         sdfa_engine: SDFAVisualEngine = None) -> VisualDictionary:
    """Generate visual dictionary for a node"""
    if sdfa_engine is None:
        sdfa_engine = create_sdfa_engine()
    return sdfa_engine.generate_visual_dictionary(node_id, node_data)

if __name__ == "__main__":
    print("🎨 Scientific Design Formula Assignment (SDFA) Visual Engine v1.0 - Demo")
    
    # Create SDFA engine
    engine = create_sdfa_engine()
    
    # Demo node data with various performance levels
    demo_nodes = [
        {
            "node_id": "V01_PEAK",
            "data": {"score": 0.95, "quality": 0.92, "stability": 0.88, "convergence": 0.94}
        },
        {
            "node_id": "V02_HIGH",
            "data": {"score": 0.82, "quality": 0.78, "stability": 0.85, "convergence": 0.80}
        },
        {
            "node_id": "V03_NEUTRAL",
            "data": {"score": 0.65, "quality": 0.58, "stability": 0.62, "convergence": 0.55}
        },
        {
            "node_id": "V04_LOW",
            "data": {"score": 0.35, "quality": 0.28, "stability": 0.32, "convergence": 0.25}
        },
        {
            "node_id": "V05_CRITICAL",
            "data": {"score": 0.08, "quality": 0.12, "stability": 0.05, "convergence": 0.15}
        }
    ]
    
    print("\n🎯 Generating Visual Dictionaries:")
    for node_info in demo_nodes:
        visual_dict = engine.generate_visual_dictionary(node_info["node_id"], node_info["data"])
        
        print(f"\n📊 {node_info['node_id']}:")
        print(f"  State: {visual_dict.node_state.value}")
        print(f"  Performance: {visual_dict.performance_score:.3f}")
        print(f"  Color: {visual_dict.color_mapping.primary_color}")
        print(f"  Animation: {visual_dict.color_mapping.animation_type}")
        print(f"  Signals: {len(visual_dict.design_signals)}")
    
    print("\n🎨 Platform-Specific Rendering:")
    
    # Web rendering
    web_data = engine.get_rendering_data("V01_PEAK", "web")
    print(f"\n🌐 Web Rendering:")
    print(f"  CSS Color: {web_data['css_color']}")
    print(f"  State Class: {web_data['state_class']}")
    print(f"  Animation: {web_data['animation_class']}")
    
    # Unreal rendering
    unreal_data = engine.get_rendering_data("V01_PEAK", "unreal")
    print(f"\n🎮 Unreal Rendering:")
    print(f"  RGBA: {unreal_data['color']}")
    print(f"  Animation Type: {unreal_data['animation']['type']}")
    
    # Cytoscape rendering
    cytoscape_data = engine.get_rendering_data("V01_PEAK", "cytoscape")
    print(f"\n🕸️ Cytoscape Rendering:")
    print(f"  Background: {cytoscape_data['style']['background-color']}")
    print(f"  Classes: {cytoscape_data['classes']}")
    
    # Export visual state
    export_data = engine.export_visual_state()
    print(f"\n📊 Visual State Summary:")
    print(f"  Total Nodes: {export_data['total_nodes']}")
    print(f"  State Distribution: {export_data['state_distribution']}")
    
    print("\n🎉 SDFA Visual Engine Demo Complete!")
    print("🎨 Color as structured, interpretable system data - not cosmetic UI!") 