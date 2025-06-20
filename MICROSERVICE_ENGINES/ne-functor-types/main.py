#!/usr/bin/env python3
"""
Functor Types Microservice - Real Implementation
Handles functor type analysis and recommendations with actual computations
"""

from flask import Flask, request, jsonify
import json
import logging
from datetime import datetime
import sys
import os

# Add shared modules to path
sys.path.append('/shared')
sys.path.append('/app/shared')

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load functor type data
functor_types_data = None

def load_functor_types():
    """Load functor types from configuration files"""
    global functor_types_data
    try:
        # Try to load functor types configuration
        with open('../allowed_functor_types_verbose_by_phase.json', 'r') as f:
            functor_types_data = json.load(f)
        logger.info(f"Loaded functor types data: {len(functor_types_data)} phases")
    except FileNotFoundError:
        # Fallback data if file not found
        functor_types_data = {
            "phase_1": ["structural", "cost", "energy"],
            "phase_2": ["structural", "cost", "energy", "mep", "time"],
            "phase_3": ["structural", "cost", "energy", "mep", "time", "fabrication"]
        }
        logger.warning("Using fallback functor types data")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if functor_types_data is None:
        load_functor_types()
    
    return jsonify({
        "status": "healthy",
        "service": "ne-functor-types",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "functor_types_loaded": functor_types_data is not None,
        "available_phases": list(functor_types_data.keys()) if functor_types_data else []
    }), 200

@app.route('/process', methods=['POST'])
def process_functor_types():
    """
    Process functor type analysis with real computations
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        request_data = data.get('data', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"Processing Functor Types request at {timestamp}")
        
        # Load functor types if not already loaded
        if functor_types_data is None:
            load_functor_types()
        
        # Extract analysis requests
        components = request_data.get('components', [])
        phase = request_data.get('phase', 'phase_2')
        analysis_type = request_data.get('analysis_type', 'compatibility')
        
        # Process functor type analysis
        results = {
            "functor_analysis_status": "completed",
            "timestamp": datetime.now().isoformat(),
            "components_analyzed": len(components),
            "phase": phase,
            "analysis_type": analysis_type
        }
        
        # Perform type compatibility analysis
        compatibility_results = analyze_type_compatibility(components, phase)
        results["compatibility_analysis"] = compatibility_results
        
        # Generate type recommendations
        type_recommendations = generate_type_recommendations(components, phase)
        results["type_recommendations"] = type_recommendations
        
        # Analyze functor affinity
        affinity_analysis = analyze_functor_affinity(components, phase)
        results["affinity_analysis"] = affinity_analysis
        
        # Calculate type scores
        type_scores = calculate_type_scores(components, phase)
        results["type_scores"] = type_scores
        
        logger.info(f"Functor Types processing completed successfully")
        
        return jsonify({
            "status": "success",
            "results": results,
            "processing_time_ms": (datetime.now() - datetime.fromisoformat(timestamp.replace('Z', '+00:00'))).total_seconds() * 1000,
            "service": "ne-functor-types"
        }), 200
        
    except Exception as e:
        logger.error(f"Functor Types processing failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "service": "ne-functor-types",
            "timestamp": datetime.now().isoformat()
        }), 500

def analyze_type_compatibility(components, phase):
    """Analyze compatibility between component types and phase requirements"""
    try:
        global functor_types_data
        
        # Get allowed types for phase
        allowed_types = functor_types_data.get(phase, functor_types_data.get("phase_2", []))
        
        compatibility_results = {
            "allowed_types": allowed_types,
            "component_analysis": [],
            "overall_compatibility": 0.0
        }
        
        total_score = 0.0
        
        for component in components:
            component_id = component.get('id', 'unknown')
            component_type = component.get('type', 'unknown')
            primary_functor = component.get('primary_functor', 'structural')
            
            # Check if component type is compatible
            is_compatible = primary_functor in allowed_types
            compatibility_score = 1.0 if is_compatible else 0.5
            
            # Bonus for having multiple compatible attributes
            if 'secondary_functor' in component:
                secondary_functor = component['secondary_functor']
                if secondary_functor in allowed_types:
                    compatibility_score = min(1.0, compatibility_score + 0.2)
            
            component_analysis = {
                "component_id": component_id,
                "component_type": component_type,
                "primary_functor": primary_functor,
                "is_compatible": is_compatible,
                "compatibility_score": compatibility_score,
                "recommendations": get_compatibility_recommendations(primary_functor, allowed_types)
            }
            
            compatibility_results["component_analysis"].append(component_analysis)
            total_score += compatibility_score
        
        # Calculate overall compatibility
        if components:
            compatibility_results["overall_compatibility"] = total_score / len(components)
        
        return compatibility_results
        
    except Exception as e:
        logger.error(f"Type compatibility analysis failed: {e}")
        return {"error": str(e)}

def generate_type_recommendations(components, phase):
    """Generate functor type recommendations based on components and phase"""
    try:
        global functor_types_data
        
        # Get allowed types for phase
        allowed_types = functor_types_data.get(phase, functor_types_data.get("phase_2", []))
        
        # Analyze component types
        component_types = [comp.get('primary_functor', 'structural') for comp in components]
        unique_types = list(set(component_types))
        
        # Generate recommendations
        recommendations = {
            "current_types": unique_types,
            "allowed_types": allowed_types,
            "recommended_additions": [],
            "type_optimization": []
        }
        
        # Find missing recommended types
        for allowed_type in allowed_types:
            if allowed_type not in unique_types:
                recommendations["recommended_additions"].append({
                    "type": allowed_type,
                    "reason": f"Phase {phase} benefits from {allowed_type} analysis",
                    "priority": get_type_priority(allowed_type, phase)
                })
        
        # Suggest optimizations for existing types
        for component_type in unique_types:
            if component_type in allowed_types:
                optimization = generate_type_optimization(component_type, components, phase)
                if optimization:
                    recommendations["type_optimization"].append(optimization)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Type recommendation generation failed: {e}")
        return {"error": str(e)}

def analyze_functor_affinity(components, phase):
    """Analyze functor affinity relationships"""
    try:
        # Define affinity relationships
        affinity_matrix = {
            "structural": {"cost": 0.8, "energy": 0.6, "mep": 0.7, "time": 0.5, "fabrication": 0.9},
            "cost": {"structural": 0.8, "energy": 0.7, "mep": 0.6, "time": 0.9, "fabrication": 0.7},
            "energy": {"structural": 0.6, "cost": 0.7, "mep": 0.9, "time": 0.5, "fabrication": 0.4},
            "mep": {"structural": 0.7, "cost": 0.6, "energy": 0.9, "time": 0.6, "fabrication": 0.8},
            "time": {"structural": 0.5, "cost": 0.9, "energy": 0.5, "mep": 0.6, "fabrication": 0.8},
            "fabrication": {"structural": 0.9, "cost": 0.7, "energy": 0.4, "mep": 0.8, "time": 0.8}
        }
        
        component_types = [comp.get('primary_functor', 'structural') for comp in components]
        unique_types = list(set(component_types))
        
        affinity_results = {
            "type_pairs": [],
            "average_affinity": 0.0,
            "affinity_matrix": {}
        }
        
        total_affinity = 0.0
        pair_count = 0
        
        # Calculate pairwise affinities
        for i, type1 in enumerate(unique_types):
            affinity_results["affinity_matrix"][type1] = {}
            for j, type2 in enumerate(unique_types):
                if i != j:
                    affinity = affinity_matrix.get(type1, {}).get(type2, 0.5)
                    affinity_results["affinity_matrix"][type1][type2] = affinity
                    
                    if i < j:  # Avoid duplicates
                        affinity_results["type_pairs"].append({
                            "type1": type1,
                            "type2": type2,
                            "affinity": affinity,
                            "synergy": "high" if affinity > 0.7 else "medium" if affinity > 0.5 else "low"
                        })
                        total_affinity += affinity
                        pair_count += 1
        
        # Calculate average affinity
        if pair_count > 0:
            affinity_results["average_affinity"] = total_affinity / pair_count
        
        return affinity_results
        
    except Exception as e:
        logger.error(f"Functor affinity analysis failed: {e}")
        return {"error": str(e)}

def calculate_type_scores(components, phase):
    """Calculate scores for each functor type"""
    try:
        global functor_types_data
        
        # Get allowed types for phase
        allowed_types = functor_types_data.get(phase, functor_types_data.get("phase_2", []))
        
        type_scores = {}
        
        for functor_type in allowed_types:
            # Count occurrences in components
            occurrences = sum(1 for comp in components 
                            if comp.get('primary_functor') == functor_type or 
                               comp.get('secondary_functor') == functor_type)
            
            # Calculate base score
            coverage_score = min(1.0, occurrences / max(1, len(components)))
            
            # Phase-specific bonuses
            phase_bonus = get_phase_bonus(functor_type, phase)
            
            # Calculate final score
            final_score = min(1.0, coverage_score + phase_bonus)
            
            type_scores[functor_type] = {
                "occurrences": occurrences,
                "coverage_score": coverage_score,
                "phase_bonus": phase_bonus,
                "final_score": final_score,
                "recommendation": "increase" if final_score < 0.5 else "maintain" if final_score < 0.8 else "optimize"
            }
        
        return type_scores
        
    except Exception as e:
        logger.error(f"Type score calculation failed: {e}")
        return {"error": str(e)}

def get_compatibility_recommendations(functor_type, allowed_types):
    """Get recommendations for improving type compatibility"""
    if functor_type in allowed_types:
        return f"{functor_type} is compatible with current phase"
    else:
        closest_type = min(allowed_types, key=lambda x: abs(hash(x) - hash(functor_type)) % 100)
        return f"Consider changing to {closest_type} for better phase compatibility"

def get_type_priority(functor_type, phase):
    """Get priority level for a functor type in a specific phase"""
    priority_map = {
        "phase_1": {"structural": "high", "cost": "high", "energy": "medium"},
        "phase_2": {"structural": "high", "cost": "high", "energy": "high", "mep": "medium", "time": "medium"},
        "phase_3": {"structural": "high", "cost": "high", "energy": "high", "mep": "high", "time": "high", "fabrication": "high"}
    }
    
    return priority_map.get(phase, {}).get(functor_type, "low")

def generate_type_optimization(component_type, components, phase):
    """Generate optimization suggestions for a component type"""
    type_count = sum(1 for comp in components if comp.get('primary_functor') == component_type)
    
    if type_count == 1:
        return {
            "type": component_type,
            "suggestion": f"Consider adding more {component_type} components for better analysis",
            "current_count": type_count,
            "recommended_count": "2-3"
        }
    elif type_count > 5:
        return {
            "type": component_type,
            "suggestion": f"High number of {component_type} components - consider consolidation",
            "current_count": type_count,
            "recommended_count": "3-5"
        }
    
    return None

def get_phase_bonus(functor_type, phase):
    """Get phase-specific bonus for functor type"""
    bonuses = {
        "phase_1": {"structural": 0.2, "cost": 0.2},
        "phase_2": {"structural": 0.15, "cost": 0.15, "energy": 0.15, "mep": 0.1},
        "phase_3": {"fabrication": 0.2, "time": 0.15, "mep": 0.15}
    }
    
    return bonuses.get(phase, {}).get(functor_type, 0.0)

if __name__ == '__main__':
    logger.info("Starting Functor Types microservice on port 5001")
    app.run(host='0.0.0.0', port=5001, debug=False) 