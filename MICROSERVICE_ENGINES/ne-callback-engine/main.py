#!/usr/bin/env python3
"""
Callback Engine Microservice - Real Implementation
Handles callback processing and event management with actual computations
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

# Load callback registry
callback_registry = None

def load_callback_registry():
    """Load callback registry from configuration files"""
    global callback_registry
    try:
        # Try to load callback registry
        with open('../callback_registry.json', 'r') as f:
            callback_registry = json.load(f)
        logger.info(f"Loaded callback registry: {len(callback_registry)} callbacks")
    except FileNotFoundError:
        # Fallback data if file not found
        callback_registry = [
            {"node_id": "callback_001", "type": "phase_transition", "priority": "high"},
            {"node_id": "callback_002", "type": "validation", "priority": "medium"},
            {"node_id": "callback_003", "type": "error_handling", "priority": "high"},
            {"node_id": "callback_004", "type": "completion", "priority": "low"},
            {"node_id": "callback_005", "type": "progress_update", "priority": "medium"}
        ]
        logger.warning("Using fallback callback registry data")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if callback_registry is None:
        load_callback_registry()
    
    return jsonify({
        "status": "healthy",
        "service": "ne-callback-engine",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "callback_registry_loaded": callback_registry is not None,
        "registered_callbacks": len(callback_registry) if callback_registry else 0
    }), 200

@app.route('/process', methods=['POST'])
def process_callbacks():
    """
    Process callback operations with real computations
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        request_data = data.get('data', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"Processing Callbacks request at {timestamp}")
        
        # Load callback registry if not already loaded
        if callback_registry is None:
            load_callback_registry()
        
        # Extract callback requests
        phase = request_data.get('phase', 'cross_phase')
        callback_types = request_data.get('callback_types', ['validation', 'progress_update'])
        components = request_data.get('components', [])
        
        # Process callback operations
        results = {
            "callback_execution_status": "completed",
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "callback_types": callback_types,
            "components_processed": len(components)
        }
        
        # Execute callbacks
        callback_results = execute_callbacks(callback_types, components, phase)
        results["callback_results"] = callback_results
        
        # Validate callback execution
        validation_results = validate_callback_execution(callback_results)
        results["validation_results"] = validation_results
        
        # Process event handling
        event_results = process_events(callback_results, phase)
        results["event_results"] = event_results
        
        # Generate callback metrics
        metrics = generate_callback_metrics(callback_results)
        results["metrics"] = metrics
        
        logger.info(f"Callback processing completed successfully")
        
        return jsonify({
            "status": "success",
            "results": results,
            "processing_time_ms": (datetime.now() - datetime.fromisoformat(timestamp.replace('Z', '+00:00'))).total_seconds() * 1000,
            "service": "ne-callback-engine"
        }), 200
        
    except Exception as e:
        logger.error(f"Callback processing failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "service": "ne-callback-engine",
            "timestamp": datetime.now().isoformat()
        }), 500

def execute_callbacks(callback_types, components, phase):
    """Execute specified callback types"""
    try:
        global callback_registry
        
        callback_results = []
        
        for callback_type in callback_types:
            # Find callbacks of this type
            matching_callbacks = [cb for cb in callback_registry 
                                if cb.get('type') == callback_type]
            
            if not matching_callbacks:
                continue
            
            # Execute each matching callback
            for callback in matching_callbacks:
                callback_id = callback.get('node_id', 'unknown')
                priority = callback.get('priority', 'medium')
                
                # Execute callback based on type
                execution_result = execute_single_callback(callback_type, callback_id, components, phase)
                
                callback_result = {
                    "callback_id": callback_id,
                    "callback_type": callback_type,
                    "priority": priority,
                    "phase": phase,
                    "execution_result": execution_result,
                    "execution_time": datetime.now().isoformat(),
                    "status": execution_result.get("status", "completed")
                }
                
                callback_results.append(callback_result)
        
        return callback_results
        
    except Exception as e:
        logger.error(f"Callback execution failed: {e}")
        return [{"error": str(e)}]

def execute_single_callback(callback_type, callback_id, components, phase):
    """Execute a single callback based on its type"""
    try:
        if callback_type == "validation":
            return execute_validation_callback(callback_id, components, phase)
        elif callback_type == "progress_update":
            return execute_progress_callback(callback_id, components, phase)
        elif callback_type == "phase_transition":
            return execute_phase_transition_callback(callback_id, components, phase)
        elif callback_type == "error_handling":
            return execute_error_handling_callback(callback_id, components, phase)
        elif callback_type == "completion":
            return execute_completion_callback(callback_id, components, phase)
        else:
            return {"status": "unknown_type", "message": f"Unknown callback type: {callback_type}"}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

def execute_validation_callback(callback_id, components, phase):
    """Execute validation callback"""
    valid_components = 0
    total_components = len(components)
    validation_errors = []
    
    for component in components:
        # Validate component structure
        if 'id' in component and 'type' in component:
            valid_components += 1
        else:
            validation_errors.append(f"Component missing required fields: {component}")
    
    validation_score = valid_components / max(1, total_components)
    
    return {
        "status": "completed",
        "validation_score": validation_score,
        "valid_components": valid_components,
        "total_components": total_components,
        "validation_errors": validation_errors,
        "passed": validation_score >= 0.8
    }

def execute_progress_callback(callback_id, components, phase):
    """Execute progress update callback"""
    # Calculate progress based on component processing
    processed_components = len([comp for comp in components 
                              if comp.get('status') in ['processed', 'completed']])
    total_components = len(components)
    progress_percentage = (processed_components / max(1, total_components)) * 100
    
    return {
        "status": "completed",
        "progress_percentage": progress_percentage,
        "processed_components": processed_components,
        "total_components": total_components,
        "phase": phase,
        "estimated_completion": calculate_estimated_completion(progress_percentage)
    }

def execute_phase_transition_callback(callback_id, components, phase):
    """Execute phase transition callback"""
    # Check if phase transition is ready
    ready_for_transition = all(comp.get('status') in ['completed', 'validated'] 
                              for comp in components)
    
    next_phase = get_next_phase(phase)
    
    return {
        "status": "completed",
        "ready_for_transition": ready_for_transition,
        "current_phase": phase,
        "next_phase": next_phase,
        "transition_requirements": check_transition_requirements(components, phase),
        "transition_score": calculate_transition_score(components)
    }

def execute_error_handling_callback(callback_id, components, phase):
    """Execute error handling callback"""
    error_components = [comp for comp in components 
                       if comp.get('status') in ['error', 'failed']]
    
    error_analysis = analyze_errors(error_components)
    recovery_plan = generate_recovery_plan(error_components, phase)
    
    return {
        "status": "completed",
        "error_count": len(error_components),
        "total_components": len(components),
        "error_rate": len(error_components) / max(1, len(components)),
        "error_analysis": error_analysis,
        "recovery_plan": recovery_plan,
        "critical_errors": [comp for comp in error_components 
                           if comp.get('priority') == 'high']
    }

def execute_completion_callback(callback_id, components, phase):
    """Execute completion callback"""
    completed_components = len([comp for comp in components 
                               if comp.get('status') == 'completed'])
    
    completion_rate = completed_components / max(1, len(components))
    
    return {
        "status": "completed",
        "completion_rate": completion_rate,
        "completed_components": completed_components,
        "total_components": len(components),
        "phase": phase,
        "phase_complete": completion_rate >= 0.95,
        "summary": generate_completion_summary(components, phase)
    }

def validate_callback_execution(callback_results):
    """Validate that callbacks executed correctly"""
    try:
        validation_results = {
            "total_callbacks": len(callback_results),
            "successful_callbacks": 0,
            "failed_callbacks": 0,
            "validation_details": []
        }
        
        for result in callback_results:
            status = result.get("status", "unknown")
            callback_id = result.get("callback_id", "unknown")
            
            if status == "completed":
                validation_results["successful_callbacks"] += 1
                validation_results["validation_details"].append({
                    "callback_id": callback_id,
                    "status": "valid",
                    "message": "Callback executed successfully"
                })
            else:
                validation_results["failed_callbacks"] += 1
                validation_results["validation_details"].append({
                    "callback_id": callback_id,
                    "status": "invalid",
                    "message": f"Callback failed with status: {status}"
                })
        
        validation_results["overall_success_rate"] = (
            validation_results["successful_callbacks"] / 
            max(1, validation_results["total_callbacks"])
        )
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Callback validation failed: {e}")
        return {"error": str(e)}

def process_events(callback_results, phase):
    """Process events generated by callbacks"""
    try:
        events = []
        
        for result in callback_results:
            callback_type = result.get("callback_type")
            execution_result = result.get("execution_result", {})
            
            # Generate events based on callback results
            if callback_type == "validation" and not execution_result.get("passed", True):
                events.append({
                    "event_type": "validation_failed",
                    "priority": "high",
                    "message": "Validation callback reported failures",
                    "data": execution_result
                })
            
            elif callback_type == "progress_update":
                progress = execution_result.get("progress_percentage", 0)
                if progress >= 100:
                    events.append({
                        "event_type": "progress_complete",
                        "priority": "medium",
                        "message": "Progress callback reports completion",
                        "data": execution_result
                    })
            
            elif callback_type == "error_handling":
                error_count = execution_result.get("error_count", 0)
                if error_count > 0:
                    events.append({
                        "event_type": "errors_detected",
                        "priority": "high",
                        "message": f"Error handling callback found {error_count} errors",
                        "data": execution_result
                    })
        
        return {
            "events_generated": len(events),
            "events": events,
            "high_priority_events": len([e for e in events if e.get("priority") == "high"]),
            "event_summary": summarize_events(events)
        }
        
    except Exception as e:
        logger.error(f"Event processing failed: {e}")
        return {"error": str(e)}

def generate_callback_metrics(callback_results):
    """Generate metrics from callback execution"""
    try:
        metrics = {
            "total_callbacks": len(callback_results),
            "execution_time_ms": 0,
            "callback_types": {},
            "success_rate": 0,
            "average_execution_time": 0
        }
        
        successful_callbacks = 0
        total_execution_time = 0
        
        for result in callback_results:
            callback_type = result.get("callback_type", "unknown")
            status = result.get("status", "unknown")
            
            # Count callback types
            metrics["callback_types"][callback_type] = metrics["callback_types"].get(callback_type, 0) + 1
            
            # Count successful callbacks
            if status == "completed":
                successful_callbacks += 1
        
        # Calculate success rate
        metrics["success_rate"] = successful_callbacks / max(1, len(callback_results))
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        return {"error": str(e)}

# Helper functions
def calculate_estimated_completion(progress_percentage):
    """Calculate estimated completion time"""
    if progress_percentage >= 100:
        return "Completed"
    elif progress_percentage >= 75:
        return "Soon (< 10 minutes)"
    elif progress_percentage >= 50:
        return "Medium (10-30 minutes)"
    else:
        return "Long (> 30 minutes)"

def get_next_phase(current_phase):
    """Get the next phase in sequence"""
    phase_sequence = ["phase_1", "phase_2", "phase_3", "cross_phase"]
    try:
        current_index = phase_sequence.index(current_phase)
        if current_index < len(phase_sequence) - 1:
            return phase_sequence[current_index + 1]
        else:
            return "completed"
    except ValueError:
        return "unknown"

def check_transition_requirements(components, phase):
    """Check requirements for phase transition"""
    requirements = {
        "all_components_processed": all(comp.get('status') in ['completed', 'validated'] for comp in components),
        "no_critical_errors": not any(comp.get('status') == 'error' and comp.get('priority') == 'high' for comp in components),
        "minimum_quality_score": calculate_quality_score(components) >= 0.8
    }
    return requirements

def calculate_transition_score(components):
    """Calculate score for phase transition readiness"""
    if not components:
        return 0.0
    
    completed_score = len([comp for comp in components if comp.get('status') == 'completed']) / len(components)
    quality_score = calculate_quality_score(components)
    
    return (completed_score * 0.6 + quality_score * 0.4)

def calculate_quality_score(components):
    """Calculate overall quality score of components"""
    if not components:
        return 0.0
    
    quality_scores = []
    for comp in components:
        # Base quality score
        if comp.get('status') == 'completed':
            quality_scores.append(0.9)
        elif comp.get('status') == 'validated':
            quality_scores.append(0.8)
        elif comp.get('status') == 'processed':
            quality_scores.append(0.7)
        else:
            quality_scores.append(0.3)
    
    return sum(quality_scores) / len(quality_scores)

def analyze_errors(error_components):
    """Analyze error patterns in components"""
    error_types = {}
    for comp in error_components:
        error_type = comp.get('error_type', 'unknown')
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    return {
        "error_types": error_types,
        "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
        "error_distribution": error_types
    }

def generate_recovery_plan(error_components, phase):
    """Generate recovery plan for error components"""
    recovery_actions = []
    
    for comp in error_components:
        error_type = comp.get('error_type', 'unknown')
        if error_type == 'validation_failed':
            recovery_actions.append(f"Re-validate component {comp.get('id', 'unknown')}")
        elif error_type == 'processing_failed':
            recovery_actions.append(f"Retry processing for component {comp.get('id', 'unknown')}")
        else:
            recovery_actions.append(f"Manual review needed for component {comp.get('id', 'unknown')}")
    
    return {
        "recovery_actions": recovery_actions,
        "estimated_recovery_time": len(recovery_actions) * 5,  # 5 minutes per action
        "priority": "high" if len(error_components) > len(recovery_actions) * 0.5 else "medium"
    }

def generate_completion_summary(components, phase):
    """Generate completion summary"""
    status_counts = {}
    for comp in components:
        status = comp.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        "phase": phase,
        "total_components": len(components),
        "status_distribution": status_counts,
        "overall_health": "good" if status_counts.get('completed', 0) >= len(components) * 0.8 else "needs_attention"
    }

def summarize_events(events):
    """Summarize generated events"""
    if not events:
        return "No events generated"
    
    event_types = [event.get('event_type') for event in events]
    high_priority_count = len([e for e in events if e.get('priority') == 'high'])
    
    return f"{len(events)} events generated, {high_priority_count} high priority"

if __name__ == '__main__':
    logger.info("Starting Callback Engine microservice on port 5002")
    app.run(host='0.0.0.0', port=5002, debug=False) 