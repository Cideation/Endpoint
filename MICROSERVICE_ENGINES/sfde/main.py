#!/usr/bin/env python3
"""
SFDE Engine Microservice - Real Implementation  
Handles SFDE utility formula computations with actual calculations
"""

from flask import Flask, request, jsonify
import json
import logging
import importlib.util
from datetime import datetime
import sys
import os

# Add shared modules to path
sys.path.append('/shared')
sys.path.append('/app/shared')
sys.path.append('..')

# Import SFDE utility functions
try:
    spec = importlib.util.spec_from_file_location("sfde_utils", "../sfde_utility_foundation_extended.py")
    sfde_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sfde_utils)
except Exception as e:
    print(f"Warning: Could not load SFDE utility functions: {e}")
    sfde_utils = None

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "sfde",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "sfde_functions_loaded": sfde_utils is not None
    }), 200

@app.route('/process', methods=['POST'])
def process_sfde():
    """
    Process SFDE utility formula computations with real calculations
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        request_data = data.get('data', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"Processing SFDE request at {timestamp}")
        
        # Extract SFDE requests and affinity types
        sfde_requests = request_data.get('sfde_requests', [])
        affinity_types = request_data.get('affinity_types', ['structural', 'cost', 'energy'])
        
        # Process each affinity type
        results = {
            "sfde_execution_status": "completed",
            "timestamp": datetime.now().isoformat(),
            "formulas_executed": len(sfde_requests),
            "affinity_types": affinity_types,
            "calculations": {}
        }
        
        # Process structural calculations
        if 'structural' in affinity_types:
            structural_results = process_structural_formulas(sfde_requests)
            results["calculations"]["structural"] = structural_results
        
        # Process cost calculations  
        if 'cost' in affinity_types:
            cost_results = process_cost_formulas(sfde_requests)
            results["calculations"]["cost"] = cost_results
        
        # Process energy calculations
        if 'energy' in affinity_types:
            energy_results = process_energy_formulas(sfde_requests)
            results["calculations"]["energy"] = energy_results
        
        # Process MEP calculations
        if 'mep' in affinity_types:
            mep_results = process_mep_formulas(sfde_requests)
            results["calculations"]["mep"] = mep_results
        
        # Process time-based calculations
        if 'time' in affinity_types:
            time_results = process_time_formulas(sfde_requests)
            results["calculations"]["time"] = time_results
        
        # Process fabrication calculations
        if 'fabrication' in affinity_types:
            fab_results = process_fabrication_formulas(sfde_requests)
            results["calculations"]["fabrication"] = fab_results
        
        logger.info(f"SFDE processing completed successfully")
        
        return jsonify({
            "status": "success",
            "results": results,
            "processing_time_ms": (datetime.now() - datetime.fromisoformat(timestamp.replace('Z', '+00:00'))).total_seconds() * 1000,
            "service": "sfde"
        }), 200
        
    except Exception as e:
        logger.error(f"SFDE processing failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "service": "sfde",
            "timestamp": datetime.now().isoformat()
        }), 500

def process_structural_formulas(sfde_requests):
    """Process structural engineering formulas"""
    try:
        results = {}
        
        for request in sfde_requests:
            component_data = request.get('component_data', {})
            
            # NSCP 2015 LRFD calculations
            if sfde_utils and hasattr(sfde_utils, 'lrfd_load_combination'):
                dead_load = component_data.get('dead_load', 100)
                live_load = component_data.get('live_load', 80)
                results['lrfd_load'] = sfde_utils.lrfd_load_combination(dead_load, live_load)
            
            # Beam deflection
            if sfde_utils and hasattr(sfde_utils, 'beam_deflection'):
                force = component_data.get('force', 1000)
                length = component_data.get('length', 10)
                elastic_modulus = component_data.get('elastic_modulus', 200000)
                moment_of_inertia = component_data.get('moment_of_inertia', 500)
                results['beam_deflection'] = sfde_utils.beam_deflection(force, length, elastic_modulus, moment_of_inertia)
            
            # Safety factor
            if sfde_utils and hasattr(sfde_utils, 'safety_factor'):
                ultimate_strength = component_data.get('ultimate_strength', 5000)
                applied_stress = component_data.get('applied_stress', 2000)
                results['safety_factor'] = sfde_utils.safety_factor(ultimate_strength, applied_stress)
            
        # Add summary
        results['total_cost'] = sum([v for v in results.values() if isinstance(v, (int, float))])
        results['unit_cost'] = results['total_cost'] / max(1, len(sfde_requests))
        
        return results
        
    except Exception as e:
        logger.error(f"Structural formula processing failed: {e}")
        return {"error": str(e)}

def process_cost_formulas(sfde_requests):
    """Process cost estimation formulas"""
    try:
        results = {}
        total_cost = 0
        
        for request in sfde_requests:
            component_data = request.get('component_data', {})
            
            # Material cost
            if sfde_utils and hasattr(sfde_utils, 'material_cost'):
                quantity = component_data.get('quantity', 10)
                unit_cost = component_data.get('unit_cost', 25)
                results['material_cost'] = sfde_utils.material_cost(quantity, unit_cost)
                total_cost += results['material_cost']
            
            # Labor cost
            if sfde_utils and hasattr(sfde_utils, 'labor_cost'):
                hours = component_data.get('labor_hours', 40)
                hourly_rate = component_data.get('hourly_rate', 50)
                results['labor_cost'] = sfde_utils.labor_cost(hours, hourly_rate)
                total_cost += results['labor_cost']
            
            # Overhead cost
            if sfde_utils and hasattr(sfde_utils, 'overhead_cost'):
                direct_cost = results.get('material_cost', 0) + results.get('labor_cost', 0)
                overhead_rate = component_data.get('overhead_rate', 0.15)
                results['overhead_cost'] = sfde_utils.overhead_cost(direct_cost, overhead_rate)
                total_cost += results['overhead_cost']
        
        results['total_cost'] = total_cost
        results['unit_cost'] = total_cost / max(1, len(sfde_requests))
        
        return results
        
    except Exception as e:
        logger.error(f"Cost formula processing failed: {e}")
        return {"error": str(e)}

def process_energy_formulas(sfde_requests):
    """Process energy efficiency formulas"""
    try:
        results = {}
        total_energy = 0
        
        for request in sfde_requests:
            component_data = request.get('component_data', {})
            
            # HVAC load
            if sfde_utils and hasattr(sfde_utils, 'hvac_load'):
                area = component_data.get('area', 100)
                cooling_load_per_sqm = component_data.get('cooling_load_per_sqm', 45)
                results['hvac_load'] = sfde_utils.hvac_load(area, cooling_load_per_sqm)
                total_energy += results['hvac_load']
            
            # Lighting load
            if sfde_utils and hasattr(sfde_utils, 'lighting_load'):
                area = component_data.get('area', 100)
                lighting_power_density = component_data.get('lighting_power_density', 12)
                results['lighting_load'] = sfde_utils.lighting_load(area, lighting_power_density)
                total_energy += results['lighting_load']
            
            # Equipment load
            equipment_power = component_data.get('equipment_power', 500)
            results['equipment_load'] = equipment_power
            total_energy += equipment_power
        
        results['total_energy'] = total_energy
        results['efficiency'] = min(1.0, 10000 / max(1, total_energy))  # Efficiency score
        
        return results
        
    except Exception as e:
        logger.error(f"Energy formula processing failed: {e}")
        return {"error": str(e)}

def process_mep_formulas(sfde_requests):
    """Process MEP (Mechanical, Electrical, Plumbing) formulas"""
    try:
        results = {}
        
        for request in sfde_requests:
            component_data = request.get('component_data', {})
            
            # Pipe sizing
            if sfde_utils and hasattr(sfde_utils, 'pipe_sizing'):
                flow_rate = component_data.get('flow_rate', 100)
                velocity = component_data.get('velocity', 2)
                results['pipe_diameter'] = sfde_utils.pipe_sizing(flow_rate, velocity)
            
            # Electrical load
            if sfde_utils and hasattr(sfde_utils, 'electrical_load'):
                power = component_data.get('power', 1000)
                voltage = component_data.get('voltage', 240)
                power_factor = component_data.get('power_factor', 0.9)
                results['electrical_current'] = sfde_utils.electrical_load(power, voltage, power_factor)
        
        return results
        
    except Exception as e:
        logger.error(f"MEP formula processing failed: {e}")
        return {"error": str(e)}

def process_time_formulas(sfde_requests):
    """Process time-based formulas"""
    try:
        results = {}
        
        for request in sfde_requests:
            component_data = request.get('component_data', {})
            
            # Project duration
            if sfde_utils and hasattr(sfde_utils, 'project_duration'):
                total_tasks = component_data.get('total_tasks', 10)
                avg_duration = component_data.get('avg_duration_per_task', 8)
                results['project_duration'] = sfde_utils.project_duration(total_tasks, avg_duration)
            
            # Productivity rate
            if sfde_utils and hasattr(sfde_utils, 'productivity_rate'):
                output_units = component_data.get('output_units', 100)
                total_time = component_data.get('total_time', 40)
                results['productivity_rate'] = sfde_utils.productivity_rate(output_units, total_time)
        
        return results
        
    except Exception as e:
        logger.error(f"Time formula processing failed: {e}")
        return {"error": str(e)}

def process_fabrication_formulas(sfde_requests):
    """Process fabrication formulas including G-code generation"""
    try:
        results = {}
        
        for request in sfde_requests:
            component_data = request.get('component_data', {})
            
            # G-code generation
            if sfde_utils and hasattr(sfde_utils, 'generate_gcode_path'):
                path = component_data.get('path', [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
                tool_radius = component_data.get('tool_radius', 1.0)
                cut_depth = component_data.get('cut_depth', -1.0)
                
                gcode = sfde_utils.generate_gcode_path(path, tool_radius, cut_depth)
                results['gcode'] = gcode
                results['path_length'] = len(path)
        
        return results
        
    except Exception as e:
        logger.error(f"Fabrication formula processing failed: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    logger.info("Starting SFDE Engine microservice on port 5003")
    app.run(host='0.0.0.0', port=5003, debug=False) 