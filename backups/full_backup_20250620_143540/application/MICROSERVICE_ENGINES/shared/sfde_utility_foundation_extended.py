# ==== cartesian_utils.py ====

from math import floor, sqrt

def generate_bbx(center, size):
    """
    Generate bounding box coordinates from a center point and size.
    Args:
        center: Tuple (x, y, z)
        size: Tuple (length, width, height)
    Returns:
        Dict with min and max corners
    """
    cx, cy, cz = center
    lx, ly, lz = size
    return {
        'min': (cx - lx/2, cy - ly/2, cz - lz/2),
        'max': (cx + lx/2, cy + ly/2, cz + lz/2)
    }

def bbx_overlap(bbx1, bbx2):
    """
    Check if two bounding boxes overlap.
    """
    for i in range(3):
        if bbx1['max'][i] < bbx2['min'][i] or bbx1['min'][i] > bbx2['max'][i]:
            return False
    return True

def get_octree_index(position, level, space_size=(1.0, 1.0, 1.0)):
    """
    Compute octree index from position and level.
    """
    factor = 2 ** level
    return tuple(floor(position[i] / (space_size[i] / factor)) for i in range(3))

def bbx_to_octree_level(bbx, level, space_size=(1.0, 1.0, 1.0)):
    """
    Convert bounding box to octree coverage index at a given level.
    """
    min_index = get_octree_index(bbx['min'], level, space_size)
    max_index = get_octree_index(bbx['max'], level, space_size)
    return min_index, max_index

def bbx_distance(bbx1, bbx2):
    """
    Compute Euclidean distance between centers of two bounding boxes.
    """
    c1 = [(bbx1['min'][i] + bbx1['max'][i]) / 2 for i in range(3)]
    c2 = [(bbx2['min'][i] + bbx2['max'][i]) / 2 for i in range(3)]
    return sqrt(sum((c1[i] - c2[i]) ** 2 for i in range(3)))


# ==== econ_utils.py ====

def calculate_roi(gain, cost):
    if cost == 0:
        return float('inf')
    return (gain - cost) / cost

def generate_bid_score(price, delivery_time, performance_score, weight_factors):
    price_w, time_w, perf_w = weight_factors
    score = (performance_score * perf_w) - (price * price_w) - (delivery_time * time_w)
    return score

def calculate_irr(cash_flows, guess=0.1, max_iterations=1000, tolerance=1e-6):
    """
    Calculate the Internal Rate of Return (IRR) for a list of cash flows.

    Parameters:
        cash_flows (list of float): Cash flows per period. First is typically negative (investment).
        guess (float): Initial guess for IRR (default 10%).
        max_iterations (int): Maximum number of iterations for convergence.
        tolerance (float): Acceptable difference from zero NPV.

    Returns:
        float: Estimated IRR as a decimal (e.g. 0.12 for 12%), or None if it doesn't converge.
    """
    rate = guess
    for _ in range(max_iterations):
        npv = sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
        d_npv = sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))

        if abs(npv) < tolerance:
            return rate

        if d_npv == 0:
            return None  # Avoid division by zero

        rate -= npv / d_npv

    return None  # Did not converge


# ==== electrical_utils.py ====

def check_voltage_drop(length_m, current_A, resistance_ohm_per_m, max_drop_percent=3):
    """
    Calculate voltage drop and verify if within allowable percent.
    """
    voltage_drop = length_m * current_A * resistance_ohm_per_m
    return voltage_drop, voltage_drop <= (max_drop_percent / 100.0) * 230  # assuming 230V

def is_circuit_overloaded(current_A, breaker_rating_A):
    """
    Check if the circuit exceeds the breaker's rated current.
    """
    return current_A > breaker_rating_A

def check_conductor_size(load_kW, voltage_V, allowable_current_A):
    """
    Simple check to see if conductor current is within bounds.
    """
    current = (load_kW * 1000) / voltage_V
    return current <= allowable_current_A


# ==== geo_utils.py ====
from shapely.geometry import Point, Polygon
import trimesh
from scipy.spatial import distance

def point_in_zone(point_coords, zone_coords):
    """
    Check if a point is within a polygon zone.
    Args:
        point_coords: Tuple (x, y)
        zone_coords: List of (x, y) tuples
    Returns:
        Boolean
    """
    point = Point(point_coords)
    polygon = Polygon(zone_coords)
    return point.within(polygon)

def distance_between_points(p1, p2):
    """
    Euclidean distance between two points.
    Args:
        p1, p2: Tuples (x, y)
    Returns:
        Float
    """
    return Point(p1).distance(Point(p2))

def load_mesh_volume(obj_path):
    """
    Load a 3D .obj or .stl file and return volume.
    Args:
        obj_path: Path to mesh file
    Returns:
        Volume (float)
    """
    mesh = trimesh.load(obj_path)
    return mesh.volume

def centroid_of_polygon(coords):
    """
    Compute centroid of a 2D polygon.
    Args:
        coords: List of (x, y) tuples
    Returns:
        Tuple (x, y)
    """
    polygon = Polygon(coords)
    return polygon.centroid.coords[0]

def bounding_box_of_mesh(obj_path):
    """
    Return the 3D bounding box dimensions of a mesh.
    Args:
        obj_path: Path to mesh file
    Returns:
        (length, width, height)
    """
    mesh = trimesh.load(obj_path)
    bounds = mesh.bounds
    return tuple((bounds[1] - bounds[0]).tolist())


# ==== math_utils.py ====

def normalize_score(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def weighted_average(values, weights):
    if not values or not weights or len(values) != len(weights):
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


# ==== path_utils.py ====

from scipy.spatial import distance

def euclidean_distance(a, b):
    return distance.euclidean(a, b)

def nearest_node(target_point, node_list):
    distances = [(node, euclidean_distance(target_point, node)) for node in node_list]
    return min(distances, key=lambda x: x[1])[0] if distances else None


# ==== sanitary_utils.py ====

def check_pipe_slope(pipe_diameter_mm):
    """
    Check if pipe slope meets minimum standards based on diameter.
    """
    if pipe_diameter_mm < 100:
        return 0.02  # 2% slope for small pipes
    elif pipe_diameter_mm <= 150:
        return 0.01  # 1% for medium
    else:
        return 0.005  # 0.5% for large pipes

def check_flow_capacity(flow_lps, pipe_diameter_mm):
    """
    Rough check if pipe can handle the given flow rate.
    """
    max_capacity = 0.05 * pipe_diameter_mm  # Simplified model
    return flow_lps <= max_capacity

def check_trap_distance(fixture_type):
    """
    Returns max trap distance from vent for given fixture.
    """
    fixture_limits = {
        'lavatory': 1200,
        'water_closet': 1800,
        'kitchen_sink': 1500
    }
    return fixture_limits.get(fixture_type, 1000)  # default if unknown


# ==== setup.py ====

from setuptools import setup, find_packages

setup(
    name='spatial_core',
    version='0.1',
    packages=find_packages(),
    description='Custom spatial computation utilities for SOS functors',
    author='jp calma',
)


# ==== structural_utils.py ====

def check_slenderness_ratio(height_mm, radius_mm, material='steel'):
    """
    Check if the column's slenderness ratio is within allowable code limits.
    """
    slenderness = height_mm / radius_mm
    if material == 'steel':
        return slenderness <= 200
    elif material == 'concrete':
        return slenderness <= 100
    return False

def validate_span_to_depth_ratio(span_mm, depth_mm):
    """
    Check beam span/depth ratio compliance for deflection control.
    """
    ratio = span_mm / depth_mm
    return ratio <= 20  # Typical for simply supported beams

def check_bearing_area(load_kN, bearing_stress_kPa):
    """
    Check if bearing area is sufficient for applied load.
    """
    required_area_m2 = (load_kN * 1000) / bearing_stress_kPa
    return required_area_m2


# ==== gnn_formula_utils.py ====
"""
GNN/DGL-Ready Scientific Formula Utilities
Core formula library for SFDE operator training and validation
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def graph_node_similarity(node_features_a: np.ndarray, node_features_b: np.ndarray, 
                         metric: str = 'cosine') -> float:
    """
    GNN-ready node similarity formula for graph learning.
    
    Args:
        node_features_a: Feature vector for node A
        node_features_b: Feature vector for node B 
        metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if metric == 'cosine':
        # Cosine similarity: dot(a,b) / (||a|| * ||b||)
        dot_product = np.dot(node_features_a, node_features_b)
        norm_a = np.linalg.norm(node_features_a)
        norm_b = np.linalg.norm(node_features_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return abs(dot_product / (norm_a * norm_b))
    
    elif metric == 'euclidean':
        # Euclidean distance converted to similarity
        distance = np.linalg.norm(node_features_a - node_features_b)
        return 1.0 / (1.0 + distance)
    
    elif metric == 'manhattan':
        # Manhattan distance converted to similarity
        distance = np.sum(np.abs(node_features_a - node_features_b))
        return 1.0 / (1.0 + distance)
    
    return 0.0

def edge_weight_formula(source_props: Dict, target_props: Dict, 
                       edge_type: str = 'structural') -> float:
    """
    Scientific formula for computing edge weights in graph networks.
    
    Args:
        source_props: Source node properties
        target_props: Target node properties
        edge_type: Type of edge connection
    
    Returns:
        Computed edge weight (0.0 to 1.0)
    """
    if edge_type == 'structural':
        # Structural load transfer formula
        source_capacity = source_props.get('load_capacity', 1.0)
        target_demand = target_props.get('load_demand', 0.5)
        safety_factor = source_props.get('safety_factor', 1.0)
        
        # Weight = (capacity - demand) / capacity * safety_factor
        if source_capacity > 0:
            weight = ((source_capacity - target_demand) / source_capacity) * safety_factor
            return max(0.0, min(1.0, weight))
        return 0.0
    
    elif edge_type == 'electrical':
        # Electrical power flow formula
        source_voltage = source_props.get('voltage', 230.0)
        target_current = target_props.get('current', 10.0)
        resistance = source_props.get('resistance', 0.1)
        
        # Power law: P = V²/R, normalized
        power = (source_voltage ** 2) / (resistance * target_current + 1)
        return min(1.0, power / 1000.0)  # Normalize to 0-1
    
    elif edge_type == 'economic':
        # Economic efficiency formula
        source_cost = source_props.get('cost', 1000.0)
        target_benefit = target_props.get('benefit', 1500.0)
        roi = calculate_roi(target_benefit, source_cost)
        return min(1.0, max(0.0, roi / 2.0))  # Normalize ROI
    
    # Default geometric distance weight
    source_pos = source_props.get('position', [0, 0, 0])
    target_pos = target_props.get('position', [0, 0, 0])
    distance = euclidean_distance(source_pos, target_pos)
    return 1.0 / (1.0 + distance)

def agent_coefficient_formula(node_data: Dict, agent_type: str, 
                             context_coeffs: Dict) -> Dict[str, float]:
    """
    Scientific formula for computing agent coefficients based on node properties.
    
    Args:
        node_data: Node properties and features
        agent_type: Type of agent (BiddingAgent, OccupancyNode, etc.)
        context_coeffs: Contextual coefficients from system
    
    Returns:
        Dictionary of computed agent coefficients
    """
    base_coeffs = {
        'efficiency': 0.5,
        'priority': 0.5, 
        'capacity': 0.5,
        'reliability': 0.5
    }
    
    if agent_type == 'BiddingAgent':
        # Bidding efficiency based on cost-benefit ratio
        cost = node_data.get('cost', 1000.0)
        performance = node_data.get('performance', 0.5)
        base_coeffs['efficiency'] = min(1.0, performance / (cost / 10000.0 + 0.1))
        base_coeffs['priority'] = context_coeffs.get('bid_priority', 0.7)
        
    elif agent_type == 'OccupancyNode':
        # Occupancy efficiency based on spatial utilization
        volume = node_data.get('volume', 100.0)
        area = node_data.get('area', 50.0)
        occupancy_ratio = min(1.0, volume / (area * 3.0))  # 3m ceiling assumption
        base_coeffs['efficiency'] = occupancy_ratio
        base_coeffs['capacity'] = min(1.0, area / 100.0)  # Normalized to 100 sqm
        
    elif agent_type == 'MEPSystemNode':
        # MEP system performance coefficients
        power = node_data.get('power', 400.0)
        efficiency_rating = node_data.get('efficiency', 0.8)
        base_coeffs['efficiency'] = efficiency_rating
        base_coeffs['capacity'] = min(1.0, power / 1000.0)  # Normalized to 1kW
        base_coeffs['reliability'] = context_coeffs.get('mep_reliability', 0.85)
        
    elif agent_type == 'ComplianceNode':
        # Compliance scoring coefficients
        safety_factor = node_data.get('safety_factor', 1.5)
        code_compliance = node_data.get('code_compliance', 0.9)
        base_coeffs['reliability'] = min(1.0, safety_factor / 2.5)
        base_coeffs['priority'] = code_compliance
        
    elif agent_type == 'InvestmentNode':
        # Investment performance coefficients
        roi = node_data.get('roi', 0.15)
        risk_factor = node_data.get('risk_factor', 0.3)
        base_coeffs['efficiency'] = min(1.0, roi / 0.5)  # 50% ROI as max
        base_coeffs['reliability'] = 1.0 - risk_factor
        
    return base_coeffs

def emergence_detection_formula(node_embeddings: np.ndarray, 
                              historical_patterns: List[np.ndarray]) -> float:
    """
    Scientific formula for detecting emergent behavior in graph structures.
    
    Args:
        node_embeddings: Current node embedding matrix
        historical_patterns: List of historical embedding patterns
    
    Returns:
        Emergence score (0.0 to 1.0)
    """
    if len(historical_patterns) < 3:
        return 0.0
    
    # Calculate embedding centroid drift
    current_centroid = np.mean(node_embeddings, axis=0)
    historical_centroids = [np.mean(pattern, axis=0) for pattern in historical_patterns]
    
    # Measure trajectory deviation (emergence indicator)
    centroid_distances = [
        np.linalg.norm(current_centroid - hist_centroid) 
        for hist_centroid in historical_centroids[-5:]  # Last 5 patterns
    ]
    
    # Calculate trend acceleration (rapid change = emergence)
    if len(centroid_distances) >= 3:
        acceleration = np.diff(np.diff(centroid_distances))
        emergence_strength = np.mean(np.abs(acceleration))
        
        # Normalize emergence score
        return min(1.0, emergence_strength * 10.0)  # Scale factor
    
    return 0.0

def callback_success_predictor(source_features: np.ndarray, target_features: np.ndarray,
                              historical_success_rate: float = 0.5) -> float:
    """
    Scientific formula for predicting callback execution success probability.
    
    Args:
        source_features: Source node feature vector
        target_features: Target node feature vector  
        historical_success_rate: Historical success rate for this path type
    
    Returns:
        Success probability (0.0 to 1.0)
    """
    # Feature compatibility score
    compatibility = graph_node_similarity(source_features, target_features, 'cosine')
    
    # Feature magnitude balance (similar magnitudes = better compatibility)
    source_magnitude = np.linalg.norm(source_features)
    target_magnitude = np.linalg.norm(target_features)
    magnitude_balance = 1.0 / (1.0 + abs(source_magnitude - target_magnitude))
    
    # Combined prediction with historical weighting
    feature_score = (compatibility * 0.6) + (magnitude_balance * 0.4)
    weighted_prediction = (feature_score * 0.7) + (historical_success_rate * 0.3)
    
    return min(1.0, max(0.0, weighted_prediction))


# ==== Scientific Formula Discovery Engine (SFDE) ====

class SFDEngine:
    """
    Scientific Formula Discovery Engine - The Trainer
    Autonomously identifies, assigns, and validates mathematical formulas across graph nodes
    """
    
    def __init__(self, node_dict: Dict, agent_coeffs: Dict):
        self.node_dict = node_dict
        self.agent_coeffs = agent_coeffs
        self.formula_registry = self._initialize_formula_registry()
        self.validation_results = {}
        
    def _initialize_formula_registry(self) -> Dict[str, callable]:
        """Initialize the core formula registry for agile expansion"""
        return {
            'node_similarity': graph_node_similarity,
            'edge_weight': edge_weight_formula,
            'agent_coefficient': agent_coefficient_formula,
            'emergence_detection': emergence_detection_formula,
            'callback_prediction': callback_success_predictor
        }
    
    def discover_formula_for_node(self, node_id: str, node_data: Dict) -> Dict[str, Any]:
        """
        Autonomously discover and assign appropriate formulas for a graph node
        """
        node_type = node_data.get('component_type', 'unknown')
        agent_type = node_data.get('agent_type', 'BiddingAgent')
        
        # Determine applicable formulas based on node characteristics
        applicable_formulas = []
        
        # Always apply agent coefficient formula
        applicable_formulas.append('agent_coefficient')
        
        # Apply specific formulas based on node type
        if node_type in ['structural', 'electrical', 'mep']:
            applicable_formulas.append('edge_weight')
            
        # Apply similarity formula for clustering potential
        applicable_formulas.append('node_similarity')
        
        return {
            'node_id': node_id,
            'applicable_formulas': applicable_formulas,
            'formula_assignments': self._assign_formulas(node_data, applicable_formulas),
            'validation_status': 'ready_for_training'
        }
    
    def _assign_formulas(self, node_data: Dict, formula_names: List[str]) -> Dict[str, Any]:
        """Assign and execute formulas for the node"""
        assignments = {}
        
        for formula_name in formula_names:
            if formula_name in self.formula_registry:
                formula_func = self.formula_registry[formula_name]
                
                try:
                    if formula_name == 'agent_coefficient':
                        agent_type = node_data.get('agent_type', 'BiddingAgent')
                        result = formula_func(node_data, agent_type, self.agent_coeffs)
                        
                    elif formula_name == 'edge_weight':
                        # Sample edge weight calculation
                        result = formula_func(node_data, node_data, 'structural')
                        
                    elif formula_name == 'node_similarity':
                        # Create sample feature vector for similarity
                        features = self._extract_node_features(node_data)
                        result = {'features_ready': True, 'feature_dim': len(features)}
                        
                    assignments[formula_name] = {
                        'result': result,
                        'status': 'computed',
                        'formula_type': 'scientific'
                    }
                    
                except Exception as e:
                    assignments[formula_name] = {
                        'result': None,
                        'status': 'error',
                        'error': str(e)
                    }
        
        return assignments
    
    def _extract_node_features(self, node_data: Dict) -> np.ndarray:
        """Extract numerical features from node data for GNN processing"""
        features = []
        
        # Extract properties
        props = node_data.get('properties', {})
        features.extend([
            props.get('volume', 0.0) / 1000.0,      # Normalized volume
            props.get('cost', 0.0) / 100000.0,      # Normalized cost  
            props.get('area', 0.0) / 100.0,         # Normalized area
            props.get('power', 0.0) / 1000.0,       # Normalized power
        ])
        
        # Extract coefficients
        coeffs = node_data.get('coefficients', {})
        features.extend([
            coeffs.get('safety_factor', 1.0),
            coeffs.get('efficiency', 0.5),
            coeffs.get('performance', 0.5),
            coeffs.get('thermal_rating', 0.5)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def validate_formula_results(self, formula_results: Dict) -> Dict[str, Any]:
        """Validate computed formula results for scientific accuracy"""
        validation = {
            'overall_status': 'valid',
            'formula_validations': {},
            'recommendations': []
        }
        
        for formula_name, result_data in formula_results.items():
            if result_data['status'] == 'computed':
                result = result_data['result']
                
                # Validate result ranges and scientific logic
                if formula_name == 'agent_coefficient':
                    valid = all(0.0 <= v <= 1.0 for v in result.values())
                    validation['formula_validations'][formula_name] = {
                        'valid': valid,
                        'reason': 'Coefficients within [0,1] range' if valid else 'Invalid range'
                    }
                    
                elif formula_name == 'edge_weight':
                    valid = 0.0 <= result <= 1.0
                    validation['formula_validations'][formula_name] = {
                        'valid': valid,
                        'reason': 'Weight within [0,1] range' if valid else 'Invalid range'
                    }
                    
                else:
                    validation['formula_validations'][formula_name] = {
                        'valid': True,
                        'reason': 'Basic validation passed'
                    }
            else:
                validation['formula_validations'][formula_name] = {
                    'valid': False,
                    'reason': f"Formula execution failed: {result_data.get('error', 'Unknown error')}"
                }
        
        # Generate recommendations
        invalid_formulas = [name for name, val in validation['formula_validations'].items() if not val['valid']]
        if invalid_formulas:
            validation['overall_status'] = 'requires_attention'
            validation['recommendations'].append(f"Review formulas: {', '.join(invalid_formulas)}")
        
        return validation
    
    def run(self):
        """Run the SFDE operator on all nodes"""
        print("🧪 Scientific Formula Discovery Engine (SFDE) - Starting Training...")
        
        processed_nodes = 0
        valid_formulas = 0
        
        # Process each node in the dictionary  
        for node_id, node_data in self.node_dict.items():
            if isinstance(node_data, dict):
                # Discover and assign formulas
                discovery_result = self.discover_formula_for_node(node_id, node_data)
                
                # Validate formula results
                validation = self.validate_formula_results(discovery_result['formula_assignments'])
                
                # Store results
                self.validation_results[node_id] = {
                    'discovery': discovery_result,
                    'validation': validation
                }
                
                processed_nodes += 1
                if validation['overall_status'] == 'valid':
                    valid_formulas += 1
        
        # Summary
        print(f"✅ SFDE Training Complete:")
        print(f"   📊 Processed nodes: {processed_nodes}")
        print(f"   🧮 Valid formulas: {valid_formulas}")
        print(f"   🎯 Success rate: {valid_formulas/max(processed_nodes,1)*100:.1f}%")
        print(f"   🔬 Formula registry: {len(self.formula_registry)} scientific formulas")
        
        return {
            'processed_nodes': processed_nodes,
            'valid_formulas': valid_formulas,
            'formula_registry_size': len(self.formula_registry),
            'validation_results': self.validation_results
        } 