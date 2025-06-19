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

</rewritten_file> 