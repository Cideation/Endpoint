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
    return required_area_m2


# ==== topologic_utils.py ====

# Placeholder: to be extended with actual TopologicPy functions
def get_cell_centroid(cell):
    return cell.Centroid().Coordinates()

def extract_edges_from_cell(cell):
    return cell.Edges()

# ==== aggregate_runner.py ====
# Performs a weighted average aggregation using defined scalar inputs and weights.

def aggregate_scalar_values(data):
    """
    Expects:
    {
        "roi_estimate": float,
        "cost_index": float,
        "investment_score": float,
        "agent_trust": float,
        "node_priority": float
    }
    """
    scalar_keys = ["roi_estimate", "cost_index", "investment_score"]
    weight_keys = ["agent_trust", "node_priority"]

    values = [data.get(k, 0.0) for k in scalar_keys]
    weights = [data.get(k, 1.0) for k in weight_keys]

    weighted_values = [v * sum(weights) for v in values]
    total_weight = len(values) * sum(weights)
    result = sum(weighted_values) / total_weight if total_weight != 0 else 0.0

    return {"aggregated_score": round(result, 3)}




# ==== STRUCTURAL FORMULAS (NSCP 2015 LRFD) ====

def factored_load(dead_load, live_load, wind_load=0, seismic_load=0, phi=0.9):
    return phi * (1.2 * dead_load + 1.6 * live_load + wind_load + seismic_load)

def bending_moment_uniform_load(w, l):
    return (w * l ** 2) / 8  # Simple beam, center load

def shear_force_uniform_load(w, l):
    return (w * l) / 2  # Max shear at supports

def deflection_uniform_load(w, l, E, I):
    return (5 * w * l**4) / (384 * E * I)

def slenderness_ratio(K, L, r):
    return (K * L) / r

def critical_buckling_load(E, I, K, L):
    from math import pi
    return (pi ** 2 * E * I) / ((K * L) ** 2)

def axial_capacity(fc_prime, Ag, fy, Ast):
    return 0.85 * fc_prime * Ag + fy * Ast

def ultimate_bearing_capacity(c, q, gamma, B, Nc, Nq, Ngamma):
    return c * Nc + q * Nq + 0.5 * gamma * B * Ngamma

# ==== COST FORMULAS ====

def material_cost(quantity, unit_price):
    return quantity * unit_price

def labor_cost(hours, rate_per_hour):
    return hours * rate_per_hour

def total_project_cost(materials, labor, equipment, contingency=0.1):
    subtotal = materials + labor + equipment
    return subtotal * (1 + contingency)

def roi_formula(gain_from_investment, cost_of_investment):
    return (gain_from_investment - cost_of_investment) / cost_of_investment

def cost_per_square_meter(total_cost, floor_area):
    return total_cost / floor_area

def depreciation_cost(initial_value, salvage_value, useful_life_years):
    return (initial_value - salvage_value) / useful_life_years

def financing_cost(principal, annual_rate, years):
    return principal * annual_rate * years

def contingency_allowance(base_cost, risk_factor):
    return base_cost * risk_factor

# ==== ENERGY FORMULAS ====

def energy_consumption(power_kw, hours):
    return power_kw * hours  # kWh

def hvac_load(area_m2, watts_per_m2=120):
    return area_m2 * watts_per_m2 / 1000  # kW

def lighting_energy(area_m2, lux_level, efficacy_lm_per_watt=80, hours=8):
    watts_required = (area_m2 * lux_level) / efficacy_lm_per_watt
    return (watts_required / 1000) * hours  # kWh

def solar_energy_output(panel_area, efficiency, solar_irradiance=5):
    return panel_area * efficiency * solar_irradiance  # kWh/day

# ==== MEP FORMULAS ====

def water_demand(occupants, daily_use_per_person=150):
    return occupants * daily_use_per_person  # Liters/day

def pipe_flow_rate(diameter_mm, velocity_mps):
    from math import pi
    radius_m = diameter_mm / 1000 / 2
    area = pi * radius_m ** 2
    return area * velocity_mps  # m^3/s

def duct_airflow(area_m2, velocity_mps):
    return area_m2 * velocity_mps  # m^3/s

def electrical_demand(load_kw, diversity_factor=0.8):
    return load_kw * diversity_factor  # adjusted load

# ==== TIME-BASED FORMULAS ====

def project_duration(total_tasks, avg_duration_per_task):
    return total_tasks * avg_duration_per_task

def schedule_slippage(planned_duration, actual_duration):
    return actual_duration - planned_duration

def cycle_time(process_time, wait_time):
    return process_time + wait_time

def productivity_rate(output_units, total_time):
    return output_units / total_time  # units per hour or similar

def gantt_overlap(tasks_parallel, duration_each):
    return duration_each / tasks_parallel  # reduced duration if parallel

# ==== FABRICATION FORMULAS ====

def generate_gcode_path(
    path,
    tool_radius=1.0,
    cut_depth=-1.0,
    travel_height=5.0,
    feed_rate=1000
):
    """
    Generate G-code from a 2D XY path for fabrication.
    
    Args:
        path: List of (x, y) coordinate tuples
        tool_radius (float): Radius of cutting/extruding tool
        cut_depth (float): Z-depth of cut (negative value)
        travel_height (float): Z-height when moving safely above workpiece
        feed_rate (int): Movement speed
        
    Returns:
        str: G-code string ready for CNC/3D printer
    """
    if not path:
        return ""

    gcode = [
        "G21 ; Set units to mm",
        "G90 ; Absolute positioning", 
        f"G1 Z{travel_height:.2f} F{feed_rate} ; Move up before travel",
        f"G1 X{path[0][0]:.2f} Y{path[0][1]:.2f} F{feed_rate} ; Move to start",
        f"G1 Z{cut_depth:.2f} F{feed_rate} ; Move down to cut"
    ]

    for x, y in path[1:]:
        gcode.append(f"G1 X{x:.2f} Y{y:.2f} F{feed_rate}")

    gcode.append(f"G1 Z{travel_height:.2f} ; Lift up after cut")
    gcode.append("M2 ; End of program")

    return "\n".join(gcode)