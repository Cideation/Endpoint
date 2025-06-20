"""
local_calc/gcode_writer.py

Utility to generate G-code from 2D geometry paths.
Used in Gamma phase as a local_calc for fabrication-ready output.
"""

from typing import List, Tuple

def generate_gcode_path(
    path: List[Tuple[float, float]],
    tool_radius: float = 1.0,
    cut_depth: float = -1.0,
    travel_height: float = 5.0,
    feed_rate: int = 1000
) -> str:
    """
    Generate G-code from a 2D XY path.

    Args:
        path (List[Tuple[float, float]]): List of (x, y) coordinates.
        tool_radius (float): Radius of cutting/extruding tool.
        cut_depth (float): Z-depth of cut.
        travel_height (float): Z-height when moving safely above workpiece.
        feed_rate (int): Movement speed.

    Returns:
        str: G-code string.
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


# 🔍 Example test
if __name__ == "__main__":
    test_path = [(0, 0), (0, 100), (100, 100), (100, 0), (0, 0)]
    gcode_output = generate_gcode_path(test_path)
    print(gcode_output) 