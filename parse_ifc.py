import ifcopenshell
import uuid

def parse_ifc_file(file_path):
    model = ifcopenshell.open(file_path)
    items = []
    for wall in model.by_type("IfcWall"):
        props = {
            "global_id": wall.GlobalId,
            "name": getattr(wall, "Name", ""),
            "predefined_type": getattr(wall, "PredefinedType", None),
        }
        item = {
            "node_label": "Item",
            "entity_id": str(uuid.uuid4()),
            "properties": props
        }
        items.append(item)
    return items 