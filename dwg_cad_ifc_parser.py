def parse_dwg(data):
    return [{"name": part.get("label", ""), "shape": part.get("geom_type", "")} for part in data]

def parse_ifc(data):
    return [{"name": entity.get("Name", ""), "shape": entity.get("Type", "")} for entity in data]
