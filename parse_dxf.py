import ezdxf
import uuid

def parse_dxf_file(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    items = []

    for entity in msp:
        props = {
            "dxftype": entity.dxftype(),
            "layer": entity.dxf.layer,
        }
        item = {
            "node_label": "Item",
            "entity_id": str(uuid.uuid4()),
            "properties": props
        }
        items.append(item)
    return items 