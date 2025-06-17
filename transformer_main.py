from dwg_cad_ifc_parser import parse_dwg
from open_clean import clean_fields
from generate_ids import assign_ids

# BEGIN FILE: parse_dxf.py
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


# BEGIN FILE: parse_ifc.py
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


# BEGIN FILE: detect_scalars.py
def detect_scalars(data):
    """
    Adds scalar type annotations to all fields in the input dictionary.
    """
    scalar_info = {}
    for key, value in data.items():
        if isinstance(value, (int, float, str, bool)):
            scalar_info[key + "_type"] = type(value).__name__
        elif value is None:
            scalar_info[key + "_type"] = "null"
        else:
            scalar_info[key + "_type"] = "complex"
    data.update(scalar_info)
    return data


# BEGIN FILE: generate_id.py
import uuid

def generate_unique_id(seed=None):
    """
    Generates a unique ID using UUID4 or seed.
    """
    if seed:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(seed)))
    return str(uuid.uuid4())


# BEGIN FILE: neo_writer.py
import json
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_password"))

def load_data_to_neo(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    with driver.session() as session:
        for item in data:
            if "label" in item:
                session.write_transaction(insert_node, item["label"], item["properties"])
            elif "type" in item:
                session.write_transaction(insert_edge, item["from_id"], item["to_id"], item["type"], item["properties"])

def insert_node(tx, label, props):
    tx.run(f"MERGE (n:{label} {{item_id: $id}}) SET n += $props", id=props["item_id"], props=props)

def insert_edge(tx, from_id, to_id, rel_type, props):
    tx.run(f"""
        MATCH (a {{item_id: $from}})
        MATCH (b {{item_id: $to}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
    """, from=from_id, to=to_id, props=props)

if __name__ == "__main__":
    load_data_to_neo("outputs/neo_ready_output.json")


# BEGIN FILE: normalize_keys.py

import re

def normalize_keys(d):
    def snake_case(k): return re.sub(r'(?<!^)(?=[A-Z])', '_', k).lower()
    return {snake_case(k): v for k, v in d.items() if isinstance(k, str)}


# BEGIN FILE: cad_parser_main.py
import os
import json
from parse_ifc import parse_ifc_file
from parse_dxf import parse_dxf_file

def cad_parser_main(input_path, output_path="neo_ready_output.json"):
    ext = os.path.splitext(input_path)[-1].lower()
    if ext == ".ifc":
        results = parse_ifc_file(input_path)
    elif ext == ".dxf":
        results = parse_dxf_file(input_path)
    else:
        raise ValueError(f"Unsupported CAD format: {ext}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved {len(results)} items to {output_path}")

if __name__ == "__main__":
    import sys
    cad_parser_main(sys.argv[1])


# OpenAI cleaner integration:
from openai_cleaner import gpt_clean_and_validate  # Optional OpenAI cleanup
