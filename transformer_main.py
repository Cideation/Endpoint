from dwg_cad_ifc_parser import parse_dwg
from open_clean import clean_fields
from generate_ids import assign_ids

def run_pipeline(raw_data):
    parsed = parse_dwg(raw_data)
    cleaned = clean_fields(parsed)
    enriched = assign_ids(cleaned)
    return enriched

def parse_data(cleaned_dict):
    # Stub: You should replace this with actual parsing logic
    return {"parsed_output": cleaned_dict}
