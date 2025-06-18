import os
import json
from parse_ifc import parse_ifc_file
from parse_dxf import parse_dxf_file

def cad_parser_main(input_path, output_path="parsed_output.json"):
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