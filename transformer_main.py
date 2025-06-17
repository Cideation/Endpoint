from dwg_cad_ifc_parser import parse_dwg
from openai_cleaner import gpt_clean_and_validate
from generate_ids import assign_ids
from parse_dxf import parse_dxf_file
from parse_ifc import parse_ifc_file
from detect_scalars import detect_scalars
from generate_id import generate_unique_id
from normalize_keys import normalize_keys
from cad_parser_main import cad_parser_main

def main():
    """
    Main transformer function that orchestrates the data processing pipeline.
    """
    # This function can be used to coordinate the various parsing and cleaning operations
    pass

if __name__ == "__main__":
    main()
