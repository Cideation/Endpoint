import re

def normalize_keys(d):
    def snake_case(k): return re.sub(r'(?<!^)(?=[A-Z])', '_', k).lower()
    return {snake_case(k): v for k, v in d.items() if isinstance(k, str)} 