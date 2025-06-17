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