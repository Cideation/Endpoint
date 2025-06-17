import uuid

def generate_unique_id(seed=None):
    """
    Generates a unique ID using UUID4 or seed.
    """
    if seed:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(seed)))
    return str(uuid.uuid4()) 