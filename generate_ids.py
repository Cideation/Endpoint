import uuid

def assign_ids(data):
    for i, item in enumerate(data):
        item['component_id'] = f"CMP-{uuid.uuid4().hex[:8]}"
        item['node_id'] = f"N-{i+1:03d}"
    return data
