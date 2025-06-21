# Global Design Parameters Integration Guide
## 🏗️ Node-Level Shared Variables for BEM Microservices

### Overview
The `global_design_parameters` dictionary serves as a **node-level shared resource** used to enrich each node or component during parsing, scoring, and UI rendering across all BEM microservices.

### 📁 File Structure
```
MICROSERVICE_ENGINES/
├── global_design_parameters.json          # JSON configuration
├── shared/
│   ├── __init__.py                       # Package initialization
│   └── global_design_parameters.py       # Python module
└── */main.py                            # Microservices (updated with GDP)
```

### 🎯 Core Parameters

#### Building Components
```python
"building_components": {
    "locus_condition": "context_driven",
    "self_weight_category": "lightweight",
    "order_type": "modular_grid",
    "form_strategy": "evolutionary_adaptive",
    "function_flexibility": 0.85,
    "material_expression": "symbolic_natural",
    "aesthetic_tension": 0.6,
    "precision_tolerance": "high",
    "detail_complexity": "prefab_ready",
    "joint_type": "slip_fit",
    "daylight_factor": 0.73,
    "evolutionary_potential": 0.92
}
```

#### Bi-Directional Procedural Model
```python
"bidirectional_procedural_model": {
    "connectivity_validated": True,
    "symbolic_depth_range": 5,
    "layout_compactness": 0.72,
    "hierarchy_type": "heterogeneous",
    "mirror_operations_enabled": True,
    "layout_success_rate": 0.99,
    "room_rectangularity_score": 0.84,
    "public_private_ratio": "1:4"
}
```

#### Generative Facade Tool
```python
"generative_facade_tool": {
    "facade_style_id": "style_2_bay_variation",
    "parametric_frame_depth": 0.3,
    "bay_arrangement_pattern": "alternating_even_odd",
    "component_modularity": "enabled",
    "floor_span_variance": "medium",
    "tilt_transformation": False,
    "generation_type": "rule_driven",
    "cad_integration_ready": True
}
```

### 🔧 Integration Methods

#### Method 1: Direct Import
```python
from shared.global_design_parameters import global_design_parameters

# Access specific parameter categories
building_params = global_design_parameters['building_components']
facade_params = global_design_parameters['generative_facade_tool']
```

#### Method 2: Helper Functions
```python
from shared.global_design_parameters import (
    enrich_node_with_global_params,
    get_building_component_params,
    get_scoring_context
)

# Enrich a node/component
enriched_node = enrich_node_with_global_params(node, 'building_components')

# Get specific parameter sets
building_params = get_building_component_params()
scoring_context = get_scoring_context()
```

#### Method 3: Category-Specific Access
```python
from shared.global_design_parameters import (
    get_building_component_params,
    get_procedural_model_params,
    get_facade_tool_params
)

# Use in different contexts
building_data = get_building_component_params()
layout_data = get_procedural_model_params()  
facade_data = get_facade_tool_params()
```

### 🚀 Microservice Integration Examples

#### 1. Node Enrichment During Parsing
```python
@app.route('/process', methods=['POST'])
def process_components():
    components = request.get_json().get('components', [])
    
    # Enrich each component with global design parameters
    enriched_components = []
    for component in components:
        enriched_component = component.copy()
        enriched_component = enrich_node_with_global_params(
            enriched_component, 
            'building_components'
        )
        enriched_component['_gdp_enriched'] = True
        enriched_components.append(enriched_component)
    
    return process_with_enriched_data(enriched_components)
```

#### 2. Scoring with Global Context
```python
def calculate_component_score(component):
    # Get global scoring context
    scoring_context = get_scoring_context()
    
    # Base score calculation
    base_score = calculate_base_score(component)
    
    # Apply global design parameter modifiers
    if component.get('form_strategy') == scoring_context['building_components']['form_strategy']:
        base_score *= 1.2  # Boost for evolutionary_adaptive
    
    if component.get('precision_tolerance') == scoring_context['building_components']['precision_tolerance']:
        base_score *= 1.1  # Boost for high precision
    
    return base_score
```

#### 3. UI Rendering Context
```python
@app.route('/render', methods=['POST'])
def render_component():
    component = request.get_json()
    
    # Get UI rendering context
    ui_context = get_ui_rendering_context('generative_facade_tool')
    
    # Apply facade parameters to rendering
    render_config = {
        'facade_style': ui_context['facade_style_id'],
        'frame_depth': ui_context['parametric_frame_depth'],
        'bay_pattern': ui_context['bay_arrangement_pattern'],
        'modularity': ui_context['component_modularity']
    }
    
    return render_with_config(component, render_config)
```

### 🔄 Microservice Update Pattern

#### Standard Integration Template
```python
#!/usr/bin/env python3
"""
[Service Name] Microservice
Enhanced with Global Design Parameters
"""

import sys
sys.path.append('../shared')

# Import global design parameters
try:
    from shared.global_design_parameters import (
        global_design_parameters,
        enrich_node_with_global_params,
        get_scoring_context
    )
    GDP_AVAILABLE = True
except ImportError as e:
    GDP_AVAILABLE = False
    # Fallback parameters
    global_design_parameters = {...}

@app.route('/process', methods=['POST'])
def process_request():
    data = request.get_json()
    components = data.get('components', [])
    
    # Enrich components if GDP available
    if GDP_AVAILABLE:
        enriched_components = [
            enrich_node_with_global_params(comp.copy(), 'building_components')
            for comp in components
        ]
    else:
        enriched_components = components
    
    # Process with enriched data
    results = perform_analysis(enriched_components)
    
    return jsonify({
        'results': results,
        'gdp_applied': GDP_AVAILABLE,
        'components_enriched': len(enriched_components)
    })
```

### 📊 Impact on Microservices

#### Updated Services
- ✅ **ne-functor-types**: Node enrichment during type analysis
- 🔄 **ne-dag-alpha**: *Ready for integration*
- 🔄 **ne-graph-runtime-engine**: *Ready for integration*
- 🔄 **ne-optimization-engine**: *Ready for integration*
- 🔄 **ne-callback-engine**: *Ready for integration*

#### Integration Benefits
1. **Consistent Parameters**: All services use same design parameters
2. **Node Enrichment**: Components automatically enhanced with global context
3. **Scoring Consistency**: Unified scoring context across services
4. **UI Coherence**: Consistent rendering parameters
5. **Maintainability**: Single source of truth for design parameters

### 🛠️ Environment Variables

#### Docker Configuration
```dockerfile
# Add to Dockerfile
ENV GLOBAL_DESIGN_PARAMS_PATH=/app/shared/global_design_parameters.json
ENV GDP_CACHE_TTL=3600
ENV GDP_RELOAD_ON_CHANGE=true
```

#### Docker Compose
```yaml
services:
  ne-functor-types:
    environment:
      - GLOBAL_DESIGN_PARAMS_PATH=/shared/global_design_parameters.json
    volumes:
      - ./shared:/shared:ro
```

### 🧪 Testing Integration

#### Test Script
```python
def test_gdp_integration():
    """Test global design parameters integration"""
    
    # Test import
    from shared.global_design_parameters import global_design_parameters
    assert 'building_components' in global_design_parameters
    
    # Test node enrichment
    test_node = {'id': 'test_component', 'type': 'structural'}
    enriched_node = enrich_node_with_global_params(test_node, 'building_components')
    
    assert 'locus_condition' in enriched_node
    assert enriched_node['locus_condition'] == 'context_driven'
    
    print("✅ GDP integration test passed")

if __name__ == "__main__":
    test_gdp_integration()
```

### 📈 Performance Considerations

#### Caching Strategy
- **In-Memory Cache**: Parameters loaded once per service startup
- **TTL**: 1 hour cache timeout (configurable)
- **Reload**: On-demand reload capability

#### Memory Usage
- **JSON Size**: ~2KB per service
- **Python Objects**: ~5KB per service
- **Total Impact**: Negligible (<1% memory overhead)

### 🔮 Future Enhancements

#### Dynamic Parameters
```python
# Future: Dynamic parameter loading
def load_dynamic_parameters(project_id, phase):
    """Load project-specific and phase-specific parameters"""
    base_params = global_design_parameters.copy()
    project_params = load_project_parameters(project_id)
    phase_params = load_phase_parameters(phase)
    
    return merge_parameters(base_params, project_params, phase_params)
```

#### Parameter Validation
```python
# Future: Parameter validation schema
GDP_SCHEMA = {
    "building_components": {
        "function_flexibility": {"type": "float", "range": [0.0, 1.0]},
        "aesthetic_tension": {"type": "float", "range": [0.0, 1.0]},
        "daylight_factor": {"type": "float", "range": [0.0, 1.0]}
    }
}
```

---

**Integration Status**: ✅ Active  
**Services Updated**: 1/6 microservices  
**Performance Impact**: Negligible  
**Maintenance**: Single source of truth 