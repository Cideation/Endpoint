# Neon Database Integration for Phase 2 Pipeline

This module provides complete integration between the Phase 2 microservice pipeline and the Neon PostgreSQL database. It enables running the full pipeline with real data from the database.

## 🚀 Quick Start

### 1. Test Database Connection

```bash
cd neon
python test_connection.py
```

### 2. Test Extract → Schema → Discard Pipeline

```bash
python -m neon.test_file_processing
```

### 3. Run Complete Pipeline with Database Data

```bash
# Run on all components (limit 100)
python run_pipeline.py

# Run on specific component type
python run_pipeline.py --type structural --limit 50

# Run with specific affinity types
python run_pipeline.py --affinity spatial structural cost

# Run without saving results
python run_pipeline.py --no-save
```

### 4. Run Full Database Integration Test

```bash
python -m neon.test_database_integration
```

## 📁 File Structure

```
neon/
├── config.py                    # Database configuration
├── db_manager.py               # Database connection and query management
├── models.py                   # Pydantic models for data validation
├── json_transformer.py         # Transform database data to JSON formats
├── orchestrator.py             # Pipeline orchestration
├── file_processor.py           # Extract → Schema → Discard file processing
├── run_pipeline.py             # Main pipeline runner with database integration
├── test_connection.py          # Simple connection test
├── test_file_processing.py     # Test extract → schema → discard pipeline
├── test_database_integration.py # Comprehensive integration test
├── test_pipeline.py            # Pipeline test with sample data
└── README.md                   # This file
```

## 🔧 Database Integration Features

### Extract → Schema → Discard Pipeline
- **No Raw File Storage**: Files are processed and discarded immediately
- **Immediate Database Integration**: Extracted data stored directly in PostgreSQL
- **Clean Architecture**: No file management complexity
- **Parser Integration**: Works with existing CAD/BIM parsers
- **Processing Metadata**: Track file processing history and performance

### Real Data Pipeline
- **Direct Database Queries**: Query components with all related data (spatial, dimensions, materials, geometry)
- **Filtered Processing**: Filter by component type, spatial bounds, or other criteria
- **Batch Processing**: Process large datasets with configurable limits
- **Error Handling**: Robust error handling with detailed logging

### Data Transformation
- **Database Row → Component**: Transform PostgreSQL rows to component dictionaries
- **JSON Harmonization**: Convert to container-specific JSON formats
- **Schema Validation**: Ensure data integrity with Pydantic models

### Pipeline Orchestration
- **Multi-Container Execution**: Route data to all Phase 2 containers
- **Affinity-Based Processing**: Process data based on SFDE affinity types
- **Execution Tracking**: Track pipeline execution history and results
- **Result Persistence**: Save pipeline results to JSON files

## 🗄️ Database Schema Integration

The integration supports the complete database schema including:

- **Components**: Core component data
- **Spatial Data**: Centroid coordinates, bounding boxes
- **Dimensions**: Length, width, height, area, volume
- **Materials**: Material properties and codes
- **Geometry Properties**: Vertex counts, face counts, surface areas
- **Relationships**: Component-material associations
- **Parsed Files**: File processing metadata and history

## 📊 Usage Examples

### File Processing (Extract → Schema → Discard)

```python
from neon.file_processor import process_file_and_discard
import asyncio

# Process a CAD file and discard the raw file
result = await process_file_and_discard(
    file_path="/path/to/building.dwg",
    original_filename="building.dwg",
    file_type="dwg"
)

print(f"Components extracted: {result['components_extracted']}")
print(f"Processing time: {result['processing_time_ms']}ms")
print(f"Raw file discarded: {result['raw_file_discarded']}")
```

### Basic Pipeline Execution

```python
from neon.run_pipeline import run_pipeline
import asyncio

# Run pipeline on all components
result = asyncio.run(run_pipeline(limit=100))

# Run on specific component type
result = asyncio.run(run_pipeline(
    component_type="structural",
    limit=50,
    affinity_types=["spatial", "structural", "cost"]
))
```

### Database Query Examples

```python
from neon.db_manager import NeonDBManager
from neon.config import NEON_CONFIG

async def query_examples():
    db_manager = NeonDBManager(NEON_CONFIG)
    await db_manager.create_pool()
    
    # Get all components with relations
    components = await db_manager.execute_query("""
        SELECT c.*, m.material_name, sp.centroid_x, d.length_mm
        FROM components c
        LEFT JOIN component_materials cm ON c.component_id = cm.component_id
        LEFT JOIN materials m ON cm.material_id = m.material_id
        LEFT JOIN spatial_data sp ON c.component_id = sp.component_id
        LEFT JOIN dimensions d ON c.component_id = d.component_id
        LIMIT 10
    """)
    
    await db_manager.close_async()
    return components
```

### Custom Data Transformation

```python
from neon.json_transformer import JSONTransformer
from neon.orchestrator import Orchestrator

# Transform database components to JSON
transformer = JSONTransformer()
orchestrator = Orchestrator()

# Get components from database (example)
components = [...]  # Your database query results

# Transform to node collection
node_collection = transformer.transform_components_to_node_collection(components)

# Run pipeline
result = await orchestrator.orchestrate_full_pipeline(
    components=components,
    affinity_types=["spatial", "structural"]
)
```

## 🔍 Testing

### Connection Test
```bash
python test_connection.py
```

### File Processing Test
```bash
python -m neon.test_file_processing
```

### Full Integration Test
```bash
python -m neon.test_database_integration
```

### Pipeline Test with Sample Data
```bash
python test_pipeline.py
```

## 📈 Pipeline Results

The pipeline generates comprehensive results including:

- **Execution Status**: Success/failure status
- **Component Counts**: Number of components processed
- **Container Results**: Individual container execution results
- **Database Metadata**: Query filters and database information
- **Timing Information**: Start/end times and execution duration
- **Error Details**: Detailed error information if failures occur
- **File Processing**: Extract → Schema → Discard metrics

## 🛠️ Configuration

Database configuration is managed in `config.py`:

```python
NEON_CONFIG = {
    'host': 'ep-white-waterfall-a85g0dgx-pooler.eastus2.azure.neon.tech',
    'port': 5432,
    'database': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_CcgA0kKeYVU2',
    'sslmode': 'require'
}
```

## 🔗 Phase 2 Container Integration

The pipeline integrates with all Phase 2 containers:

- **ne-dag-alpha**: Alpha-phase DAG execution
- **ne-functor-types**: Spatial and aggregation calculations
- **ne-callback-engine**: Beta/Gamma callback processing
- **sfde-engine**: AI-augmented scientific formula discovery
- **ne-graph-runtime-engine**: NetworkX graph operations
- **api-gateway**: HTTP API exposure

## 📝 Next Steps

1. **Verify Database Schema**: Ensure database tables match expected schema
2. **Insert Sample Data**: Add test data if database is empty
3. **Connect Containers**: Deploy and connect to actual Phase 2 containers
4. **Production Deployment**: Deploy to production environment
5. **Monitoring**: Add monitoring and alerting for pipeline execution

## 🚨 Troubleshooting

### Common Issues

1. **Connection Failed**: Check database credentials and network connectivity
2. **No Components Found**: Verify database has data or insert sample data
3. **Schema Mismatch**: Ensure database schema matches expected structure
4. **Container Errors**: Check Phase 2 container availability and configuration
5. **File Processing Errors**: Check parser availability and file format support

### Debug Mode

Enable detailed logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Related Documentation

- [Phase 2 Microservice Architecture](../MICROSERVICE_ENGINES/README_MICROSERVICE_ENGINES-Phase%202.md)
- [SFDE Utility Foundation](../MICROSERVICE_ENGINES/sfde_utility_foundation_extended.py)
- [Pipeline Summary](PIPELINE_SUMMARY.md)
- [Database Schema](../postgre/enhanced_schema.sql) 