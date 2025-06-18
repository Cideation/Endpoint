# Neon PostgreSQL Integration for Enhanced CAD Parser

This directory contains the Neon PostgreSQL integration layer for the Enhanced CAD Parser system. It provides a robust, scalable database backend optimized for CAD/BIM data storage and querying.

## 🏗️ Architecture Overview

```
Enhanced CAD Parser
├── Parser Layer (DXF, DWG, IFC, PDF, OBJ, STEP)
├── Neon PostgreSQL (Staging Layer)
└── Neo4j Aura (Graph Modeling)
```

## 📁 File Structure

```
neon/
├── postgresql_schema.sql      # Complete database schema
├── csv_migration.py           # CSV data migration script
├── db_manager.py              # Database connection manager
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Set Up Neon PostgreSQL

1. **Create Neon Account**: Sign up at [neon.tech](https://neon.tech)
2. **Create Database**: Create a new project in Neon console
3. **Get Connection Details**: Copy your connection string

### 2. Environment Configuration

Create a `.env` file in your project root:

```bash
# Neon PostgreSQL Configuration
NEON_HOST=your-neon-host.neon.tech
NEON_PORT=5432
NEON_DATABASE=cad_parser
NEON_USER=your-username
NEON_PASSWORD=your-password

# Optional: Connection Pool Settings
NEON_POOL_MIN_SIZE=5
NEON_POOL_MAX_SIZE=20
```

### 3. Install Dependencies

```bash
cd neon
pip install -r requirements.txt
```

### 4. Initialize Database

```bash
# Connect to your Neon database and run the schema
psql "postgresql://user:password@host:port/database" -f postgresql_schema.sql
```

### 5. Migrate CSV Data

```bash
# Run the CSV migration script
python csv_migration.py
```

## 📊 Database Schema

### Core Tables

| Table | Description | Key Features |
|-------|-------------|--------------|
| `components` | Main component entities | UUID primary keys, timestamps |
| `materials` | Material definitions | Base materials, variants, codes |
| `suppliers` | Supplier information | Contact details, lead times |
| `functions` | Functional classifications | Functors, types, descriptions |
| `families` | Product families | Variants, product families |
| `spatial_data` | Spatial coordinates | PostGIS geometry, centroids |
| `dimensions` | Physical measurements | Length, width, height, volume |
| `properties` | Component properties | Key-value pairs, units |

### Parser Integration Tables

| Table | Description | Purpose |
|-------|-------------|---------|
| `parsed_files` | File processing metadata | Track parsing status, performance |
| `parsed_components` | Parser output mapping | Link parser data to database entities |

### Views

| View | Description | Use Case |
|------|-------------|----------|
| `component_summary` | Complete component data | Reporting, analysis |
| `parser_statistics` | Parser performance metrics | Monitoring, optimization |

## 🔧 Usage Examples

### Basic Database Operations

```python
from neon.db_manager import NeonDBManager
import asyncio

async def main():
    # Initialize database manager
    db_manager = NeonDBManager()
    
    try:
        # Get database statistics
        stats = await db_manager.get_database_stats()
        print(f"Total components: {stats['components']}")
        
        # Search for components
        results = await db_manager.search_components("wall", limit=10)
        for component in results:
            print(f"Found: {component['component_name']}")
            
        # Get spatial components
        spatial_components = await db_manager.get_spatial_components()
        print(f"Spatial components: {len(spatial_components)}")
        
    finally:
        await db_manager.close_async()

# Run the example
asyncio.run(main())
```

### Parser Integration

```python
async def process_cad_file(file_path: str):
    db_manager = NeonDBManager()
    
    try:
        # Insert file metadata
        file_data = {
            'file_name': 'example.dxf',
            'file_path': file_path,
            'file_type': 'DXF',
            'file_size': 1024000,
            'parsing_status': 'processing'
        }
        
        file_id = await db_manager.insert_parsed_file(file_data)
        
        # Process components (example from parser output)
        component_data = {
            'name': 'Wall_001',
            'type': 'Wall',
            'description': 'Exterior wall',
            'geometry': {
                'has_position': True,
                'position': [100.0, 200.0, 0.0],
                'dimensions': {
                    'length': 5000.0,
                    'width': 200.0,
                    'height': 3000.0,
                    'volume': 3000000.0
                },
                'properties': {
                    'vertex_count': 8,
                    'face_count': 6,
                    'edge_count': 12,
                    'surface_area': 30.0
                }
            }
        }
        
        component_id = await db_manager.insert_parsed_component(file_id, component_data)
        
        # Update parsing status
        await db_manager.update_parsing_status(file_id, 'success')
        
        print(f"Processed component: {component_id}")
        
    finally:
        await db_manager.close_async()
```

## 🔍 Query Examples

### Component Search

```sql
-- Search components by name or type
SELECT * FROM component_summary 
WHERE component_name ILIKE '%wall%' 
   OR component_type ILIKE '%wall%'
ORDER BY created_at DESC 
LIMIT 10;
```

### Spatial Queries

```sql
-- Find components within bounding box
SELECT c.component_name, sp.centroid_x, sp.centroid_y, sp.centroid_z
FROM components c
JOIN spatial_data sp ON c.component_id = sp.component_id
WHERE sp.centroid_x BETWEEN 0 AND 1000
  AND sp.centroid_y BETWEEN 0 AND 1000
  AND sp.centroid_z BETWEEN 0 AND 100;
```

### Parser Statistics

```sql
-- Get parser performance metrics
SELECT 
    file_type,
    COUNT(*) as total_files,
    AVG(processing_time_ms) as avg_processing_time,
    SUM(components_extracted) as total_components
FROM parsed_files
WHERE parsing_status = 'success'
GROUP BY file_type
ORDER BY total_files DESC;
```

## 🚀 Performance Optimization

### Indexes

The schema includes optimized indexes for:

- **Spatial queries**: GIST indexes on geometry columns
- **Component searches**: B-tree indexes on names and types
- **Parser data**: GIN indexes on JSONB columns
- **Timestamps**: B-tree indexes for time-based queries

### Connection Pooling

The `NeonDBManager` uses connection pooling for optimal performance:

```python
# Configure pool size
await db_manager.create_pool(min_size=5, max_size=20)
```

### Query Optimization

- Use parameterized queries to prevent SQL injection
- Leverage the `component_summary` view for complex joins
- Use spatial indexes for location-based queries
- Implement pagination for large result sets

## 🔒 Security Considerations

### Environment Variables

- Store database credentials in environment variables
- Use `.env` files for local development
- Never commit credentials to version control

### Connection Security

- Use SSL connections to Neon
- Implement connection pooling to prevent connection exhaustion
- Use parameterized queries to prevent SQL injection

### Data Validation

- Validate all input data before database insertion
- Use appropriate data types and constraints
- Implement proper error handling

## 📈 Monitoring and Maintenance

### Database Statistics

```python
# Get comprehensive database stats
stats = await db_manager.get_database_stats()
print(json.dumps(stats, indent=2))
```

### Cleanup Operations

```python
# Clean up old parsed data (older than 30 days)
await db_manager.cleanup_old_data(days=30)
```

### Performance Monitoring

- Monitor query execution times
- Track connection pool usage
- Monitor disk space usage
- Set up alerts for failed parsing operations

## 🔗 Integration with Enhanced CAD Parsers

The Neon database integrates seamlessly with the enhanced CAD parsers:

1. **Parser Output**: Parsers generate structured JSON data
2. **Database Storage**: `NeonDBManager` stores parsed data efficiently
3. **Query Interface**: Rich query capabilities for analysis
4. **Spatial Support**: PostGIS integration for spatial queries

### Parser Integration Flow

```
CAD File → Parser → JSON Output → NeonDBManager → PostgreSQL
                                    ↓
                              Neo4j Aura (Graph)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify Neon credentials
   - Check network connectivity
   - Ensure SSL is enabled

2. **Performance Issues**
   - Monitor connection pool usage
   - Check query execution plans
   - Optimize indexes

3. **Data Migration Issues**
   - Verify CSV file format
   - Check data types compatibility
   - Review error logs

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Additional Resources

- [Neon PostgreSQL Documentation](https://neon.tech/docs)
- [PostGIS Documentation](https://postgis.net/documentation/)
- [Enhanced CAD Parser Documentation](../README.md)
- [Neo4j Aura Integration](../neo4j/README.md)

## 🤝 Contributing

1. Follow the existing code style
2. Add comprehensive tests
3. Update documentation
4. Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the main project LICENSE file for details. 