# Neon PostgreSQL Integration

This directory contains the Neon PostgreSQL integration for the CAD Parser system.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CAD Parser    │───▶│  Neon PostgreSQL │───▶│  Enhanced       │
│   Application   │    │  (Primary DB)    │    │  Schema (132    │
│                 │    │                  │    │  Tables)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Database Schema

### Enhanced Schema Features
- **132 enhanced tables** with standardized field names
- **Proper data types** (INTEGER, DOUBLE PRECISION, BOOLEAN, TEXT)
- **Audit trails** (created_at, updated_at)
- **Performance indexes** for fast querying
- **Analytics views** for business intelligence

### Key Tables
- `components` - Main component storage
- `component_hierarchy` - Parent-child relationships
- `material_variants` - Material variant management
- `manufacturing_methods` - Manufacturing capabilities
- `spatial_references` - PostGIS spatial data

### Analytics Views
- `component_analytics` - Component analysis
- `supply_chain_analytics` - Supply chain insights

## Configuration

### Database Connection
```python
DB_CONFIG = {
    'host': 'ep-white-waterfall-a85g0dgx-pooler.eastus2.azure.neon.tech',
    'port': '5432',
    'database': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_CcgA0kKeYVU2',
    'sslmode': 'require'
}
```

### Environment Variables
- `DB_HOST` - PostgreSQL host
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name
- `DB_USER` - Database username
- `DB_PASSWORD` - Database password

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Schema**
   ```bash
   python schema_enrichment.py
   ```

3. **Test Connection**
   ```bash
   python -c "from config import DB_CONFIG; import psycopg2; conn = psycopg2.connect(**DB_CONFIG); print('Connected!')"
   ```

## Migration Process

### CSV Migration
The system supports migration from CSV files with:
- **Content-based deduplication**
- **Schema harmonization**
- **Smart merging**
- **Batch processing**

### Schema Enrichment
The schema enrichment process:
1. Loads property mapping data
2. Analyzes schema structure
3. Maps types to PostgreSQL types
4. Generates enhanced schema SQL
5. Applies schema to database

## Performance Optimization

### Database Features
- **Automatic indexes** for common queries
- **Materialized views** for performance
- **Connection pooling** for efficiency
- **SSL encryption** for security

### Query Optimization
- **Indexed foreign keys**
- **Composite indexes** for complex queries
- **Partitioning** for large tables
- **Query caching** for repeated operations

## Monitoring and Maintenance

### Health Checks
- Database connectivity monitoring
- Query performance tracking
- Connection pool status
- Schema validation

### Backup and Recovery
- Automated backups
- Point-in-time recovery
- Schema versioning
- Data integrity checks

## Security

### Data Protection
- **SSL/TLS encryption** for all connections
- **Credential management** via environment variables
- **Input validation** and sanitization
- **Audit logging** for all operations

### Access Control
- **Role-based permissions**
- **Connection limiting**
- **Query timeout protection**
- **SQL injection prevention**

## Troubleshooting

### Common Issues
1. **Connection Timeout**: Check network connectivity and credentials
2. **Schema Errors**: Verify schema enrichment process completed successfully
3. **Performance Issues**: Check indexes and query optimization
4. **Permission Errors**: Verify user permissions and role assignments

### Debug Mode
Enable debug logging for detailed error information:
```python
logging.basicConfig(level=logging.DEBUG)
```

## API Integration

### Database Operations
```python
from config import DB_CONFIG
import psycopg2

# Connect to database
conn = psycopg2.connect(**DB_CONFIG)

# Execute queries
cursor = conn.cursor()
cursor.execute("SELECT * FROM components LIMIT 10")
results = cursor.fetchall()

# Close connection
cursor.close()
conn.close()
```

### Component Storage
```python
def store_component(component_data):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO components (component_id, component_type, properties, geometry, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (component_id) DO UPDATE SET
            component_type = EXCLUDED.component_type,
            properties = EXCLUDED.properties,
            geometry = EXCLUDED.geometry,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP
    """, (
        component_data['component_id'],
        component_data['component_type'],
        json.dumps(component_data['properties']),
        json.dumps(component_data['geometry']),
        json.dumps(component_data['metadata'])
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
```

## Future Enhancements

### Planned Features
- **Real-time replication** for high availability
- **Advanced analytics** with machine learning
- **Graph-like queries** using PostgreSQL's recursive CTEs
- **Spatial indexing** for 3D geometry
- **Time-series analysis** for component lifecycle

### Performance Improvements
- **Query optimization** with execution plan analysis
- **Connection pooling** improvements
- **Caching layer** for frequently accessed data
- **Parallel processing** for bulk operations

## Support

For issues and questions:
- Check the main documentation in `../DOCUMENTATION.md`
- Review the schema enrichment logs
- Test with the provided examples
- Contact the development team 