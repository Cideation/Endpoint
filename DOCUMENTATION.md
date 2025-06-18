# CAD Parser Documentation

## Overview
Enhanced CAD/BIM file parser with PostgreSQL integration capabilities. The system integrates with PostgreSQL for data storage and processing.

## Features
- **Multi-format Support**: DWG, IFC, DXF, PDF, OBJ, STEP files
- **Advanced Parsing**: Geometry extraction, property analysis, metadata processing
- **AI Integration**: OpenAI-powered data cleaning and validation
- **PostgreSQL Storage**: Robust database integration with Neon PostgreSQL
- **Real-time Processing**: Parallel processing for large files
- **Advanced Analytics**: Volume calculations, spatial analysis, component relationships

## Architecture

### Core Components
- `dwg_cad_ifc_parser.py`: Main parser for DWG and IFC files
- `parse_dxf.py`: DXF file parser with advanced features
- `parse_pdf.py`: PDF content extraction and analysis
- `openai_cleaner.py`: AI-powered data cleaning and validation
- `generate_ids.py`: Unique ID generation for components
- `normalize_keys.py`: Data normalization utilities

### Database Integration
- `postgre/`: PostgreSQL integration and schema management
- `neon/`: Neon PostgreSQL specific configurations
- Enhanced schema with 132 tables and analytics views

## API Endpoints

### File Processing
- `POST /parse`: Parse CAD/BIM files
- `POST /push`: Push data to PostgreSQL
- `POST /push_enhanced`: Push with AI cleaning

### Data Retrieval
- `GET /db_data`: Get database data
- `GET /health`: Health check
- `GET /test`: Test endpoint

## Database Schema

### Enhanced Tables
- **132 enhanced tables** with standardized field names
- **Proper data types** (INTEGER, DOUBLE PRECISION, BOOLEAN, TEXT)
- **Audit trails** (created_at, updated_at)
- **Performance indexes** for fast querying

### Key Tables
- `components`: Main component storage
- `component_hierarchy`: Parent-child relationships
- `material_variants`: Material variant management
- `manufacturing_methods`: Manufacturing capabilities
- `spatial_references`: PostGIS spatial data

### Analytics Views
- `component_analytics`: Component analysis
- `supply_chain_analytics`: Supply chain insights

## Environment Variables

### Database Configuration
- `DB_HOST`: PostgreSQL host
- `DB_PORT`: PostgreSQL port (default: 5432)
- `DB_NAME`: Database name
- `DB_USER`: Database username
- `DB_PASSWORD`: Database password

### OpenAI Configuration
- `OPENAI_API_KEY`: OpenAI API key for data cleaning

## Usage Examples

### Basic Parsing
```python
from dwg_cad_ifc_parser import parse_dwg_file

result = parse_dwg_file('path/to/file.dwg')
print(result)
```

### Database Integration
```python
from postgre.config import DB_CONFIG
import psycopg2

conn = psycopg2.connect(**DB_CONFIG)
# Use connection for database operations
```

### AI Data Cleaning
```python
from openai_cleaner import clean_data_with_ai

cleaned_data = clean_data_with_ai(raw_data)
```

## Installation

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL**:
   - Configure Neon PostgreSQL connection
   - Run schema enrichment: `python postgre/schema_enrichment.py`

4. **Configure environment variables**:
   - Set database credentials
   - Configure OpenAI API key

5. **Run the application**:
   ```bash
   python src/app.py
   ```

## Development

### Project Structure
```
├── src/                    # Main application
│   ├── app.py             # Flask application
│   ├── static/            # Frontend assets
│   └── templates/         # HTML templates
├── postgre/               # PostgreSQL integration
│   ├── schema_enrichment.py
│   ├── config.py
│   └── enhanced_schema.sql
├── neon/                  # Neon PostgreSQL config
├── parsers/               # File parsers
└── requirements.txt       # Dependencies
```

### Adding New Parsers
1. Create parser function in appropriate module
2. Add file type detection in `app.py`
3. Update frontend to support new format
4. Test with sample files

### Database Schema Updates
1. Modify `schema_enrichment.py`
2. Run schema update: `python postgre/schema_enrichment.py`
3. Test with sample data

## Performance Optimization

### Database
- **Indexes**: Automatic index creation for common queries
- **Materialized Views**: Pre-computed analytics for performance
- **Connection Pooling**: Efficient database connections

### File Processing
- **Parallel Processing**: Multi-threaded parsing for large files
- **Batch Operations**: Efficient database writes
- **Memory Management**: Optimized for large file handling

## Monitoring and Logging

### Health Checks
- Database connectivity monitoring
- API endpoint status
- File processing metrics

### Logging
- Structured logging with timestamps
- Error tracking and debugging
- Performance metrics

## Security

### Data Protection
- Secure database connections with SSL
- Input validation and sanitization
- API rate limiting

### Access Control
- Environment-based configuration
- Secure credential management
- Audit trail logging

## Troubleshooting

### Common Issues
1. **Database Connection**: Check credentials and network
2. **File Parsing**: Verify file format and size
3. **AI Integration**: Ensure OpenAI API key is valid

### Debug Mode
Enable debug logging for detailed error information:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is licensed under the MIT License. 