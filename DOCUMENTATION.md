# CAD Parser Documentation

## Overview
The CAD Parser is a powerful tool for processing and analyzing CAD files, supporting multiple formats and providing AI-powered cleaning capabilities. The system integrates with Neo4j for data storage and visualization.

## Supported File Formats
- **DXF** (`parse_dxf.py`): AutoCAD Drawing Exchange Format
- **DWG** (`dwg_cad_ifc_parser.py`): AutoCAD Drawing Database
- **IFC** (`parse_ifc.py`): Industry Foundation Classes
- **PDF** (`parse_pdf.py`): Portable Document Format

## Core Components

### File Parsers
- `parse_dxf.py`: Handles DXF file parsing and extraction
- `dwg_cad_ifc_parser.py`: Processes DWG and IFC files
- `parse_ifc.py`: Specialized IFC file parser
- `parse_pdf.py`: PDF file parser for CAD drawings

### Data Processing
- `normalize_keys.py`: Standardizes data keys across different formats
- `detect_scalars.py`: Identifies and processes scalar values in CAD data
- `openai_cleaner.py`: AI-powered data cleaning and standardization
- `generate_id.py` & `generate_ids.py`: Component ID generation utilities

### Database Integration
- `neo_writer.py`: Neo4j database writer for storing parsed data
- `transformer_main.py`: Data transformation pipeline

### Main Applications
- `app.py`: Flask API server
- `cad_parser_main.py`: Main CAD parsing application

## API Endpoints

### File Operations
- `POST /upload`: Upload CAD files (supports DXF, DWG, IFC, PDF)
- `POST /parse`: Parse and clean CAD data
- `POST /clean_with_ai`: AI-powered data cleaning

### Data Management
- `POST /evaluate_and_push`: Evaluate and push data to database
- `POST /push`: Direct push to Neo4j

## Web Interface
The web interface (`railway/index.html`) provides:
- File upload with drag-and-drop support
- Real-time parsing and cleaning
- AI-powered data cleaning
- Operation history tracking
- Direct Neo4j integration

## Deployment
The application is configured for deployment on Railway with:
- `railway.json`: Railway configuration
- `requirements.txt`: Python dependencies
- `Procfile`: Process management
- `render.yaml`: Render deployment configuration

## Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - `FLASK_ENV`: Development/Production mode
   - `NEO4J_URI`: Neo4j database URI
   - `NEO4J_USER`: Neo4j username
   - `NEO4J_PASSWORD`: Neo4j password
   - `OPENAI_API_KEY`: OpenAI API key for AI cleaning

## Usage Examples

### Basic File Parsing
```python
from parse_dxf import parse_dxf_file
result = parse_dxf_file("example.dxf")
```

### AI-Powered Cleaning
```python
from openai_cleaner import clean_with_ai
cleaned_data = clean_with_ai(raw_data)
```

### Neo4j Integration
```python
from neo_writer import write_to_neo4j
write_to_neo4j(parsed_data)
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details. 