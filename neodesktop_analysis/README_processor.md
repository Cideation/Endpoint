# NeoDesktop Records Processor

This Python script processes large JSON files containing graph database records from NeoDesktop and provides comprehensive analysis capabilities.

## Features

- **Data Loading**: Efficiently loads large JSON files (140K+ records)
- **Structure Analysis**: Analyzes the structure of graph data
- **Node Extraction**: Extracts unique nodes with their labels and properties
- **Relationship Analysis**: Processes relationships between nodes
- **Statistics Generation**: Provides comprehensive statistics about the data
- **CSV Export**: Exports data to CSV files for further analysis
- **Connected Components**: Finds connected components in the graph
- **Graph Analysis**: Performs various graph analysis tasks

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_processor.txt
```

## Usage

### Basic Usage

Run the script to perform a complete analysis:

```bash
python process_neodesktop_records.py
```

This will:
- Load the `records from neodesktop.json` file
- Perform structure analysis
- Generate statistics
- Find connected components
- Export data to CSV files in an `output/` directory

### Programmatic Usage

You can also use the `NeoDesktopProcessor` class in your own code:

```python
from process_neodesktop_records import NeoDesktopProcessor

# Initialize processor
processor = NeoDesktopProcessor("records from neodesktop.json")

# Load data
processor.load_data()

# Analyze structure
analysis = processor.analyze_structure()

# Extract nodes and relationships
nodes = processor.extract_nodes()
relationships = processor.extract_relationships()

# Generate statistics
stats = processor.generate_statistics()

# Export to CSV
processor.export_to_csv("my_output_directory")
```

## Output Files

The script generates several output files in the `output/` directory:

- **`nodes.csv`**: All unique nodes with their properties
- **`relationships.csv`**: All relationships between nodes
- **`statistics.json`**: Comprehensive statistics about the data

## Data Structure

The JSON file contains records with the following structure:

```json
{
  "from_labels": ["Label1", "Label2"],
  "from_props": {"property1": "value1"},
  "relationship_type": "RELATIONSHIP_TYPE",
  "rel_props": {"rel_property": "value"},
  "to_labels": ["Label3"],
  "to_props": {"property2": "value2"},
  "from_id": 0,
  "to_id": 1
}
```

## Analysis Capabilities

### Structure Analysis
- Counts total records, relationship records, and node-only records
- Identifies unique labels and relationship types
- Analyzes property distributions

### Node Analysis
- Extracts unique nodes from the graph
- Categorizes nodes by their labels
- Analyzes node properties

### Relationship Analysis
- Processes all relationships between nodes
- Categorizes relationships by type
- Analyzes relationship properties

### Graph Analysis
- Finds connected components in the graph
- Identifies isolated nodes
- Analyzes graph connectivity

## Performance

The script is optimized to handle large datasets:
- Memory-efficient processing
- Streaming JSON parsing for very large files
- Progress reporting for long operations

## Error Handling

The script includes comprehensive error handling:
- File not found errors
- JSON parsing errors
- Memory errors for very large files
- Graceful degradation for missing data

## Customization

You can easily customize the script by:
- Modifying the output directory
- Adding new analysis methods
- Changing the CSV export format
- Adding new graph analysis algorithms

## Dependencies

- **pandas**: For data manipulation and CSV export
- **numpy**: For numerical operations
- **json**: For JSON parsing (built-in)
- **collections**: For data structures (built-in)
- **pathlib**: For file path handling (built-in) 