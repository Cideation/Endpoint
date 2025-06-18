# NeoDesktop Analysis

This folder contains tools and data for analyzing NeoDesktop graph database records.

## Contents

- `records from neodesktop.json` - The original graph database export (140K+ records)
- `process_neodesktop_records.py` - Main processing script
- `requirements_processor.txt` - Python dependencies
- `README_processor.md` - Detailed documentation for the processor

## Quick Start

1. Navigate to this folder:
```bash
cd neodesktop_analysis
```

2. Install dependencies:
```bash
pip install -r requirements_processor.txt
```

3. Run the analysis:
```bash
python process_neodesktop_records.py
```

## What it does

The processor will:
- Load and analyze the large JSON file
- Extract nodes and relationships
- Generate comprehensive statistics
- Export data to CSV files
- Find connected components in the graph

## Output

The script creates an `output/` folder with:
- `nodes.csv` - All unique nodes
- `relationships.csv` - All relationships
- `statistics.json` - Analysis statistics

## Data Structure

The JSON contains graph database records with:
- Node labels and properties
- Relationship types and properties
- Node IDs for connections

See `README_processor.md` for detailed documentation. 