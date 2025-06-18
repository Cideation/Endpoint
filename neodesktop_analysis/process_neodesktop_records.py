#!/usr/bin/env python3
"""
Process NeoDesktop Records JSON File

This script processes the large JSON file containing graph database records
from NeoDesktop and provides various analysis and processing capabilities.
"""

import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
import re


class NeoDesktopProcessor:
    """Process NeoDesktop graph database records."""
    
    def __init__(self, json_file_path: str):
        """Initialize the processor with the JSON file path."""
        self.json_file_path = json_file_path
        self.records = []
        self.nodes = {}
        self.relationships = []
        
    def load_data(self) -> None:
        """Load the JSON data from file."""
        print(f"Loading data from {self.json_file_path}...")
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8-sig') as file:
                content = file.read()
            
            # Fix unescaped newlines in JSON strings
            # Replace unescaped newlines in string values with escaped newlines
            content = re.sub(r'"([^"]*)\n([^"]*)"', r'"\1\\n\2"', content)
            content = re.sub(r'"([^"]*)\r([^"]*)"', r'"\1\\r\2"', content)
            
            # Remove any remaining control characters
            content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
            
            self.records = json.loads(content)
            print(f"Successfully loaded {len(self.records)} records")
            
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            print("Trying alternative parsing method...")
            
            # Fallback: try to parse line by line
            try:
                self.records = []
                with open(self.json_file_path, 'r', encoding='utf-8-sig') as file:
                    lines = file.readlines()
                
                # Skip the first line (opening bracket) and last line (closing bracket)
                for line in lines[1:-1]:
                    line = line.strip()
                    if line.endswith(','):
                        line = line[:-1]
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            # Clean the line
                            line = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', line)
                            record = json.loads(line)
                            self.records.append(record)
                        except:
                            continue
                
                print(f"Successfully loaded {len(self.records)} records using line-by-line parsing")
                
            except Exception as e2:
                print(f"All parsing methods failed: {e2}")
                sys.exit(1)
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the data."""
        print("\n=== Data Structure Analysis ===")
        
        # Count different types of records
        relationship_records = [r for r in self.records if r.get('relationship_type')]
        node_only_records = [r for r in self.records if not r.get('relationship_type')]
        
        analysis = {
            'total_records': len(self.records),
            'relationship_records': len(relationship_records),
            'node_only_records': len(node_only_records),
            'unique_from_labels': set(),
            'unique_to_labels': set(),
            'unique_relationship_types': set(),
            'from_props_keys': set(),
            'to_props_keys': set(),
            'rel_props_keys': set()
        }
        
        # Collect unique labels and properties
        for record in self.records:
            if record.get('from_labels'):
                analysis['unique_from_labels'].update(record['from_labels'])
            if record.get('to_labels'):
                analysis['unique_to_labels'].update(record['to_labels'])
            if record.get('relationship_type'):
                analysis['unique_relationship_types'].add(record['relationship_type'])
            
            if record.get('from_props'):
                analysis['from_props_keys'].update(record['from_props'].keys())
            if record.get('to_props'):
                analysis['to_props_keys'].update(record['to_props'].keys())
            if record.get('rel_props'):
                analysis['rel_props_keys'].update(record['rel_props'].keys())
        
        # Convert sets to lists for JSON serialization
        analysis['unique_from_labels'] = list(analysis['unique_from_labels'])
        analysis['unique_to_labels'] = list(analysis['unique_to_labels'])
        analysis['unique_relationship_types'] = list(analysis['unique_relationship_types'])
        analysis['from_props_keys'] = list(analysis['from_props_keys'])
        analysis['to_props_keys'] = list(analysis['to_props_keys'])
        analysis['rel_props_keys'] = list(analysis['rel_props_keys'])
        
        # Print analysis
        print(f"Total records: {analysis['total_records']}")
        print(f"Relationship records: {analysis['relationship_records']}")
        print(f"Node-only records: {analysis['node_only_records']}")
        print(f"Unique from labels: {len(analysis['unique_from_labels'])}")
        print(f"Unique to labels: {len(analysis['unique_to_labels'])}")
        print(f"Unique relationship types: {len(analysis['unique_relationship_types'])}")
        
        return analysis
    
    def extract_nodes(self) -> Dict[int, Dict[str, Any]]:
        """Extract all unique nodes from the records."""
        print("\n=== Extracting Nodes ===")
        nodes = {}
        
        for record in self.records:
            # Extract from node
            if record.get('from_id') is not None:
                from_id = record['from_id']
                if from_id not in nodes:
                    nodes[from_id] = {
                        'id': from_id,
                        'labels': record.get('from_labels', []),
                        'properties': record.get('from_props', {})
                    }
            
            # Extract to node
            if record.get('to_id') is not None:
                to_id = record['to_id']
                if to_id not in nodes:
                    nodes[to_id] = {
                        'id': to_id,
                        'labels': record.get('to_labels', []),
                        'properties': record.get('to_props', {})
                    }
        
        print(f"Extracted {len(nodes)} unique nodes")
        return nodes
    
    def extract_relationships(self) -> List[Dict[str, Any]]:
        """Extract all relationships from the records."""
        print("\n=== Extracting Relationships ===")
        relationships = []
        
        for record in self.records:
            if record.get('relationship_type'):
                relationship = {
                    'from_id': record['from_id'],
                    'to_id': record['to_id'],
                    'type': record['relationship_type'],
                    'properties': record.get('rel_props', {})
                }
                relationships.append(relationship)
        
        print(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics about the data."""
        print("\n=== Generating Statistics ===")
        
        nodes = self.extract_nodes()
        relationships = self.extract_relationships()
        
        # Node statistics
        node_labels = Counter()
        for node in nodes.values():
            for label in node['labels']:
                node_labels[label] += 1
        
        # Relationship statistics
        rel_types = Counter()
        for rel in relationships:
            rel_types[rel['type']] += 1
        
        # Property statistics
        from_props = Counter()
        to_props = Counter()
        rel_props = Counter()
        
        for record in self.records:
            if record.get('from_props'):
                for key in record['from_props'].keys():
                    from_props[key] += 1
            if record.get('to_props'):
                for key in record['to_props'].keys():
                    to_props[key] += 1
            if record.get('rel_props'):
                for key in record['rel_props'].keys():
                    rel_props[key] += 1
        
        stats = {
            'total_nodes': len(nodes),
            'total_relationships': len(relationships),
            'node_labels_distribution': dict(node_labels),
            'relationship_types_distribution': dict(rel_types),
            'from_properties_distribution': dict(from_props),
            'to_properties_distribution': dict(to_props),
            'relationship_properties_distribution': dict(rel_props)
        }
        
        return stats
    
    def export_to_csv(self, output_dir: str = "output") -> None:
        """Export the data to CSV files for analysis."""
        print(f"\n=== Exporting to CSV in {output_dir} ===")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Extract data
        nodes = self.extract_nodes()
        relationships = self.extract_relationships()
        
        # Export nodes
        nodes_data = []
        for node in nodes.values():
            node_row = {
                'id': node['id'],
                'labels': ';'.join(node['labels']),
                **node['properties']
            }
            nodes_data.append(node_row)
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(f"{output_dir}/nodes.csv", index=False)
        print(f"Exported {len(nodes_data)} nodes to {output_dir}/nodes.csv")
        
        # Export relationships
        if relationships:
            rel_df = pd.DataFrame(relationships)
            rel_df.to_csv(f"{output_dir}/relationships.csv", index=False)
            print(f"Exported {len(relationships)} relationships to {output_dir}/relationships.csv")
        
        # Export summary statistics
        stats = self.generate_statistics()
        with open(f"{output_dir}/statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Exported statistics to {output_dir}/statistics.json")
    
    def find_connected_components(self) -> List[List[int]]:
        """Find connected components in the graph."""
        print("\n=== Finding Connected Components ===")
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for record in self.records:
            if record.get('relationship_type') and record.get('from_id') is not None and record.get('to_id') is not None:
                adjacency[record['from_id']].append(record['to_id'])
                adjacency[record['to_id']].append(record['from_id'])
        
        # DFS to find connected components
        visited = set()
        components = []
        
        def dfs(node_id, component):
            visited.add(node_id)
            component.append(node_id)
            for neighbor in adjacency[node_id]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node_id in adjacency:
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                components.append(component)
        
        print(f"Found {len(components)} connected components")
        for i, component in enumerate(components[:5]):  # Show first 5
            print(f"Component {i+1}: {len(component)} nodes")
        
        return components
    
    def run_full_analysis(self) -> None:
        """Run a complete analysis of the data."""
        print("=== NeoDesktop Records Analysis ===")
        
        # Load data
        self.load_data()
        
        # Analyze structure
        structure_analysis = self.analyze_structure()
        
        # Generate statistics
        stats = self.generate_statistics()
        
        # Find connected components
        components = self.find_connected_components()
        
        # Export to CSV
        self.export_to_csv()
        
        print("\n=== Analysis Complete ===")
        print("Check the 'output' directory for exported CSV files and statistics.")


def main():
    """Main function to run the analysis."""
    json_file = "records from neodesktop.json"
    
    if not Path(json_file).exists():
        print(f"Error: {json_file} not found in current directory")
        sys.exit(1)
    
    processor = NeoDesktopProcessor(json_file)
    processor.run_full_analysis()


if __name__ == "__main__":
    main() 