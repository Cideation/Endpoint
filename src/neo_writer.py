from py2neo import Graph, Node, Relationship
import os
from datetime import datetime

def get_neo4j_connection():
    """Get Neo4j connection"""
    # Get Neo4j credentials from environment variables
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
    
    try:
        graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        return graph
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def write_to_neo4j(data):
    """Write enhanced data to Neo4j database with relationships"""
    try:
        graph = get_neo4j_connection()
        if not graph:
            return {"error": "Could not connect to Neo4j"}
        
        # Create component node with enhanced properties
        component = Node("Component",
                        component_id=data.get('component_id'),
                        name=data.get('name', 'Unknown'),
                        shape=data.get('shape', 'Unknown'),
                        material=data.get('material', 'Unknown'),
                        quantity=data.get('quantity', 1),
                        estimated_cost=data.get('estimated_cost', 0.0),
                        quality_score=data.get('quality_score', 0.0),
                        created_at=datetime.now().isoformat())
        
        # Create material node
        material = Node("Material", 
                       name=data.get('material', 'Unknown'),
                       type='construction')
        
        # Create relationships
        component_material = Relationship(component, "MADE_OF", material)
        
        # Create cost analysis node
        cost_analysis = Node("CostAnalysis",
                            component_id=data.get('component_id'),
                            estimated_cost=data.get('estimated_cost', 0.0),
                            quantity=data.get('quantity', 1),
                            total_cost=data.get('estimated_cost', 0.0) * data.get('quantity', 1),
                            analyzed_at=datetime.now().isoformat())
        
        cost_relationship = Relationship(component, "HAS_COST", cost_analysis)
        
        # Create all nodes and relationships
        graph.create(component)
        graph.create(material)
        graph.create(component_material)
        graph.create(cost_analysis)
        graph.create(cost_relationship)
        
        return {
            "message": "Enhanced data written to Neo4j successfully", 
            "data": data,
            "relationships_created": ["MADE_OF", "HAS_COST"],
            "nodes_created": ["Component", "Material", "CostAnalysis"]
        }
    except Exception as e:
        return {"error": f"Failed to write to Neo4j: {str(e)}"}

def push_to_neo4j(data):
    """Push enhanced data to Neo4j database with batch processing"""
    try:
        graph = get_neo4j_connection()
        if not graph:
            return {"error": "Could not connect to Neo4j"}
        
        created_nodes = []
        created_relationships = []
        
        # Process multiple components
        for item in data:
            # Create component node
            component = Node("Component",
                            component_id=item.get('component_id'),
                            name=item.get('name', 'Unknown'),
                            shape=item.get('shape', 'Unknown'),
                            material=item.get('material', 'Unknown'),
                            quantity=item.get('quantity', 1),
                            estimated_cost=item.get('estimated_cost', 0.0),
                            quality_score=item.get('quality_score', 0.0),
                            created_at=datetime.now().isoformat())
            
            # Create material node
            material = Node("Material", 
                           name=item.get('material', 'Unknown'),
                           type='construction')
            
            # Create relationships
            component_material = Relationship(component, "MADE_OF", material)
            
            # Create cost analysis
            cost_analysis = Node("CostAnalysis",
                                component_id=item.get('component_id'),
                                estimated_cost=item.get('estimated_cost', 0.0),
                                quantity=item.get('quantity', 1),
                                total_cost=item.get('estimated_cost', 0.0) * item.get('quantity', 1),
                                analyzed_at=datetime.now().isoformat())
            
            cost_relationship = Relationship(component, "HAS_COST", cost_analysis)
            
            # Create all nodes and relationships
            graph.create(component)
            graph.create(material)
            graph.create(component_material)
            graph.create(cost_analysis)
            graph.create(cost_relationship)
            
            created_nodes.extend([component, material, cost_analysis])
            created_relationships.extend([component_material, cost_relationship])
        
        return {
            "message": f"Successfully pushed {len(data)} enhanced components to Neo4j",
            "nodes_created": len(created_nodes),
            "relationships_created": len(created_relationships),
            "components_processed": len(data)
        }
    except Exception as e:
        return {"error": f"Failed to push to Neo4j: {str(e)}"} 