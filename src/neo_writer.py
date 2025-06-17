from py2neo import Graph, Node, Relationship
import os

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
    """Write data to Neo4j database"""
    try:
        graph = get_neo4j_connection()
        if not graph:
            return {"error": "Could not connect to Neo4j"}
        
        # Create a component node
        component = Node("Component",
                        component_id=data.get('component_id'),
                        quantity=data.get('quantity', 1),
                        estimated_cost=data.get('estimated_cost', 0.0))
        
        graph.create(component)
        
        return {"message": "Data written to Neo4j successfully", "data": data}
    except Exception as e:
        return {"error": f"Failed to write to Neo4j: {str(e)}"}

def push_to_neo4j(data):
    """Push data to Neo4j database"""
    try:
        graph = get_neo4j_connection()
        if not graph:
            return {"error": "Could not connect to Neo4j"}
        
        # Process multiple components
        for item in data:
            component = Node("Component",
                            component_id=item.get('component_id'),
                            name=item.get('name'),
                            shape=item.get('shape'),
                            quantity=item.get('quantity', 1),
                            estimated_cost=item.get('estimated_cost', 0.0))
            
            graph.create(component)
        
        return {"message": f"Successfully pushed {len(data)} components to Neo4j"}
    except Exception as e:
        return {"error": f"Failed to push to Neo4j: {str(e)}"} 