from neo4j import GraphDatabase
import os

# Use environment variables for cloud deployment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def push_to_neo4j(data):
    with driver.session() as session:
        for item in data:
            session.run(
                """
                MERGE (c:Component {component_id: $component_id})
                SET c.name = $name,
                    c.shape = $shape,
                    c.node_id = $node_id
                """,
                component_id=item['component_id'],
                name=item['name'],
                shape=item['shape'],
                node_id=item['node_id']
            )
    print(f"[Neo4j] Pushed {len(data)} components.")
