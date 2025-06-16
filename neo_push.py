from py2neo import Graph
import os

def push_to_neo(cleaned_data):
    graph = Graph(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
    query = "CREATE (n:AgentData {data: $data})"
    graph.run(query, data=cleaned_data)
