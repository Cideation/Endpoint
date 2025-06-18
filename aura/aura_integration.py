#!/usr/bin/env python3
"""
Neo4j Aura Integration - ETL/Sync Engine
Connects Neon PostgreSQL to Neo4j Aura for graph processing
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
from neo4j import GraphDatabase
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuraIntegration:
    def __init__(self, neon_config: Dict[str, str], aura_config: Dict[str, str]):
        self.neon_config = neon_config
        self.aura_config = aura_config
        self.neon_conn = None
        self.aura_driver = None
        
    def connect_neon(self):
        """Connect to Neon PostgreSQL"""
        try:
            self.neon_conn = psycopg2.connect(**self.neon_config)
            logger.info("✅ Connected to Neon PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neon: {e}")
            return False
    
    def connect_aura(self):
        """Connect to Neo4j Aura"""
        try:
            uri = self.aura_config['uri']
            username = self.aura_config['username']
            password = self.aura_config['password']
            
            self.aura_driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            with self.aura_driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("✅ Connected to Neo4j Aura")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Aura: {e}")
            return False
    
    def read_from_neon(self, table_name: str, limit: int = None) -> pd.DataFrame:
        """Read data from Neon PostgreSQL"""
        try:
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, self.neon_conn)
            logger.info(f"📖 Read {len(df)} rows from {table_name}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to read from {table_name}: {e}")
            return pd.DataFrame()
    
    def build_cypher_nodes(self, df: pd.DataFrame, node_label: str, id_field: str) -> List[str]:
        """Build Cypher queries for creating nodes"""
        cypher_queries = []
        
        for _, row in df.iterrows():
            # Convert row to dict and handle NaN values
            properties = {}
            for col, value in row.items():
                if pd.notna(value):
                    if isinstance(value, (int, float)):
                        properties[col] = value
                    else:
                        properties[col] = str(value)
            
            # Build Cypher query
            props_str = ', '.join([f"{k}: ${k}" for k in properties.keys()])
            query = f"""
            MERGE (n:{node_label} {{{id_field}: ${id_field}}})
            SET n += {{{props_str}}}
            """
            cypher_queries.append((query, properties))
        
        return cypher_queries
    
    def build_cypher_relationships(self, df: pd.DataFrame, 
                                 source_label: str, target_label: str,
                                 relationship_type: str,
                                 source_id_field: str, target_id_field: str) -> List[str]:
        """Build Cypher queries for creating relationships"""
        cypher_queries = []
        
        for _, row in df.iterrows():
            source_id = row[source_id_field]
            target_id = row[target_id_field]
            
            if pd.notna(source_id) and pd.notna(target_id):
                # Build relationship properties
                properties = {}
                for col, value in row.items():
                    if col not in [source_id_field, target_id_field] and pd.notna(value):
                        if isinstance(value, (int, float)):
                            properties[col] = value
                        else:
                            properties[col] = str(value)
                
                props_str = ', '.join([f"{k}: ${k}" for k in properties.keys()]) if properties else ""
                rel_props = f"{{{props_str}}}" if props_str else ""
                
                query = f"""
                MATCH (a:{source_label} {{{source_id_field}: $source_id}})
                MATCH (b:{target_label} {{{target_id_field}: $target_id}})
                MERGE (a)-[r:{relationship_type} {rel_props}]->(b)
                """
                
                params = {
                    'source_id': source_id,
                    'target_id': target_id,
                    **properties
                }
                cypher_queries.append((query, params))
        
        return cypher_queries
    
    def execute_cypher_batch(self, queries: List[tuple], batch_size: int = 100):
        """Execute Cypher queries in batches"""
        if not self.aura_driver:
            logger.error("❌ Aura driver not connected")
            return False
        
        try:
            with self.aura_driver.session() as session:
                total_queries = len(queries)
                for i in range(0, total_queries, batch_size):
                    batch = queries[i:i + batch_size]
                    
                    # Execute batch
                    for query, params in batch:
                        session.run(query, params)
                    
                    logger.info(f"✅ Executed batch {i//batch_size + 1}/{(total_queries + batch_size - 1)//batch_size}")
                    
            logger.info(f"✅ Successfully executed {total_queries} Cypher queries")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to execute Cypher batch: {e}")
            return False
    
    def sync_component_data(self):
        """Sync component data from Neon to Aura"""
        logger.info("🔄 Starting component data sync...")
        
        # Read component data from Neon
        components_df = self.read_from_neon('component_enhanced')
        if components_df.empty:
            logger.error("❌ No component data found")
            return False
        
        # Build node creation queries
        node_queries = self.build_cypher_nodes(
            components_df, 
            'Component', 
            'component_id'
        )
        
        # Execute node creation
        if self.execute_cypher_batch(node_queries):
            logger.info(f"✅ Created {len(node_queries)} Component nodes")
            return True
        return False
    
    def sync_relationships(self):
        """Sync relationship data from Neon to Aura"""
        logger.info("🔄 Starting relationship sync...")
        
        # Read relationship data
        hierarchy_df = self.read_from_neon('component_hierarchy')
        if not hierarchy_df.empty:
            rel_queries = self.build_cypher_relationships(
                hierarchy_df,
                'Component', 'Component',
                'HAS_CHILD',
                'parent_component_id', 'child_component_id'
            )
            self.execute_cypher_batch(rel_queries)
            logger.info(f"✅ Created {len(rel_queries)} hierarchy relationships")
        
        # Add more relationship types as needed
        # Example: material relationships, supplier relationships, etc.
    
    def create_graph_indexes(self):
        """Create indexes for better graph performance"""
        index_queries = [
            "CREATE INDEX component_id_index IF NOT EXISTS FOR (c:Component) ON (c.component_id)",
            "CREATE INDEX material_id_index IF NOT EXISTS FOR (m:Material) ON (m.material_id)",
            "CREATE INDEX supplier_id_index IF NOT EXISTS FOR (s:Supplier) ON (s.supplier_id)",
        ]
        
        try:
            with self.aura_driver.session() as session:
                for query in index_queries:
                    session.run(query)
            logger.info("✅ Created graph indexes")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
            return False
    
    def run_graph_analytics(self) -> Dict[str, Any]:
        """Run graph analytics and return results"""
        analytics_queries = {
            "component_count": "MATCH (c:Component) RETURN count(c) as count",
            "relationship_count": "MATCH ()-[r]-() RETURN count(r) as count",
            "component_degrees": """
            MATCH (c:Component)
            RETURN c.component_id as component_id, 
                   size((c)-[]-()) as degree
            ORDER BY degree DESC
            LIMIT 10
            """,
            "connected_components": """
            CALL gds.weaklyConnectedComponents.stream('component-graph')
            YIELD nodeId, componentId
            RETURN componentId, count(nodeId) as size
            ORDER BY size DESC
            """
        }
        
        results = {}
        try:
            with self.aura_driver.session() as session:
                for name, query in analytics_queries.items():
                    result = session.run(query)
                    if name in ["component_count", "relationship_count"]:
                        results[name] = result.single()[0]
                    else:
                        results[name] = [dict(record) for record in result]
            
            logger.info("✅ Graph analytics completed")
            return results
        except Exception as e:
            logger.error(f"❌ Failed to run analytics: {e}")
            return {}
    
    def write_results_to_neon(self, results: Dict[str, Any]):
        """Write graph analysis results back to Neon"""
        try:
            cursor = self.neon_conn.cursor()
            
            # Create results table if it doesn't exist
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS graph_edge_results (
                id SERIAL PRIMARY KEY,
                analysis_type TEXT,
                component_id TEXT,
                edge_weight DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            
            # Insert results
            for analysis_type, data in results.items():
                if isinstance(data, list):
                    for item in data:
                        cursor.execute("""
                        INSERT INTO graph_edge_results 
                        (analysis_type, component_id, edge_weight, confidence, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        """, (
                            analysis_type,
                            item.get('component_id'),
                            item.get('degree', 0),
                            item.get('confidence', 1.0),
                            json.dumps(item)
                        ))
                else:
                    cursor.execute("""
                    INSERT INTO graph_edge_results 
                    (analysis_type, component_id, edge_weight, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """, (
                        analysis_type,
                        'summary',
                        float(data),
                        1.0,
                        json.dumps({'value': data})
                    ))
            
            self.neon_conn.commit()
            logger.info("✅ Results written back to Neon")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to write results: {e}")
            self.neon_conn.rollback()
            return False
    
    def full_sync_pipeline(self):
        """Run the complete sync pipeline"""
        logger.info("🚀 Starting full sync pipeline...")
        
        # 1. Connect to both databases
        if not self.connect_neon() or not self.connect_aura():
            return False
        
        # 2. Create indexes
        self.create_graph_indexes()
        
        # 3. Sync component data
        if not self.sync_component_data():
            return False
        
        # 4. Sync relationships
        self.sync_relationships()
        
        # 5. Run graph analytics
        results = self.run_graph_analytics()
        
        # 6. Write results back to Neon
        if results:
            self.write_results_to_neon(results)
        
        logger.info("🎉 Full sync pipeline completed!")
        return True
    
    def close_connections(self):
        """Close database connections"""
        if self.neon_conn:
            self.neon_conn.close()
        if self.aura_driver:
            self.aura_driver.close()
        logger.info("🔌 Database connections closed")

def main():
    # Load configurations
    try:
        from aura_config import AURA_CONFIG
        from postgre.config import NEON_CONFIG
        
        # Initialize integration
        integration = AuraIntegration(NEON_CONFIG, AURA_CONFIG)
        
        # Run full pipeline
        success = integration.full_sync_pipeline()
        
        if success:
            logger.info("✅ Integration completed successfully!")
        else:
            logger.error("❌ Integration failed")
        
        # Clean up
        integration.close_connections()
        
    except ImportError as e:
        logger.error(f"❌ Config file not found: {e}")
    except Exception as e:
        logger.error(f"❌ Integration error: {e}")

if __name__ == "__main__":
    main() 