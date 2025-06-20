"""
PostgreSQL Training Database Configuration
Dedicated connection management for BEM DGL training database
Segregated from main system database for training isolation
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDatabaseConfig:
    """
    Dedicated configuration for BEM training database
    Handles connection, schema validation, and training data preparation
    """
    
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('TRAINING_DB_HOST', 'localhost'),
            'port': os.getenv('TRAINING_DB_PORT', '5432'),
            'database': os.getenv('TRAINING_DB_NAME', 'bem_training'),
            'user': os.getenv('TRAINING_DB_USER', 'bem_trainer'),
            'password': os.getenv('TRAINING_DB_PASSWORD', 'training_password')
        }
        
        # Main database connection params (for data sync if needed)
        self.main_db_params = {
            'host': os.getenv('MAIN_DB_HOST', 'localhost'),
            'port': os.getenv('MAIN_DB_PORT', '5432'),
            'database': os.getenv('MAIN_DB_NAME', 'bem_system'),
            'user': os.getenv('MAIN_DB_USER', 'bem_user'),
            'password': os.getenv('MAIN_DB_PASSWORD', 'main_password')
        }
        
        self.training_connection = None
        self.main_connection = None
    
    def connect_training_db(self) -> bool:
        """Connect to dedicated training database"""
        try:
            self.training_connection = psycopg2.connect(**self.connection_params)
            logger.info("✅ Connected to BEM training database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to training database: {e}")
            return False
    
    def connect_main_db(self) -> bool:
        """Connect to main system database (for data sync)"""
        try:
            self.main_connection = psycopg2.connect(**self.main_db_params)
            logger.info("✅ Connected to main BEM database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to main database: {e}")
            return False
    
    def initialize_training_schema(self) -> bool:
        """Initialize training database schema"""
        try:
            if not self.training_connection:
                if not self.connect_training_db():
                    return False
            
            # Read and execute schema file
            schema_path = os.path.join(os.path.dirname(__file__), 'training_database_schema.sql')
            
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            with self.training_connection.cursor() as cursor:
                cursor.execute(schema_sql)
                self.training_connection.commit()
            
            logger.info("✅ Training database schema initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize training schema: {e}")
            return False
    
    def get_training_data(self) -> Optional[Dict[str, Any]]:
        """Retrieve DGL training data from database"""
        try:
            if not self.training_connection:
                if not self.connect_training_db():
                    return None
            
            with self.training_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get training data using the database function
                cursor.execute("SELECT * FROM get_dgl_training_data();")
                result = cursor.fetchone()
                
                if result:
                    return dict(result)
                else:
                    logger.warning("No training data found in database")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to retrieve training data: {e}")
            return None
    
    def insert_training_run(self, run_name: str, model_config: Dict, training_params: Dict) -> Optional[str]:
        """Insert new training run record"""
        try:
            if not self.training_connection:
                if not self.connect_training_db():
                    return None
            
            with self.training_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO training_runs (run_name, model_config, training_params)
                    VALUES (%s, %s, %s)
                    RETURNING run_id::text
                """, (run_name, json.dumps(model_config), json.dumps(training_params)))
                
                run_id = cursor.fetchone()[0]
                self.training_connection.commit()
                
                logger.info(f"✅ Created training run: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"❌ Failed to insert training run: {e}")
            return None
    
    def insert_training_metric(self, run_id: str, epoch: int, metric_name: str, 
                              metric_value: float, metadata: Optional[Dict] = None):
        """Insert training metric for a specific run"""
        try:
            if not self.training_connection:
                if not self.connect_training_db():
                    return False
            
            with self.training_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO training_metrics (run_id, epoch, metric_name, metric_value, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (run_id, epoch, metric_name, metric_value, 
                     json.dumps(metadata) if metadata else None))
                
                self.training_connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to insert training metric: {e}")
            return False
    
    def save_model_embeddings(self, run_id: str, node_embeddings: Dict[str, List[float]]) -> bool:
        """Save learned node embeddings"""
        try:
            if not self.training_connection:
                if not self.connect_training_db():
                    return False
            
            with self.training_connection.cursor() as cursor:
                for node_id, embedding in node_embeddings.items():
                    cursor.execute("""
                        INSERT INTO model_embeddings (run_id, node_id, embedding_vector, embedding_dim)
                        VALUES (%s, %s, %s, %s)
                    """, (run_id, node_id, embedding, len(embedding)))
                
                self.training_connection.commit()
                logger.info(f"✅ Saved {len(node_embeddings)} node embeddings for run {run_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings: {e}")
            return False
    
    def update_training_status(self, run_id: str, status: str) -> bool:
        """Update training run status"""
        try:
            if not self.training_connection:
                if not self.connect_training_db():
                    return False
            
            with self.training_connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE training_runs 
                    SET status = %s 
                    WHERE run_id = %s
                """, (status, run_id))
                
                self.training_connection.commit()
                logger.info(f"✅ Updated training run {run_id} status to {status}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to update training status: {e}")
            return False
    
    def sync_from_main_database(self) -> bool:
        """Sync relevant data from main database to training database"""
        try:
            if not self.main_connection:
                if not self.connect_main_db():
                    return False
            
            if not self.training_connection:
                if not self.connect_training_db():
                    return False
            
            logger.info("🔄 Syncing data from main database to training database...")
            
            # Example sync - adapt based on your main database schema
            with self.main_connection.cursor(cursor_factory=RealDictCursor) as main_cursor:
                with self.training_connection.cursor() as training_cursor:
                    
                    # Sync nodes (example - adapt to your main schema)
                    main_cursor.execute("""
                        SELECT node_id, node_type, phase, properties as dictionary
                        FROM nodes 
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                    """)
                    
                    new_nodes = main_cursor.fetchall()
                    
                    for node in new_nodes:
                        training_cursor.execute("""
                            INSERT INTO nodes (node_id, node_type, phase, dictionary)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (node_id) DO UPDATE SET
                                node_type = EXCLUDED.node_type,
                                phase = EXCLUDED.phase,
                                dictionary = EXCLUDED.dictionary
                        """, (node['node_id'], node['node_type'], 
                             node['phase'], json.dumps(dict(node['dictionary']))))
                    
                    self.training_connection.commit()
                    logger.info(f"✅ Synced {len(new_nodes)} nodes from main database")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to sync from main database: {e}")
            return False
    
    def get_formula_training_readiness(self) -> List[Dict[str, Any]]:
        """Check which formulas are ready for training"""
        try:
            if not self.training_connection:
                if not self.connect_training_db():
                    return []
            
            with self.training_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM formula_training_readiness")
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"❌ Failed to get formula readiness: {e}")
            return []
    
    def close_connections(self):
        """Close all database connections"""
        if self.training_connection:
            self.training_connection.close()
            logger.info("🔒 Training database connection closed")
        
        if self.main_connection:
            self.main_connection.close()
            logger.info("🔒 Main database connection closed")

# Global instance for easy access
training_db = TrainingDatabaseConfig()

def get_training_db() -> TrainingDatabaseConfig:
    """Get global training database instance"""
    return training_db

def setup_training_environment() -> bool:
    """Complete setup of training environment"""
    logger.info("🚀 Setting up BEM training environment...")
    
    # Initialize database connections and schema
    if not training_db.connect_training_db():
        return False
    
    if not training_db.initialize_training_schema():
        return False
    
    # Optionally sync from main database
    if training_db.connect_main_db():
        training_db.sync_from_main_database()
    
    logger.info("✅ BEM training environment ready!")
    return True

if __name__ == "__main__":
    # Test setup
    if setup_training_environment():
        
        # Test data retrieval
        training_data = training_db.get_training_data()
        if training_data:
            logger.info(f"📊 Training data available: {training_data}")
        
        # Test formula readiness
        formula_readiness = training_db.get_formula_training_readiness()
        logger.info(f"🧮 Formula readiness: {len(formula_readiness)} formulas")
        
        # Cleanup
        training_db.close_connections()
    else:
        logger.error("❌ Failed to setup training environment") 