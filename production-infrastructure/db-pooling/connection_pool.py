#!/usr/bin/env python3
"""
Production Database Connection Pooling System
Provides optimized connection management, monitoring, and performance tuning
"""

import os
import time
import logging
import threading
import psycopg2
from psycopg2 import pool, extras
from contextlib import contextmanager
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class PoolConfig:
    """Database pool configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    idle_timeout: int = 300  # 5 minutes
    max_lifetime: int = 3600  # 1 hour
    retry_attempts: int = 3
    retry_delay: int = 5

class DatabaseConnectionPool:
    """Production-grade database connection pool"""
    
    def __init__(self, config: PoolConfig):
        self.config = config
        self.pool = None
        self.stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'connections_active': 0,
            'connections_idle': 0,
            'queries_executed': 0,
            'query_errors': 0,
            'pool_exhausted': 0,
            'average_query_time': 0.0,
            'last_health_check': None
        }
        self.query_times = []
        self.lock = threading.Lock()
        
        self._create_pool()
        self._start_health_monitor()
    
    def _create_pool(self):
        """Create database connection pool"""
        try:
            # Build connection string
            conn_string = f"host={self.config.host} " \
                         f"port={self.config.port} " \
                         f"dbname={self.config.database} " \
                         f"user={self.config.user} " \
                         f"password={self.config.password}"
            
            # Create connection pool
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                dsn=conn_string,
                connection_factory=None,
                cursor_factory=extras.RealDictCursor
            )
            
            # Test pool creation
            test_conn = self.pool.getconn()
            if test_conn:
                self.pool.putconn(test_conn)
                logger.info(f"Database pool created: {self.config.min_connections}-{self.config.max_connections} connections")
            
            self.stats['connections_created'] = self.config.min_connections
            
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic return"""
        connection = None
        start_time = time.time()
        
        try:
            # Get connection from pool
            connection = self.pool.getconn()
            
            if connection is None:
                self.stats['pool_exhausted'] += 1
                raise Exception("No connections available in pool")
            
            # Set connection properties
            connection.autocommit = False
            
            with self.lock:
                self.stats['connections_active'] += 1
            
            yield connection
            
        except psycopg2.Error as e:
            if connection:
                connection.rollback()
            self.stats['query_errors'] += 1
            logger.error(f"Database error: {e}")
            raise
            
        except Exception as e:
            if connection:
                connection.rollback()
            self.stats['query_errors'] += 1
            logger.error(f"Connection error: {e}")
            raise
            
        finally:
            if connection:
                # Return connection to pool
                self.pool.putconn(connection)
                
                with self.lock:
                    self.stats['connections_active'] -= 1
                
                # Track query time
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                
                # Keep only last 1000 query times
                if len(self.query_times) > 1000:
                    self.query_times = self.query_times[-1000:]
                
                # Update average query time
                self.stats['average_query_time'] = sum(self.query_times) / len(self.query_times)
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = True) -> Any:
        """Execute query with connection management"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                
                self.stats['queries_executed'] += 1
                
                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                else:
                    return cursor.rowcount
    
    def execute_transaction(self, queries: List[Dict]) -> List[Any]:
        """Execute multiple queries in a transaction"""
        results = []
        
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    for query_info in queries:
                        query = query_info['query']
                        params = query_info.get('params')
                        
                        cursor.execute(query, params)
                        
                        if query_info.get('fetch_one'):
                            results.append(cursor.fetchone())
                        elif query_info.get('fetch_all', True):
                            results.append(cursor.fetchall())
                        else:
                            results.append(cursor.rowcount)
                
                conn.commit()
                self.stats['queries_executed'] += len(queries)
                
            except Exception as e:
                conn.rollback()
                self.stats['query_errors'] += 1
                raise
        
        return results
    
    def health_check(self) -> Dict:
        """Perform health check on pool"""
        health_status = {
            'healthy': False,
            'timestamp': datetime.now().isoformat(),
            'connections': {
                'total': 0,
                'active': self.stats['connections_active'],
                'idle': 0,
                'available': 0
            },
            'performance': {
                'queries_per_second': 0.0,
                'average_query_time': self.stats['average_query_time'],
                'error_rate': 0.0
            },
            'errors': []
        }
        
        try:
            # Test connection
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    
                    if result and result[0] == 1:
                        health_status['healthy'] = True
            
            # Calculate performance metrics
            if self.stats['queries_executed'] > 0:
                total_time = time.time() - getattr(self, 'start_time', time.time())
                health_status['performance']['queries_per_second'] = self.stats['queries_executed'] / max(total_time, 1)
                health_status['performance']['error_rate'] = self.stats['query_errors'] / self.stats['queries_executed']
            
            # Get pool info
            if hasattr(self.pool, '_pool'):
                total_connections = len(self.pool._pool) + len(self.pool._used)
                health_status['connections']['total'] = total_connections
                health_status['connections']['idle'] = len(self.pool._pool)
                health_status['connections']['available'] = self.config.max_connections - total_connections
            
            self.stats['last_health_check'] = datetime.now().isoformat()
            
        except Exception as e:
            health_status['healthy'] = False
            health_status['errors'].append(str(e))
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def get_stats(self) -> Dict:
        """Get pool statistics"""
        with self.lock:
            stats_copy = self.stats.copy()
        
        # Add calculated metrics
        stats_copy['pool_config'] = {
            'min_connections': self.config.min_connections,
            'max_connections': self.config.max_connections,
            'connection_timeout': self.config.connection_timeout
        }
        
        # Add recent query times
        if self.query_times:
            recent_times = self.query_times[-100:]  # Last 100 queries
            stats_copy['recent_performance'] = {
                'min_query_time': min(recent_times),
                'max_query_time': max(recent_times),
                'median_query_time': sorted(recent_times)[len(recent_times)//2],
                'p95_query_time': sorted(recent_times)[int(len(recent_times)*0.95)]
            }
        
        return stats_copy
    
    def _start_health_monitor(self):
        """Start background health monitoring"""
        def monitor():
            while True:
                try:
                    # Perform health check every 60 seconds
                    time.sleep(60)
                    health = self.health_check()
                    
                    if not health['healthy']:
                        logger.warning("Database pool health check failed")
                    
                    # Clean up old connections if needed
                    self._cleanup_connections()
                    
                except Exception as e:
                    logger.error(f"Health monitor error: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.start_time = time.time()
    
    def _cleanup_connections(self):
        """Clean up idle connections beyond timeout"""
        try:
            # This would implement connection cleanup logic
            # For now, we'll just log the cleanup attempt
            logger.debug("Connection cleanup check performed")
        except Exception as e:
            logger.error(f"Connection cleanup error: {e}")
    
    def close_pool(self):
        """Close all connections in pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database pool closed")
            
            with self.lock:
                self.stats['connections_closed'] += self.stats['connections_active']
                self.stats['connections_active'] = 0

class DatabaseManager:
    """High-level database manager with multiple pools"""
    
    def __init__(self):
        self.pools = {}
        self.default_pool = None
        self._load_configs()
    
    def _load_configs(self):
        """Load database configurations"""
        # Main production database
        main_config = PoolConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'bem_production'),
            user=os.getenv('DB_USER', 'bem_user'),
            password=os.getenv('DB_PASSWORD', ''),
            min_connections=int(os.getenv('DB_MIN_CONN', '5')),
            max_connections=int(os.getenv('DB_MAX_CONN', '20'))
        )
        
        self.add_pool('main', main_config, is_default=True)
        
        # Training database (if different)
        training_db = os.getenv('TRAINING_DB_NAME')
        if training_db and training_db != main_config.database:
            training_config = PoolConfig(
                host=os.getenv('TRAINING_DB_HOST', main_config.host),
                port=int(os.getenv('TRAINING_DB_PORT', str(main_config.port))),
                database=training_db,
                user=os.getenv('TRAINING_DB_USER', main_config.user),
                password=os.getenv('TRAINING_DB_PASSWORD', main_config.password),
                min_connections=3,
                max_connections=10
            )
            self.add_pool('training', training_config)
    
    def add_pool(self, pool_name: str, config: PoolConfig, is_default: bool = False):
        """Add database pool"""
        try:
            pool = DatabaseConnectionPool(config)
            self.pools[pool_name] = pool
            
            if is_default or not self.default_pool:
                self.default_pool = pool
            
            logger.info(f"Added database pool: {pool_name}")
            
        except Exception as e:
            logger.error(f"Failed to add pool {pool_name}: {e}")
            raise
    
    def get_pool(self, pool_name: str = None) -> DatabaseConnectionPool:
        """Get database pool by name"""
        if pool_name:
            if pool_name not in self.pools:
                raise ValueError(f"Unknown pool: {pool_name}")
            return self.pools[pool_name]
        
        if not self.default_pool:
            raise ValueError("No default pool configured")
        
        return self.default_pool
    
    def execute_query(self, query: str, params: tuple = None, pool_name: str = None, **kwargs) -> Any:
        """Execute query on specified pool"""
        pool = self.get_pool(pool_name)
        return pool.execute_query(query, params, **kwargs)
    
    def execute_transaction(self, queries: List[Dict], pool_name: str = None) -> List[Any]:
        """Execute transaction on specified pool"""
        pool = self.get_pool(pool_name)
        return pool.execute_transaction(queries)
    
    @contextmanager
    def get_connection(self, pool_name: str = None):
        """Get connection from specified pool"""
        pool = self.get_pool(pool_name)
        with pool.get_connection() as conn:
            yield conn
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all pools"""
        stats = {}
        for pool_name, pool in self.pools.items():
            stats[pool_name] = pool.get_stats()
        return stats
    
    def health_check_all(self) -> Dict:
        """Health check for all pools"""
        health = {}
        overall_healthy = True
        
        for pool_name, pool in self.pools.items():
            pool_health = pool.health_check()
            health[pool_name] = pool_health
            
            if not pool_health['healthy']:
                overall_healthy = False
        
        health['overall_healthy'] = overall_healthy
        return health
    
    def close_all_pools(self):
        """Close all database pools"""
        for pool_name, pool in self.pools.items():
            try:
                pool.close_pool()
                logger.info(f"Closed pool: {pool_name}")
            except Exception as e:
                logger.error(f"Error closing pool {pool_name}: {e}")

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
def get_db_connection(pool_name: str = None):
    """Get database connection (context manager)"""
    return db_manager.get_connection(pool_name)

def execute_query(query: str, params: tuple = None, pool_name: str = None, **kwargs):
    """Execute database query"""
    return db_manager.execute_query(query, params, pool_name, **kwargs)

def execute_transaction(queries: List[Dict], pool_name: str = None):
    """Execute database transaction"""
    return db_manager.execute_transaction(queries, pool_name)

# Flask integration
class FlaskDatabaseIntegration:
    """Flask integration for database pooling"""
    
    def __init__(self, app=None):
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize Flask app with database pooling"""
        app.teardown_appcontext(self.close_db)
        
        # Add database health check endpoint
        @app.route('/admin/db/health')
        def db_health_check():
            return jsonify(db_manager.health_check_all())
        
        @app.route('/admin/db/stats')
        def db_stats():
            return jsonify(db_manager.get_all_stats())
    
    def close_db(self, error):
        """Close database connections on app context teardown"""
        # Connections are automatically returned to pool
        pass

if __name__ == "__main__":
    # Test the database connection pool
    try:
        # Test connection
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                result = cursor.fetchone()
                print(f"Database version: {result[0] if result else 'Unknown'}")
        
        # Test query execution
        result = execute_query("SELECT 1 as test", fetch_all=True)
        print(f"Test query result: {result}")
        
        # Get statistics
        stats = db_manager.get_all_stats()
        print(f"Pool stats: {json.dumps(stats, indent=2, default=str)}")
        
        # Health check
        health = db_manager.health_check_all()
        print(f"Health check: {json.dumps(health, indent=2, default=str)}")
        
    except Exception as e:
        print(f"Database test failed: {e}")
    
    finally:
        db_manager.close_all_pools() 