#!/usr/bin/env python3
"""
BEM Emergence Database Optimizer
High-performance database operations for VaaS, PaaS, P2P financial modes
Optimized queries, connection pooling, and caching for emergence transactions
"""

import asyncio
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis
from contextlib import asynccontextmanager
import uuid

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration for emergence operations"""
    host: str = "localhost"
    port: int = 5432
    database: str = "bem_emergence"
    user: str = "bem_user"
    password: str = ""
    min_connections: int = 10
    max_connections: int = 50
    command_timeout: int = 60

class EmergenceDBOptimizer:
    """
    High-performance database optimizer for BEM emergence financial operations
    Handles VaaS billing, PaaS pool management, P2P trust tracking
    """
    
    def __init__(self, config: DatabaseConfig = None, redis_client=None):
        self.config = config or DatabaseConfig()
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        self.pool = None
        self.sync_pool = None
        
        # Performance metrics
        self.query_metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'slow_queries': 0,
            'query_times': [],
            'vaas_transactions': 0,
            'paas_transactions': 0,
            'p2p_transactions': 0
        }
        
        # Query cache TTL settings
        self.cache_ttl = {
            'user_profile': 300,      # 5 minutes
            'pool_status': 60,        # 1 minute
            'trust_scores': 1800,     # 30 minutes
            'emergence_outputs': 3600, # 1 hour
            'billing_history': 600    # 10 minutes
        }
        
        logger.info("Emergence DB Optimizer initialized")
    
    async def initialize_pools(self):
        """Initialize async and sync connection pools"""
        try:
            # Async pool for high-performance operations
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout
            )
            
            # Sync pool for compatibility
            self.sync_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=5,
                maxconn=20,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            
            logger.info("Database pools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pools: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get async database connection"""
        if not self.pool:
            await self.initialize_pools()
        
        async with self.pool.acquire() as conn:
            yield conn
    
    async def _execute_cached_query(self, query: str, params: tuple, cache_key: str, ttl: int) -> List[Dict]:
        """Execute query with Redis caching"""
        start_time = time.time()
        
        # Try cache first
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            self.query_metrics['cache_hits'] += 1
            return json.loads(cached_result)
        
        # Execute query
        self.query_metrics['cache_misses'] += 1
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, *params)
            result = [dict(row) for row in rows]
        
        # Cache result
        self.redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
        
        # Track performance
        query_time = time.time() - start_time
        self._track_query_performance(query_time)
        
        return result
    
    def _track_query_performance(self, query_time: float):
        """Track query performance metrics"""
        self.query_metrics['total_queries'] += 1
        self.query_metrics['query_times'].append(query_time)
        
        # Track slow queries (>500ms)
        if query_time > 0.5:
            self.query_metrics['slow_queries'] += 1
        
        # Keep only last 1000 query times
        if len(self.query_metrics['query_times']) > 1000:
            self.query_metrics['query_times'] = self.query_metrics['query_times'][-1000:]
    
    # ==================== VaaS (Value-as-a-Service) Operations ====================
    
    async def create_vaas_transaction(self, user_id: str, emergence_type: str, 
                                     amount: float, payment_method: str, 
                                     output_data: Dict) -> str:
        """Create VaaS transaction record"""
        transaction_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO vaas_transactions (
            transaction_id, user_id, emergence_type, amount, 
            payment_method, output_data, status, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING transaction_id
        """
        
        async with self.get_connection() as conn:
            result = await conn.fetchrow(
                query, transaction_id, user_id, emergence_type, amount,
                payment_method, json.dumps(output_data), 'completed', datetime.now()
            )
        
        self.query_metrics['vaas_transactions'] += 1
        
        # Invalidate user billing cache
        self.redis_client.delete(f"billing_history:{user_id}")
        
        return result['transaction_id']
    
    async def get_user_vaas_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's VaaS transaction history with caching"""
        cache_key = f"billing_history:{user_id}:{limit}"
        
        query = """
        SELECT transaction_id, emergence_type, amount, payment_method, 
               output_data, status, created_at
        FROM vaas_transactions 
        WHERE user_id = $1 
        ORDER BY created_at DESC 
        LIMIT $2
        """
        
        return await self._execute_cached_query(
            query, (user_id, limit), cache_key, self.cache_ttl['billing_history']
        )
    
    async def get_vaas_revenue_stats(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get VaaS revenue statistics"""
        cache_key = f"vaas_revenue:{start_date.date()}:{end_date.date()}"
        
        query = """
        SELECT 
            emergence_type,
            COUNT(*) as transaction_count,
            SUM(amount) as total_revenue,
            AVG(amount) as average_amount,
            payment_method
        FROM vaas_transactions 
        WHERE created_at BETWEEN $1 AND $2 AND status = 'completed'
        GROUP BY emergence_type, payment_method
        ORDER BY total_revenue DESC
        """
        
        return await self._execute_cached_query(
            query, (start_date, end_date), cache_key, 300  # 5 minute cache
        )
    
    # ==================== PaaS (Paluwagan-as-a-Service) Operations ====================
    
    async def create_paas_pool(self, pool_id: str, target_amount: float, 
                              emergence_type: str, metadata: Dict) -> bool:
        """Create new PaaS pool"""
        query = """
        INSERT INTO paas_pools (
            pool_id, target_amount, current_amount, emergence_type, 
            metadata, status, created_at, contributors_count
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        async with self.get_connection() as conn:
            await conn.execute(
                query, pool_id, target_amount, 0.0, emergence_type,
                json.dumps(metadata), 'active', datetime.now(), 0
            )
        
        return True
    
    async def add_pool_contribution(self, pool_id: str, user_id: str, 
                                   amount: float, payment_method: str) -> Dict:
        """Add contribution to PaaS pool"""
        contribution_id = str(uuid.uuid4())
        
        async with self.get_connection() as conn:
            async with conn.transaction():
                # Add contribution record
                await conn.execute("""
                    INSERT INTO paas_contributions (
                        contribution_id, pool_id, user_id, amount, 
                        payment_method, status, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, contribution_id, pool_id, user_id, amount, 
                    payment_method, 'confirmed', datetime.now())
                
                # Update pool totals
                pool_update = await conn.fetchrow("""
                    UPDATE paas_pools 
                    SET current_amount = current_amount + $1,
                        contributors_count = contributors_count + 1,
                        updated_at = $2
                    WHERE pool_id = $3
                    RETURNING current_amount, target_amount, contributors_count
                """, amount, datetime.now(), pool_id)
        
        self.query_metrics['paas_transactions'] += 1
        
        # Invalidate pool cache
        self.redis_client.delete(f"pool_status:{pool_id}")
        
        # Check if pool is now fulfilled
        pool_fulfilled = pool_update['current_amount'] >= pool_update['target_amount']
        
        if pool_fulfilled:
            await self._mark_pool_fulfilled(pool_id)
        
        return {
            'contribution_id': contribution_id,
            'pool_fulfilled': pool_fulfilled,
            'current_amount': float(pool_update['current_amount']),
            'target_amount': float(pool_update['target_amount']),
            'contributors_count': pool_update['contributors_count']
        }
    
    async def get_pool_status(self, pool_id: str) -> Dict:
        """Get PaaS pool status with caching"""
        cache_key = f"pool_status:{pool_id}"
        
        query = """
        SELECT pool_id, target_amount, current_amount, emergence_type,
               metadata, status, created_at, contributors_count,
               (current_amount >= target_amount) as fulfilled
        FROM paas_pools 
        WHERE pool_id = $1
        """
        
        result = await self._execute_cached_query(
            query, (pool_id,), cache_key, self.cache_ttl['pool_status']
        )
        
        return result[0] if result else None
    
    async def get_pool_contributors(self, pool_id: str) -> List[Dict]:
        """Get list of pool contributors"""
        query = """
        SELECT user_id, amount, payment_method, created_at
        FROM paas_contributions 
        WHERE pool_id = $1 AND status = 'confirmed'
        ORDER BY created_at ASC
        """
        
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, pool_id)
            return [dict(row) for row in rows]
    
    async def _mark_pool_fulfilled(self, pool_id: str):
        """Mark pool as fulfilled and trigger emergence generation"""
        async with self.get_connection() as conn:
            await conn.execute("""
                UPDATE paas_pools 
                SET status = 'fulfilled', fulfilled_at = $1
                WHERE pool_id = $2
            """, datetime.now(), pool_id)
        
        # Trigger emergence generation (would integrate with your emergence system)
        logger.info(f"Pool {pool_id} fulfilled - triggering emergence generation")
    
    # ==================== P2P (Peer-to-Peer) Operations ====================
    
    async def record_p2p_exchange(self, from_user_id: str, to_user_id: str,
                                 emergence_type: str, output_data: Dict,
                                 trust_score: float) -> str:
        """Record P2P emergence exchange"""
        exchange_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO p2p_exchanges (
            exchange_id, from_user_id, to_user_id, emergence_type,
            output_data, trust_score, status, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING exchange_id
        """
        
        async with self.get_connection() as conn:
            result = await conn.fetchrow(
                query, exchange_id, from_user_id, to_user_id, emergence_type,
                json.dumps(output_data), trust_score, 'completed', datetime.now()
            )
        
        self.query_metrics['p2p_transactions'] += 1
        
        # Update trust scores
        await self._update_trust_scores(from_user_id, to_user_id, trust_score)
        
        return result['exchange_id']
    
    async def get_trust_score(self, user_a: str, user_b: str) -> float:
        """Get trust score between two users"""
        cache_key = f"trust_score:{min(user_a, user_b)}:{max(user_a, user_b)}"
        
        cached_score = self.redis_client.get(cache_key)
        if cached_score:
            self.query_metrics['cache_hits'] += 1
            return float(cached_score)
        
        self.query_metrics['cache_misses'] += 1
        
        query = """
        SELECT trust_score
        FROM user_trust_scores 
        WHERE (user_a = $1 AND user_b = $2) OR (user_a = $2 AND user_b = $1)
        ORDER BY updated_at DESC
        LIMIT 1
        """
        
        async with self.get_connection() as conn:
            result = await conn.fetchrow(query, user_a, user_b)
        
        trust_score = float(result['trust_score']) if result else 0.5  # Default neutral trust
        
        # Cache the result
        self.redis_client.setex(cache_key, self.cache_ttl['trust_scores'], str(trust_score))
        
        return trust_score
    
    async def _update_trust_scores(self, user_a: str, user_b: str, interaction_score: float):
        """Update trust scores based on successful interaction"""
        async with self.get_connection() as conn:
            # Get current trust score
            current_score = await self.get_trust_score(user_a, user_b)
            
            # Calculate new trust score (weighted average)
            new_score = (current_score * 0.8) + (interaction_score * 0.2)
            
            # Upsert trust score
            await conn.execute("""
                INSERT INTO user_trust_scores (user_a, user_b, trust_score, updated_at, interaction_count)
                VALUES ($1, $2, $3, $4, 1)
                ON CONFLICT (user_a, user_b) 
                DO UPDATE SET 
                    trust_score = $3,
                    updated_at = $4,
                    interaction_count = user_trust_scores.interaction_count + 1
            """, min(user_a, user_b), max(user_a, user_b), new_score, datetime.now())
        
        # Invalidate cache
        cache_key = f"trust_score:{min(user_a, user_b)}:{max(user_a, user_b)}"
        self.redis_client.delete(cache_key)
    
    async def get_user_p2p_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's P2P exchange history"""
        query = """
        SELECT exchange_id, from_user_id, to_user_id, emergence_type,
               output_data, trust_score, status, created_at
        FROM p2p_exchanges 
        WHERE from_user_id = $1 OR to_user_id = $1
        ORDER BY created_at DESC 
        LIMIT $2
        """
        
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, user_id, limit)
            return [dict(row) for row in rows]
    
    # ==================== Analytics and Reporting ====================
    
    async def get_emergence_analytics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get comprehensive emergence analytics across all modes"""
        
        # VaaS analytics
        vaas_stats = await self.get_vaas_revenue_stats(start_date, end_date)
        
        # PaaS analytics
        paas_query = """
        SELECT 
            COUNT(DISTINCT pool_id) as total_pools,
            COUNT(DISTINCT pool_id) FILTER (WHERE status = 'fulfilled') as fulfilled_pools,
            SUM(target_amount) as total_target_amount,
            SUM(current_amount) as total_raised_amount,
            AVG(contributors_count) as avg_contributors_per_pool
        FROM paas_pools 
        WHERE created_at BETWEEN $1 AND $2
        """
        
        async with self.get_connection() as conn:
            paas_stats = await conn.fetchrow(paas_query, start_date, end_date)
        
        # P2P analytics
        p2p_query = """
        SELECT 
            COUNT(*) as total_exchanges,
            COUNT(DISTINCT from_user_id) as unique_senders,
            COUNT(DISTINCT to_user_id) as unique_receivers,
            AVG(trust_score) as average_trust_score,
            emergence_type
        FROM p2p_exchanges 
        WHERE created_at BETWEEN $1 AND $2
        GROUP BY emergence_type
        """
        
        async with self.get_connection() as conn:
            p2p_stats = await conn.fetch(p2p_query, start_date, end_date)
        
        return {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'vaas': {
                'revenue_by_type': vaas_stats,
                'total_transactions': sum(item['transaction_count'] for item in vaas_stats),
                'total_revenue': sum(item['total_revenue'] for item in vaas_stats)
            },
            'paas': dict(paas_stats) if paas_stats else {},
            'p2p': [dict(row) for row in p2p_stats],
            'database_performance': self.get_performance_metrics()
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get database performance metrics"""
        avg_query_time = sum(self.query_metrics['query_times']) / max(len(self.query_metrics['query_times']), 1)
        
        cache_hit_rate = 0
        total_cache_ops = self.query_metrics['cache_hits'] + self.query_metrics['cache_misses']
        if total_cache_ops > 0:
            cache_hit_rate = (self.query_metrics['cache_hits'] / total_cache_ops) * 100
        
        return {
            'total_queries': self.query_metrics['total_queries'],
            'average_query_time_ms': avg_query_time * 1000,
            'slow_queries': self.query_metrics['slow_queries'],
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'transaction_counts': {
                'vaas': self.query_metrics['vaas_transactions'],
                'paas': self.query_metrics['paas_transactions'],
                'p2p': self.query_metrics['p2p_transactions']
            }
        }
    
    async def health_check(self) -> Dict:
        """Perform database health check"""
        try:
            start_time = time.time()
            
            async with self.get_connection() as conn:
                # Test basic connectivity
                await conn.fetchrow("SELECT 1 as test")
                
                # Check table sizes
                table_stats = await conn.fetch("""
                    SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
                    FROM pg_stat_user_tables 
                    WHERE schemaname = 'public'
                """)
            
            response_time = time.time() - start_time
            
            return {
                'healthy': True,
                'response_time_ms': response_time * 1000,
                'table_statistics': [dict(row) for row in table_stats],
                'connection_pool_size': self.pool.get_size() if self.pool else 0,
                'performance_metrics': self.get_performance_metrics()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }

# Database schema creation
CREATE_SCHEMA_SQL = """
-- VaaS Transactions Table
CREATE TABLE IF NOT EXISTS vaas_transactions (
    transaction_id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    emergence_type VARCHAR(100) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    output_data JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- PaaS Pools Table
CREATE TABLE IF NOT EXISTS paas_pools (
    pool_id UUID PRIMARY KEY,
    target_amount DECIMAL(10,2) NOT NULL,
    current_amount DECIMAL(10,2) DEFAULT 0.0,
    emergence_type VARCHAR(100) NOT NULL,
    metadata JSONB,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    fulfilled_at TIMESTAMP WITH TIME ZONE,
    contributors_count INTEGER DEFAULT 0
);

-- PaaS Contributions Table
CREATE TABLE IF NOT EXISTS paas_contributions (
    contribution_id UUID PRIMARY KEY,
    pool_id UUID REFERENCES paas_pools(pool_id),
    user_id VARCHAR(255) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- P2P Exchanges Table
CREATE TABLE IF NOT EXISTS p2p_exchanges (
    exchange_id UUID PRIMARY KEY,
    from_user_id VARCHAR(255) NOT NULL,
    to_user_id VARCHAR(255) NOT NULL,
    emergence_type VARCHAR(100) NOT NULL,
    output_data JSONB NOT NULL,
    trust_score DECIMAL(3,2) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User Trust Scores Table
CREATE TABLE IF NOT EXISTS user_trust_scores (
    user_a VARCHAR(255) NOT NULL,
    user_b VARCHAR(255) NOT NULL,
    trust_score DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    interaction_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (user_a, user_b),
    CHECK (user_a < user_b)  -- Ensure consistent ordering
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_vaas_user_created ON vaas_transactions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vaas_type_status ON vaas_transactions(emergence_type, status);
CREATE INDEX IF NOT EXISTS idx_paas_pool_status ON paas_pools(status, created_at);
CREATE INDEX IF NOT EXISTS idx_paas_contributions_pool ON paas_contributions(pool_id, status);
CREATE INDEX IF NOT EXISTS idx_p2p_users ON p2p_exchanges(from_user_id, to_user_id);
CREATE INDEX IF NOT EXISTS idx_trust_lookup ON user_trust_scores(user_a, user_b);
"""

# Global optimizer instance
db_optimizer = None

async def get_db_optimizer() -> EmergenceDBOptimizer:
    """Get global database optimizer instance"""
    global db_optimizer
    if not db_optimizer:
        db_optimizer = EmergenceDBOptimizer()
        await db_optimizer.initialize_pools()
    return db_optimizer

if __name__ == "__main__":
    # Test the database optimizer
    async def test_optimizer():
        optimizer = EmergenceDBOptimizer()
        await optimizer.initialize_pools()
        
        # Test VaaS transaction
        vaas_tx_id = await optimizer.create_vaas_transaction(
            user_id="test_user_001",
            emergence_type="CAD",
            amount=99.99,
            payment_method="credit_card",
            output_data={"file_url": "/outputs/test_cad.dwg"}
        )
        print(f"Created VaaS transaction: {vaas_tx_id}")
        
        # Test PaaS pool
        await optimizer.create_paas_pool(
            pool_id="test_pool_001",
            target_amount=5000.0,
            emergence_type="ROI",
            metadata={"project": "Community Building"}
        )
        
        contribution_result = await optimizer.add_pool_contribution(
            pool_id="test_pool_001",
            user_id="contributor_001",
            amount=1500.0,
            payment_method="bank_transfer"
        )
        print(f"Pool contribution result: {contribution_result}")
        
        # Test P2P exchange
        p2p_exchange_id = await optimizer.record_p2p_exchange(
            from_user_id="agent_001",
            to_user_id="agent_002",
            emergence_type="BOM",
            output_data={"components": ["steel", "concrete"]},
            trust_score=0.85
        )
        print(f"P2P exchange recorded: {p2p_exchange_id}")
        
        # Performance metrics
        metrics = optimizer.get_performance_metrics()
        print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
        # Health check
        health = await optimizer.health_check()
        print(f"Health check: {json.dumps(health, indent=2, default=str)}")
    
    asyncio.run(test_optimizer())
