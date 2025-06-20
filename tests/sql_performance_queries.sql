-- BEM System PostgreSQL Performance Profiling Queries
-- Use with EXPLAIN ANALYZE to find expensive operations and optimization opportunities

-- ============================================================================
-- CORE PERFORMANCE QUERIES
-- ============================================================================

-- Query 1: Alpha Phase Node Analysis (from user example)
-- Tests: Sequential scan vs index scan, ORDER BY performance
EXPLAIN ANALYZE
SELECT * FROM node_data
WHERE phase = 'alpha'
ORDER BY created_at DESC
LIMIT 50;

-- Recommended Index:
-- CREATE INDEX idx_node_data_phase_created ON node_data(phase, created_at DESC);

-- Query 2: Recent Interaction Logs (Time-based filtering)
-- Tests: Time range queries, index effectiveness
EXPLAIN ANALYZE
SELECT user_id, action_type, timestamp, pulse_type
FROM interaction_logs
WHERE timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;

-- Recommended Index:
-- CREATE INDEX idx_interaction_logs_timestamp ON interaction_logs(timestamp);

-- Query 3: Active Agent State Lookup (Pattern matching)
-- Tests: LIKE query performance, text pattern operations
EXPLAIN ANALYZE
SELECT node_id, status, last_update, functor_type
FROM agent_state
WHERE node_id LIKE 'V%'
AND status = 'active';

-- Recommended Index:
-- CREATE INDEX idx_agent_state_node_pattern ON agent_state(node_id text_pattern_ops, status);

-- Query 4: Complex Edge Relationship Join (Multi-table performance)
-- Tests: JOIN performance, cross-phase relationships
EXPLAIN ANALYZE
SELECT n1.node_id as source_node, n1.phase as source_phase,
       n2.node_id as target_node, n2.phase as target_phase,
       e.edge_type, e.weight
FROM node_data n1
JOIN edge_data e ON n1.node_id = e.source_node
JOIN node_data n2 ON e.target_node = n2.node_id
WHERE n1.phase = 'beta' AND n2.phase = 'gamma';

-- Recommended Indexes:
-- CREATE INDEX idx_edge_data_source_node ON edge_data(source_node);
-- CREATE INDEX idx_edge_data_target_node ON edge_data(target_node);
-- CREATE INDEX idx_node_data_phase_id ON node_data(phase, node_id);

-- ============================================================================
-- ADDITIONAL PERFORMANCE TEST QUERIES
-- ============================================================================

-- Query 5: Functor Type Distribution Analysis
-- Tests: Aggregation performance, GROUP BY optimization
EXPLAIN ANALYZE
SELECT functor_type, phase, COUNT(*) as node_count,
       AVG(EXTRACT(EPOCH FROM (last_update - created_at))) as avg_lifetime_seconds
FROM node_data
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY functor_type, phase
ORDER BY node_count DESC;

-- Recommended Index:
-- CREATE INDEX idx_node_data_created_functor_phase ON node_data(created_at, functor_type, phase);

-- Query 6: User Behavior Pattern Analysis
-- Tests: Window functions, complex aggregations
EXPLAIN ANALYZE
SELECT user_id,
       COUNT(DISTINCT action_type) as unique_actions,
       COUNT(*) as total_interactions,
       MIN(timestamp) as first_interaction,
       MAX(timestamp) as last_interaction,
       AVG(EXTRACT(EPOCH FROM (LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) - timestamp))) as avg_interval_seconds
FROM interaction_logs
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY user_id
HAVING COUNT(*) >= 5
ORDER BY total_interactions DESC
LIMIT 100;

-- Recommended Index:
-- CREATE INDEX idx_interaction_logs_user_timestamp ON interaction_logs(user_id, timestamp);

-- Query 7: Phase Transition Performance
-- Tests: Subquery performance, EXISTS vs IN
EXPLAIN ANALYZE
SELECT n1.node_id, n1.phase, n1.functor_type
FROM node_data n1
WHERE EXISTS (
    SELECT 1 FROM edge_data e
    WHERE e.source_node = n1.node_id
    AND EXISTS (
        SELECT 1 FROM node_data n2
        WHERE n2.node_id = e.target_node
        AND n2.phase != n1.phase
    )
)
ORDER BY n1.phase, n1.created_at;

-- Query 8: Heavy Aggregation with Multiple JOINs
-- Tests: Complex join performance, aggregation optimization
EXPLAIN ANALYZE
SELECT n.phase,
       e.edge_type,
       COUNT(*) as edge_count,
       AVG(e.weight) as avg_weight,
       COUNT(DISTINCT n.functor_type) as unique_functors,
       COUNT(DISTINCT il.user_id) as unique_users
FROM node_data n
JOIN edge_data e ON (n.node_id = e.source_node OR n.node_id = e.target_node)
LEFT JOIN interaction_logs il ON il.target_nodes::text LIKE '%' || n.node_id || '%'
WHERE n.created_at >= NOW() - INTERVAL '30 days'
GROUP BY n.phase, e.edge_type
HAVING COUNT(*) >= 10
ORDER BY edge_count DESC;

-- ============================================================================
-- PERFORMANCE MONITORING QUERIES
-- ============================================================================

-- Check table sizes and bloat
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
       n_live_tup, n_dead_tup,
       ROUND((n_dead_tup::float / NULLIF(n_live_tup + n_dead_tup, 0)) * 100, 2) as dead_tuple_percent
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage statistics
SELECT schemaname, tablename, indexname,
       idx_scan, idx_tup_read, idx_tup_fetch,
       pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check slow queries (requires pg_stat_statements extension)
-- SELECT query, calls, total_time, mean_time, rows
-- FROM pg_stat_statements
-- WHERE mean_time > 100  -- queries taking more than 100ms on average
-- ORDER BY mean_time DESC
-- LIMIT 20;

-- ============================================================================
-- RECOMMENDED INDEXES FOR BEM SYSTEM
-- ============================================================================

-- Core indexes for optimal performance:

-- Node data indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_data_phase_created ON node_data(phase, created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_data_functor_type ON node_data(functor_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_data_status ON node_data(status);

-- Edge data indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_edge_data_source_node ON edge_data(source_node);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_edge_data_target_node ON edge_data(target_node);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_edge_data_type_weight ON edge_data(edge_type, weight);

-- Interaction logs indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_interaction_logs_timestamp ON interaction_logs(timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_interaction_logs_user_timestamp ON interaction_logs(user_id, timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_interaction_logs_action_type ON interaction_logs(action_type);

-- Agent state indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_state_node_pattern ON agent_state(node_id text_pattern_ops, status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_state_last_update ON agent_state(last_update);

-- Composite indexes for complex queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_data_phase_functor_created ON node_data(phase, functor_type, created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_edge_data_source_target_type ON edge_data(source_node, target_node, edge_type);

-- ============================================================================
-- PERFORMANCE OPTIMIZATION COMMANDS
-- ============================================================================

-- Update table statistics
ANALYZE node_data;
ANALYZE edge_data;
ANALYZE interaction_logs;
ANALYZE agent_state;

-- Vacuum tables to reclaim space
VACUUM ANALYZE node_data;
VACUUM ANALYZE edge_data;
VACUUM ANALYZE interaction_logs;
VACUUM ANALYZE agent_state;

-- Check for missing indexes on foreign keys
SELECT c.conname as constraint_name,
       t.relname as table_name,
       a.attname as column_name
FROM pg_constraint c
JOIN pg_class t ON c.conrelid = t.oid
JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(c.conkey)
WHERE c.contype = 'f'
AND NOT EXISTS (
    SELECT 1 FROM pg_index i
    WHERE i.indrelid = t.oid
    AND a.attnum = ANY(i.indkey)
)
ORDER BY t.relname, a.attname; 