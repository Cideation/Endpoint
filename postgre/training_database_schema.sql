-- PostgreSQL Schema for BEM System + DGL Training
-- Dedicated training database segregated from main system
-- Supports Alpha/Beta/Gamma phases, coefficients, formulas, and pulse training

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- =============================================================================
-- CORE TABLES FOR BEM GRAPH STRUCTURE
-- =============================================================================

-- Nodes table: Core graph nodes with phase classification
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    phase TEXT, -- Alpha, Beta, Gamma
    dictionary JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_phase CHECK (phase IN ('Alpha', 'Beta', 'Gamma', 'Cross-Phase'))
);

-- Create indexes for efficient querying
CREATE INDEX idx_nodes_phase ON nodes(phase);
CREATE INDEX idx_nodes_type ON nodes(node_type);
CREATE INDEX idx_nodes_dictionary ON nodes USING GIN(dictionary);

-- Edges table: All edges across phases for complete graph learning
CREATE TABLE edges (
    edge_id SERIAL PRIMARY KEY,
    source_node TEXT REFERENCES nodes(node_id),
    target_node TEXT REFERENCES nodes(node_id),
    edge_type TEXT,
    pulse_type TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_edge_type CHECK (edge_type IN (
        'structural_support', 'system_integration', 'load_transfer',
        'functional_relation', 'many_to_many', 'hierarchical_flow',
        'peer_connection', 'combinatorial', 'emergent_relation',
        'network_effect', 'cross_phase', 'pulse_flow'
    ))
);

-- Create indexes for edge queries
CREATE INDEX idx_edges_source ON edges(source_node);
CREATE INDEX idx_edges_target ON edges(target_node);
CREATE INDEX idx_edges_type ON edges(edge_type);
CREATE INDEX idx_edges_pulse_type ON edges(pulse_type);
CREATE INDEX idx_edges_metadata ON edges USING GIN(metadata);

-- =============================================================================
-- COEFFICIENT TABLES FOR SFDE TRAINING
-- =============================================================================

-- Coefficients table: Agent coefficients for SFDE formulas
CREATE TABLE coefficients (
    node_id TEXT REFERENCES nodes(node_id),
    coefficient_key TEXT,
    value NUMERIC,
    source TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (node_id, coefficient_key),
    CONSTRAINT valid_coefficient_source CHECK (source IN (
        'SFDE_formula', 'agent_coefficient_formula', 'manual', 'computed', 'learned'
    ))
);

-- Create indexes for coefficient queries
CREATE INDEX idx_coefficients_key ON coefficients(coefficient_key);
CREATE INDEX idx_coefficients_source ON coefficients(source);
CREATE INDEX idx_coefficients_updated ON coefficients(updated_at);

-- =============================================================================
-- LABELS TABLE FOR DGL TRAINING TARGETS
-- =============================================================================

-- Labels table: Training targets for DGL learning
CREATE TABLE labels (
    node_id TEXT REFERENCES nodes(node_id),
    label_key TEXT,
    value NUMERIC,
    label_source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (node_id, label_key),
    CONSTRAINT valid_label_source CHECK (label_source IN (
        'ground_truth', 'SFDE_computed', 'cross_phase_derived', 'expert_labeled'
    ))
);

-- Create indexes for label queries
CREATE INDEX idx_labels_key ON labels(label_key);
CREATE INDEX idx_labels_source ON labels(label_source);
CREATE INDEX idx_labels_created ON labels(created_at);

-- =============================================================================
-- PULSE SYSTEM TABLES
-- =============================================================================

-- Pulses table: 7-pulse system for training
CREATE TABLE pulses (
    pulse_id SERIAL PRIMARY KEY,
    node_id TEXT REFERENCES nodes(node_id),
    pulse_type TEXT,
    direction TEXT,
    color TEXT,
    payload JSONB,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_pulse_type CHECK (pulse_type IN (
        'bid_pulse', 'occupancy_pulse', 'compliancy_pulse', 
        'fit_pulse', 'investment_pulse', 'decay_pulse', 'reject_pulse'
    )),
    CONSTRAINT valid_direction CHECK (direction IN (
        'downward', 'upward', 'lateral', 'reflexive', 'cross_phase'
    ))
);

-- Create indexes for pulse queries
CREATE INDEX idx_pulses_node ON pulses(node_id);
CREATE INDEX idx_pulses_type ON pulses(pulse_type);
CREATE INDEX idx_pulses_direction ON pulses(direction);
CREATE INDEX idx_pulses_triggered ON pulses(triggered_at);
CREATE INDEX idx_pulses_payload ON pulses USING GIN(payload);

-- =============================================================================
-- FORMULA TABLES FOR SFDE INTEGRATION
-- =============================================================================

-- Formulas table: SFDE scientific formulas for training
CREATE TABLE formulas (
    formula_id TEXT PRIMARY KEY,
    label_target TEXT,
    inputs TEXT[],
    pulse_dependencies TEXT[],
    applicable_node_types TEXT[],
    raw_formula TEXT,
    is_trainable BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_inputs CHECK (array_length(inputs, 1) > 0)
);

-- Create indexes for formula queries
CREATE INDEX idx_formulas_target ON formulas(label_target);
CREATE INDEX idx_formulas_trainable ON formulas(is_trainable);
CREATE INDEX idx_formulas_node_types ON formulas USING GIN(applicable_node_types);
CREATE INDEX idx_formulas_inputs ON formulas USING GIN(inputs);

-- =============================================================================
-- TRAINING TABLES FOR DGL OPTIMIZATION
-- =============================================================================

-- Training runs: Track DGL training sessions
CREATE TABLE training_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_name TEXT,
    model_config JSONB,
    training_params JSONB,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT DEFAULT 'running',
    CONSTRAINT valid_status CHECK (status IN ('running', 'completed', 'failed', 'stopped'))
);

-- Training metrics: Store training progress and results
CREATE TABLE training_metrics (
    metric_id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES training_runs(run_id),
    epoch INTEGER,
    metric_name TEXT,
    metric_value NUMERIC,
    metadata JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model embeddings: Store learned node embeddings
CREATE TABLE model_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES training_runs(run_id),
    node_id TEXT REFERENCES nodes(node_id),
    embedding_vector NUMERIC[],
    embedding_dim INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- CROSS-PHASE ANALYSIS TABLES
-- =============================================================================

-- Phase interactions: Track cross-phase learning patterns
CREATE TABLE phase_interactions (
    interaction_id SERIAL PRIMARY KEY,
    source_phase TEXT,
    target_phase TEXT,
    interaction_type TEXT,
    strength NUMERIC,
    metadata JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_source_phase CHECK (source_phase IN ('Alpha', 'Beta', 'Gamma')),
    CONSTRAINT valid_target_phase CHECK (target_phase IN ('Alpha', 'Beta', 'Gamma'))
);

-- Emergence events: Track emergent behavior detection
CREATE TABLE emergence_events (
    event_id SERIAL PRIMARY KEY,
    event_type TEXT,
    affected_nodes TEXT[],
    emergence_score NUMERIC,
    phase_distribution JSONB,
    detected_by TEXT,
    event_data JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- VIEWS FOR TRAINING DATA PREPARATION
-- =============================================================================

-- Complete graph view for DGL training
CREATE VIEW training_graph AS
SELECT 
    n.node_id,
    n.node_type,
    n.phase,
    n.dictionary,
    COALESCE(
        json_agg(
            json_build_object(
                'coefficient_key', c.coefficient_key,
                'value', c.value,
                'source', c.source
            )
        ) FILTER (WHERE c.coefficient_key IS NOT NULL),
        '[]'::json
    ) as coefficients,
    COALESCE(
        json_agg(
            json_build_object(
                'label_key', l.label_key,
                'value', l.value,
                'source', l.label_source
            )
        ) FILTER (WHERE l.label_key IS NOT NULL),
        '[]'::json
    ) as labels
FROM nodes n
LEFT JOIN coefficients c ON n.node_id = c.node_id
LEFT JOIN labels l ON n.node_id = l.node_id
GROUP BY n.node_id, n.node_type, n.phase, n.dictionary;

-- Cross-phase edge view
CREATE VIEW cross_phase_edges AS
SELECT 
    e.edge_id,
    e.source_node,
    e.target_node,
    e.edge_type,
    e.pulse_type,
    e.metadata,
    n1.phase as source_phase,
    n2.phase as target_phase,
    CASE 
        WHEN n1.phase != n2.phase THEN TRUE 
        ELSE FALSE 
    END as is_cross_phase
FROM edges e
JOIN nodes n1 ON e.source_node = n1.node_id
JOIN nodes n2 ON e.target_node = n2.node_id;

-- Formula readiness view
CREATE VIEW formula_training_readiness AS
SELECT 
    f.formula_id,
    f.label_target,
    f.is_trainable,
    COUNT(DISTINCT l.node_id) as nodes_with_labels,
    COUNT(DISTINCT c.node_id) as nodes_with_coefficients,
    array_agg(DISTINCT n.phase) as covered_phases
FROM formulas f
LEFT JOIN labels l ON f.label_target = l.label_key
LEFT JOIN coefficients c ON c.coefficient_key = ANY(f.inputs)
LEFT JOIN nodes n ON (l.node_id = n.node_id OR c.node_id = n.node_id)
WHERE f.is_trainable = TRUE
GROUP BY f.formula_id, f.label_target, f.is_trainable;

-- =============================================================================
-- FUNCTIONS FOR TRAINING DATA GENERATION
-- =============================================================================

-- Function to generate DGL training data
CREATE OR REPLACE FUNCTION get_dgl_training_data()
RETURNS TABLE (
    node_features JSONB,
    edge_list JSONB,
    node_labels JSONB,
    phase_info JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        json_agg(
            json_build_object(
                'node_id', tg.node_id,
                'features', array[
                    -- Extract numerical features from dictionary and coefficients
                    COALESCE((tg.dictionary->>'volume')::numeric / 1000.0, 0),
                    COALESCE((tg.dictionary->>'cost')::numeric / 100000.0, 0),
                    COALESCE((tg.dictionary->>'area')::numeric / 100.0, 0),
                    COALESCE((tg.dictionary->>'power')::numeric / 1000.0, 0)
                ],
                'phase', CASE 
                    WHEN tg.phase = 'Alpha' THEN 0
                    WHEN tg.phase = 'Beta' THEN 1  
                    WHEN tg.phase = 'Gamma' THEN 2
                    ELSE 1
                END
            )
        ) as node_features,
        
        json_agg(
            json_build_object(
                'source', cpe.source_node,
                'target', cpe.target_node,
                'edge_type', cpe.edge_type,
                'is_cross_phase', cpe.is_cross_phase
            )
        ) as edge_list,
        
        json_agg(
            json_build_object(
                'node_id', tg.node_id,
                'labels', tg.labels
            )
        ) as node_labels,
        
        json_build_object(
            'total_nodes', COUNT(DISTINCT tg.node_id),
            'total_edges', COUNT(DISTINCT cpe.edge_id),
            'phases', json_build_object(
                'Alpha', COUNT(*) FILTER (WHERE tg.phase = 'Alpha'),
                'Beta', COUNT(*) FILTER (WHERE tg.phase = 'Beta'),
                'Gamma', COUNT(*) FILTER (WHERE tg.phase = 'Gamma')
            )
        ) as phase_info
        
    FROM training_graph tg
    CROSS JOIN cross_phase_edges cpe;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- INITIAL DATA AND CONSTRAINTS
-- =============================================================================

-- Add constraint to ensure edge endpoints exist
ALTER TABLE edges ADD CONSTRAINT fk_edges_source_exists 
    FOREIGN KEY (source_node) REFERENCES nodes(node_id) ON DELETE CASCADE;
    
ALTER TABLE edges ADD CONSTRAINT fk_edges_target_exists 
    FOREIGN KEY (target_node) REFERENCES nodes(node_id) ON DELETE CASCADE;

-- Add constraint for valid embedding dimensions
ALTER TABLE model_embeddings ADD CONSTRAINT valid_embedding_dim 
    CHECK (embedding_dim > 0 AND embedding_dim <= 1024);

-- Add trigger to update training run completion
CREATE OR REPLACE FUNCTION update_training_completion()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('completed', 'failed', 'stopped') AND OLD.completed_at IS NULL THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_training_completion
    BEFORE UPDATE ON training_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_training_completion();

-- =============================================================================
-- SAMPLE DATA FOR TESTING
-- =============================================================================

-- Insert sample SFDE formulas
INSERT INTO formulas (formula_id, label_target, inputs, pulse_dependencies, applicable_node_types, raw_formula, is_trainable) VALUES
('node_similarity', 'similarity_score', ARRAY['volume', 'cost', 'efficiency'], ARRAY[], ARRAY['structural', 'electrical', 'mep'], 'cosine_similarity(features_a, features_b)', TRUE),
('edge_weight', 'edge_strength', ARRAY['source_capacity', 'target_demand', 'safety_factor'], ARRAY['fit_pulse'], ARRAY['structural'], '(capacity - demand) / capacity * safety_factor', TRUE),
('agent_coefficient', 'agent_efficiency', ARRAY['cost', 'performance'], ARRAY['bid_pulse'], ARRAY['BiddingAgent'], 'performance / (cost / 10000 + 0.1)', TRUE),
('emergence_detection', 'emergence_score', ARRAY['embedding_drift', 'variance'], ARRAY['decay_pulse'], ARRAY['all'], 'trend_acceleration(centroid_distances)', TRUE),
('cross_phase_learning', 'phase_interaction', ARRAY['alpha_features', 'beta_features', 'gamma_features'], ARRAY['occupancy_pulse', 'compliancy_pulse'], ARRAY['all'], 'phase_fusion(alpha_h, beta_h, gamma_h)', TRUE);

COMMENT ON DATABASE CURRENT_DATABASE IS 'BEM System DGL Training Database - Segregated training environment for cross-phase graph learning with SFDE scientific foundation';

-- Grant permissions for training applications
-- Note: Adjust these permissions based on your application user
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO bem_training_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO bem_training_app; 