-- PostgreSQL Schema for Enhanced CAD Parser - Neon Database (FIXED)
-- Based on auraDB Migration CSV structure
-- Optimized for Neon PostgreSQL serverless architecture

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- =====================================================
-- CORE ENTITY TABLES
-- =====================================================

-- Main component table (from component.csv)
CREATE TABLE components (
    component_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_name VARCHAR(255),
    component_type VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Item identity nodes (from item_identity_nodes.csv)
CREATE TABLE item_identity_nodes (
    item_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_family VARCHAR(255),
    variant VARCHAR(255),
    group_id UUID,
    manufacture_score DECIMAL(5,2),
    spec_tag VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Item identity connections (from item_identity_connections.csv)
CREATE TABLE item_identity_connections (
    connection_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES item_identity_nodes(item_id),
    target_id UUID REFERENCES item_identity_nodes(item_id),
    connection_type VARCHAR(100),
    connection_tag VARCHAR(100),
    connection_role VARCHAR(100),
    connection_score DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- GEOMETRY AND SPATIAL TABLES
-- =====================================================

-- Geometry properties (from geometryformat.csv, vertexproperty.csv)
CREATE TABLE geometry_properties (
    geometry_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_id UUID REFERENCES components(component_id),
    geometry_format VARCHAR(50),
    geometry_type VARCHAR(50),
    vertex_count INTEGER,
    face_count INTEGER,
    edge_count INTEGER,
    bounding_box_volume_cm3 DECIMAL(15,3),
    surface_area_m2 DECIMAL(15,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Spatial coordinates and locations (FIXED)
CREATE TABLE spatial_data (
    spatial_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_id UUID REFERENCES components(component_id),
    centroid_x DECIMAL(15,6),
    centroid_y DECIMAL(15,6),
    centroid_z DECIMAL(15,6),
    bbox_x_mm DECIMAL(15,3),
    bbox_y_mm DECIMAL(15,3),
    bbox_z_mm DECIMAL(15,3),
    location_geom GEOMETRY(POINTZ, 4326),
    bounding_box_geom GEOMETRY(POLYGON, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- MATERIAL AND PROPERTY TABLES
-- =====================================================

-- Materials (from material.csv, basematerial.csv)
CREATE TABLE materials (
    material_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    material_name VARCHAR(255),
    base_material VARCHAR(255),
    material_variant VARCHAR(255),
    material_code VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Component materials relationship
CREATE TABLE component_materials (
    component_id UUID REFERENCES components(component_id),
    material_id UUID REFERENCES materials(material_id),
    quantity DECIMAL(15,3),
    unit VARCHAR(50),
    PRIMARY KEY (component_id, material_id)
);

-- Properties (from propertykey.csv, propertyname.csv)
CREATE TABLE properties (
    property_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_key VARCHAR(255),
    property_name VARCHAR(255),
    property_type VARCHAR(100),
    property_value TEXT,
    unit VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Component properties relationship
CREATE TABLE component_properties (
    component_id UUID REFERENCES components(component_id),
    property_id UUID REFERENCES properties(property_id),
    PRIMARY KEY (component_id, property_id)
);

-- =====================================================
-- DIMENSIONS AND MEASUREMENTS
-- =====================================================

-- Dimensions table
CREATE TABLE dimensions (
    dimension_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_id UUID REFERENCES components(component_id),
    length_mm DECIMAL(15,3),
    width_mm DECIMAL(15,3),
    height_mm DECIMAL(15,3),
    area_m2 DECIMAL(15,3),
    volume_cm3 DECIMAL(15,3),
    tolerance_mm DECIMAL(15,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- ANCHOR AND TOPOLOGY TABLES
-- =====================================================

-- Anchors (from anchor.csv, anchorname.csv, anchortype.csv)
CREATE TABLE anchors (
    anchor_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    anchor_name VARCHAR(255),
    anchor_type VARCHAR(100),
    anchor_configuration TEXT,
    anchor_constraints TEXT,
    octree_depth INTEGER,
    octree_size DECIMAL(15,3),
    topologic_vertex_anchor BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Component anchors relationship
CREATE TABLE component_anchors (
    component_id UUID REFERENCES components(component_id),
    anchor_id UUID REFERENCES anchors(anchor_id),
    anchor_reference VARCHAR(255),
    PRIMARY KEY (component_id, anchor_id)
);

-- =====================================================
-- SUPPLIER AND SOURCING TABLES
-- =====================================================

-- Suppliers (from supplier.csv, suppliername.csv, suppliertype.csv)
CREATE TABLE suppliers (
    supplier_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    supplier_name VARCHAR(255),
    supplier_type VARCHAR(100),
    contact_person VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    lead_time_days INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Component suppliers relationship
CREATE TABLE component_suppliers (
    component_id UUID REFERENCES components(component_id),
    supplier_id UUID REFERENCES suppliers(supplier_id),
    unit_cost DECIMAL(15,2),
    quantity_available INTEGER,
    PRIMARY KEY (component_id, supplier_id)
);

-- =====================================================
-- FUNCTIONAL AND CLASSIFICATION TABLES
-- =====================================================

-- Functions (from function.csv, functionname.csv, functortype.csv)
CREATE TABLE functions (
    function_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    function_name VARCHAR(255),
    function_type VARCHAR(100),
    functor_name VARCHAR(255),
    functor_type VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Component functions relationship
CREATE TABLE component_functions (
    component_id UUID REFERENCES components(component_id),
    function_id UUID REFERENCES functions(function_id),
    PRIMARY KEY (component_id, function_id)
);

-- Families and variants
CREATE TABLE families (
    family_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    family_name VARCHAR(255),
    family_type VARCHAR(100),
    product_family VARCHAR(255),
    product_family_type VARCHAR(100),
    variant VARCHAR(255),
    variant_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Component families relationship
CREATE TABLE component_families (
    component_id UUID REFERENCES components(component_id),
    family_id UUID REFERENCES families(family_id),
    PRIMARY KEY (component_id, family_id)
);

-- =====================================================
-- PARSER INTEGRATION TABLES
-- =====================================================

-- File parsing metadata
CREATE TABLE parsed_files (
    file_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_name VARCHAR(255),
    file_path TEXT,
    file_type VARCHAR(10), -- DXF, DWG, IFC, PDF, OBJ, STEP
    file_size BIGINT,
    parsed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    parsing_status VARCHAR(50), -- success, error, processing
    error_message TEXT,
    components_extracted INTEGER,
    processing_time_ms INTEGER
);

-- Parsed components mapping
CREATE TABLE parsed_components (
    parsed_component_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_id UUID REFERENCES parsed_files(file_id),
    component_id UUID REFERENCES components(component_id),
    parser_component_id VARCHAR(255), -- Original parser component ID
    parser_type VARCHAR(50), -- DXF, DWG, IFC, PDF, OBJ, STEP
    parser_data JSONB, -- Full parser output for this component
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Spatial indexes
CREATE INDEX idx_spatial_data_location ON spatial_data USING GIST (location_geom);
CREATE INDEX idx_spatial_data_bbox ON spatial_data USING GIST (bounding_box_geom);

-- Component indexes
CREATE INDEX idx_components_type ON components(component_type);
CREATE INDEX idx_components_name ON components(component_name);

-- Material indexes
CREATE INDEX idx_materials_name ON materials(material_name);
CREATE INDEX idx_materials_base ON materials(base_material);

-- Parser indexes
CREATE INDEX idx_parsed_files_type ON parsed_files(file_type);
CREATE INDEX idx_parsed_files_status ON parsed_files(parsing_status);
CREATE INDEX idx_parsed_components_parser ON parsed_components(parser_type);

-- JSONB indexes for parser data
CREATE INDEX idx_parsed_components_data ON parsed_components USING GIN (parser_data);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for component summary with all related data
CREATE VIEW component_summary AS
SELECT 
    c.component_id,
    c.component_name,
    c.component_type,
    c.description,
    m.material_name,
    f.function_name,
    fam.family_name,
    s.supplier_name,
    sp.centroid_x,
    sp.centroid_y,
    sp.centroid_z,
    d.length_mm,
    d.width_mm,
    d.height_mm,
    d.volume_cm3,
    gp.vertex_count,
    gp.face_count,
    c.created_at
FROM components c
LEFT JOIN component_materials cm ON c.component_id = cm.component_id
LEFT JOIN materials m ON cm.material_id = m.material_id
LEFT JOIN component_functions cf ON c.component_id = cf.component_id
LEFT JOIN functions f ON cf.function_id = f.function_id
LEFT JOIN component_families cfa ON c.component_id = cfa.component_id
LEFT JOIN families fam ON cfa.family_id = fam.family_id
LEFT JOIN component_suppliers cs ON c.component_id = cs.component_id
LEFT JOIN suppliers s ON cs.supplier_id = s.supplier_id
LEFT JOIN spatial_data sp ON c.component_id = sp.component_id
LEFT JOIN dimensions d ON c.component_id = d.component_id
LEFT JOIN geometry_properties gp ON c.component_id = gp.component_id;

-- View for parser statistics
CREATE VIEW parser_statistics AS
SELECT 
    file_type,
    COUNT(*) as total_files,
    COUNT(CASE WHEN parsing_status = 'success' THEN 1 END) as successful_parses,
    COUNT(CASE WHEN parsing_status = 'error' THEN 1 END) as failed_parses,
    AVG(processing_time_ms) as avg_processing_time,
    SUM(components_extracted) as total_components_extracted
FROM parsed_files
GROUP BY file_type;

-- =====================================================
-- FUNCTIONS FOR DATA INTEGRATION
-- =====================================================

-- Function to insert parsed component data
CREATE OR REPLACE FUNCTION insert_parsed_component(
    p_file_id UUID,
    p_parser_component_id VARCHAR,
    p_parser_type VARCHAR,
    p_parser_data JSONB
) RETURNS UUID AS $$
DECLARE
    v_component_id UUID;
    v_parsed_component_id UUID;
BEGIN
    -- Extract component data from parser output
    INSERT INTO components (component_name, component_type, description)
    VALUES (
        p_parser_data->>'name',
        p_parser_data->>'type',
        p_parser_data->'properties'->>'description'
    ) RETURNING component_id INTO v_component_id;
    
    -- Insert parsed component mapping
    INSERT INTO parsed_components (file_id, component_id, parser_component_id, parser_type, parser_data)
    VALUES (p_file_id, v_component_id, p_parser_component_id, p_parser_type, p_parser_data)
    RETURNING parsed_component_id INTO v_parsed_component_id;
    
    -- Extract and insert spatial data if available
    IF p_parser_data->'geometry'->>'has_position' = 'true' THEN
        INSERT INTO spatial_data (component_id, centroid_x, centroid_y, centroid_z)
        VALUES (
            v_component_id,
            (p_parser_data->'geometry'->'position'->>0)::DECIMAL,
            (p_parser_data->'geometry'->'position'->>1)::DECIMAL,
            (p_parser_data->'geometry'->'position'->>2)::DECIMAL
        );
    END IF;
    
    RETURN v_component_id;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- TRIGGERS FOR DATA INTEGRITY
-- =====================================================

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_components_updated_at
    BEFORE UPDATE ON components
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- COMMENTS FOR DOCUMENTATION
-- =====================================================

COMMENT ON TABLE components IS 'Main components extracted from CAD files';
COMMENT ON TABLE parsed_files IS 'Metadata for parsed CAD files';
COMMENT ON TABLE parsed_components IS 'Mapping between parser output and database components';
COMMENT ON TABLE spatial_data IS 'Spatial coordinates and geometry data';
COMMENT ON TABLE materials IS 'Material definitions and properties';
COMMENT ON TABLE functions IS 'Functional classifications for components';
COMMENT ON TABLE families IS 'Product families and variants';
COMMENT ON TABLE suppliers IS 'Supplier information and sourcing data';

COMMENT ON VIEW component_summary IS 'Comprehensive view of components with all related data';
COMMENT ON VIEW parser_statistics IS 'Statistics on parser performance and file processing'; 