
-- Enhanced table: AcPropertyLabel
CREATE TABLE IF NOT EXISTS acpropertylabel_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ac_property_label TEXT
);
CREATE INDEX IF NOT EXISTS idx_acpropertylabel_id ON acpropertylabel_enhanced(id);

-- Enhanced table: AgentClass
CREATE TABLE IF NOT EXISTS agentclass_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_class TEXT
);
CREATE INDEX IF NOT EXISTS idx_agentclass_id ON agentclass_enhanced(id);

-- Enhanced table: AgentSubclass
CREATE TABLE IF NOT EXISTS agentsubclass_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_subclass TEXT
);
CREATE INDEX IF NOT EXISTS idx_agentsubclass_id ON agentsubclass_enhanced(id);

-- Enhanced table: Anchor
CREATE TABLE IF NOT EXISTS anchor_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    anchor_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_anchor_id ON anchor_enhanced(id);

-- Enhanced table: AnchorConfiguration
CREATE TABLE IF NOT EXISTS anchorconfiguration_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    anchor_configuration TEXT
);
CREATE INDEX IF NOT EXISTS idx_anchorconfiguration_id ON anchorconfiguration_enhanced(id);

-- Enhanced table: AnchorConstraints
CREATE TABLE IF NOT EXISTS anchorconstraints_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    anchor_constraints TEXT
);
CREATE INDEX IF NOT EXISTS idx_anchorconstraints_id ON anchorconstraints_enhanced(id);

-- Enhanced table: AnchorName
CREATE TABLE IF NOT EXISTS anchorname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    anchor_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_anchorname_id ON anchorname_enhanced(id);

-- Enhanced table: AnchorReference
CREATE TABLE IF NOT EXISTS anchorreference_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    anchor_reference TEXT
);
CREATE INDEX IF NOT EXISTS idx_anchorreference_id ON anchorreference_enhanced(id);

-- Enhanced table: AnchorType
CREATE TABLE IF NOT EXISTS anchortype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    anchor_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_anchortype_id ON anchortype_enhanced(id);

-- Enhanced table: AreaM2
CREATE TABLE IF NOT EXISTS aream2_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    area_m2 INTEGER
);
CREATE INDEX IF NOT EXISTS idx_aream2_id ON aream2_enhanced(id);

-- Enhanced table: AssignComponent
CREATE TABLE IF NOT EXISTS assigncomponent_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assign_component_id BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_assigncomponent_id ON assigncomponent_enhanced(id);

-- Enhanced table: AvgElevationM
CREATE TABLE IF NOT EXISTS avgelevationm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    avg_elevation_m DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_avgelevationm_id ON avgelevationm_enhanced(id);

-- Enhanced table: Axis
CREATE TABLE IF NOT EXISTS axis_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    axis TEXT
);
CREATE INDEX IF NOT EXISTS idx_axis_id ON axis_enhanced(id);

-- Enhanced table: BaseMaterial
CREATE TABLE IF NOT EXISTS basematerial_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    base_material TEXT
);
CREATE INDEX IF NOT EXISTS idx_basematerial_id ON basematerial_enhanced(id);

-- Enhanced table: BboxXMm
CREATE TABLE IF NOT EXISTS bboxxmm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bbox_x_mm INTEGER
);
CREATE INDEX IF NOT EXISTS idx_bboxxmm_id ON bboxxmm_enhanced(id);

-- Enhanced table: BboxYMm
CREATE TABLE IF NOT EXISTS bboxymm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bbox_y_mm INTEGER
);
CREATE INDEX IF NOT EXISTS idx_bboxymm_id ON bboxymm_enhanced(id);

-- Enhanced table: BboxZMm
CREATE TABLE IF NOT EXISTS bboxzmm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bbox_z_mm INTEGER
);
CREATE INDEX IF NOT EXISTS idx_bboxzmm_id ON bboxzmm_enhanced(id);

-- Enhanced table: BoundaryCoordinates
CREATE TABLE IF NOT EXISTS boundarycoordinates_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    boundary_coordinates TEXT
);
CREATE INDEX IF NOT EXISTS idx_boundarycoordinates_id ON boundarycoordinates_enhanced(id);

-- Enhanced table: BoundingBoxType
CREATE TABLE IF NOT EXISTS boundingboxtype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bounding_box_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_boundingboxtype_id ON boundingboxtype_enhanced(id);

-- Enhanced table: BoundingBoxVolumeCm3
CREATE TABLE IF NOT EXISTS boundingboxvolumecm3_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bounding_box_volume_cm3 DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_boundingboxvolumecm3_id ON boundingboxvolumecm3_enhanced(id);

-- Enhanced table: CentroidX
CREATE TABLE IF NOT EXISTS centroidx_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    centroid_x_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_centroidx_id ON centroidx_enhanced(id);

-- Enhanced table: CentroidY
CREATE TABLE IF NOT EXISTS centroidy_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    centroid_y_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_centroidy_id ON centroidy_enhanced(id);

-- Enhanced table: CentroidZ
CREATE TABLE IF NOT EXISTS centroidz_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    centroid_z_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_centroidz_id ON centroidz_enhanced(id);

-- Enhanced table: CoefficientName
CREATE TABLE IF NOT EXISTS coefficientname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    coefficient_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_coefficientname_id ON coefficientname_enhanced(id);

-- Enhanced table: Component
CREATE TABLE IF NOT EXISTS component_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    component_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_component_id ON component_enhanced(id);

-- Enhanced table: ContactPerson
CREATE TABLE IF NOT EXISTS contactperson_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    contact_person TEXT
);
CREATE INDEX IF NOT EXISTS idx_contactperson_id ON contactperson_enhanced(id);

-- Enhanced table: Contract
CREATE TABLE IF NOT EXISTS contract_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    contract_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_contract_id ON contract_enhanced(id);

-- Enhanced table: Crs
CREATE TABLE IF NOT EXISTS crs_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    crs TEXT
);
CREATE INDEX IF NOT EXISTS idx_crs_id ON crs_enhanced(id);

-- Enhanced table: DataType
CREATE TABLE IF NOT EXISTS datatype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_datatype_id ON datatype_enhanced(id);

-- Enhanced table: Description
CREATE TABLE IF NOT EXISTS description_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);
CREATE INDEX IF NOT EXISTS idx_description_id ON description_enhanced(id);

-- Enhanced table: EdgeType
CREATE TABLE IF NOT EXISTS edgetype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    edge_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_edgetype_id ON edgetype_enhanced(id);

-- Enhanced table: EdgeValue
CREATE TABLE IF NOT EXISTS edgevalue_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    edge_value TEXT
);
CREATE INDEX IF NOT EXISTS idx_edgevalue_id ON edgevalue_enhanced(id);

-- Enhanced table: ElevationRange
CREATE TABLE IF NOT EXISTS elevationrange_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    elevation_range TEXT
);
CREATE INDEX IF NOT EXISTS idx_elevationrange_id ON elevationrange_enhanced(id);

-- Enhanced table: Email
CREATE TABLE IF NOT EXISTS email_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    email TEXT
);
CREATE INDEX IF NOT EXISTS idx_email_id ON email_enhanced(id);

-- Enhanced table: Family
CREATE TABLE IF NOT EXISTS family_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    family_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_family_id ON family_enhanced(id);

-- Enhanced table: FamilyName
CREATE TABLE IF NOT EXISTS familyname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    family_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_familyname_id ON familyname_enhanced(id);

-- Enhanced table: FasteningType
CREATE TABLE IF NOT EXISTS fasteningtype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fastening_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_fasteningtype_id ON fasteningtype_enhanced(id);

-- Enhanced table: FieldName
CREATE TABLE IF NOT EXISTS fieldname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    field_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_fieldname_id ON fieldname_enhanced(id);

-- Enhanced table: Formula
CREATE TABLE IF NOT EXISTS formula_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    formula TEXT
);
CREATE INDEX IF NOT EXISTS idx_formula_id ON formula_enhanced(id);

-- Enhanced table: Function
CREATE TABLE IF NOT EXISTS function_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    function_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_function_id ON function_enhanced(id);

-- Enhanced table: FunctionName
CREATE TABLE IF NOT EXISTS functionname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    function_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_functionname_id ON functionname_enhanced(id);

-- Enhanced table: Functor
CREATE TABLE IF NOT EXISTS functor_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    functor_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_functor_id ON functor_enhanced(id);

-- Enhanced table: FunctorName
CREATE TABLE IF NOT EXISTS functorname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    functor_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_functorname_id ON functorname_enhanced(id);

-- Enhanced table: FunctorType
CREATE TABLE IF NOT EXISTS functortype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    functor_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_functortype_id ON functortype_enhanced(id);

-- Enhanced table: GeometryFormat
CREATE TABLE IF NOT EXISTS geometryformat_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    geometry_format TEXT
);
CREATE INDEX IF NOT EXISTS idx_geometryformat_id ON geometryformat_enhanced(id);

-- Enhanced table: HeightMm
CREATE TABLE IF NOT EXISTS heightmm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    height_mm DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_heightmm_id ON heightmm_enhanced(id);

-- Enhanced table: InputFieldNames
CREATE TABLE IF NOT EXISTS inputfieldnames_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_field_names TEXT
);
CREATE INDEX IF NOT EXISTS idx_inputfieldnames_id ON inputfieldnames_enhanced(id);

-- Enhanced table: InputFields
CREATE TABLE IF NOT EXISTS inputfields_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_field_ids_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_inputfields_id ON inputfields_enhanced(id);

-- Enhanced table: IsRequired
CREATE TABLE IF NOT EXISTS isrequired_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_required BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_isrequired_id ON isrequired_enhanced(id);

-- Enhanced table: IsReusable
CREATE TABLE IF NOT EXISTS isreusable_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_reusable BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_isreusable_id ON isreusable_enhanced(id);

-- Enhanced table: IsRootNode
CREATE TABLE IF NOT EXISTS isrootnode_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_root_node BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_isrootnode_id ON isrootnode_enhanced(id);

-- Enhanced table: ItemType
CREATE TABLE IF NOT EXISTS itemtype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    item_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_itemtype_id ON itemtype_enhanced(id);

-- Enhanced table: JointName
CREATE TABLE IF NOT EXISTS jointname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    joint_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_jointname_id ON jointname_enhanced(id);

-- Enhanced table: JointType
CREATE TABLE IF NOT EXISTS jointtype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    joint_type_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_jointtype_id ON jointtype_enhanced(id);

-- Enhanced table: LastUpdated
CREATE TABLE IF NOT EXISTS lastupdated_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TEXT
);
CREATE INDEX IF NOT EXISTS idx_lastupdated_id ON lastupdated_enhanced(id);

-- Enhanced table: LeadTimeDays
CREATE TABLE IF NOT EXISTS leadtimedays_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    lead_time_days TEXT
);
CREATE INDEX IF NOT EXISTS idx_leadtimedays_id ON leadtimedays_enhanced(id);

-- Enhanced table: LengthMm
CREATE TABLE IF NOT EXISTS lengthmm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    length_mm DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_lengthmm_id ON lengthmm_enhanced(id);

-- Enhanced table: LoadRatingN
CREATE TABLE IF NOT EXISTS loadratingn_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    load_rating_n TEXT
);
CREATE INDEX IF NOT EXISTS idx_loadratingn_id ON loadratingn_enhanced(id);

-- Enhanced table: Location
CREATE TABLE IF NOT EXISTS location_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location TEXT,
    location_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_location_id ON location_enhanced(id);

-- Enhanced table: Material
CREATE TABLE IF NOT EXISTS material_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    material_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_material_id ON material_enhanced(id);

-- Enhanced table: MaxSlopeDeg
CREATE TABLE IF NOT EXISTS maxslopedeg_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    max_slope_deg DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_maxslopedeg_id ON maxslopedeg_enhanced(id);

-- Enhanced table: Method
CREATE TABLE IF NOT EXISTS method_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    method_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_method_id ON method_enhanced(id);

-- Enhanced table: MethodName
CREATE TABLE IF NOT EXISTS methodname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    method_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_methodname_id ON methodname_enhanced(id);

-- Enhanced table: NodeName
CREATE TABLE IF NOT EXISTS nodename_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    node_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodename_id ON nodename_enhanced(id);

-- Enhanced table: Nodeid
CREATE TABLE IF NOT EXISTS nodeid_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    nodeid_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodeid_id ON nodeid_enhanced(id);

-- Enhanced table: OctreeDepthAnchor
CREATE TABLE IF NOT EXISTS octreedepthanchor_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    octree_depth_anchor TEXT
);
CREATE INDEX IF NOT EXISTS idx_octreedepthanchor_id ON octreedepthanchor_enhanced(id);

-- Enhanced table: OctreeLevel
CREATE TABLE IF NOT EXISTS octreelevel_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    octree_level INTEGER
);
CREATE INDEX IF NOT EXISTS idx_octreelevel_id ON octreelevel_enhanced(id);

-- Enhanced table: OctreeSizeAnchor
CREATE TABLE IF NOT EXISTS octreesizeanchor_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    octree_size_anchor TEXT
);
CREATE INDEX IF NOT EXISTS idx_octreesizeanchor_id ON octreesizeanchor_enhanced(id);

-- Enhanced table: OutputFields
CREATE TABLE IF NOT EXISTS outputfields_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    output_fields TEXT
);
CREATE INDEX IF NOT EXISTS idx_outputfields_id ON outputfields_enhanced(id);

-- Enhanced table: OutputType
CREATE TABLE IF NOT EXISTS outputtype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    output_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_outputtype_id ON outputtype_enhanced(id);

-- Enhanced table: Owner
CREATE TABLE IF NOT EXISTS owner_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    owner_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_owner_id ON owner_enhanced(id);

-- Enhanced table: OwnershipType
CREATE TABLE IF NOT EXISTS ownershiptype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ownership_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_ownershiptype_id ON ownershiptype_enhanced(id);

-- Enhanced table: Phase
CREATE TABLE IF NOT EXISTS phase_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phase TEXT
);
CREATE INDEX IF NOT EXISTS idx_phase_id ON phase_enhanced(id);

-- Enhanced table: Phone
CREATE TABLE IF NOT EXISTS phone_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phone TEXT
);
CREATE INDEX IF NOT EXISTS idx_phone_id ON phone_enhanced(id);

-- Enhanced table: PointType
CREATE TABLE IF NOT EXISTS pointtype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    point_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_pointtype_id ON pointtype_enhanced(id);

-- Enhanced table: ProductFamily
CREATE TABLE IF NOT EXISTS productfamily_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    product_family TEXT
);
CREATE INDEX IF NOT EXISTS idx_productfamily_id ON productfamily_enhanced(id);

-- Enhanced table: ProductFamilyType
CREATE TABLE IF NOT EXISTS productfamilytype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    product_family_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_productfamilytype_id ON productfamilytype_enhanced(id);

-- Enhanced table: PropertyKey
CREATE TABLE IF NOT EXISTS propertykey_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    property_key TEXT
);
CREATE INDEX IF NOT EXISTS idx_propertykey_id ON propertykey_enhanced(id);

-- Enhanced table: PropertyName
CREATE TABLE IF NOT EXISTS propertyname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    property_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_propertyname_id ON propertyname_enhanced(id);

-- Enhanced table: Purpose
CREATE TABLE IF NOT EXISTS purpose_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    purpose TEXT
);
CREATE INDEX IF NOT EXISTS idx_purpose_id ON purpose_enhanced(id);

-- Enhanced table: Quantity
CREATE TABLE IF NOT EXISTS quantity_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quantity TEXT
);
CREATE INDEX IF NOT EXISTS idx_quantity_id ON quantity_enhanced(id);

-- Enhanced table: QuantityAvailable
CREATE TABLE IF NOT EXISTS quantityavailable_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quantity_available TEXT
);
CREATE INDEX IF NOT EXISTS idx_quantityavailable_id ON quantityavailable_enhanced(id);

-- Enhanced table: RangeMm
CREATE TABLE IF NOT EXISTS rangemm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    range_mm TEXT
);
CREATE INDEX IF NOT EXISTS idx_rangemm_id ON rangemm_enhanced(id);

-- Enhanced table: Requirement
CREATE TABLE IF NOT EXISTS requirement_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    requirement_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_requirement_id ON requirement_enhanced(id);

-- Enhanced table: Reversible
CREATE TABLE IF NOT EXISTS reversible_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reversible TEXT
);
CREATE INDEX IF NOT EXISTS idx_reversible_id ON reversible_enhanced(id);

-- Enhanced table: SlopeClass
CREATE TABLE IF NOT EXISTS slopeclass_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    slope_class TEXT
);
CREATE INDEX IF NOT EXISTS idx_slopeclass_id ON slopeclass_enhanced(id);

-- Enhanced table: SoilCategory
CREATE TABLE IF NOT EXISTS soilcategory_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    soil_category TEXT
);
CREATE INDEX IF NOT EXISTS idx_soilcategory_id ON soilcategory_enhanced(id);

-- Enhanced table: SourceComponent
CREATE TABLE IF NOT EXISTS sourcecomponent_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_component_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourcecomponent_id ON sourcecomponent_enhanced(id);

-- Enhanced table: SourceInventory
CREATE TABLE IF NOT EXISTS sourceinventory_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_inventory_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourceinventory_id ON sourceinventory_enhanced(id);

-- Enhanced table: SourceItem
CREATE TABLE IF NOT EXISTS sourceitem_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_item_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourceitem_id ON sourceitem_enhanced(id);

-- Enhanced table: SourceJoint
CREATE TABLE IF NOT EXISTS sourcejoint_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_joint_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourcejoint_id ON sourcejoint_enhanced(id);

-- Enhanced table: SourceLand
CREATE TABLE IF NOT EXISTS sourceland_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_land_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourceland_id ON sourceland_enhanced(id);

-- Enhanced table: SourceMaterialVariant
CREATE TABLE IF NOT EXISTS sourcematerialvariant_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_material_variant_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourcematerialvariant_id ON sourcematerialvariant_enhanced(id);

-- Enhanced table: SourceProductFamily
CREATE TABLE IF NOT EXISTS sourceproductfamily_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_product_family_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourceproductfamily_id ON sourceproductfamily_enhanced(id);

-- Enhanced table: SourceSubpartList
CREATE TABLE IF NOT EXISTS sourcesubpartlist_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_subpart_id_list_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourcesubpartlist_id ON sourcesubpartlist_enhanced(id);

-- Enhanced table: SourceType
CREATE TABLE IF NOT EXISTS sourcetype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourcetype_id ON sourcetype_enhanced(id);

-- Enhanced table: SourceWithdrawal
CREATE TABLE IF NOT EXISTS sourcewithdrawal_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_withdrawal_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourcewithdrawal_id ON sourcewithdrawal_enhanced(id);

-- Enhanced table: SpecLevel
CREATE TABLE IF NOT EXISTS speclevel_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    spec_level TEXT
);
CREATE INDEX IF NOT EXISTS idx_speclevel_id ON speclevel_enhanced(id);

-- Enhanced table: StandardReference
CREATE TABLE IF NOT EXISTS standardreference_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    standard_reference TEXT
);
CREATE INDEX IF NOT EXISTS idx_standardreference_id ON standardreference_enhanced(id);

-- Enhanced table: Subpart
CREATE TABLE IF NOT EXISTS subpart_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    subpart_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_subpart_id ON subpart_enhanced(id);

-- Enhanced table: SubpartFamily
CREATE TABLE IF NOT EXISTS subpartfamily_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    subpart_family TEXT
);
CREATE INDEX IF NOT EXISTS idx_subpartfamily_id ON subpartfamily_enhanced(id);

-- Enhanced table: SubpartName
CREATE TABLE IF NOT EXISTS subpartname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    subpart_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_subpartname_id ON subpartname_enhanced(id);

-- Enhanced table: Subparts
CREATE TABLE IF NOT EXISTS subparts_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    subpart_ids_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_subparts_id ON subparts_enhanced(id);

-- Enhanced table: Supplier
CREATE TABLE IF NOT EXISTS supplier_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    supplier_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_supplier_id ON supplier_enhanced(id);

-- Enhanced table: SupplierName
CREATE TABLE IF NOT EXISTS suppliername_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    supplier_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_suppliername_id ON suppliername_enhanced(id);

-- Enhanced table: SupplierType
CREATE TABLE IF NOT EXISTS suppliertype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    supplier_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_suppliertype_id ON suppliertype_enhanced(id);

-- Enhanced table: TargetBboxXMm
CREATE TABLE IF NOT EXISTS targetbboxxmm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_bbox_x_mm INTEGER
);
CREATE INDEX IF NOT EXISTS idx_targetbboxxmm_id ON targetbboxxmm_enhanced(id);

-- Enhanced table: TargetComponentName
CREATE TABLE IF NOT EXISTS targetcomponentname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_component_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_targetcomponentname_id ON targetcomponentname_enhanced(id);

-- Enhanced table: TargetFromSubpart
CREATE TABLE IF NOT EXISTS targetfromsubpart_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_from_subpart_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_targetfromsubpart_id ON targetfromsubpart_enhanced(id);

-- Enhanced table: TargetItem
CREATE TABLE IF NOT EXISTS targetitem_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_item_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_targetitem_id ON targetitem_enhanced(id);

-- Enhanced table: TargetItemType
CREATE TABLE IF NOT EXISTS targetitemtype_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_item_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_targetitemtype_id ON targetitemtype_enhanced(id);

-- Enhanced table: TargetMaterialVariantCode
CREATE TABLE IF NOT EXISTS targetmaterialvariantcode_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_material_variant_code_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_targetmaterialvariantcode_id ON targetmaterialvariantcode_enhanced(id);

-- Enhanced table: TargetParcelName
CREATE TABLE IF NOT EXISTS targetparcelname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_parcel_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_targetparcelname_id ON targetparcelname_enhanced(id);

-- Enhanced table: TargetTopologicCandidate
CREATE TABLE IF NOT EXISTS targettopologiccandidate_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_topologic_candidate_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_targettopologiccandidate_id ON targettopologiccandidate_enhanced(id);

-- Enhanced table: TerrainNotes
CREATE TABLE IF NOT EXISTS terrainnotes_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    terrain_notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_terrainnotes_id ON terrainnotes_enhanced(id);

-- Enhanced table: Timestamp
CREATE TABLE IF NOT EXISTS timestamp_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    timestamp TEXT
);
CREATE INDEX IF NOT EXISTS idx_timestamp_id ON timestamp_enhanced(id);

-- Enhanced table: ToSubpart
CREATE TABLE IF NOT EXISTS tosubpart_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    to_subpart_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_tosubpart_id ON tosubpart_enhanced(id);

-- Enhanced table: ToleranceLabel
CREATE TABLE IF NOT EXISTS tolerancelabel_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tolerance_label TEXT
);
CREATE INDEX IF NOT EXISTS idx_tolerancelabel_id ON tolerancelabel_enhanced(id);

-- Enhanced table: ToleranceMm
CREATE TABLE IF NOT EXISTS tolerancemm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tolerance_mm DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_tolerancemm_id ON tolerancemm_enhanced(id);

-- Enhanced table: TopologicReference
CREATE TABLE IF NOT EXISTS topologicreference_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    topologic_reference TEXT
);
CREATE INDEX IF NOT EXISTS idx_topologicreference_id ON topologicreference_enhanced(id);

-- Enhanced table: TopologicRole
CREATE TABLE IF NOT EXISTS topologicrole_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    topologic_role TEXT
);
CREATE INDEX IF NOT EXISTS idx_topologicrole_id ON topologicrole_enhanced(id);

-- Enhanced table: TopologicVertexAnchor
CREATE TABLE IF NOT EXISTS topologicvertexanchor_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    topologic_vertex_anchor TEXT
);
CREATE INDEX IF NOT EXISTS idx_topologicvertexanchor_id ON topologicvertexanchor_enhanced(id);

-- Enhanced table: Unit
CREATE TABLE IF NOT EXISTS unit_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    unit TEXT
);
CREATE INDEX IF NOT EXISTS idx_unit_id ON unit_enhanced(id);

-- Enhanced table: UnitCost
CREATE TABLE IF NOT EXISTS unitcost_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    unit_cost TEXT
);
CREATE INDEX IF NOT EXISTS idx_unitcost_id ON unitcost_enhanced(id);

-- Enhanced table: Value
CREATE TABLE IF NOT EXISTS value_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    value INTEGER
);
CREATE INDEX IF NOT EXISTS idx_value_id ON value_enhanced(id);

-- Enhanced table: VariableDomain
CREATE TABLE IF NOT EXISTS variabledomain_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    variable_domain TEXT
);
CREATE INDEX IF NOT EXISTS idx_variabledomain_id ON variabledomain_enhanced(id);

-- Enhanced table: VariableName
CREATE TABLE IF NOT EXISTS variablename_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    variable_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_variablename_id ON variablename_enhanced(id);

-- Enhanced table: Variant
CREATE TABLE IF NOT EXISTS variant_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    variant_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_variant_id ON variant_enhanced(id);

-- Enhanced table: VariantName
CREATE TABLE IF NOT EXISTS variantname_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    variant_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_variantname_id ON variantname_enhanced(id);

-- Enhanced table: VertexProperty
CREATE TABLE IF NOT EXISTS vertexproperty_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vertex_property TEXT
);
CREATE INDEX IF NOT EXISTS idx_vertexproperty_id ON vertexproperty_enhanced(id);

-- Enhanced table: WidthMm
CREATE TABLE IF NOT EXISTS widthmm_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    width_mm_id DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_widthmm_id ON widthmm_enhanced(id);

-- Enhanced table: ZoningClass
CREATE TABLE IF NOT EXISTS zoningclass_enhanced (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    zoning_class TEXT
);
CREATE INDEX IF NOT EXISTS idx_zoningclass_id ON zoningclass_enhanced(id);

-- Relationship tables for complex associations
CREATE TABLE IF NOT EXISTS component_hierarchy (
    id SERIAL PRIMARY KEY,
    parent_component_id TEXT,
    child_component_id TEXT,
    relationship_type TEXT,
    hierarchy_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS material_variants (
    id SERIAL PRIMARY KEY,
    base_material_id TEXT,
    variant_code TEXT,
    variant_name TEXT,
    properties JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS manufacturing_methods (
    id SERIAL PRIMARY KEY,
    method_id TEXT,
    method_name TEXT,
    capabilities JSONB,
    tolerances JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS spatial_references (
    id SERIAL PRIMARY KEY,
    reference_id TEXT,
    reference_type TEXT,
    coordinates GEOMETRY,
    properties JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Materialized views for analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS component_analytics AS
SELECT 
    c.component_id,
    c.product_family,
    c.material_id,
    c.bbox_x_mm,
    c.bbox_y_mm,
    c.bbox_z_mm,
    (c.bbox_x_mm * c.bbox_y_mm * c.bbox_z_mm) / 1000.0 as volume_cm3,
    c.is_root_node,
    c.is_reusable
FROM component_enhanced c;

CREATE MATERIALIZED VIEW IF NOT EXISTS supply_chain_analytics AS
SELECT 
    s.supplier_id,
    s.supplier_name,
    s.supplier_type,
    COUNT(i.source_item_id) as item_count,
    AVG(CAST(i.unit_cost AS INTEGER)) as avg_unit_cost
FROM supplier_enhanced s
LEFT JOIN source_item_enhanced i ON s.supplier_id = i.supplier_id
GROUP BY s.supplier_id, s.supplier_name, s.supplier_type;

-- Refresh functions
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW component_analytics;
    REFRESH MATERIALIZED VIEW supply_chain_analytics;
END;
$$ LANGUAGE plpgsql;
