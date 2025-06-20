#!/usr/bin/env python3
"""
Fabrication Bundle Generator - Emergence-Driven CAD Automation
Only activates when emergence criteria are met:
- node_state.finalized == True
- emergence_flag == "ready_for_fabrication" 
- fit_score > threshold
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import traceback
from datetime import datetime
from pathlib import Path
import logging
import structlog

# Import modular components
from modules.blueprint_gen import BlueprintGenerator
from modules.bom_resolver import BOMResolver
from modules.compliance_checker import ComplianceChecker
from modules.supply_map import SupplyMapper

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = Flask(__name__)
CORS(app)

class FabricationBundleGenerator:
    """
    Emergence-driven fabrication bundle generator
    Converts finalized node states into complete fabrication packages
    """
    
    def __init__(self):
        self.blueprint_generator = BlueprintGenerator()
        self.bom_resolver = BOMResolver()
        self.compliance_checker = ComplianceChecker()
        self.supply_mapper = SupplyMapper()
        
        # Emergence thresholds
        self.fit_score_threshold = float(os.getenv("FIT_SCORE_THRESHOLD", "0.85"))
        self.base_output_dir = Path("/app/outputs")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Fabrication Bundle Generator initialized", 
                   fit_threshold=self.fit_score_threshold)
    
    def check_emergence_criteria(self, request_data: dict) -> tuple[bool, str]:
        """
        Check if emergence criteria are met for fabrication bundle generation
        
        Returns:
            (should_process: bool, reason: str)
        """
        # Extract key fields
        node_state = request_data.get("node_state", {})
        emergence_flag = request_data.get("emergence_flag")
        dictionary = request_data.get("dictionary", {})
        fit_score = dictionary.get("fit_score", 0.0)
        
        # Condition 1: Node state finalized
        if node_state.get("finalized") == True:
            return True, "node_state.finalized == True"
        
        # Condition 2: Explicit emergence flag
        if emergence_flag == "ready_for_fabrication":
            return True, f"emergence_flag == 'ready_for_fabrication'"
        
        # Condition 3: Fit score above threshold
        if fit_score > self.fit_score_threshold:
            return True, f"fit_score ({fit_score}) > threshold ({self.fit_score_threshold})"
        
        # No emergence criteria met
        return False, f"No emergence criteria met (finalized={node_state.get('finalized')}, flag={emergence_flag}, fit_score={fit_score})"
    
    def generate_fabrication_bundle(self, request_data: dict) -> dict:
        """
        Generate complete fabrication bundle for emergent component
        
        Expected input format:
        {
            "project_id": "ClusterAlpha",
            "geometry": "mesh.obj",
            "dictionary": {
                "material_type": "plywood",
                "component_id": "WALL_034",
                "fit_score": 0.96
            },
            "region": "PH-NCR", 
            "emergence_flag": "ready_for_fabrication"
        }
        """
        project_id = request_data.get("project_id", "unknown_project")
        component_id = request_data.get("dictionary", {}).get("component_id", "unknown_component")
        
        # Create project-specific output directory
        project_output_dir = self.base_output_dir / project_id
        project_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting fabrication bundle generation", 
                   project_id=project_id, 
                   component_id=component_id)
        
        try:
            # Generate all bundle components
            bundle_results = {}
            
            # 1. Blueprint Generation (2D DXF, 3D IFC/OBJ, STL, annotated PDFs)
            logger.info("Generating blueprints and CAD files")
            blueprint_result = self.blueprint_generator.generate_blueprints(
                request_data, project_output_dir
            )
            bundle_results["blueprints"] = blueprint_result
            
            # 2. BOM Resolution (live Bill of Materials)
            logger.info("Resolving Bill of Materials")
            bom_result = self.bom_resolver.generate_bom(
                request_data, project_output_dir
            )
            bundle_results["bom"] = bom_result
            
            # 3. Compliance Checking (regulatory validation)
            logger.info("Performing compliance validation")
            compliance_result = self.compliance_checker.validate_compliance(
                request_data, project_output_dir
            )
            bundle_results["compliance"] = compliance_result
            
            # 4. Supply Chain Mapping (verified suppliers)
            logger.info("Mapping verified suppliers")
            supply_result = self.supply_mapper.map_suppliers(
                bom_result["bom_data"], 
                request_data.get("region"), 
                project_output_dir
            )
            bundle_results["suppliers"] = supply_result
            
            # Generate bundle manifest
            bundle_manifest = self.create_bundle_manifest(
                request_data, bundle_results, project_output_dir
            )
            
            logger.info("Fabrication bundle generation completed", 
                       project_id=project_id,
                       files_generated=len(bundle_manifest["files"]))
            
            return {
                "status": "success",
                "project_id": project_id,
                "component_id": component_id,
                "bundle_path": str(project_output_dir),
                "manifest": bundle_manifest,
                "generated_at": datetime.now().isoformat(),
                "emergence_reason": self.check_emergence_criteria(request_data)[1]
            }
            
        except Exception as e:
            logger.error("Fabrication bundle generation failed", 
                        project_id=project_id,
                        error=str(e),
                        traceback=traceback.format_exc())
            raise
    
    def create_bundle_manifest(self, request_data: dict, bundle_results: dict, output_dir: Path) -> dict:
        """Create comprehensive bundle manifest"""
        manifest = {
            "bundle_info": {
                "project_id": request_data.get("project_id"),
                "component_id": request_data.get("dictionary", {}).get("component_id"),
                "material_type": request_data.get("dictionary", {}).get("material_type"),
                "fit_score": request_data.get("dictionary", {}).get("fit_score"),
                "region": request_data.get("region"),
                "generated_at": datetime.now().isoformat()
            },
            "files": {
                "blueprints": {
                    "layout_dxf": "layout.dxf",
                    "model_ifc": "model.ifc", 
                    "mesh_obj": "model.obj",
                    "print_stl": "model.stl",
                    "assembly_spec": "assembly_spec.pdf"
                },
                "bom": {
                    "bill_of_materials": "bill_of_materials.json",
                    "materials_list": "materials_list.xlsx"
                },
                "suppliers": {
                    "verified_suppliers": "verified_suppliers.csv",
                    "supplier_contacts": "supplier_contacts.json"
                },
                "compliance": {
                    "compliance_package": "compliance_package.zip",
                    "certifications": "certifications.pdf",
                    "regulatory_report": "regulatory_report.pdf"
                }
            },
            "fabrication_readiness": {
                "cad_complete": bundle_results.get("blueprints", {}).get("success", False),
                "bom_resolved": bundle_results.get("bom", {}).get("success", False),
                "suppliers_verified": bundle_results.get("suppliers", {}).get("success", False),
                "compliance_validated": bundle_results.get("compliance", {}).get("success", False),
                "ready_for_production": all([
                    bundle_results.get("blueprints", {}).get("success", False),
                    bundle_results.get("bom", {}).get("success", False),
                    bundle_results.get("suppliers", {}).get("success", False),
                    bundle_results.get("compliance", {}).get("success", False)
                ])
            },
            "delivery_targets": {
                "contractors": bundle_results.get("blueprints", {}).get("contractor_ready", False),
                "cnc_machines": bundle_results.get("blueprints", {}).get("cnc_ready", False),
                "3d_printers": bundle_results.get("blueprints", {}).get("print_ready", False),
                "permitting_authorities": bundle_results.get("compliance", {}).get("permit_ready", False)
            }
        }
        
        # Save manifest
        manifest_path = output_dir / "bundle_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest

# Initialize the generator
fabrication_generator = FabricationBundleGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "fabrication-bundle-gen",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "emergence_threshold": fabrication_generator.fit_score_threshold
    }), 200

@app.route('/process', methods=['POST'])
def process_fabrication_request():
    """
    Main endpoint for processing fabrication bundle requests
    Only processes when emergence criteria are met
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        request_data = data.get('data', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info("Received fabrication request", 
                   project_id=request_data.get("project_id"),
                   timestamp=timestamp)
        
        # Check emergence criteria
        should_process, reason = fabrication_generator.check_emergence_criteria(request_data)
        
        if not should_process:
            logger.info("Emergence criteria not met, skipping fabrication", 
                       reason=reason,
                       project_id=request_data.get("project_id"))
            
            return jsonify({
                "status": "skipped",
                "reason": reason,
                "emergence_criteria": {
                    "node_finalized": request_data.get("node_state", {}).get("finalized"),
                    "emergence_flag": request_data.get("emergence_flag"),
                    "fit_score": request_data.get("dictionary", {}).get("fit_score"),
                    "threshold": fabrication_generator.fit_score_threshold
                },
                "timestamp": datetime.now().isoformat()
            }), 200
        
        # Process fabrication bundle
        logger.info("Emergence criteria met, generating fabrication bundle", 
                   reason=reason)
        
        result = fabrication_generator.generate_fabrication_bundle(request_data)
        
        return jsonify({
            "status": "success",
            "message": "Fabrication bundle generated successfully",
            "result": result,
            "processing_time_ms": (datetime.now() - datetime.fromisoformat(timestamp.replace('Z', '+00:00'))).total_seconds() * 1000,
            "service": "fabrication-bundle-gen"
        }), 200
        
    except Exception as e:
        logger.error("Fabrication processing failed", 
                    error=str(e),
                    traceback=traceback.format_exc())
        
        return jsonify({
            "status": "error",
            "error": str(e),
            "service": "fabrication-bundle-gen",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/check-emergence', methods=['POST'])
def check_emergence_criteria():
    """
    Endpoint to check if emergence criteria are met without processing
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        request_data = data.get('data', {})
        
        should_process, reason = fabrication_generator.check_emergence_criteria(request_data)
        
        return jsonify({
            "emergence_ready": should_process,
            "reason": reason,
            "criteria_check": {
                "node_finalized": request_data.get("node_state", {}).get("finalized"),
                "emergence_flag": request_data.get("emergence_flag"),
                "fit_score": request_data.get("dictionary", {}).get("fit_score"),
                "threshold": fabrication_generator.fit_score_threshold
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error("Emergence check failed", error=str(e))
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/list-bundles', methods=['GET'])
def list_fabrication_bundles():
    """List all generated fabrication bundles"""
    try:
        bundles = []
        
        if fabrication_generator.base_output_dir.exists():
            for project_dir in fabrication_generator.base_output_dir.iterdir():
                if project_dir.is_dir():
                    manifest_path = project_dir / "bundle_manifest.json"
                    if manifest_path.exists():
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                        bundles.append({
                            "project_id": project_dir.name,
                            "path": str(project_dir),
                            "manifest": manifest
                        })
        
        return jsonify({
            "bundles": bundles,
            "total_count": len(bundles),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error("Bundle listing failed", error=str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Fabrication Bundle Generator service")
    app.run(host='0.0.0.0', port=5006, debug=False) 