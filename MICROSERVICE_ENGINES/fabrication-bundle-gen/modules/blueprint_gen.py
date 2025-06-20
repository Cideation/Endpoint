#!/usr/bin/env python3
"""
Blueprint Generator Module
Converts emergent geometry into fabrication-ready CAD formats:
- 2D DXF layouts for CNC/laser cutting
- 3D IFC models for BIM integration  
- OBJ meshes for visualization
- STL files for 3D printing scenarios
- Annotated PDF specifications
"""

import os
import json
import numpy as np
import trimesh
import ezdxf
from pathlib import Path
from datetime import datetime
import structlog
from typing import Dict, Any, List

try:
    import ifcopenshell
    import ifcopenshell.api
    HAS_IFC = True
except ImportError:
    HAS_IFC = False

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

logger = structlog.get_logger()

class BlueprintGenerator:
    """
    Generates complete CAD blueprint packages from emergent geometry
    """
    
    def __init__(self):
        self.supported_formats = ['dxf', 'ifc', 'obj', 'stl', 'pdf']
        self.logger = logger.bind(module="blueprint_gen")
        
    def generate_blueprints(self, request_data: dict, output_dir: Path) -> dict:
        """
        Generate complete blueprint package from emergent component data
        
        Output structure:
        /outputs/ProjectID/
        ├── layout.dxf          # 2D CNC/laser cutting layout
        ├── model.ifc           # BIM-ready 3D model
        ├── model.obj           # 3D mesh for visualization
        ├── model.stl           # 3D printing ready
        └── assembly_spec.pdf   # Technical specifications
        """
        try:
            self.logger.info("Starting blueprint generation")
            
            # Extract component data
            project_id = request_data.get("project_id", "unknown")
            geometry_file = request_data.get("geometry", "")
            dictionary = request_data.get("dictionary", {})
            component_id = dictionary.get("component_id", "unknown_component")
            material_type = dictionary.get("material_type", "unknown_material")
            
            results = {
                "success": False,
                "files_generated": [],
                "errors": [],
                "contractor_ready": False,
                "cnc_ready": False,
                "print_ready": False
            }
            
            # Load or generate base geometry
            mesh_data = self._load_or_generate_geometry(geometry_file, dictionary)
            
            if mesh_data is None:
                results["errors"].append("Failed to load/generate geometry")
                return results
            
            # Generate 2D DXF layout
            dxf_success = self._generate_dxf_layout(
                mesh_data, component_id, material_type, output_dir
            )
            if dxf_success:
                results["files_generated"].append("layout.dxf")
                results["cnc_ready"] = True
            
            # Generate 3D IFC model
            ifc_success = self._generate_ifc_model(
                mesh_data, component_id, dictionary, output_dir
            )
            if ifc_success:
                results["files_generated"].append("model.ifc")
            
            # Generate OBJ mesh
            obj_success = self._generate_obj_mesh(
                mesh_data, component_id, output_dir
            )
            if obj_success:
                results["files_generated"].append("model.obj")
            
            # Generate STL for 3D printing
            stl_success = self._generate_stl_print(
                mesh_data, component_id, material_type, output_dir
            )
            if stl_success:
                results["files_generated"].append("model.stl")
                results["print_ready"] = True
            
            # Generate annotated PDF specifications
            pdf_success = self._generate_assembly_spec_pdf(
                request_data, mesh_data, output_dir
            )
            if pdf_success:
                results["files_generated"].append("assembly_spec.pdf")
                results["contractor_ready"] = True
            
            results["success"] = len(results["files_generated"]) > 0
            
            self.logger.info("Blueprint generation completed", 
                           files_count=len(results["files_generated"]),
                           cnc_ready=results["cnc_ready"],
                           print_ready=results["print_ready"])
            
            return results
            
        except Exception as e:
            self.logger.error("Blueprint generation failed", error=str(e))
            return {
                "success": False,
                "files_generated": [],
                "errors": [str(e)],
                "contractor_ready": False,
                "cnc_ready": False,
                "print_ready": False
            }
    
    def _load_or_generate_geometry(self, geometry_file: str, dictionary: dict) -> trimesh.Trimesh:
        """Load existing geometry or generate from dictionary parameters"""
        try:
            # Try to load existing geometry file
            if geometry_file and os.path.exists(geometry_file):
                mesh = trimesh.load(geometry_file)
                self.logger.info("Loaded existing geometry", file=geometry_file)
                return mesh
            
            # Generate geometry from dictionary parameters
            component_type = dictionary.get("component_type", "box")
            material_type = dictionary.get("material_type", "plywood")
            
            # Extract dimensions
            width = dictionary.get("width", 1000.0)  # mm
            height = dictionary.get("height", 2400.0)  # mm
            depth = dictionary.get("depth", 200.0)  # mm
            
            if component_type.lower() in ["wall", "panel"]:
                # Generate wall/panel geometry
                mesh = trimesh.creation.box(extents=[width, height, depth])
                
                # Add material-specific features
                if material_type.lower() == "plywood":
                    # Add typical plywood joinery features
                    mesh = self._add_plywood_features(mesh, width, height, depth)
                
            elif component_type.lower() in ["beam", "column"]:
                # Generate structural member
                if component_type.lower() == "beam":
                    mesh = trimesh.creation.box(extents=[width, depth, height])
                else:  # column
                    mesh = trimesh.creation.box(extents=[depth, depth, height])
            
            else:
                # Default box geometry
                mesh = trimesh.creation.box(extents=[width, height, depth])
            
            self.logger.info("Generated geometry from parameters", 
                           component_type=component_type,
                           dimensions=f"{width}x{height}x{depth}")
            
            return mesh
            
        except Exception as e:
            self.logger.error("Geometry loading/generation failed", error=str(e))
            return None
    
    def _add_plywood_features(self, mesh: trimesh.Trimesh, width: float, height: float, depth: float) -> trimesh.Trimesh:
        """Add plywood-specific joinery features"""
        # This would add finger joints, dados, etc. for plywood fabrication
        # For now, return the base mesh - real implementation would add CAM features
        return mesh
    
    def _generate_dxf_layout(self, mesh: trimesh.Trimesh, component_id: str, material_type: str, output_dir: Path) -> bool:
        """Generate 2D DXF layout for CNC/laser cutting"""
        try:
            # Create new DXF document
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Get mesh bounds
            bounds = mesh.bounds
            width = bounds[1][0] - bounds[0][0]
            height = bounds[1][1] - bounds[0][1]
            depth = bounds[1][2] - bounds[0][2]
            
            # Create cutting layout based on material type
            if material_type.lower() == "plywood":
                self._add_plywood_cutting_layout(msp, width, height, depth, component_id)
            else:
                self._add_generic_cutting_layout(msp, width, height, depth, component_id)
            
            # Add dimensions and annotations
            self._add_dxf_dimensions(msp, width, height, depth)
            self._add_dxf_annotations(msp, component_id, material_type)
            
            # Save DXF file
            dxf_path = output_dir / "layout.dxf"
            doc.saveas(dxf_path)
            
            self.logger.info("DXF layout generated", path=str(dxf_path))
            return True
            
        except Exception as e:
            self.logger.error("DXF generation failed", error=str(e))
            return False
    
    def _add_plywood_cutting_layout(self, msp, width: float, height: float, depth: float, component_id: str):
        """Add plywood-specific cutting layout with joinery"""
        # Main panel
        msp.add_lwpolyline([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
        
        # Side panels (if depth > thickness)
        if depth > 20:  # Assuming 20mm plywood thickness
            side_y_offset = height + 50
            msp.add_lwpolyline([(0, side_y_offset), (depth, side_y_offset), 
                               (depth, side_y_offset + height), (0, side_y_offset + height), (0, side_y_offset)])
            msp.add_lwpolyline([(depth + 50, side_y_offset), (depth + 50 + depth, side_y_offset), 
                               (depth + 50 + depth, side_y_offset + height), (depth + 50, side_y_offset + height), (depth + 50, side_y_offset)])
        
        # Add finger joints
        self._add_finger_joints(msp, width, height, depth)
    
    def _add_generic_cutting_layout(self, msp, width: float, height: float, depth: float, component_id: str):
        """Add generic cutting layout"""
        # Simple rectangular layout
        msp.add_lwpolyline([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
        
        # Add cut lines if needed
        if depth > 50:  # Add depth cuts for thick materials
            cut_spacing = width / 4
            for i in range(1, 4):
                x = i * cut_spacing
                msp.add_line((x, 0), (x, height))
    
    def _add_finger_joints(self, msp, width: float, height: float, depth: float):
        """Add finger joint patterns for plywood assembly"""
        finger_width = 20  # mm
        finger_count = int(width / finger_width)
        
        # Top edge fingers
        for i in range(finger_count):
            if i % 2 == 0:  # Create alternating fingers
                x_start = i * finger_width
                x_end = min((i + 1) * finger_width, width)
                # Add finger cutout
                msp.add_lwpolyline([
                    (x_start, height - depth), 
                    (x_end, height - depth),
                    (x_end, height), 
                    (x_start, height), 
                    (x_start, height - depth)
                ])
    
    def _add_dxf_dimensions(self, msp, width: float, height: float, depth: float):
        """Add dimensions to DXF layout"""
        # Width dimension
        msp.add_linear_dim(
            base=(width/2, -50),
            p1=(0, 0),
            p2=(width, 0),
            text=f"{width:.1f}mm"
        )
        
        # Height dimension  
        msp.add_linear_dim(
            base=(-50, height/2),
            p1=(0, 0),
            p2=(0, height),
            text=f"{height:.1f}mm"
        )
    
    def _add_dxf_annotations(self, msp, component_id: str, material_type: str):
        """Add text annotations to DXF"""
        # Component ID
        msp.add_text(
            component_id,
            dxfattribs={'height': 20, 'style': 'OpenSans'}
        ).set_pos((10, 10))
        
        # Material specification
        msp.add_text(
            f"Material: {material_type}",
            dxfattribs={'height': 15, 'style': 'OpenSans'}
        ).set_pos((10, 35))
    
    def _generate_ifc_model(self, mesh: trimesh.Trimesh, component_id: str, dictionary: dict, output_dir: Path) -> bool:
        """Generate IFC model for BIM integration"""
        if not HAS_IFC:
            self.logger.warning("IFC support not available, skipping IFC generation")
            return False
        
        try:
            # Create IFC file
            ifc_file = ifcopenshell.api.run("root.create_entity", model=ifcopenshell.file(), ifc_class="IfcProject")
            
            # Add basic project structure
            project = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcProject", name="BEM Fabrication Project")
            site = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcSite", name="Project Site")
            building = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcBuilding", name="Building")
            storey = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcBuildingStorey", name="Ground Floor")
            
            # Create building element
            element_type = dictionary.get("component_type", "wall").upper()
            ifc_class = f"Ifc{element_type}" if element_type in ["WALL", "BEAM", "COLUMN", "SLAB"] else "IfcBuildingElementProxy"
            
            element = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class=ifc_class, name=component_id)
            
            # Add geometry representation
            # This would include the actual mesh geometry - simplified for now
            
            # Save IFC file
            ifc_path = output_dir / "model.ifc"
            ifc_file.write(str(ifc_path))
            
            self.logger.info("IFC model generated", path=str(ifc_path))
            return True
            
        except Exception as e:
            self.logger.error("IFC generation failed", error=str(e))
            return False
    
    def _generate_obj_mesh(self, mesh: trimesh.Trimesh, component_id: str, output_dir: Path) -> bool:
        """Generate OBJ mesh for visualization"""
        try:
            obj_path = output_dir / "model.obj"
            mesh.export(str(obj_path))
            
            self.logger.info("OBJ mesh generated", path=str(obj_path))
            return True
            
        except Exception as e:
            self.logger.error("OBJ generation failed", error=str(e))
            return False
    
    def _generate_stl_print(self, mesh: trimesh.Trimesh, component_id: str, material_type: str, output_dir: Path) -> bool:
        """Generate STL file optimized for 3D printing"""
        try:
            # Optimize mesh for 3D printing
            print_mesh = mesh.copy()
            
            # Ensure mesh is watertight
            if not print_mesh.is_watertight:
                print_mesh.fill_holes()
            
            # Scale for printing if needed (convert mm to printer units)
            scale_factor = self._get_print_scale_factor(material_type)
            if scale_factor != 1.0:
                print_mesh.apply_scale(scale_factor)
            
            # Apply print-specific optimizations
            if material_type.lower() in ["pla", "abs", "petg"]:
                # Plastic printing optimizations
                print_mesh = self._optimize_for_plastic_printing(print_mesh)
            elif material_type.lower() in ["metal", "steel", "aluminum"]:
                # Metal printing optimizations
                print_mesh = self._optimize_for_metal_printing(print_mesh)
            
            # Export STL
            stl_path = output_dir / "model.stl"
            print_mesh.export(str(stl_path))
            
            # Generate print settings file
            self._generate_print_settings(material_type, output_dir)
            
            self.logger.info("STL file generated for 3D printing", 
                           path=str(stl_path),
                           material=material_type,
                           watertight=print_mesh.is_watertight)
            return True
            
        except Exception as e:
            self.logger.error("STL generation failed", error=str(e))
            return False
    
    def _get_print_scale_factor(self, material_type: str) -> float:
        """Get appropriate scale factor for printing material"""
        # Scale factors to optimize for different printing materials
        scale_factors = {
            "pla": 1.0,         # Standard scale
            "abs": 1.02,        # Slight scale up for shrinkage
            "petg": 1.0,        # Standard scale
            "metal": 1.05,      # Scale up for metal shrinkage
            "steel": 1.05,      # Scale up for metal shrinkage
            "aluminum": 1.03,   # Scale up for aluminum shrinkage
            "resin": 1.0        # Standard scale for resin
        }
        return scale_factors.get(material_type.lower(), 1.0)
    
    def _optimize_for_plastic_printing(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply optimizations for plastic 3D printing"""
        # Add support structures if needed (simplified)
        # Check for overhangs > 45 degrees
        # Ensure minimum wall thickness
        return mesh
    
    def _optimize_for_metal_printing(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply optimizations for metal 3D printing"""
        # Add powder escape holes
        # Ensure minimum wall thickness for metal printing
        # Optimize for heat dissipation
        return mesh
    
    def _generate_print_settings(self, material_type: str, output_dir: Path):
        """Generate 3D printer settings file"""
        print_settings = {
            "material": material_type,
            "layer_height": self._get_layer_height(material_type),
            "infill_density": self._get_infill_density(material_type),
            "print_speed": self._get_print_speed(material_type),
            "bed_temperature": self._get_bed_temp(material_type),
            "nozzle_temperature": self._get_nozzle_temp(material_type),
            "supports": self._needs_supports(material_type)
        }
        
        settings_path = output_dir / "print_settings.json"
        with open(settings_path, 'w') as f:
            json.dump(print_settings, f, indent=2)
    
    def _get_layer_height(self, material_type: str) -> float:
        """Get recommended layer height for material"""
        layer_heights = {
            "pla": 0.2, "abs": 0.2, "petg": 0.25,
            "metal": 0.05, "steel": 0.05, "aluminum": 0.08,
            "resin": 0.05
        }
        return layer_heights.get(material_type.lower(), 0.2)
    
    def _get_infill_density(self, material_type: str) -> int:
        """Get recommended infill density percentage"""
        infill_densities = {
            "pla": 20, "abs": 25, "petg": 20,
            "metal": 80, "steel": 90, "aluminum": 70,
            "resin": 100
        }
        return infill_densities.get(material_type.lower(), 20)
    
    def _get_print_speed(self, material_type: str) -> int:
        """Get recommended print speed in mm/s"""
        print_speeds = {
            "pla": 60, "abs": 50, "petg": 45,
            "metal": 20, "steel": 15, "aluminum": 25,
            "resin": 30
        }
        return print_speeds.get(material_type.lower(), 50)
    
    def _get_bed_temp(self, material_type: str) -> int:
        """Get recommended bed temperature in Celsius"""
        bed_temps = {
            "pla": 60, "abs": 100, "petg": 80,
            "metal": 200, "steel": 250, "aluminum": 180,
            "resin": 25
        }
        return bed_temps.get(material_type.lower(), 60)
    
    def _get_nozzle_temp(self, material_type: str) -> int:
        """Get recommended nozzle temperature in Celsius"""
        nozzle_temps = {
            "pla": 210, "abs": 250, "petg": 235,
            "metal": 400, "steel": 450, "aluminum": 380,
            "resin": 25
        }
        return nozzle_temps.get(material_type.lower(), 210)
    
    def _needs_supports(self, material_type: str) -> bool:
        """Determine if supports are typically needed"""
        needs_supports = {
            "pla": True, "abs": True, "petg": True,
            "metal": False, "steel": False, "aluminum": False,
            "resin": True
        }
        return needs_supports.get(material_type.lower(), True)
    
    def _generate_assembly_spec_pdf(self, request_data: dict, mesh: trimesh.Trimesh, output_dir: Path) -> bool:
        """Generate annotated PDF with assembly specifications"""
        try:
            pdf_path = output_dir / "assembly_spec.pdf"
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            
            project_id = request_data.get("project_id", "Unknown Project")
            component_id = request_data.get("dictionary", {}).get("component_id", "Unknown Component")
            
            story.append(Paragraph(f"Assembly Specification", title_style))
            story.append(Paragraph(f"Project: {project_id}", styles['Heading2']))
            story.append(Paragraph(f"Component: {component_id}", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Component details table
            dictionary = request_data.get("dictionary", {})
            details_data = [
                ['Property', 'Value'],
                ['Component ID', component_id],
                ['Material Type', dictionary.get("material_type", "Unknown")],
                ['Fit Score', f"{dictionary.get('fit_score', 0.0):.3f}"],
                ['Region', request_data.get("region", "Unknown")],
                ['Generated At', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            ]
            
            if mesh:
                bounds = mesh.bounds
                details_data.extend([
                    ['Width (mm)', f"{bounds[1][0] - bounds[0][0]:.1f}"],
                    ['Height (mm)', f"{bounds[1][1] - bounds[0][1]:.1f}"],
                    ['Depth (mm)', f"{bounds[1][2] - bounds[0][2]:.1f}"],
                    ['Volume (cm³)', f"{mesh.volume / 1000:.2f}"],
                    ['Surface Area (cm²)', f"{mesh.area / 100:.2f}"]
                ])
            
            table = Table(details_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 12))
            
            # Assembly instructions
            story.append(Paragraph("Assembly Instructions", styles['Heading2']))
            
            assembly_instructions = self._generate_assembly_instructions(dictionary)
            for instruction in assembly_instructions:
                story.append(Paragraph(f"• {instruction}", styles['Normal']))
            
            story.append(Spacer(1, 12))
            
            # Fabrication notes
            story.append(Paragraph("Fabrication Notes", styles['Heading2']))
            
            fab_notes = self._generate_fabrication_notes(dictionary)
            for note in fab_notes:
                story.append(Paragraph(f"• {note}", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info("Assembly specification PDF generated", path=str(pdf_path))
            return True
            
        except Exception as e:
            self.logger.error("PDF generation failed", error=str(e))
            return False
    
    def _generate_assembly_instructions(self, dictionary: dict) -> List[str]:
        """Generate assembly instructions based on component properties"""
        material_type = dictionary.get("material_type", "").lower()
        component_type = dictionary.get("component_type", "").lower()
        
        instructions = []
        
        if material_type == "plywood":
            instructions.extend([
                "Ensure all finger joints are dry-fitted before applying adhesive",
                "Use wood glue on all joint surfaces",
                "Clamp assembly for minimum 2 hours",
                "Sand all surfaces with 220-grit sandpaper after assembly"
            ])
        
        if component_type in ["wall", "panel"]:
            instructions.extend([
                "Check for square using diagonal measurements",
                "Install according to architectural drawings",
                "Ensure proper alignment with adjacent components"
            ])
        
        if "3d" in dictionary.get("fabrication_method", "").lower():
            instructions.extend([
                "Remove support material carefully with flush cutters",
                "Post-process according to printer manufacturer guidelines",
                "Test fit before final installation"
            ])
        
        return instructions
    
    def _generate_fabrication_notes(self, dictionary: dict) -> List[str]:
        """Generate fabrication-specific notes"""
        material_type = dictionary.get("material_type", "").lower()
        
        notes = []
        
        if material_type == "plywood":
            notes.extend([
                "Use sharp carbide bits to prevent tear-out",
                "Machine with grain direction where possible",
                "Seal cut edges before assembly"
            ])
        elif material_type in ["pla", "abs", "petg"]:
            notes.extend([
                "Print with 0.2mm layer height for optimal surface finish",
                "Use supports for overhangs greater than 45 degrees",
                "Allow print to cool completely before removal"
            ])
        elif material_type in ["metal", "steel", "aluminum"]:
            notes.extend([
                "Post-process with stress relief heat treatment",
                "Machine critical dimensions after printing",
                "Inspect for porosity in load-bearing areas"
            ])
        
        notes.extend([
            "Verify all dimensions before proceeding with fabrication",
            "Follow all safety protocols for material handling",
            "Document any deviations from specifications"
        ])
        
        return notes 