#!/usr/bin/env python3
"""
BOM Resolver Module
Generates live Bill of Materials from emergent component specifications
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import structlog
from typing import Dict, Any, List

logger = structlog.get_logger()

class BOMResolver:
    """
    Resolves component specifications into detailed Bill of Materials
    """
    
    def __init__(self):
        self.logger = logger.bind(module="bom_resolver")
        self.material_database = self._load_material_database()
    
    def _load_material_database(self) -> dict:
        """Load material specifications database"""
        # In production, this would load from a comprehensive database
        return {
            "plywood": {
                "thickness_12mm": {"cost_per_m2": 45.50, "density_kg_m3": 600, "supplier_code": "PLY-12"},
                "thickness_18mm": {"cost_per_m2": 68.25, "density_kg_m3": 600, "supplier_code": "PLY-18"},
                "thickness_25mm": {"cost_per_m2": 95.00, "density_kg_m3": 600, "supplier_code": "PLY-25"}
            },
            "steel": {
                "grade_a36": {"cost_per_kg": 2.15, "density_kg_m3": 7850, "supplier_code": "STL-A36"},
                "grade_304": {"cost_per_kg": 8.50, "density_kg_m3": 8000, "supplier_code": "STL-304"}
            },
            "aluminum": {
                "6061_t6": {"cost_per_kg": 4.25, "density_kg_m3": 2700, "supplier_code": "ALU-6061"},
                "7075_t6": {"cost_per_kg": 12.50, "density_kg_m3": 2810, "supplier_code": "ALU-7075"}
            },
            "pla": {
                "standard": {"cost_per_kg": 25.00, "density_kg_m3": 1240, "supplier_code": "PLA-STD"},
                "high_temp": {"cost_per_kg": 45.00, "density_kg_m3": 1240, "supplier_code": "PLA-HT"}
            },
            "hardware": {
                "wood_screws_4x50": {"cost_per_unit": 0.15, "supplier_code": "SCR-W450"},
                "metal_bolts_m8x30": {"cost_per_unit": 1.25, "supplier_code": "BLT-M830"},
                "hinges_standard": {"cost_per_unit": 8.50, "supplier_code": "HNG-STD"}
            }
        }
    
    def generate_bom(self, request_data: dict, output_dir: Path) -> dict:
        """
        Generate comprehensive Bill of Materials from component data
        
        Output:
        - bill_of_materials.json: Structured BOM data
        - materials_list.xlsx: Excel spreadsheet for procurement
        """
        try:
            self.logger.info("Starting BOM generation")
            
            dictionary = request_data.get("dictionary", {})
            project_id = request_data.get("project_id", "unknown")
            
            # Calculate material requirements
            bom_items = self._calculate_material_requirements(dictionary)
            
            # Add hardware and fasteners
            hardware_items = self._calculate_hardware_requirements(dictionary)
            bom_items.extend(hardware_items)
            
            # Add labor estimates
            labor_items = self._calculate_labor_requirements(dictionary)
            bom_items.extend(labor_items)
            
            # Generate BOM structure
            bom_data = {
                "project_id": project_id,
                "component_id": dictionary.get("component_id", "unknown"),
                "generated_at": datetime.now().isoformat(),
                "total_material_cost": sum(item.get("total_cost", 0) for item in bom_items if item.get("category") == "material"),
                "total_hardware_cost": sum(item.get("total_cost", 0) for item in bom_items if item.get("category") == "hardware"),
                "total_labor_cost": sum(item.get("total_cost", 0) for item in bom_items if item.get("category") == "labor"),
                "items": bom_items
            }
            
            bom_data["total_project_cost"] = (
                bom_data["total_material_cost"] + 
                bom_data["total_hardware_cost"] + 
                bom_data["total_labor_cost"]
            )
            
            # Save JSON BOM
            json_path = output_dir / "bill_of_materials.json"
            with open(json_path, 'w') as f:
                json.dump(bom_data, f, indent=2)
            
            # Generate Excel spreadsheet
            excel_path = output_dir / "materials_list.xlsx"
            self._generate_excel_bom(bom_data, excel_path)
            
            self.logger.info("BOM generation completed", 
                           items_count=len(bom_items),
                           total_cost=bom_data["total_project_cost"])
            
            return {
                "success": True,
                "bom_data": bom_data,
                "files_generated": ["bill_of_materials.json", "materials_list.xlsx"]
            }
            
        except Exception as e:
            self.logger.error("BOM generation failed", error=str(e))
            return {
                "success": False,
                "bom_data": {},
                "files_generated": []
            }
    
    def _calculate_material_requirements(self, dictionary: dict) -> List[dict]:
        """Calculate primary material requirements"""
        items = []
        
        material_type = dictionary.get("material_type", "plywood").lower()
        
        # Calculate dimensions and volume
        width = dictionary.get("width", 1000.0) / 1000  # Convert to meters
        height = dictionary.get("height", 2400.0) / 1000
        depth = dictionary.get("depth", 200.0) / 1000
        
        area_m2 = width * height
        volume_m3 = width * height * depth
        
        if material_type == "plywood":
            # Determine plywood thickness based on component requirements
            thickness = self._determine_plywood_thickness(dictionary)
            thickness_key = f"thickness_{thickness}mm"
            
            if thickness_key in self.material_database["plywood"]:
                material_spec = self.material_database["plywood"][thickness_key]
                
                # Calculate sheets needed (assuming 1220x2440mm standard sheets)
                sheet_area = 1.22 * 2.44  # m²
                sheets_needed = max(1, int(area_m2 / sheet_area) + (1 if area_m2 % sheet_area > 0 else 0))
                
                cost_per_sheet = material_spec["cost_per_m2"] * sheet_area
                total_cost = cost_per_sheet * sheets_needed
                
                items.append({
                    "category": "material",
                    "description": f"Plywood {thickness}mm - Standard Sheet (1220x2440mm)",
                    "quantity": sheets_needed,
                    "unit": "sheets",
                    "unit_cost": cost_per_sheet,
                    "total_cost": total_cost,
                    "supplier_code": material_spec["supplier_code"],
                    "specifications": {
                        "thickness_mm": thickness,
                        "grade": "B/C or better",
                        "sheet_size": "1220x2440mm"
                    }
                })
        
        elif material_type in ["steel", "aluminum"]:
            metal_spec = self._determine_metal_specification(dictionary, material_type)
            if metal_spec:
                density = metal_spec["density_kg_m3"]
                weight_kg = volume_m3 * density
                total_cost = weight_kg * metal_spec["cost_per_kg"]
                
                items.append({
                    "category": "material",
                    "description": f"{material_type.title()} - {metal_spec.get('grade', 'Standard')}",
                    "quantity": weight_kg,
                    "unit": "kg",
                    "unit_cost": metal_spec["cost_per_kg"],
                    "total_cost": total_cost,
                    "supplier_code": metal_spec["supplier_code"],
                    "specifications": {
                        "grade": metal_spec.get('grade', 'Standard'),
                        "density_kg_m3": density,
                        "estimated_weight_kg": weight_kg
                    }
                })
        
        elif material_type in ["pla", "abs", "petg"]:
            # 3D printing materials
            filament_spec = self.material_database.get(material_type, {}).get("standard", {})
            if filament_spec:
                # Estimate filament weight (assuming 15% infill)
                infill_factor = 0.15
                density = filament_spec["density_kg_m3"]
                weight_kg = volume_m3 * density * infill_factor
                
                # Add 20% waste factor
                weight_kg *= 1.2
                
                total_cost = weight_kg * filament_spec["cost_per_kg"]
                
                items.append({
                    "category": "material",
                    "description": f"{material_type.upper()} Filament - 1.75mm",
                    "quantity": weight_kg,
                    "unit": "kg",
                    "unit_cost": filament_spec["cost_per_kg"],
                    "total_cost": total_cost,
                    "supplier_code": filament_spec["supplier_code"],
                    "specifications": {
                        "diameter": "1.75mm",
                        "infill_density": "15%",
                        "waste_factor": "20%"
                    }
                })
        
        return items
    
    def _determine_plywood_thickness(self, dictionary: dict) -> int:
        """Determine appropriate plywood thickness based on component requirements"""
        component_type = dictionary.get("component_type", "").lower()
        width = dictionary.get("width", 1000.0)
        height = dictionary.get("height", 2400.0)
        
        # Structural requirements
        if component_type in ["beam", "structural"]:
            return 25
        elif component_type in ["wall", "panel"] and max(width, height) > 1500:
            return 18
        else:
            return 12
    
    def _determine_metal_specification(self, dictionary: dict, material_type: str) -> dict:
        """Determine metal grade based on component requirements"""
        component_type = dictionary.get("component_type", "").lower()
        
        if material_type == "steel":
            if component_type in ["structural", "beam", "column"]:
                return {**self.material_database["steel"]["grade_a36"], "grade": "A36"}
            else:
                return {**self.material_database["steel"]["grade_304"], "grade": "304"}
        
        elif material_type == "aluminum":
            if component_type in ["aerospace", "high_strength"]:
                return {**self.material_database["aluminum"]["7075_t6"], "grade": "7075-T6"}
            else:
                return {**self.material_database["aluminum"]["6061_t6"], "grade": "6061-T6"}
        
        return {}
    
    def _calculate_hardware_requirements(self, dictionary: dict) -> List[dict]:
        """Calculate fasteners and hardware requirements"""
        items = []
        
        material_type = dictionary.get("material_type", "").lower()
        component_type = dictionary.get("component_type", "").lower()
        
        # Calculate component perimeter for fastener spacing
        width = dictionary.get("width", 1000.0)
        height = dictionary.get("height", 2400.0)
        perimeter = 2 * (width + height)
        
        if material_type == "plywood":
            # Wood screws for plywood assembly
            screw_spacing = 200  # mm
            screws_needed = max(8, int(perimeter / screw_spacing))
            
            screw_spec = self.material_database["hardware"]["wood_screws_4x50"]
            items.append({
                "category": "hardware",
                "description": "Wood Screws 4x50mm - Phillips Head",
                "quantity": screws_needed,
                "unit": "pieces",
                "unit_cost": screw_spec["cost_per_unit"],
                "total_cost": screws_needed * screw_spec["cost_per_unit"],
                "supplier_code": screw_spec["supplier_code"],
                "specifications": {
                    "size": "4x50mm",
                    "head_type": "Phillips",
                    "material": "Stainless Steel"
                }
            })
        
        elif material_type in ["steel", "aluminum"]:
            # Metal bolts for metal assembly
            bolt_spacing = 300  # mm
            bolts_needed = max(4, int(perimeter / bolt_spacing))
            
            bolt_spec = self.material_database["hardware"]["metal_bolts_m8x30"]
            items.append({
                "category": "hardware",
                "description": "Hex Bolts M8x30 with Nuts and Washers",
                "quantity": bolts_needed,
                "unit": "sets",
                "unit_cost": bolt_spec["cost_per_unit"],
                "total_cost": bolts_needed * bolt_spec["cost_per_unit"],
                "supplier_code": bolt_spec["supplier_code"],
                "specifications": {
                    "size": "M8x30",
                    "grade": "8.8",
                    "includes": "Bolt, Nut, 2x Washers"
                }
            })
        
        # Add hinges for door/panel components
        if component_type in ["door", "panel", "cabinet"]:
            hinges_needed = 2 if height < 1500 else 3
            
            hinge_spec = self.material_database["hardware"]["hinges_standard"]
            items.append({
                "category": "hardware",
                "description": "Standard Hinges - Heavy Duty",
                "quantity": hinges_needed,
                "unit": "pieces",
                "unit_cost": hinge_spec["cost_per_unit"],
                "total_cost": hinges_needed * hinge_spec["cost_per_unit"],
                "supplier_code": hinge_spec["supplier_code"],
                "specifications": {
                    "type": "Heavy Duty",
                    "load_rating": "50kg",
                    "finish": "Satin Chrome"
                }
            })
        
        return items
    
    def _calculate_labor_requirements(self, dictionary: dict) -> List[dict]:
        """Calculate labor cost estimates"""
        items = []
        
        material_type = dictionary.get("material_type", "").lower()
        component_type = dictionary.get("component_type", "").lower()
        
        # Base labor rates (per hour)
        labor_rates = {
            "fabrication": 65.00,    # Skilled fabrication
            "assembly": 45.00,       # Assembly work
            "finishing": 55.00,      # Finishing work
            "machining": 85.00,      # CNC/machining
            "3d_printing": 25.00     # 3D printing operation
        }
        
        # Estimate fabrication time based on complexity
        complexity_factor = self._calculate_complexity_factor(dictionary)
        base_hours = 2.0  # Base fabrication time
        
        if material_type == "plywood":
            # CNC cutting + assembly
            cnc_hours = base_hours * complexity_factor * 0.5
            assembly_hours = base_hours * complexity_factor * 1.0
            finishing_hours = base_hours * complexity_factor * 0.3
            
            items.extend([
                {
                    "category": "labor",
                    "description": "CNC Cutting and Machining",
                    "quantity": cnc_hours,
                    "unit": "hours",
                    "unit_cost": labor_rates["machining"],
                    "total_cost": cnc_hours * labor_rates["machining"],
                    "specifications": {"skill_level": "CNC Operator"}
                },
                {
                    "category": "labor",
                    "description": "Assembly and Joinery",
                    "quantity": assembly_hours,
                    "unit": "hours",
                    "unit_cost": labor_rates["assembly"],
                    "total_cost": assembly_hours * labor_rates["assembly"],
                    "specifications": {"skill_level": "Carpenter"}
                },
                {
                    "category": "labor",
                    "description": "Sanding and Finishing",
                    "quantity": finishing_hours,
                    "unit": "hours",
                    "unit_cost": labor_rates["finishing"],
                    "total_cost": finishing_hours * labor_rates["finishing"],
                    "specifications": {"skill_level": "Finisher"}
                }
            ])
        
        elif material_type in ["steel", "aluminum"]:
            # Metal fabrication
            fab_hours = base_hours * complexity_factor * 1.5
            assembly_hours = base_hours * complexity_factor * 0.8
            
            items.extend([
                {
                    "category": "labor",
                    "description": "Metal Fabrication and Welding",
                    "quantity": fab_hours,
                    "unit": "hours",
                    "unit_cost": labor_rates["fabrication"],
                    "total_cost": fab_hours * labor_rates["fabrication"],
                    "specifications": {"skill_level": "Certified Welder"}
                },
                {
                    "category": "labor",
                    "description": "Assembly and Finishing",
                    "quantity": assembly_hours,
                    "unit": "hours",
                    "unit_cost": labor_rates["assembly"],
                    "total_cost": assembly_hours * labor_rates["assembly"],
                    "specifications": {"skill_level": "Metal Worker"}
                }
            ])
        
        elif material_type in ["pla", "abs", "petg"]:
            # 3D printing
            print_hours = base_hours * complexity_factor * 2.0  # Printing is slow
            post_hours = base_hours * complexity_factor * 0.5   # Post-processing
            
            items.extend([
                {
                    "category": "labor",
                    "description": "3D Printing Operation",
                    "quantity": print_hours,
                    "unit": "hours",
                    "unit_cost": labor_rates["3d_printing"],
                    "total_cost": print_hours * labor_rates["3d_printing"],
                    "specifications": {"skill_level": "3D Print Operator"}
                },
                {
                    "category": "labor",
                    "description": "Post-Processing and Finishing",
                    "quantity": post_hours,
                    "unit": "hours",
                    "unit_cost": labor_rates["finishing"],
                    "total_cost": post_hours * labor_rates["finishing"],
                    "specifications": {"skill_level": "Technician"}
                }
            ])
        
        return items
    
    def _calculate_complexity_factor(self, dictionary: dict) -> float:
        """Calculate complexity multiplier based on component characteristics"""
        base_factor = 1.0
        
        # Size complexity
        width = dictionary.get("width", 1000.0)
        height = dictionary.get("height", 2400.0)
        max_dimension = max(width, height)
        
        if max_dimension > 2000:
            base_factor *= 1.3
        elif max_dimension > 3000:
            base_factor *= 1.6
        
        # Fit score complexity (higher fit = more precise = more complex)
        fit_score = dictionary.get("fit_score", 0.5)
        if fit_score > 0.9:
            base_factor *= 1.4
        elif fit_score > 0.8:
            base_factor *= 1.2
        
        # Component type complexity
        component_type = dictionary.get("component_type", "").lower()
        if component_type in ["structural", "precision"]:
            base_factor *= 1.5
        elif component_type in ["decorative", "simple"]:
            base_factor *= 0.8
        
        return base_factor
    
    def _generate_excel_bom(self, bom_data: dict, output_path: Path):
        """Generate Excel spreadsheet for procurement"""
        try:
            # Create DataFrame from BOM items
            df_items = pd.DataFrame(bom_data["items"])
            
            # Create summary data
            summary_data = {
                "Project Information": [
                    f"Project ID: {bom_data['project_id']}",
                    f"Component ID: {bom_data['component_id']}",
                    f"Generated: {bom_data['generated_at'][:10]}",
                    "",
                    f"Material Cost: ${bom_data['total_material_cost']:.2f}",
                    f"Hardware Cost: ${bom_data['total_hardware_cost']:.2f}",
                    f"Labor Cost: ${bom_data['total_labor_cost']:.2f}",
                    f"TOTAL COST: ${bom_data['total_project_cost']:.2f}"
                ]
            }
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Summary sheet
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Full BOM sheet
                df_items.to_excel(writer, sheet_name='Bill of Materials', index=False)
                
                # Material-specific sheets
                for category in ['material', 'hardware', 'labor']:
                    category_items = df_items[df_items['category'] == category]
                    if not category_items.empty:
                        category_items.to_excel(writer, sheet_name=category.title(), index=False)
            
            self.logger.info("Excel BOM generated", path=str(output_path))
            
        except Exception as e:
            self.logger.error("Excel BOM generation failed", error=str(e)) 