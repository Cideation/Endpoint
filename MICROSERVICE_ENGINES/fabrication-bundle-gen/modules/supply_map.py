#!/usr/bin/env python3
"""
Supply Mapper Module
Maps BOM items to verified suppliers
"""

import csv
import json
from pathlib import Path
from datetime import datetime
import structlog

logger = structlog.get_logger()

class SupplyMapper:
    """Maps materials to verified suppliers"""
    
    def __init__(self):
        self.logger = logger.bind(module="supply_mapper")
        self.supplier_database = self._load_supplier_database()
    
    def _load_supplier_database(self) -> dict:
        """Load verified supplier database"""
        return {
            "PLY-12": {"name": "Manila Plywood Corp", "contact": "sales@mpc.ph", "lead_time": "3-5 days"},
            "STL-A36": {"name": "Philippine Steel Corp", "contact": "orders@psc.com.ph", "lead_time": "7-10 days"},
            "PLA-STD": {"name": "3D Printing Solutions", "contact": "info@3dps.ph", "lead_time": "1-2 days"}
        }
    
    def map_suppliers(self, bom_data: dict, region: str, output_dir: Path) -> dict:
        """Map BOM items to verified suppliers"""
        try:
            suppliers = []
            
            for item in bom_data.get("items", []):
                supplier_code = item.get("supplier_code")
                if supplier_code and supplier_code in self.supplier_database:
                    supplier_info = self.supplier_database[supplier_code]
                    suppliers.append({
                        "item_description": item["description"],
                        "supplier_name": supplier_info["name"],
                        "contact": supplier_info["contact"],
                        "lead_time": supplier_info["lead_time"],
                        "quantity": item["quantity"],
                        "estimated_cost": item["total_cost"]
                    })
            
            # Save CSV
            csv_path = output_dir / "verified_suppliers.csv"
            with open(csv_path, 'w', newline='') as f:
                if suppliers:
                    writer = csv.DictWriter(f, fieldnames=suppliers[0].keys())
                    writer.writeheader()
                    writer.writerows(suppliers)
            
            return {"success": True, "suppliers_count": len(suppliers)}
            
        except Exception as e:
            self.logger.error("Supplier mapping failed", error=str(e))
            return {"success": False, "suppliers_count": 0} 