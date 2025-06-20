#!/usr/bin/env python3
"""
BEM System Backup & Recovery Procedures
Comprehensive data protection and disaster recovery implementation
"""

import os
import shutil
import datetime
import subprocess
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BEMBackupManager:
    def __init__(self, backup_root="./backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_full_backup(self):
        """Create complete system backup"""
        backup_dir = self.backup_root / f"full_backup_{self.timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔄 Creating full system backup in {backup_dir}")
        
        # Backup configuration files
        self._backup_config_files(backup_dir)
        
        # Backup database schemas
        self._backup_database_schemas(backup_dir)
        
        # Backup application data
        self._backup_application_data(backup_dir)
        
        # Backup logs
        self._backup_logs(backup_dir)
        
        # Create backup manifest
        self._create_backup_manifest(backup_dir)
        
        logger.info("✅ Full backup completed successfully")
        return backup_dir
    
    def _backup_config_files(self, backup_dir):
        """Backup configuration and environment files"""
        config_dir = backup_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        config_files = [
            "docker-compose.yml",
            "requirements.txt",
            "start_services.py",
            "MICROSERVICE_ENGINES/docker-compose.yml",
            "neon/config.py",
            "postgre/config.py",
            "cert.pem",
            "key.pem"
        ]
        
        for file_path in config_files:
            if os.path.exists(file_path):
                dest = config_dir / Path(file_path).name
                shutil.copy2(file_path, dest)
                logger.info(f"📁 Backed up config: {file_path}")
    
    def _backup_database_schemas(self, backup_dir):
        """Backup database schemas and migration scripts"""
        db_dir = backup_dir / "database"
        db_dir.mkdir(exist_ok=True)
        
        schema_files = [
            "neon/postgresql_schema.sql",
            "postgre/enhanced_schema.sql",
            "Final_Phase/training_database_schema.sql",
            "neon/csv_migration.py"
        ]
        
        for schema_file in schema_files:
            if os.path.exists(schema_file):
                dest = db_dir / Path(schema_file).name
                shutil.copy2(schema_file, dest)
                logger.info(f"🗄️ Backed up schema: {schema_file}")
    
    def _backup_application_data(self, backup_dir):
        """Backup application code and data files"""
        app_dir = backup_dir / "application"
        app_dir.mkdir(exist_ok=True)
        
        # Backup critical application directories
        critical_dirs = [
            "Final_Phase",
            "frontend", 
            "neon",
            "MICROSERVICE_ENGINES",
            "shared"
        ]
        
        for dir_name in critical_dirs:
            if os.path.exists(dir_name):
                dest_dir = app_dir / dir_name
                shutil.copytree(dir_name, dest_dir, dirs_exist_ok=True)
                logger.info(f"📦 Backed up directory: {dir_name}")
        
        # Backup individual critical files
        critical_files = [
            "test_phase3_production.py",
            "test_phase2_integration.py", 
            "PHASE3_FINAL_REPORT.md"
        ]
        
        for file_name in critical_files:
            if os.path.exists(file_name):
                dest = app_dir / file_name
                shutil.copy2(file_name, dest)
                logger.info(f"📄 Backed up file: {file_name}")
    
    def _backup_logs(self, backup_dir):
        """Backup log files and monitoring data"""
        logs_dir = backup_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        log_files = [
            "ecm_log.txt",
            "error.log",
            "app.log"
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                dest = logs_dir / log_file
                shutil.copy2(log_file, dest)
                logger.info(f"📋 Backed up log: {log_file}")
    
    def _create_backup_manifest(self, backup_dir):
        """Create backup manifest with metadata"""
        manifest = {
            "backup_timestamp": self.timestamp,
            "backup_type": "full_system",
            "system_info": {
                "python_version": subprocess.getoutput("python --version"),
                "git_commit": subprocess.getoutput("git rev-parse HEAD"),
                "git_branch": subprocess.getoutput("git branch --show-current")
            },
            "backed_up_components": [
                "configuration_files",
                "database_schemas", 
                "application_code",
                "log_files"
            ]
        }
        
        manifest_file = backup_dir / "backup_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📝 Created backup manifest: {manifest_file}")
    
    def restore_from_backup(self, backup_path):
        """Restore system from backup"""
        backup_dir = Path(backup_path)
        
        if not backup_dir.exists():
            logger.error(f"❌ Backup directory not found: {backup_dir}")
            return False
        
        logger.info(f"🔄 Restoring system from backup: {backup_dir}")
        
        # Load backup manifest
        manifest_file = backup_dir / "backup_manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            logger.info(f"📋 Restoring backup from {manifest['backup_timestamp']}")
        
        # Restore configuration files
        config_dir = backup_dir / "config"
        if config_dir.exists():
            for config_file in config_dir.iterdir():
                dest = Path(".") / config_file.name
                shutil.copy2(config_file, dest)
                logger.info(f"📁 Restored config: {config_file.name}")
        
        # Restore database schemas
        db_dir = backup_dir / "database"
        if db_dir.exists():
            for schema_file in db_dir.iterdir():
                # Restore to appropriate locations based on filename
                if "postgresql" in schema_file.name:
                    dest = Path("neon") / schema_file.name
                elif "enhanced" in schema_file.name:
                    dest = Path("postgre") / schema_file.name
                elif "training" in schema_file.name:
                    dest = Path("Final_Phase") / schema_file.name
                else:
                    dest = Path(".") / schema_file.name
                
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(schema_file, dest)
                logger.info(f"🗄️ Restored schema: {schema_file.name}")
        
        # Restore application data
        app_dir = backup_dir / "application"
        if app_dir.exists():
            for item in app_dir.iterdir():
                if item.is_dir():
                    dest_dir = Path(".") / item.name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(item, dest_dir)
                    logger.info(f"📦 Restored directory: {item.name}")
                else:
                    dest = Path(".") / item.name
                    shutil.copy2(item, dest)
                    logger.info(f"📄 Restored file: {item.name}")
        
        logger.info("✅ System restore completed successfully")
        return True
    
    def list_backups(self):
        """List available backups"""
        backups = []
        for item in self.backup_root.iterdir():
            if item.is_dir() and item.name.startswith("full_backup_"):
                manifest_file = item / "backup_manifest.json"
                if manifest_file.exists():
                    with open(manifest_file) as f:
                        manifest = json.load(f)
                    backups.append({
                        "path": str(item),
                        "timestamp": manifest.get("backup_timestamp"),
                        "type": manifest.get("backup_type")
                    })
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

def main():
    """Main backup/recovery interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BEM System Backup & Recovery")
    parser.add_argument("action", choices=["backup", "restore", "list"], 
                       help="Action to perform")
    parser.add_argument("--backup-path", help="Path to backup for restore operation")
    
    args = parser.parse_args()
    
    manager = BEMBackupManager()
    
    if args.action == "backup":
        backup_dir = manager.create_full_backup()
        print(f"✅ Backup created: {backup_dir}")
    
    elif args.action == "restore":
        if not args.backup_path:
            print("❌ --backup-path required for restore operation")
            return
        success = manager.restore_from_backup(args.backup_path)
        if success:
            print("✅ Restore completed successfully")
        else:
            print("❌ Restore failed")
    
    elif args.action == "list":
        backups = manager.list_backups()
        if backups:
            print("📋 Available backups:")
            for backup in backups:
                print(f"  {backup['timestamp']}: {backup['path']}")
        else:
            print("📭 No backups found")

if __name__ == "__main__":
    main() 