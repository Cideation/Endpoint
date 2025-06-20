#!/usr/bin/env python3
"""
BEM System Backup & Recovery Procedures
Enhanced with encryption, verification, and remote storage support
"""

import os
import shutil
import datetime
import subprocess
import json
import logging
import hashlib
import boto3
import cryptography
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from base64 import b64encode, b64decode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BEMBackupManager:
    def __init__(self, backup_root="./backups", encryption_key=None, remote_storage=None):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize encryption
        if encryption_key:
            self.encryption_key = encryption_key
        else:
            self.encryption_key = self._generate_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Initialize remote storage (AWS S3)
        self.remote_storage = remote_storage
        if remote_storage:
            self.s3_client = boto3.client('s3',
                aws_access_key_id=remote_storage.get('aws_access_key_id'),
                aws_secret_access_key=remote_storage.get('aws_secret_access_key'),
                region_name=remote_storage.get('region_name', 'us-east-1')
            )
            self.bucket_name = remote_storage.get('bucket_name')
    
    def _generate_encryption_key(self):
        """Generate a secure encryption key"""
        key = Fernet.generate_key()
        key_file = self.backup_root / ".encryption_key"
        if not key_file.exists():
            with open(key_file, 'wb') as f:
                f.write(key)
        return key
    
    def _encrypt_file(self, file_path):
        """Encrypt a file using Fernet symmetric encryption"""
        with open(file_path, 'rb') as f:
            data = f.read()
        encrypted_data = self.fernet.encrypt(data)
        with open(str(file_path) + '.encrypted', 'wb') as f:
            f.write(encrypted_data)
        os.remove(file_path)  # Remove unencrypted file
        return str(file_path) + '.encrypted'
    
    def _decrypt_file(self, encrypted_file_path):
        """Decrypt a file using Fernet symmetric encryption"""
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = self.fernet.decrypt(encrypted_data)
        original_path = encrypted_file_path.replace('.encrypted', '')
        with open(original_path, 'wb') as f:
            f.write(decrypted_data)
        os.remove(encrypted_file_path)  # Remove encrypted file
        return original_path
    
    def _calculate_checksum(self, file_path):
        """Calculate SHA-256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _upload_to_remote(self, local_path, remote_path):
        """Upload backup to remote storage (S3)"""
        if self.remote_storage:
            try:
                self.s3_client.upload_file(str(local_path), self.bucket_name, remote_path)
                logger.info(f"☁️ Uploaded to remote storage: {remote_path}")
                return True
            except Exception as e:
                logger.error(f"❌ Remote upload failed: {str(e)}")
                return False
        return False
    
    def _download_from_remote(self, remote_path, local_path):
        """Download backup from remote storage (S3)"""
        if self.remote_storage:
            try:
                self.s3_client.download_file(self.bucket_name, remote_path, str(local_path))
                logger.info(f"☁️ Downloaded from remote storage: {remote_path}")
                return True
            except Exception as e:
                logger.error(f"❌ Remote download failed: {str(e)}")
                return False
        return False
    
    def create_full_backup(self, encrypt=True, upload_remote=True):
        """Create complete system backup with encryption and remote storage"""
        backup_dir = self.backup_root / f"full_backup_{self.timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔄 Creating full system backup in {backup_dir}")
        
        # Track files and checksums
        checksums = {}
        
        # Backup configuration files
        config_files = self._backup_config_files(backup_dir)
        checksums.update(config_files)
        
        # Backup database schemas
        db_files = self._backup_database_schemas(backup_dir)
        checksums.update(db_files)
        
        # Backup application data
        app_files = self._backup_application_data(backup_dir)
        checksums.update(app_files)
        
        # Backup logs
        log_files = self._backup_logs(backup_dir)
        checksums.update(log_files)
        
        # Create backup manifest with checksums
        manifest = self._create_backup_manifest(backup_dir, checksums)
        
        # Encrypt backup if requested
        if encrypt:
            logger.info("🔒 Encrypting backup files")
            for root, _, files in os.walk(backup_dir):
                for file in files:
                    file_path = Path(root) / file
                    if not file.endswith('.encrypted'):
                        self._encrypt_file(file_path)
        
        # Create backup archive
        archive_name = f"backup_{self.timestamp}.tar.gz"
        archive_path = self.backup_root / archive_name
        shutil.make_archive(
            str(archive_path).replace('.tar.gz', ''),
            'gztar',
            backup_dir
        )
        
        # Upload to remote storage if configured
        if upload_remote and self.remote_storage:
            self._upload_to_remote(archive_path, archive_name)
        
        # Cleanup old backups (keep last 5 by default)
        self._cleanup_old_backups(keep_last=5)
        
        logger.info("✅ Full backup completed successfully")
        return backup_dir
    
    def _backup_config_files(self, backup_dir):
        """Backup configuration and environment files"""
        config_dir = backup_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        checksums = {}
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
                checksums[str(dest)] = self._calculate_checksum(dest)
                logger.info(f"📁 Backed up config: {file_path}")
        
        return checksums
    
    def _backup_database_schemas(self, backup_dir):
        """Backup database schemas and migration scripts"""
        db_dir = backup_dir / "database"
        db_dir.mkdir(exist_ok=True)
        
        checksums = {}
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
                checksums[str(dest)] = self._calculate_checksum(dest)
                logger.info(f"🗄️ Backed up schema: {schema_file}")
        
        return checksums
    
    def _backup_application_data(self, backup_dir):
        """Backup application code and data files"""
        app_dir = backup_dir / "application"
        app_dir.mkdir(exist_ok=True)
        
        checksums = {}
        
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
                # Calculate checksums for all files in directory
                for root, _, files in os.walk(dest_dir):
                    for file in files:
                        file_path = Path(root) / file
                        checksums[str(file_path)] = self._calculate_checksum(file_path)
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
                checksums[str(dest)] = self._calculate_checksum(dest)
                logger.info(f"📄 Backed up file: {file_name}")
        
        return checksums
    
    def _backup_logs(self, backup_dir):
        """Backup log files and monitoring data"""
        logs_dir = backup_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        checksums = {}
        log_files = [
            "ecm_log.txt",
            "error.log",
            "app.log"
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                dest = logs_dir / log_file
                shutil.copy2(log_file, dest)
                checksums[str(dest)] = self._calculate_checksum(dest)
                logger.info(f"📋 Backed up log: {log_file}")
        
        return checksums
    
    def _create_backup_manifest(self, backup_dir, checksums):
        """Create backup manifest with metadata and checksums"""
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
            ],
            "file_checksums": checksums,
            "encryption_enabled": True,
            "remote_storage_enabled": bool(self.remote_storage)
        }
        
        manifest_file = backup_dir / "backup_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📝 Created backup manifest: {manifest_file}")
        return manifest
    
    def restore_from_backup(self, backup_path, decrypt=True, verify_checksums=True):
        """Restore system from backup with integrity verification"""
        backup_dir = Path(backup_path)
        
        if not backup_dir.exists():
            logger.error(f"❌ Backup directory not found: {backup_dir}")
            return False
        
        logger.info(f"🔄 Restoring system from backup: {backup_dir}")
        
        # Load and verify backup manifest
        manifest_file = backup_dir / "backup_manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            logger.info(f"📋 Restoring backup from {manifest['backup_timestamp']}")
            
            # Verify file checksums if enabled
            if verify_checksums and "file_checksums" in manifest:
                logger.info("🔍 Verifying backup integrity")
                for file_path, stored_checksum in manifest["file_checksums"].items():
                    if os.path.exists(file_path):
                        current_checksum = self._calculate_checksum(file_path)
                        if current_checksum != stored_checksum:
                            logger.error(f"❌ Checksum mismatch for {file_path}")
                            return False
        
        # Decrypt files if needed
        if decrypt:
            logger.info("🔓 Decrypting backup files")
            for root, _, files in os.walk(backup_dir):
                for file in files:
                    if file.endswith('.encrypted'):
                        file_path = Path(root) / file
                        self._decrypt_file(file_path)
        
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
    
    def _cleanup_old_backups(self, keep_last=5):
        """Remove old backups keeping only the specified number of recent ones"""
        backups = self.list_backups()
        if len(backups) > keep_last:
            for backup in backups[keep_last:]:
                backup_path = Path(backup["path"])
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                    logger.info(f"🗑️ Removed old backup: {backup_path}")
                
                # Remove from remote storage if configured
                if self.remote_storage:
                    try:
                        self.s3_client.delete_object(
                            Bucket=self.bucket_name,
                            Key=f"backup_{backup['timestamp']}.tar.gz"
                        )
                        logger.info(f"☁️ Removed old backup from remote storage")
                    except Exception as e:
                        logger.error(f"❌ Failed to remove remote backup: {str(e)}")
    
    def verify_backup_integrity(self, backup_path):
        """Verify the integrity of a backup using stored checksums"""
        backup_dir = Path(backup_path)
        manifest_file = backup_dir / "backup_manifest.json"
        
        if not manifest_file.exists():
            logger.error("❌ Backup manifest not found")
            return False
        
        with open(manifest_file) as f:
            manifest = json.load(f)
        
        if "file_checksums" not in manifest:
            logger.error("❌ No checksums found in backup manifest")
            return False
        
        logger.info("🔍 Verifying backup integrity")
        mismatches = []
        
        for file_path, stored_checksum in manifest["file_checksums"].items():
            if os.path.exists(file_path):
                current_checksum = self._calculate_checksum(file_path)
                if current_checksum != stored_checksum:
                    mismatches.append(file_path)
                    logger.error(f"❌ Checksum mismatch for {file_path}")
        
        if mismatches:
            logger.error(f"❌ Found {len(mismatches)} integrity issues")
            return False
        
        logger.info("✅ Backup integrity verified successfully")
        return True
    
    def list_backups(self):
        """List available backups with detailed information"""
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
                        "type": manifest.get("backup_type"),
                        "git_commit": manifest.get("system_info", {}).get("git_commit"),
                        "encryption_enabled": manifest.get("encryption_enabled", False),
                        "remote_storage_enabled": manifest.get("remote_storage_enabled", False)
                    })
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

def main():
    """Main backup/recovery interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BEM System Backup & Recovery")
    parser.add_argument("action", choices=["backup", "restore", "list", "verify"], 
                       help="Action to perform")
    parser.add_argument("--backup-path", help="Path to backup for restore/verify operation")
    parser.add_argument("--no-encrypt", action="store_true", help="Disable backup encryption")
    parser.add_argument("--no-remote", action="store_true", help="Disable remote storage")
    parser.add_argument("--keep-last", type=int, default=5, help="Number of backups to keep")
    
    args = parser.parse_args()
    
    # Load remote storage configuration from environment variables
    remote_storage = None
    if os.getenv("AWS_ACCESS_KEY_ID"):
        remote_storage = {
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "region_name": os.getenv("AWS_REGION", "us-east-1"),
            "bucket_name": os.getenv("AWS_BACKUP_BUCKET")
        }
    
    manager = BEMBackupManager(remote_storage=remote_storage)
    
    if args.action == "backup":
        backup_dir = manager.create_full_backup(
            encrypt=not args.no_encrypt,
            upload_remote=not args.no_remote
        )
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
    
    elif args.action == "verify":
        if not args.backup_path:
            print("❌ --backup-path required for verify operation")
            return
        success = manager.verify_backup_integrity(args.backup_path)
        if success:
            print("✅ Backup integrity verified")
        else:
            print("❌ Backup integrity check failed")
    
    elif args.action == "list":
        backups = manager.list_backups()
        if backups:
            print("📋 Available backups:")
            for backup in backups:
                print(f"  {backup['timestamp']}: {backup['path']}")
                print(f"    Git Commit: {backup['git_commit']}")
                print(f"    Encryption: {'✅' if backup['encryption_enabled'] else '❌'}")
                print(f"    Remote Storage: {'✅' if backup['remote_storage_enabled'] else '❌'}")
        else:
            print("📭 No backups found")

if __name__ == "__main__":
    main() 