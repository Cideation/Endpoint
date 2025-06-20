#!/usr/bin/env python3
"""
BEM System Deployment & Security Testing Suite
Tests container orchestration, audit trails, and security measures
"""

import os
import re
import json
import time
import docker
import logging
import asyncio
import aiohttp
import psycopg2
import websockets
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('deployment_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentSecurityTester:
    """Tests deployment readiness and security measures"""
    
    def __init__(self):
        """Initialize test suite"""
        self.docker_client = docker.from_env()
        self.config = {
            'ecm_ws_url': 'ws://localhost:8765',
            'db_connection': {
                'host': 'localhost',
                'port': 5432,
                'database': 'bem_training',
                'user': 'bem_user',
                'password': os.getenv('DB_PASSWORD'),
                'sslmode': 'verify-full',
                'sslcert': '/certs/cert.pem',
                'sslkey': '/certs/key.pem'
            },
            'required_services': [
                'ecm-gateway',
                'callback-engine',
                'dag-alpha',
                'functor-types',
                'graph-runtime',
                'optimization'
            ]
        }
    
    async def test_container_orchestration(self) -> Dict[str, bool]:
        """Test Docker container orchestration"""
        results = {}
        
        # Check all required services are running
        containers = self.docker_client.containers.list()
        running_services = [c.name for c in containers]
        
        for service in self.config['required_services']:
            is_running = any(service in name for name in running_services)
            results[f"service_{service}"] = is_running
            
            if is_running:
                container = next(c for c in containers if service in c.name)
                # Check container health
                if hasattr(container, 'health'):
                    results[f"health_{service}"] = container.health == 'healthy'
                # Check logs for errors
                logs = container.logs(tail=100).decode('utf-8')
                results[f"errors_{service}"] = 'ERROR' not in logs
        
        # Test network connectivity
        for service in self.config['required_services']:
            if service != 'ecm-gateway':
                # Test connection to ECM Gateway
                container = next(c for c in containers if service in c.name)
                exit_code, _ = container.exec_run(
                    ["ping", "-c", "1", "ecm-gateway"],
                    privileged=True
                )
                results[f"network_{service}_to_ecm"] = exit_code == 0
        
        # Check volume persistence
        volumes = self.docker_client.volumes.list()
        required_volumes = [
            'training-db-data',
            'redis-data',
            'ecm-logs',
            'dag-data',
            'graph-data'
        ]
        
        for volume in required_volumes:
            results[f"volume_{volume}"] = any(volume in v.name for v in volumes)
        
        return results
    
    async def test_ecm_audit_trail(self) -> Dict[str, bool]:
        """Test ECM audit trail compliance"""
        results = {}
        
        # Get ECM container
        ecm_container = next(
            c for c in self.docker_client.containers.list()
            if 'ecm-gateway' in c.name
        )
        
        # Get audit log
        logs = ecm_container.exec_run(
            ["cat", "/logs/ecm_audit.log"]
        ).output.decode('utf-8')
        
        # Check timestamp format
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        has_timestamps = bool(re.search(timestamp_pattern, logs))
        results['timestamps_present'] = has_timestamps
        
        # Check for sensitive data
        sensitive_patterns = [
            r'password',
            r'secret',
            r'token',
            r'key',
            r'credential'
        ]
        has_sensitive = any(
            re.search(pattern, logs, re.IGNORECASE)
            for pattern in sensitive_patterns
        )
        results['no_sensitive_data'] = not has_sensitive
        
        # Check log immutability
        # Try to modify log file
        exit_code, _ = ecm_container.exec_run(
            ["bash", "-c", "echo 'test' >> /logs/ecm_audit.log"]
        )
        results['logs_immutable'] = exit_code != 0
        
        # Check retention policy
        # Get oldest log entry
        first_line = logs.split('\n')[0]
        if first_line:
            match = re.search(timestamp_pattern, first_line)
            if match:
                oldest_date = datetime.strptime(
                    match.group(), '%Y-%m-%dT%H:%M:%S'
                )
                # Check if older than retention period (e.g., 90 days)
                retention_days = 90
                results['retention_policy_enforced'] = \
                    datetime.now() - oldest_date <= timedelta(days=retention_days)
        
        return results
    
    async def test_database_security(self) -> Dict[str, bool]:
        """Test database security measures"""
        results = {}
        
        # Test SSL connection
        try:
            conn = psycopg2.connect(**self.config['db_connection'])
            results['ssl_connection'] = True
            
            # Check SSL is actually being used
            cur = conn.cursor()
            cur.execute("SHOW ssl;")
            ssl_on = cur.fetchone()[0] == 'on'
            results['ssl_active'] = ssl_on
            
            # Check user permissions
            cur.execute("""
                SELECT r.rolname, r.rolsuper, r.rolinherit,
                    r.rolcreaterole, r.rolcreatedb, r.rolcanlogin,
                    r.rolreplication, r.rolconnlimit
                FROM pg_roles r
                WHERE r.rolname = current_user;
            """)
            user_perms = cur.fetchone()
            
            # Ensure user doesn't have superuser privileges
            results['proper_permissions'] = not user_perms[1]
            
            # Check database isolation
            cur.execute("""
                SELECT datname FROM pg_database
                WHERE datistemplate = false;
            """)
            databases = cur.fetchall()
            # Ensure only necessary databases exist
            results['db_isolation'] = len(databases) <= 2  # template0 and our db
            
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Database security test error: {str(e)}")
            results['ssl_connection'] = False
            results['ssl_active'] = False
            results['proper_permissions'] = False
            results['db_isolation'] = False
        
        # Test backup security
        backup_dir = Path('/backups')
        if backup_dir.exists():
            # Check backup file permissions
            backup_files = list(backup_dir.glob('*.backup'))
            if backup_files:
                latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
                # Check file permissions (should be 600)
                perms = oct(latest_backup.stat().st_mode)[-3:]
                results['backup_permissions'] = perms == '600'
                
                # Check backup encryption
                # Assuming backups are encrypted if they have .enc extension
                results['backup_encrypted'] = latest_backup.suffix == '.enc'
            else:
                results['backup_permissions'] = False
                results['backup_encrypted'] = False
        else:
            results['backup_permissions'] = False
            results['backup_encrypted'] = False
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Dict[str, bool]]:
        """Run all deployment and security tests"""
        logger.info("Starting deployment and security tests...")
        
        results = {
            'container_orchestration': await self.test_container_orchestration(),
            'ecm_audit_trail': await self.test_ecm_audit_trail(),
            'database_security': await self.test_database_security()
        }
        
        # Generate test report
        report = "Deployment & Security Test Report\n"
        report += "================================\n\n"
        
        for category, tests in results.items():
            report += f"\n{category.replace('_', ' ').title()}\n"
            report += "-" * len(category) + "\n"
            for test, passed in tests.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                report += f"{test.replace('_', ' ').title()}: {status}\n"
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"test_results/security_report_{timestamp}.txt"
        os.makedirs("test_results", exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Test report saved to {report_path}")
        return results

async def main():
    """Run deployment and security tests"""
    tester = DeploymentSecurityTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 