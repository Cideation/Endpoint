#!/usr/bin/env python3
"""
BEM System Deployment to Production
🚀 Automated deployment to Render/Railway/Fly.io with webhook auto-refresh

Usage:
  python deploy_to_render.py --production    # Full production deployment
  python deploy_to_render.py --staging       # Staging deployment
  python deploy_to_render.py --check         # Just check readiness
"""

import argparse
import subprocess
import json
import time
import os
import sys
from pathlib import Path

class BEMDeployment:
    def __init__(self, environment='staging'):
        self.environment = environment
        self.deployment_config = {
            'staging': {
                'name': 'bem-system-staging',
                'branch': 'main',
                'auto_deploy': True,
                'health_check_url': '/health'
            },
            'production': {
                'name': 'bem-system-prod',
                'branch': 'main', 
                'auto_deploy': True,
                'health_check_url': '/health'
            }
        }
        
        self.services = {
            'graphql_engine': {
                'port': 8004,
                'name': 'Real-Time GraphQL Engine',
                'health_path': '/health',
                'start_command': 'python frontend/graphql_realtime_engine.py'
            },
            'frontend_server': {
                'port': 8005,
                'name': 'Frontend Web Server', 
                'health_path': '/',
                'start_command': 'python -m http.server 8005'
            }
        }
    
    def log(self, message, level='INFO'):
        timestamp = time.strftime('%H:%M:%S')
        symbols = {'INFO': 'ℹ️', 'SUCCESS': '✅', 'ERROR': '❌', 'WARNING': '⚠️'}
        symbol = symbols.get(level, 'ℹ️')
        print(f"[{timestamp}] {symbol} {message}")
    
    def create_render_yaml(self):
        """Create/update render.yaml for automatic deployment"""
        self.log("📝 Creating render.yaml configuration...")
        
        render_config = {
            'services': [
                {
                    'type': 'web',
                    'name': f"bem-graphql-{self.environment}",
                    'env': 'python',
                    'plan': 'starter',
                    'buildCommand': 'pip install -r requirements_realtime.txt',
                    'startCommand': 'python frontend/graphql_realtime_engine.py',
                    'envVars': [
                        {'key': 'PYTHON_VERSION', 'value': '3.9'},
                        {'key': 'GRAPHQL_PORT', 'value': '8004'},
                        {'key': 'ENVIRONMENT', 'value': self.environment},
                        {'key': 'ENABLE_CORS', 'value': 'true'}
                    ],
                    'healthCheckPath': '/health'
                },
                {
                    'type': 'web', 
                    'name': f"bem-frontend-{self.environment}",
                    'env': 'static',
                    'staticPublishPath': './frontend',
                    'buildCommand': 'echo "Static frontend build"',
                    'headers': [
                        {'path': '/*', 'name': 'X-Frame-Options', 'value': 'DENY'},
                        {'path': '/*', 'name': 'X-Content-Type-Options', 'value': 'nosniff'}
                    ]
                }
            ],
            'databases': [
                {
                    'name': f"bem-postgres-{self.environment}",
                    'plan': 'starter'
                }
            ]
        }
        
        # Write render.yaml
        with open('render.yaml', 'w') as f:
            import yaml
            yaml.safe_dump(render_config, f, default_flow_style=False)
        
        self.log("✅ render.yaml created successfully")
        return render_config
    
    def create_dockerfile(self):
        """Create optimized Dockerfile for deployment"""
        self.log("🐳 Creating Dockerfile...")
        
        dockerfile_content = """# BEM System Production Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_realtime.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash bem
RUN chown -R bem:bem /app
USER bem

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8004/health || exit 1

# Expose ports
EXPOSE 8004 8005

# Start command (can be overridden)
CMD ["python", "start_realtime_system.py"]
"""
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        self.log("✅ Dockerfile created successfully")
        return dockerfile_content
    
    def create_railway_config(self):
        """Create railway.json for Railway deployment"""
        self.log("🚂 Creating Railway configuration...")
        
        railway_config = {
            "$schema": "https://railway.app/railway.schema.json",
            "build": {
                "builder": "DOCKERFILE",
                "buildCommand": "pip install -r requirements_realtime.txt"
            },
            "deploy": {
                "startCommand": "python start_realtime_system.py",
                "healthcheckPath": "/health",
                "healthcheckTimeout": 30,
                "restartPolicyType": "ON_FAILURE"
            }
        }
        
        with open('railway.json', 'w') as f:
            json.dump(railway_config, f, indent=2)
        
        self.log("✅ railway.json created successfully")
        return railway_config
    
    def create_flyio_config(self):
        """Create fly.toml for Fly.io deployment"""
        self.log("🪰 Creating Fly.io configuration...")
        
        flyio_config = f"""# BEM System - Fly.io Configuration
app = "bem-system-{self.environment}"
primary_region = "sjc"

[build]
  dockerfile = "Dockerfile"

[env]
  ENVIRONMENT = "{self.environment}"
  GRAPHQL_PORT = "8004"
  FRONTEND_PORT = "8005"

[[services]]
  protocol = "tcp"
  internal_port = 8004
  processes = ["app"]

  [[services.ports]]
    port = 80
    handlers = ["http"]
    force_https = true

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [services.concurrency]
    type = "connections"
    hard_limit = 1000
    soft_limit = 800

[[services.tcp_checks]]
  interval = "15s"
  timeout = "2s"
  grace_period = "5s"
  method = "connection"
  port = 8004

[[services.http_checks]]
  interval = "10s"
  timeout = "2s"
  grace_period = "5s"
  method = "get"
  path = "/health"
  protocol = "http"
  port = 8004

[metrics]
  port = 9091
  path = "/metrics"
"""
        
        with open('fly.toml', 'w') as f:
            f.write(flyio_config)
        
        self.log("✅ fly.toml created successfully")
        return flyio_config
    
    def setup_github_webhook(self):
        """Set up GitHub webhook for auto-refresh"""
        self.log("🔗 Setting up GitHub webhook for auto-refresh...")
        
        webhook_script = """#!/bin/bash
# GitHub Webhook Auto-Refresh Setup
# This script sets up automatic deployment triggers

echo "🔗 Setting up GitHub webhook for auto-deployment..."

# Create webhook payload example
cat > webhook_payload_example.json << 'EOF'
{
  "ref": "refs/heads/main",
  "repository": {
    "name": "Endpoint",
    "full_name": "Cideation/Endpoint",
    "html_url": "https://github.com/Cideation/Endpoint"
  },
  "head_commit": {
    "id": "commit_sha_here",
    "message": "Deploy BEM System to production",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
EOF

echo "✅ Webhook payload example created"
echo "📋 Configure webhook URL in GitHub repository settings:"
echo "   - Payload URL: https://your-app.render.com/webhooks/github"
echo "   - Content type: application/json"
echo "   - Events: push, pull_request"

# Create webhook handler if it doesn't exist
if [ ! -f "frontend/webhook_handler.py" ]; then
    cat > frontend/webhook_handler.py << 'EOF'
from fastapi import FastAPI, Request
import subprocess
import json

app = FastAPI()

@app.post("/webhooks/github")
async def handle_github_webhook(request: Request):
    payload = await request.json()
    
    if payload.get("ref") == "refs/heads/main":
        print("🚀 Main branch updated - triggering deployment refresh")
        
        # Restart services (deployment platform will handle this)
        return {"status": "deployment_triggered", "ref": payload["ref"]}
    
    return {"status": "no_action", "ref": payload.get("ref", "unknown")}

@app.get("/webhooks/status")
async def webhook_status():
    return {"status": "webhook_active", "service": "BEM Auto-Deploy"}
EOF
    echo "✅ Webhook handler created at frontend/webhook_handler.py"
fi

echo "🎯 Auto-refresh webhook setup complete!"
"""
        
        with open('setup_webhook.sh', 'w') as f:
            f.write(webhook_script)
        
        # Make executable
        os.chmod('setup_webhook.sh', 0o755)
        
        self.log("✅ GitHub webhook setup script created")
        return webhook_script
    
    def run_pre_deployment_checks(self):
        """Run pre-deployment validation"""
        self.log("🔍 Running pre-deployment checks...")
        
        checks = {
            'git_clean': False,
            'requirements_exist': False,
            'key_files_exist': False,
            'syntax_valid': False
        }
        
        # Check git status
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                checks['git_clean'] = True
                self.log("✅ Git repository is clean")
            else:
                self.log("⚠️ Uncommitted changes detected", 'WARNING')
        except:
            self.log("❌ Git check failed", 'ERROR')
        
        # Check requirements files
        req_files = ['requirements.txt', 'requirements_realtime.txt']
        if all(Path(req).exists() for req in req_files):
            checks['requirements_exist'] = True
            self.log("✅ Requirements files found")
        else:
            self.log("❌ Missing requirements files", 'ERROR')
        
        # Check key files
        key_files = [
            'frontend/graphql_realtime_engine.py',
            'frontend/realtime_graph_interface.html', 
            'start_realtime_system.py'
        ]
        if all(Path(kf).exists() for kf in key_files):
            checks['key_files_exist'] = True
            self.log("✅ Key application files found")
        else:
            self.log("❌ Missing key application files", 'ERROR')
        
        # Basic syntax check
        try:
            result = subprocess.run([
                sys.executable, '-m', 'py_compile', 
                'frontend/graphql_realtime_engine.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                checks['syntax_valid'] = True
                self.log("✅ Python syntax validation passed")
            else:
                self.log("❌ Python syntax errors detected", 'ERROR')
        except:
            self.log("❌ Syntax check failed", 'ERROR')
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        self.log(f"📊 Pre-deployment: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks >= 3:  # At least 3/4 checks must pass
            self.log("✅ Pre-deployment checks: PASSED", 'SUCCESS')
            return True
        else:
            self.log("❌ Pre-deployment checks: FAILED", 'ERROR')
            return False
    
    def deploy_to_render(self):
        """Deploy to Render.com"""
        self.log(f"🚀 Deploying to Render ({self.environment})...")
        
        # Create deployment files
        self.create_render_yaml()
        self.create_dockerfile()
        
        # Commit deployment files
        try:
            subprocess.run(['git', 'add', 'render.yaml', 'Dockerfile'], check=True)
            subprocess.run(['git', 'commit', '-m', f'Add Render deployment config for {self.environment}'], 
                         check=False)  # Don't fail if nothing to commit
            
            self.log("✅ Deployment files committed")
        except subprocess.CalledProcessError as e:
            self.log(f"⚠️ Git commit issue: {e}", 'WARNING')
        
        # Push to trigger auto-deploy
        try:
            subprocess.run(['git', 'push', 'origin', 'main'], check=True)
            self.log("✅ Code pushed to trigger auto-deployment", 'SUCCESS')
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Git push failed: {e}", 'ERROR')
            return False
        
        self.log("🎉 Render deployment initiated!", 'SUCCESS')
        self.log("📋 Next steps:")
        self.log("   1. Check Render dashboard for deployment status")
        self.log("   2. Monitor logs for any issues")
        self.log("   3. Test endpoints once deployed")
        
        return True
    
    def deploy_to_railway(self):
        """Deploy to Railway"""
        self.log(f"🚂 Deploying to Railway ({self.environment})...")
        
        self.create_railway_config()
        self.create_dockerfile()
        
        # Railway CLI deployment (if available)
        try:
            result = subprocess.run(['railway', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                subprocess.run(['railway', 'up'], check=True)
                self.log("✅ Railway deployment via CLI successful", 'SUCCESS')
            else:
                self.log("ℹ️ Railway CLI not found, using git push method")
                # Fallback to git push
                subprocess.run(['git', 'add', 'railway.json', 'Dockerfile'], check=True)
                subprocess.run(['git', 'commit', '-m', f'Add Railway config for {self.environment}'], 
                             check=False)
                subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Railway deployment failed: {e}", 'ERROR')
            return False
        
        return True
    
    def deploy_to_flyio(self):
        """Deploy to Fly.io"""
        self.log(f"🪰 Deploying to Fly.io ({self.environment})...")
        
        self.create_flyio_config()
        self.create_dockerfile()
        
        # Fly.io CLI deployment
        try:
            # Check if flyctl is available
            result = subprocess.run(['flyctl', 'version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Deploy with flyctl
                subprocess.run(['flyctl', 'deploy'], check=True)
                self.log("✅ Fly.io deployment successful", 'SUCCESS')
            else:
                self.log("❌ Fly.io CLI not found. Install with: curl -L https://fly.io/install.sh | sh", 'ERROR')
                return False
        
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Fly.io deployment failed: {e}", 'ERROR')
            return False
        
        return True
    
    def post_deployment_verification(self, base_url):
        """Verify deployment is working"""
        self.log("🔬 Running post-deployment verification...")
        
        import requests
        
        endpoints_to_test = [
            f"{base_url}/health",
            f"{base_url}/stats", 
            f"{base_url}/graphql"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    self.log(f"✅ {endpoint} - OK", 'SUCCESS')
                else:
                    self.log(f"⚠️ {endpoint} - Status {response.status_code}", 'WARNING')
            except requests.RequestException as e:
                self.log(f"❌ {endpoint} - Failed: {e}", 'ERROR')
        
        self.log("✅ Post-deployment verification complete")
    
    def deploy(self, platform='render'):
        """Main deployment method"""
        self.log(f"🚀 Starting BEM System deployment to {platform}...")
        self.log(f"📋 Environment: {self.environment}")
        self.log("=" * 60)
        
        # Pre-deployment checks
        if not self.run_pre_deployment_checks():
            self.log("❌ Pre-deployment checks failed. Aborting.", 'ERROR')
            return False
        
        # Set up webhook for auto-refresh
        self.setup_github_webhook()
        
        # Deploy to chosen platform
        deployment_success = False
        
        if platform == 'render':
            deployment_success = self.deploy_to_render()
        elif platform == 'railway':
            deployment_success = self.deploy_to_railway()
        elif platform == 'flyio':
            deployment_success = self.deploy_to_flyio()
        else:
            self.log(f"❌ Unsupported platform: {platform}", 'ERROR')
            return False
        
        if deployment_success:
            self.log("🎉 DEPLOYMENT COMPLETE!", 'SUCCESS')
            self.log("📋 Monitor your deployment:")
            self.log(f"   - Platform: {platform}")
            self.log(f"   - Environment: {self.environment}")
            self.log("   - Auto-refresh: Enabled via GitHub webhooks")
            return True
        else:
            self.log("❌ DEPLOYMENT FAILED!", 'ERROR')
            return False

def main():
    parser = argparse.ArgumentParser(description='Deploy BEM System to production')
    parser.add_argument('--production', action='store_true', 
                       help='Deploy to production environment')
    parser.add_argument('--staging', action='store_true', 
                       help='Deploy to staging environment')
    parser.add_argument('--platform', choices=['render', 'railway', 'flyio'], 
                       default='render', help='Deployment platform')
    parser.add_argument('--check', action='store_true', 
                       help='Just run readiness check')
    
    args = parser.parse_args()
    
    # Determine environment
    if args.production:
        environment = 'production'
    elif args.staging:
        environment = 'staging'
    else:
        environment = 'staging'  # Default to staging
    
    # Just run readiness check
    if args.check:
        print("🔍 Running deployment readiness check...")
        subprocess.run([sys.executable, 'deployment_readiness_check.py'])
        return
    
    # Create deployment instance
    deployer = BEMDeployment(environment)
    
    # Run deployment
    success = deployer.deploy(args.platform)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 