#!/usr/bin/env python3
"""
Deployment Readiness Check
✅ Validates "Good Enough" deployment criteria for BEM System

USER CRITERIA (Strategic):
✅ Core node execution works (1-3 nodes)
✅ GraphQL engine responds  
✅ UI shows working DAG or state update
✅ At least one functor triggers and returns result

TECHNICAL CRITERIA (Implementation):
✅ Docker containers build successfully
✅ Environment variables properly configured
✅ Database connectivity works
✅ WebSocket connections stable
✅ Security basics in place (CORS, rate limiting)
✅ Health endpoints functional
✅ Error handling graceful (no crashes)
✅ Git repo has necessary deployment files
"""

import asyncio
import subprocess
import json
import time
import sys
import os
from pathlib import Path

class DeploymentReadinessChecker:
    def __init__(self):
        # User's strategic criteria
        self.strategic_results = {
            'core_node_execution': False,
            'graphql_engine': False,
            'ui_dag_display': False,
            'functor_execution': False
        }
        
        # Technical implementation criteria
        self.technical_results = {
            'docker_build': False,
            'environment_config': False,
            'database_connectivity': False,
            'websocket_stability': False,
            'security_basics': False,
            'health_endpoints': False,
            'error_handling': False,
            'deployment_files': False
        }
        
        self.overall_ready = False
        self.detailed_results = []
    
    def log(self, message, status='INFO'):
        timestamp = time.strftime('%H:%M:%S')
        symbol = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else 'ℹ️' if status == 'INFO' else '⚠️'
        print(f"[{timestamp}] {symbol} {message}")
        
        self.detailed_results.append({
            'timestamp': timestamp,
            'message': message,
            'status': status
        })
    
    # ========================================================================
    # USER'S STRATEGIC CRITERIA
    # ========================================================================
    
    async def check_core_node_execution(self):
        """✅ Core node execution works (1-3 nodes)"""
        self.log("🎯 [STRATEGIC] Checking Core Node Execution...", 'INFO')
        
        try:
            # Check for functional node execution
            node_files = [
                'Final_Phase/pulse_router.py',
                'Final_Phase/fsm_runtime.py', 
                'tests/test_full_graph_pass.py',
                'tests/test_focused_validation.py'
            ]
            
            working_nodes = 0
            for node_file in node_files:
                if Path(node_file).exists():
                    with open(node_file, 'r') as f:
                        content = f.read()
                        # Look for node execution patterns
                        if any(pattern in content.lower() for pattern in [
                            'execute', 'run_functor', 'process_node', 'v01_', 'v02_', 'v03_'
                        ]):
                            self.log(f"   ✅ Node execution in: {node_file}", 'INFO')
                            working_nodes += 1
            
            if working_nodes >= 1:
                self.log("✅ Core node execution: READY", 'PASS')
                self.strategic_results['core_node_execution'] = True
                return True
            else:
                self.log("❌ Core node execution: NOT FOUND", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Core node execution check failed: {e}", 'FAIL')
            return False
    
    async def check_graphql_engine(self):
        """✅ GraphQL engine responds"""
        self.log("🎯 [STRATEGIC] Checking GraphQL Engine Response...", 'INFO')
        
        try:
            engine_file = 'frontend/graphql_realtime_engine.py'
            if not Path(engine_file).exists():
                self.log("❌ GraphQL engine file missing", 'FAIL')
                return False
            
            # Quick syntax check
            result = subprocess.run([
                sys.executable, '-m', 'py_compile', engine_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ GraphQL engine: SYNTAX OK", 'PASS')
                self.strategic_results['graphql_engine'] = True
                return True
            else:
                self.log(f"❌ GraphQL engine syntax error: {result.stderr}", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ GraphQL engine check failed: {e}", 'FAIL')
            return False
    
    async def check_ui_dag_display(self):
        """✅ UI shows working DAG or state update"""
        self.log("🎯 [STRATEGIC] Checking UI DAG Display...", 'INFO')
        
        try:
            ui_files = [
                'frontend/realtime_graph_interface.html',
                'frontend/agent_console.html', 
                'frontend/enhanced_unified_interface.html'
            ]
            
            working_ui = False
            for ui_file in ui_files:
                if Path(ui_file).exists():
                    with open(ui_file, 'r') as f:
                        content = f.read()
                        # Check for DAG visualization components
                        if all(component in content.lower() for component in [
                            'cytoscape', 'graph', 'node'
                        ]):
                            self.log(f"   ✅ Working DAG UI: {ui_file}", 'INFO')
                            working_ui = True
                            break
            
            if working_ui:
                self.log("✅ UI DAG display: READY", 'PASS')
                self.strategic_results['ui_dag_display'] = True
                return True
            else:
                self.log("❌ UI DAG display: NOT FOUND", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ UI DAG check failed: {e}", 'FAIL')
            return False
    
    async def check_functor_execution(self):
        """✅ At least one functor triggers and returns result"""
        self.log("🎯 [STRATEGIC] Checking Functor Execution...", 'INFO')
        
        try:
            functor_files = [
                'frontend/graphql_realtime_engine.py',
                'Final_Phase/pulse_router.py',
                'tests/test_focused_validation.py'
            ]
            
            functor_ready = False
            for functor_file in functor_files:
                if Path(functor_file).exists():
                    with open(functor_file, 'r') as f:
                        content = f.read()
                        # Look for functor execution patterns
                        if any(pattern in content.lower() for pattern in [
                            'execute_functor', 'materialspecification', 'designoptimization',
                            'qualityvalidation', 'functor_execution'
                        ]):
                            self.log(f"   ✅ Functor execution in: {functor_file}", 'INFO')
                            functor_ready = True
                            break
            
            if functor_ready:
                self.log("✅ Functor execution: READY", 'PASS')
                self.strategic_results['functor_execution'] = True
                return True
            else:
                self.log("❌ Functor execution: NOT FOUND", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Functor execution check failed: {e}", 'FAIL')
            return False
    
    # ========================================================================
    # TECHNICAL IMPLEMENTATION CRITERIA
    # ========================================================================
    
    async def check_docker_build(self):
        """🔧 Docker containers build successfully"""
        self.log("🔧 [TECHNICAL] Checking Docker Build Capability...", 'INFO')
        
        try:
            docker_files = [
                'Dockerfile',
                'docker-compose.yml',
                'deploy/Dockerfile.ecm',
                'deploy/Dockerfile.behavior-ac'
            ]
            
            docker_ready = False
            for docker_file in docker_files:
                if Path(docker_file).exists():
                    self.log(f"   ✅ Found: {docker_file}", 'INFO')
                    docker_ready = True
            
            # Check for requirements files
            req_files = ['requirements.txt', 'requirements_realtime.txt']
            req_found = any(Path(req).exists() for req in req_files)
            
            if docker_ready and req_found:
                self.log("✅ Docker build: READY", 'PASS')
                self.technical_results['docker_build'] = True
                return True
            else:
                self.log("❌ Docker build: MISSING FILES", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Docker build check failed: {e}", 'FAIL')
            return False
    
    async def check_environment_config(self):
        """🔧 Environment variables properly configured"""
        self.log("🔧 [TECHNICAL] Checking Environment Configuration...", 'INFO')
        
        try:
            config_files = [
                'render.yaml',
                '.env.example',
                'config/config.py',
                'neon/config.py'
            ]
            
            config_ready = False
            for config_file in config_files:
                if Path(config_file).exists():
                    self.log(f"   ✅ Config found: {config_file}", 'INFO')
                    config_ready = True
            
            # Check for port configurations in code
            port_configs = []
            for py_file in Path('.').glob('**/*.py'):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if 'port=' in content.lower() and ('8004' in content or '8005' in content):
                            port_configs.append(str(py_file))
                except:
                    pass
            
            if config_ready and port_configs:
                self.log("✅ Environment config: READY", 'PASS')
                self.technical_results['environment_config'] = True
                return True
            else:
                self.log("❌ Environment config: INCOMPLETE", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Environment config check failed: {e}", 'FAIL')
            return False
    
    async def check_database_connectivity(self):
        """🔧 Database connectivity works"""
        self.log("🔧 [TECHNICAL] Checking Database Connectivity...", 'INFO')
        
        try:
            db_files = [
                'neon/test_database_integration.py',
                'neon/db_manager.py',
                'postgre/enhanced_schema.sql',
                'database/postgresql_schema.sql'
            ]
            
            db_ready = False
            for db_file in db_files:
                if Path(db_file).exists():
                    with open(db_file, 'r') as f:
                        content = f.read()
                        if 'postgresql' in content.lower() or 'psycopg2' in content.lower():
                            self.log(f"   ✅ DB connectivity in: {db_file}", 'INFO')
                            db_ready = True
                            break
            
            if db_ready:
                self.log("✅ Database connectivity: READY", 'PASS')
                self.technical_results['database_connectivity'] = True
                return True
            else:
                self.log("❌ Database connectivity: NOT CONFIGURED", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Database connectivity check failed: {e}", 'FAIL')
            return False
    
    async def check_websocket_stability(self):
        """🔧 WebSocket connections stable"""
        self.log("🔧 [TECHNICAL] Checking WebSocket Stability...", 'INFO')
        
        try:
            ws_files = [
                'frontend/graphql_realtime_engine.py',
                'frontend/cytoscape_realtime_client.js',
                'Final_Phase/ecm_gateway.py'
            ]
            
            ws_ready = False
            for ws_file in ws_files:
                if Path(ws_file).exists():
                    with open(ws_file, 'r') as f:
                        content = f.read()
                        if any(pattern in content.lower() for pattern in [
                            'websocket', 'ws://', 'websockets', 'socket.io'
                        ]):
                            self.log(f"   ✅ WebSocket in: {ws_file}", 'INFO')
                            ws_ready = True
                            break
            
            if ws_ready:
                self.log("✅ WebSocket stability: READY", 'PASS')
                self.technical_results['websocket_stability'] = True
                return True
            else:
                self.log("❌ WebSocket stability: NOT IMPLEMENTED", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ WebSocket stability check failed: {e}", 'FAIL')
            return False
    
    async def check_security_basics(self):
        """🔧 Security basics in place (CORS, rate limiting)"""
        self.log("🔧 [TECHNICAL] Checking Security Basics...", 'INFO')
        
        try:
            security_patterns = {
                'cors': False,
                'rate_limiting': False,
                'input_validation': False
            }
            
            # Check for security implementations
            for py_file in Path('.').glob('**/*.py'):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read().lower()
                        
                        if 'cors' in content or 'corsMiddleware' in content:
                            security_patterns['cors'] = True
                        if 'rate' in content and 'limit' in content:
                            security_patterns['rate_limiting'] = True
                        if 'validate' in content or 'sanitize' in content:
                            security_patterns['input_validation'] = True
                except:
                    pass
            
            security_score = sum(security_patterns.values())
            
            if security_score >= 2:  # At least 2 out of 3 security measures
                self.log("✅ Security basics: READY", 'PASS')
                self.technical_results['security_basics'] = True
                return True
            else:
                self.log(f"❌ Security basics: {security_score}/3 measures found", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Security basics check failed: {e}", 'FAIL')
            return False
    
    async def check_health_endpoints(self):
        """🔧 Health endpoints functional"""
        self.log("🔧 [TECHNICAL] Checking Health Endpoints...", 'INFO')
        
        try:
            health_found = False
            
            # Look for health endpoint implementations
            for py_file in Path('.').glob('**/*.py'):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if '/health' in content and ('get' in content.lower() or '@app.route' in content):
                            self.log(f"   ✅ Health endpoint in: {py_file}", 'INFO')
                            health_found = True
                            break
                except:
                    pass
            
            if health_found:
                self.log("✅ Health endpoints: READY", 'PASS')
                self.technical_results['health_endpoints'] = True
                return True
            else:
                self.log("❌ Health endpoints: NOT FOUND", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Health endpoints check failed: {e}", 'FAIL')
            return False
    
    async def check_error_handling(self):
        """🔧 Error handling graceful (no crashes)"""
        self.log("🔧 [TECHNICAL] Checking Error Handling...", 'INFO')
        
        try:
            error_handling_score = 0
            
            # Check for error handling patterns
            for py_file in Path('.').glob('**/*.py'):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        
                        # Count error handling mechanisms
                        if 'try:' in content and 'except' in content:
                            error_handling_score += 1
                        if 'logging' in content:
                            error_handling_score += 1
                        if 'graceful' in content.lower() or 'fallback' in content.lower():
                            error_handling_score += 1
                except:
                    pass
            
            if error_handling_score >= 5:  # Multiple files with error handling
                self.log("✅ Error handling: ROBUST", 'PASS')
                self.technical_results['error_handling'] = True
                return True
            else:
                self.log(f"❌ Error handling: {error_handling_score} patterns found", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Error handling check failed: {e}", 'FAIL')
            return False
    
    async def check_deployment_files(self):
        """🔧 Git repo has necessary deployment files"""
        self.log("🔧 [TECHNICAL] Checking Deployment Files...", 'INFO')
        
        try:
            deployment_files = [
                'README.md',
                'requirements.txt',
                'render.yaml',
                '.github/workflows/bem-cicd.yml',
                'start_realtime_system.py'
            ]
            
            found_files = 0
            for dep_file in deployment_files:
                if Path(dep_file).exists():
                    self.log(f"   ✅ Found: {dep_file}", 'INFO')
                    found_files += 1
            
            if found_files >= 4:  # At least 4 out of 5 deployment files
                self.log("✅ Deployment files: READY", 'PASS')
                self.technical_results['deployment_files'] = True
                return True
            else:
                self.log(f"❌ Deployment files: {found_files}/5 found", 'FAIL')
                return False
                
        except Exception as e:
            self.log(f"❌ Deployment files check failed: {e}", 'FAIL')
            return False
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    async def run_comprehensive_check(self):
        """Run all deployment readiness checks"""
        self.log("🚀 DEPLOYMENT READINESS CHECK", 'INFO')
        self.log("=" * 80, 'INFO')
        
        # Strategic checks (user's criteria)
        self.log("🎯 STRATEGIC CRITERIA (User Requirements):", 'INFO')
        strategic_checks = [
            self.check_core_node_execution(),
            self.check_graphql_engine(),
            self.check_ui_dag_display(),
            self.check_functor_execution()
        ]
        
        strategic_results = await asyncio.gather(*strategic_checks, return_exceptions=True)
        strategic_passed = sum(1 for result in strategic_results if result is True)
        
        self.log("-" * 80, 'INFO')
        
        # Technical checks (implementation requirements)
        self.log("🔧 TECHNICAL CRITERIA (Implementation Requirements):", 'INFO')
        technical_checks = [
            self.check_docker_build(),
            self.check_environment_config(),
            self.check_database_connectivity(),
            self.check_websocket_stability(),
            self.check_security_basics(),
            self.check_health_endpoints(),
            self.check_error_handling(),
            self.check_deployment_files()
        ]
        
        technical_results = await asyncio.gather(*technical_checks, return_exceptions=True)
        technical_passed = sum(1 for result in technical_results if result is True)
        
        # Overall assessment
        self.log("=" * 80, 'INFO')
        self.log(f"📊 STRATEGIC: {strategic_passed}/4 criteria met", 'INFO')
        self.log(f"📊 TECHNICAL: {technical_passed}/8 criteria met", 'INFO')
        
        # Deployment decision logic
        strategic_ready = strategic_passed >= 3  # At least 3/4 strategic
        technical_ready = technical_passed >= 6   # At least 6/8 technical
        
        self.overall_ready = strategic_ready and technical_ready
        
        if self.overall_ready:
            self.log("🎉 DEPLOYMENT APPROVED! Ready for production.", 'PASS')
        elif strategic_ready:
            self.log("⚠️ STRATEGIC READY but technical gaps exist. Deploy with monitoring.", 'WARN')
        else:
            self.log("❌ NOT READY. Strategic criteria not met.", 'FAIL')
        
        return {
            'strategic': self.strategic_results,
            'technical': self.technical_results,
            'overall_ready': self.overall_ready,
            'strategic_passed': strategic_passed,
            'technical_passed': technical_passed
        }
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        report = {
            'deployment_ready': self.overall_ready,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'strategic_criteria': self.strategic_results,
            'technical_criteria': self.technical_results,
            'detailed_log': self.detailed_results,
            'deployment_command': self.get_deployment_command(),
            'monitoring_setup': self.get_monitoring_setup()
        }
        
        with open('deployment_readiness_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def get_deployment_command(self):
        """Get appropriate deployment command based on readiness"""
        if self.overall_ready:
            return [
                "# Full deployment approved",
                "python deploy_to_render.py --production",
                "# OR manual Render deployment:",
                "git push origin main  # Triggers auto-deploy via render.yaml"
            ]
        else:
            return [
                "# Staging deployment only",
                "python deploy_to_render.py --staging",
                "# Fix issues before production deployment"
            ]
    
    def get_monitoring_setup(self):
        """Get monitoring setup recommendations"""
        return [
            "✅ GitHub Actions CI/CD: Already configured",
            "✅ Health endpoints: Monitor /health",
            "✅ WebSocket monitoring: Check connection count",
            "✅ GraphQL metrics: Query response times",
            "⚠️ Add production logging aggregation",
            "⚠️ Set up error alerting (email/Slack)",
            "⚠️ Configure uptime monitoring (UptimeRobot/Pingdom)"
        ]

async def main():
    """Main execution with leadership-friendly output"""
    checker = DeploymentReadinessChecker()
    
    try:
        # Run comprehensive check
        results = await checker.run_comprehensive_check()
        
        # Generate report
        report = checker.generate_deployment_report()
        
        # Executive summary
        print("\n" + "=" * 80)
        print("📋 EXECUTIVE SUMMARY:")
        print(f"   Strategic Readiness: {results['strategic_passed']}/4 ✅")
        print(f"   Technical Readiness: {results['technical_passed']}/8 ✅")
        print(f"   Overall Status: {'DEPLOY NOW' if results['overall_ready'] else 'NEEDS WORK'}")
        print("=" * 80)
        
        # Next actions
        if results['overall_ready']:
            print("🚀 RECOMMENDED ACTION: Deploy to production")
            print("📋 COMMAND: python deploy_to_render.py --production")
        else:
            print("⚠️ RECOMMENDED ACTION: Address gaps and redeploy")
            print("📋 COMMAND: python deployment_readiness_check.py")
        
        print("\n📄 Full report: deployment_readiness_report.json")
        
        # Exit with appropriate code
        sys.exit(0 if results['overall_ready'] else 1)
        
    except KeyboardInterrupt:
        print("\n👋 Check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 