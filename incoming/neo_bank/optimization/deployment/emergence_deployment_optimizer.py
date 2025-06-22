#!/usr/bin/env python3
"""
BEM Emergence Deployment Optimizer
Optimized deployment configuration for VaaS, PaaS, P2P financial modes
Docker, Kubernetes, and cloud infrastructure optimization
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import time

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Service configuration for deployment"""
    name: str
    image: str
    replicas: int
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    ports: List[int]
    env_vars: Dict[str, str]
    volumes: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.volumes is None:
            self.volumes = []

class EmergenceDeploymentOptimizer:
    """
    Deployment optimizer for BEM emergence financial system
    Generates optimized configurations for different environments
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.base_config = self._load_base_config()
        
        # Resource allocation based on financial mode requirements
        self.service_specs = {
            'vaas': {
                'cpu_intensive': True,    # Payment processing
                'memory_intensive': False,
                'network_intensive': True, # API calls
                'storage_intensive': False
            },
            'paas': {
                'cpu_intensive': False,
                'memory_intensive': True,  # Pool calculations
                'network_intensive': False,
                'storage_intensive': True  # Contribution tracking
            },
            'p2p': {
                'cpu_intensive': False,
                'memory_intensive': False,
                'network_intensive': True, # Agent communications
                'storage_intensive': False
            },
            'emergence_generator': {
                'cpu_intensive': True,    # DGL computations
                'memory_intensive': True, # Graph processing
                'network_intensive': False,
                'storage_intensive': True  # Model storage
            }
        }
        
        logger.info(f"Deployment optimizer initialized for {environment}")
    
    def _load_base_config(self) -> Dict:
        """Load base configuration settings"""
        return {
            'development': {
                'replicas': 1,
                'cpu_request': '100m',
                'cpu_limit': '500m',
                'memory_request': '128Mi',
                'memory_limit': '512Mi'
            },
            'staging': {
                'replicas': 2,
                'cpu_request': '200m',
                'cpu_limit': '1000m',
                'memory_request': '256Mi',
                'memory_limit': '1Gi'
            },
            'production': {
                'replicas': 3,
                'cpu_request': '500m',
                'cpu_limit': '2000m',
                'memory_request': '512Mi',
                'memory_limit': '2Gi'
            }
        }
    
    def generate_service_configs(self) -> Dict[str, ServiceConfig]:
        """Generate optimized service configurations"""
        base = self.base_config[self.environment]
        configs = {}
        
        # VaaS Service (Payment Processing)
        configs['vaas_service'] = ServiceConfig(
            name='vaas-service',
            image='bem/vaas-service:latest',
            replicas=base['replicas'] * 2,  # Higher availability for payments
            cpu_request=base['cpu_request'],
            cpu_limit='3000m' if self.environment == 'production' else base['cpu_limit'],
            memory_request=base['memory_request'],
            memory_limit=base['memory_limit'],
            ports=[8080, 8443],  # HTTP and HTTPS
            env_vars={
                'PAYMENT_GATEWAY_URL': os.getenv('PAYMENT_GATEWAY_URL', ''),
                'REDIS_URL': 'redis://redis-cluster:6379',
                'DATABASE_URL': 'postgresql://vaas-db:5432/vaas_db',
                'LOG_LEVEL': 'INFO',
                'MAX_CONCURRENT_PAYMENTS': '100'
            }
        )
        
        # PaaS Service (Pool Management)
        configs['paas_service'] = ServiceConfig(
            name='paas-service',
            image='bem/paas-service:latest',
            replicas=base['replicas'],
            cpu_request=base['cpu_request'],
            cpu_limit=base['cpu_limit'],
            memory_request='1Gi' if self.environment == 'production' else base['memory_request'],
            memory_limit='4Gi' if self.environment == 'production' else base['memory_limit'],
            ports=[8081],
            env_vars={
                'REDIS_URL': 'redis://redis-cluster:6379',
                'DATABASE_URL': 'postgresql://paas-db:5432/paas_db',
                'POOL_CALCULATION_WORKERS': '10',
                'CONTRIBUTION_BATCH_SIZE': '50'
            }
        )
        
        # P2P Service (Agent Exchange)
        configs['p2p_service'] = ServiceConfig(
            name='p2p-service',
            image='bem/p2p-service:latest',
            replicas=base['replicas'],
            cpu_request='50m',  # Lower CPU for P2P
            cpu_limit='500m',
            memory_request='128Mi',
            memory_limit='512Mi',
            ports=[8082, 9000],  # HTTP and WebSocket
            env_vars={
                'REDIS_URL': 'redis://redis-cluster:6379',
                'DATABASE_URL': 'postgresql://p2p-db:5432/p2p_db',
                'TRUST_CALCULATION_INTERVAL': '300',  # 5 minutes
                'MAX_CONCURRENT_EXCHANGES': '200'
            }
        )
        
        # Emergence Generator (DGL/AI Service)
        configs['emergence_generator'] = ServiceConfig(
            name='emergence-generator',
            image='bem/emergence-generator:latest',
            replicas=2 if self.environment == 'production' else 1,
            cpu_request='1000m',  # CPU intensive
            cpu_limit='4000m' if self.environment == 'production' else '2000m',
            memory_request='2Gi',  # Memory intensive
            memory_limit='8Gi' if self.environment == 'production' else '4Gi',
            ports=[8083],
            env_vars={
                'DGL_WORKERS': '4',
                'MODEL_CACHE_SIZE': '1000',
                'GRAPH_PROCESSING_TIMEOUT': '300',
                'GPU_ENABLED': 'true' if self.environment == 'production' else 'false'
            },
            volumes=[
                {'name': 'model-storage', 'mountPath': '/app/models'},
                {'name': 'graph-data', 'mountPath': '/app/data'}
            ]
        )
        
        # Emergence Router (Load Balancer)
        configs['emergence_router'] = ServiceConfig(
            name='emergence-router',
            image='bem/emergence-router:latest',
            replicas=base['replicas'],
            cpu_request=base['cpu_request'],
            cpu_limit=base['cpu_limit'],
            memory_request=base['memory_request'],
            memory_limit=base['memory_limit'],
            ports=[8000, 8443],
            env_vars={
                'VAAS_SERVICE_URL': 'http://vaas-service:8080',
                'PAAS_SERVICE_URL': 'http://paas-service:8081',
                'P2P_SERVICE_URL': 'http://p2p-service:8082',
                'EMERGENCE_GENERATOR_URL': 'http://emergence-generator:8083',
                'REDIS_URL': 'redis://redis-cluster:6379',
                'ROUTING_STRATEGY': 'least_connections'
            }
        )
        
        # Performance Monitor
        configs['performance_monitor'] = ServiceConfig(
            name='performance-monitor',
            image='bem/performance-monitor:latest',
            replicas=1,  # Single instance for monitoring
            cpu_request='100m',
            cpu_limit='500m',
            memory_request='256Mi',
            memory_limit='1Gi',
            ports=[8090, 9090],  # Dashboard and metrics
            env_vars={
                'MONITORING_INTERVAL': '10',
                'ALERT_WEBHOOK_URL': os.getenv('ALERT_WEBHOOK_URL', ''),
                'METRICS_RETENTION_HOURS': '168'  # 1 week
            }
        )
        
        return configs
    
    def generate_kubernetes_manifests(self) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests"""
        services = self.generate_service_configs()
        manifests = {}
        
        for service_name, config in services.items():
            manifest = self._create_k8s_deployment(config)
            manifests[f"{service_name}-deployment.yaml"] = yaml.dump(manifest, default_flow_style=False)
            
            # Create service manifest
            service_manifest = self._create_k8s_service(config)
            manifests[f"{service_name}-service.yaml"] = yaml.dump(service_manifest, default_flow_style=False)
        
        # Add infrastructure components
        manifests.update(self._generate_infrastructure_manifests())
        
        return manifests
    
    def _create_k8s_deployment(self, config: ServiceConfig) -> Dict:
        """Create Kubernetes deployment manifest"""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': config.name,
                'labels': {
                    'app': config.name,
                    'environment': self.environment,
                    'component': 'bem-emergence'
                }
            },
            'spec': {
                'replicas': config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': config.name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': config.name,
                            'environment': self.environment
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': config.name,
                            'image': config.image,
                            'ports': [{'containerPort': port} for port in config.ports],
                            'env': [
                                {'name': key, 'value': value}
                                for key, value in config.env_vars.items()
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': config.cpu_request,
                                    'memory': config.memory_request
                                },
                                'limits': {
                                    'cpu': config.cpu_limit,
                                    'memory': config.memory_limit
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': config.ports[0]
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': config.ports[0]
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            },
                            'volumeMounts': [
                                {
                                    'name': vol['name'],
                                    'mountPath': vol['mountPath']
                                }
                                for vol in config.volumes
                            ] if config.volumes else []
                        }],
                        'volumes': [
                            {
                                'name': vol['name'],
                                'persistentVolumeClaim': {
                                    'claimName': f"{vol['name']}-pvc"
                                }
                            }
                            for vol in config.volumes
                        ] if config.volumes else []
                    }
                }
            }
        }
    
    def _create_k8s_service(self, config: ServiceConfig) -> Dict:
        """Create Kubernetes service manifest"""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': config.name,
                'labels': {
                    'app': config.name,
                    'environment': self.environment
                }
            },
            'spec': {
                'selector': {
                    'app': config.name
                },
                'ports': [
                    {
                        'name': f"port-{port}",
                        'port': port,
                        'targetPort': port,
                        'protocol': 'TCP'
                    }
                    for port in config.ports
                ],
                'type': 'ClusterIP'
            }
        }
    
    def _generate_infrastructure_manifests(self) -> Dict[str, str]:
        """Generate infrastructure component manifests"""
        manifests = {}
        
        # Redis Cluster
        manifests['redis-cluster.yaml'] = yaml.dump({
            'apiVersion': 'apps/v1',
            'kind': 'StatefulSet',
            'metadata': {
                'name': 'redis-cluster',
                'labels': {'app': 'redis-cluster'}
            },
            'spec': {
                'serviceName': 'redis-cluster',
                'replicas': 3 if self.environment == 'production' else 1,
                'selector': {'matchLabels': {'app': 'redis-cluster'}},
                'template': {
                    'metadata': {'labels': {'app': 'redis-cluster'}},
                    'spec': {
                        'containers': [{
                            'name': 'redis',
                            'image': 'redis:7-alpine',
                            'ports': [{'containerPort': 6379}],
                            'resources': {
                                'requests': {'cpu': '100m', 'memory': '128Mi'},
                                'limits': {'cpu': '500m', 'memory': '512Mi'}
                            },
                            'volumeMounts': [{
                                'name': 'redis-data',
                                'mountPath': '/data'
                            }]
                        }]
                    }
                },
                'volumeClaimTemplates': [{
                    'metadata': {'name': 'redis-data'},
                    'spec': {
                        'accessModes': ['ReadWriteOnce'],
                        'resources': {'requests': {'storage': '10Gi'}}
                    }
                }]
            }
        }, default_flow_style=False)
        
        # PostgreSQL Databases
        for db_name in ['vaas-db', 'paas-db', 'p2p-db']:
            manifests[f'{db_name}.yaml'] = yaml.dump({
                'apiVersion': 'apps/v1',
                'kind': 'StatefulSet',
                'metadata': {
                    'name': db_name,
                    'labels': {'app': db_name}
                },
                'spec': {
                    'serviceName': db_name,
                    'replicas': 1,
                    'selector': {'matchLabels': {'app': db_name}},
                    'template': {
                        'metadata': {'labels': {'app': db_name}},
                        'spec': {
                            'containers': [{
                                'name': 'postgres',
                                'image': 'postgres:15',
                                'ports': [{'containerPort': 5432}],
                                'env': [
                                    {'name': 'POSTGRES_DB', 'value': f'{db_name.replace("-", "_")}'},
                                    {'name': 'POSTGRES_USER', 'value': 'bem_user'},
                                    {'name': 'POSTGRES_PASSWORD', 'valueFrom': {
                                        'secretKeyRef': {
                                            'name': 'postgres-secret',
                                            'key': 'password'
                                        }
                                    }}
                                ],
                                'resources': {
                                    'requests': {'cpu': '200m', 'memory': '256Mi'},
                                    'limits': {'cpu': '1000m', 'memory': '1Gi'}
                                },
                                'volumeMounts': [{
                                    'name': 'postgres-data',
                                    'mountPath': '/var/lib/postgresql/data'
                                }]
                            }]
                        }
                    },
                    'volumeClaimTemplates': [{
                        'metadata': {'name': 'postgres-data'},
                        'spec': {
                            'accessModes': ['ReadWriteOnce'],
                            'resources': {'requests': {'storage': '20Gi'}}
                        }
                    }]
                }
            }, default_flow_style=False)
        
        # Ingress Controller
        manifests['ingress.yaml'] = yaml.dump({
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': 'bem-emergence-ingress',
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': ['api.bem-emergence.com'],
                    'secretName': 'bem-emergence-tls'
                }],
                'rules': [{
                    'host': 'api.bem-emergence.com',
                    'http': {
                        'paths': [
                            {
                                'path': '/vaas',
                                'pathType': 'Prefix',
                                'backend': {
                                    'service': {
                                        'name': 'vaas-service',
                                        'port': {'number': 8080}
                                    }
                                }
                            },
                            {
                                'path': '/paas',
                                'pathType': 'Prefix',
                                'backend': {
                                    'service': {
                                        'name': 'paas-service',
                                        'port': {'number': 8081}
                                    }
                                }
                            },
                            {
                                'path': '/p2p',
                                'pathType': 'Prefix',
                                'backend': {
                                    'service': {
                                        'name': 'p2p-service',
                                        'port': {'number': 8082}
                                    }
                                }
                            },
                            {
                                'path': '/',
                                'pathType': 'Prefix',
                                'backend': {
                                    'service': {
                                        'name': 'emergence-router',
                                        'port': {'number': 8000}
                                    }
                                }
                            }
                        ]
                    }
                }]
            }
        }, default_flow_style=False)
        
        return manifests
    
    def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration"""
        services = self.generate_service_configs()
        
        compose_config = {
            'version': '3.8',
            'services': {},
            'networks': {
                'bem-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'redis-data': {},
                'postgres-vaas-data': {},
                'postgres-paas-data': {},
                'postgres-p2p-data': {},
                'model-storage': {},
                'graph-data': {}
            }
        }
        
        # Add services
        for service_name, config in services.items():
            compose_config['services'][service_name.replace('_', '-')] = {
                'image': config.image,
                'ports': [f"{port}:{port}" for port in config.ports],
                'environment': config.env_vars,
                'networks': ['bem-network'],
                'restart': 'unless-stopped',
                'deploy': {
                    'replicas': config.replicas,
                    'resources': {
                        'limits': {
                            'cpus': config.cpu_limit.replace('m', ''),
                            'memory': config.memory_limit
                        },
                        'reservations': {
                            'cpus': config.cpu_request.replace('m', ''),
                            'memory': config.memory_request
                        }
                    }
                }
            }
            
            # Add volumes if specified
            if config.volumes:
                compose_config['services'][service_name.replace('_', '-')]['volumes'] = [
                    f"{vol['name']}:{vol['mountPath']}"
                    for vol in config.volumes
                ]
        
        # Add infrastructure services
        compose_config['services'].update({
            'redis-cluster': {
                'image': 'redis:7-alpine',
                'ports': ['6379:6379'],
                'volumes': ['redis-data:/data'],
                'networks': ['bem-network'],
                'restart': 'unless-stopped'
            },
            'vaas-db': {
                'image': 'postgres:15',
                'environment': {
                    'POSTGRES_DB': 'vaas_db',
                    'POSTGRES_USER': 'bem_user',
                    'POSTGRES_PASSWORD': '${POSTGRES_PASSWORD}'
                },
                'volumes': ['postgres-vaas-data:/var/lib/postgresql/data'],
                'networks': ['bem-network'],
                'restart': 'unless-stopped'
            },
            'paas-db': {
                'image': 'postgres:15',
                'environment': {
                    'POSTGRES_DB': 'paas_db',
                    'POSTGRES_USER': 'bem_user',
                    'POSTGRES_PASSWORD': '${POSTGRES_PASSWORD}'
                },
                'volumes': ['postgres-paas-data:/var/lib/postgresql/data'],
                'networks': ['bem-network'],
                'restart': 'unless-stopped'
            },
            'p2p-db': {
                'image': 'postgres:15',
                'environment': {
                    'POSTGRES_DB': 'p2p_db',
                    'POSTGRES_USER': 'bem_user',
                    'POSTGRES_PASSWORD': '${POSTGRES_PASSWORD}'
                },
                'volumes': ['postgres-p2p-data:/var/lib/postgresql/data'],
                'networks': ['bem-network'],
                'restart': 'unless-stopped'
            }
        })
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def generate_helm_chart(self) -> Dict[str, str]:
        """Generate Helm chart for deployment"""
        chart_files = {}
        
        # Chart.yaml
        chart_files['Chart.yaml'] = yaml.dump({
            'apiVersion': 'v2',
            'name': 'bem-emergence',
            'description': 'BEM Emergence Financial System',
            'type': 'application',
            'version': '1.0.0',
            'appVersion': '1.0.0',
            'dependencies': [
                {
                    'name': 'redis',
                    'version': '17.0.0',
                    'repository': 'https://charts.bitnami.com/bitnami'
                },
                {
                    'name': 'postgresql',
                    'version': '12.0.0',
                    'repository': 'https://charts.bitnami.com/bitnami'
                }
            ]
        }, default_flow_style=False)
        
        # values.yaml
        services = self.generate_service_configs()
        values = {
            'environment': self.environment,
            'services': {
                name: {
                    'image': config.image,
                    'replicas': config.replicas,
                    'resources': {
                        'requests': {
                            'cpu': config.cpu_request,
                            'memory': config.memory_request
                        },
                        'limits': {
                            'cpu': config.cpu_limit,
                            'memory': config.memory_limit
                        }
                    },
                    'env': config.env_vars
                }
                for name, config in services.items()
            },
            'ingress': {
                'enabled': True,
                'host': 'api.bem-emergence.com',
                'tls': True
            },
            'redis': {
                'enabled': True,
                'auth': {'enabled': False}
            },
            'postgresql': {
                'enabled': True,
                'auth': {
                    'postgresPassword': '${POSTGRES_PASSWORD}',
                    'database': 'bem_emergence'
                }
            }
        }
        
        chart_files['values.yaml'] = yaml.dump(values, default_flow_style=False)
        
        return chart_files
    
    def optimize_for_cloud_provider(self, provider: str) -> Dict[str, Any]:
        """Generate cloud-specific optimizations"""
        optimizations = {
            'aws': {
                'instance_types': {
                    'vaas_service': 'c5.large',      # CPU optimized for payments
                    'paas_service': 'r5.large',      # Memory optimized for pools
                    'p2p_service': 't3.medium',      # Balanced for P2P
                    'emergence_generator': 'p3.xlarge'  # GPU for AI/DGL
                },
                'storage': {
                    'type': 'gp3',
                    'iops': 3000,
                    'throughput': 125
                },
                'networking': {
                    'load_balancer': 'application',
                    'ssl_policy': 'ELBSecurityPolicy-TLS-1-2-2017-01'
                }
            },
            'gcp': {
                'machine_types': {
                    'vaas_service': 'c2-standard-2',
                    'paas_service': 'n2-highmem-2',
                    'p2p_service': 'e2-standard-2',
                    'emergence_generator': 'a2-highgpu-1g'
                },
                'storage': {
                    'type': 'pd-ssd',
                    'size': '100GB'
                },
                'networking': {
                    'load_balancer': 'global',
                    'cdn_enabled': True
                }
            },
            'azure': {
                'vm_sizes': {
                    'vaas_service': 'Standard_F2s_v2',
                    'paas_service': 'Standard_E2s_v3',
                    'p2p_service': 'Standard_B2s',
                    'emergence_generator': 'Standard_NC6s_v3'
                },
                'storage': {
                    'type': 'Premium_LRS',
                    'tier': 'P10'
                },
                'networking': {
                    'load_balancer': 'standard',
                    'application_gateway': True
                }
            }
        }
        
        return optimizations.get(provider, {})
    
    def save_configurations(self, output_dir: str = "./deployment_configs"):
        """Save all generated configurations to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Kubernetes manifests
        k8s_path = output_path / "kubernetes"
        k8s_path.mkdir(exist_ok=True)
        
        manifests = self.generate_kubernetes_manifests()
        for filename, content in manifests.items():
            (k8s_path / filename).write_text(content)
        
        # Docker Compose
        docker_compose = self.generate_docker_compose()
        (output_path / "docker-compose.yml").write_text(docker_compose)
        
        # Helm Chart
        helm_path = output_path / "helm"
        helm_path.mkdir(exist_ok=True)
        
        helm_files = self.generate_helm_chart()
        for filename, content in helm_files.items():
            (helm_path / filename).write_text(content)
        
        # Cloud optimizations
        for provider in ['aws', 'gcp', 'azure']:
            optimizations = self.optimize_for_cloud_provider(provider)
            if optimizations:
                (output_path / f"{provider}_optimizations.json").write_text(
                    json.dumps(optimizations, indent=2)
                )
        
        logger.info(f"Deployment configurations saved to {output_path}")

if __name__ == "__main__":
    # Generate deployment configurations
    optimizer = EmergenceDeploymentOptimizer(environment="production")
    
    # Generate and save all configurations
    optimizer.save_configurations("./emergence_deployment_configs")
    
    # Display service configurations
    services = optimizer.generate_service_configs()
    print("Generated Service Configurations:")
    for name, config in services.items():
        print(f"\n{name}:")
        print(f"  Image: {config.image}")
        print(f"  Replicas: {config.replicas}")
        print(f"  CPU: {config.cpu_request} - {config.cpu_limit}")
        print(f"  Memory: {config.memory_request} - {config.memory_limit}")
        print(f"  Ports: {config.ports}")
    
    print(f"\nDeployment configurations generated for production environment")
