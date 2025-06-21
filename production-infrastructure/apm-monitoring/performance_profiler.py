"""
Performance Profiling System
Advanced profiling capabilities for production performance analysis
"""

import cProfile
import pstats
import time
import threading
import tracemalloc
import gc
import sys
import os
import json
import psutil
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import functools
from contextlib import contextmanager
import logging

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

class PerformanceProfiler:
    """Comprehensive performance profiling system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.profiles = {}
        self.memory_snapshots = deque(maxlen=100)
        self.cpu_profiles = deque(maxlen=50)
        self.execution_stats = defaultdict(list)
        self.start_time = datetime.now()
        self._lock = threading.Lock()
        
        # Initialize memory tracking
        tracemalloc.start()
        
        # Configuration
        self.enable_memory_profiling = self.config.get('memory_profiling', True)
        self.enable_cpu_profiling = self.config.get('cpu_profiling', True)
        self.enable_line_profiling = self.config.get('line_profiling', False)
        self.profile_interval = self.config.get('profile_interval', 60)  # seconds
        
        # Start background profiling
        self._init_background_profiling()
        
        logging.info("Performance Profiler initialized")
    
    def _init_background_profiling(self):
        """Initialize background profiling threads"""
        
        def memory_monitor():
            """Background memory monitoring"""
            while True:
                try:
                    if self.enable_memory_profiling:
                        self._capture_memory_snapshot()
                    time.sleep(self.profile_interval)
                except Exception as e:
                    logging.error(f"Memory monitor error: {e}")
        
        def cpu_monitor():
            """Background CPU monitoring"""
            while True:
                try:
                    if self.enable_cpu_profiling:
                        self._capture_cpu_snapshot()
                    time.sleep(self.profile_interval)
                except Exception as e:
                    logging.error(f"CPU monitor error: {e}")
        
        # Start monitoring threads
        memory_thread = threading.Thread(target=memory_monitor, daemon=True)
        cpu_thread = threading.Thread(target=cpu_monitor, daemon=True)
        
        memory_thread.start()
        cpu_thread.start()
    
    def _capture_memory_snapshot(self):
        """Capture current memory usage snapshot"""
        try:
            # Memory usage details
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Tracemalloc snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Python memory details
            gc_stats = gc.get_stats()
            
            memory_snapshot = {
                'timestamp': datetime.now(),
                'process_memory': {
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'percent': process.memory_percent()
                },
                'python_memory': {
                    'traced_mb': sum(stat.size for stat in top_stats) / (1024 * 1024),
                    'top_allocations': [
                        {
                            'file': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                            'size_mb': stat.size / (1024 * 1024),
                            'count': stat.count
                        }
                        for stat in top_stats[:10]
                    ]
                },
                'gc_stats': gc_stats,
                'object_counts': {
                    'total_objects': len(gc.get_objects()),
                    'generation_0': gc_stats[0]['count'] if gc_stats else 0,
                    'generation_1': gc_stats[1]['count'] if len(gc_stats) > 1 else 0,
                    'generation_2': gc_stats[2]['count'] if len(gc_stats) > 2 else 0
                }
            }
            
            with self._lock:
                self.memory_snapshots.append(memory_snapshot)
                
        except Exception as e:
            logging.error(f"Memory snapshot error: {e}")
    
    def _capture_cpu_snapshot(self):
        """Capture current CPU usage snapshot"""
        try:
            process = psutil.Process()
            
            cpu_snapshot = {
                'timestamp': datetime.now(),
                'cpu_percent': process.cpu_percent(),
                'cpu_times': process.cpu_times()._asdict(),
                'threads': process.num_threads(),
                'context_switches': process.num_ctx_switches()._asdict(),
                'system_cpu': psutil.cpu_percent(percpu=True),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            with self._lock:
                self.cpu_profiles.append(cpu_snapshot)
                
        except Exception as e:
            logging.error(f"CPU snapshot error: {e}")
    
    @contextmanager
    def profile_block(self, name: str, enable_line_profile: bool = False):
        """Context manager for profiling code blocks"""
        start_time = time.time()
        start_memory = tracemalloc.take_snapshot()
        
        # CPU profiling
        profiler = None
        if self.enable_cpu_profiling:
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Line profiling
        line_prof = None
        if enable_line_profile and LINE_PROFILER_AVAILABLE:
            line_prof = line_profiler.LineProfiler()
        
        try:
            yield {
                'profiler': profiler,
                'line_profiler': line_prof
            }
        finally:
            # Stop profiling
            end_time = time.time()
            duration = end_time - start_time
            
            if profiler:
                profiler.disable()
            
            # Memory analysis
            end_memory = tracemalloc.take_snapshot()
            memory_diff = end_memory.compare_to(start_memory, 'lineno')
            
            # Store results
            profile_result = {
                'name': name,
                'timestamp': datetime.now(),
                'duration_seconds': duration,
                'memory_delta': {
                    'top_changes': [
                        {
                            'file': diff.traceback.format()[-1] if diff.traceback else 'unknown',
                            'size_delta_mb': diff.size_diff / (1024 * 1024),
                            'count_delta': diff.count_diff
                        }
                        for diff in memory_diff[:10]
                    ],
                    'total_delta_mb': sum(diff.size_diff for diff in memory_diff) / (1024 * 1024)
                }
            }
            
            if profiler:
                # Save CPU profile
                profile_result['cpu_profile'] = self._analyze_cpu_profile(profiler)
            
            with self._lock:
                self.execution_stats[name].append(profile_result)
                
                # Keep only last 100 entries per operation
                if len(self.execution_stats[name]) > 100:
                    self.execution_stats[name] = self.execution_stats[name][-100:]
    
    def _analyze_cpu_profile(self, profiler: cProfile.Profile) -> Dict[str, Any]:
        """Analyze CPU profile and extract key statistics"""
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Get top functions
        top_functions = []
        for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:20]:
            filename, line_no, func_name = func_info
            top_functions.append({
                'function': f"{filename}:{line_no}({func_name})",
                'calls': cc,
                'total_time': tt,
                'cumulative_time': ct,
                'per_call': tt/cc if cc > 0 else 0
            })
        
        return {
            'total_calls': stats.total_calls,
            'total_time': stats.total_tt,
            'top_functions': top_functions
        }
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a single function execution"""
        func_name = f"{func.__module__}.{func.__name__}"
        
        with self.profile_block(func_name) as profile_ctx:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
            end_time = time.time()
        
        # Get the latest profile result
        with self._lock:
            latest_profile = self.execution_stats[func_name][-1] if self.execution_stats[func_name] else {}
        
        return {
            'result': result,
            'success': success,
            'error': error,
            'duration_seconds': end_time - start_time,
            'profile_data': latest_profile
        }
    
    def get_performance_report(self, hours_back: int = 1) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            # Filter recent data
            recent_memory = [s for s in self.memory_snapshots if s['timestamp'] > cutoff_time]
            recent_cpu = [s for s in self.cpu_profiles if s['timestamp'] > cutoff_time]
            
            # Execution statistics
            recent_execution = {}
            for operation, stats in self.execution_stats.items():
                recent_stats = [s for s in stats if s['timestamp'] > cutoff_time]
                if recent_stats:
                    recent_execution[operation] = {
                        'count': len(recent_stats),
                        'avg_duration': sum(s['duration_seconds'] for s in recent_stats) / len(recent_stats),
                        'max_duration': max(s['duration_seconds'] for s in recent_stats),
                        'min_duration': min(s['duration_seconds'] for s in recent_stats),
                        'total_memory_delta_mb': sum(
                            s['memory_delta']['total_delta_mb'] for s in recent_stats
                        )
                    }
        
        # Memory analysis
        memory_analysis = {}
        if recent_memory:
            memory_values = [s['process_memory']['rss_mb'] for s in recent_memory]
            memory_analysis = {
                'current_mb': memory_values[-1] if memory_values else 0,
                'avg_mb': sum(memory_values) / len(memory_values),
                'max_mb': max(memory_values),
                'min_mb': min(memory_values),
                'trend': 'increasing' if len(memory_values) > 1 and memory_values[-1] > memory_values[0] else 'stable'
            }
        
        # CPU analysis
        cpu_analysis = {}
        if recent_cpu:
            cpu_values = [s['cpu_percent'] for s in recent_cpu]
            cpu_analysis = {
                'current_percent': cpu_values[-1] if cpu_values else 0,
                'avg_percent': sum(cpu_values) / len(cpu_values),
                'max_percent': max(cpu_values),
                'min_percent': min(cpu_values)
            }
        
        return {
            'report_period_hours': hours_back,
            'timestamp': datetime.now(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'memory_analysis': memory_analysis,
            'cpu_analysis': cpu_analysis,
            'execution_analysis': recent_execution,
            'total_snapshots': {
                'memory': len(recent_memory),
                'cpu': len(recent_cpu),
                'operations': sum(len(stats) for stats in self.execution_stats.values())
            },
            'recommendations': self._generate_recommendations(memory_analysis, cpu_analysis, recent_execution)
        }
    
    def _generate_recommendations(self, memory_analysis: Dict, cpu_analysis: Dict, execution_analysis: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        if memory_analysis.get('avg_mb', 0) > 500:
            recommendations.append("High memory usage detected. Consider memory optimization.")
        
        if memory_analysis.get('trend') == 'increasing':
            recommendations.append("Memory usage is increasing. Check for memory leaks.")
        
        # CPU recommendations
        if cpu_analysis.get('avg_percent', 0) > 70:
            recommendations.append("High CPU usage detected. Consider performance optimization.")
        
        # Execution recommendations
        slow_operations = [
            op for op, stats in execution_analysis.items()
            if stats.get('avg_duration', 0) > 1.0
        ]
        
        if slow_operations:
            recommendations.append(f"Slow operations detected: {', '.join(slow_operations)}")
        
        if not recommendations:
            recommendations.append("System performance looks good!")
        
        return recommendations
    
    def export_profile_data(self, filepath: str):
        """Export all profile data to JSON file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'memory_snapshots': [
                {**snapshot, 'timestamp': snapshot['timestamp'].isoformat()}
                for snapshot in self.memory_snapshots
            ],
            'cpu_profiles': [
                {**profile, 'timestamp': profile['timestamp'].isoformat()}
                for profile in self.cpu_profiles
            ],
            'execution_stats': {
                operation: [
                    {**stat, 'timestamp': stat['timestamp'].isoformat()}
                    for stat in stats
                ]
                for operation, stats in self.execution_stats.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logging.info(f"Profile data exported to {filepath}")

# Global profiler instance
performance_profiler = None

def init_profiler(config: Dict[str, Any] = None) -> PerformanceProfiler:
    """Initialize global performance profiler"""
    global performance_profiler
    performance_profiler = PerformanceProfiler(config)
    return performance_profiler

def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    global performance_profiler
    if performance_profiler is None:
        performance_profiler = init_profiler()
    return performance_profiler

# Decorators for easy profiling
def profile_performance(operation_name: str = None):
    """Decorator to automatically profile function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            profiler = get_profiler()
            
            with profiler.profile_block(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def profile_memory(func: Callable) -> Callable:
    """Decorator to specifically profile memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = get_profiler()
        return profiler.profile_function(func, *args, **kwargs)
    
    return wrapper 