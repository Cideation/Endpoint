#!/usr/bin/env python3
"""
Engineering Traceability Engine v1.0
🔍 Complete audit trail system for BEM design decisions and node transformations
Makes every design decision fully explainable and backtrackable from agent intent to final output
"""

import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OriginTag:
    """Origin tag for tracking component score sources"""
    source_functor: str
    input_nodes: List[str]
    agent: str
    design_param: str
    timestamp: str = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class TraceLog:
    """Complete trace log for node transformations"""
    timestamp: str
    node_id: str
    functor: str
    input_snapshot: Dict[str, Any]
    output_snapshot: Dict[str, Any]
    dependencies: List[str]
    agent_triggered: str
    global_params_used: List[str]
    decision_path: List[str]
    trace_id: str = None
    chain_id: str = None
    execution_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.trace_id is None:
            self.trace_id = f"trace_{uuid.uuid4().hex[:8]}"
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class ComponentScore:
    """Component score with full origin tracking"""
    value: float
    origin_tag: OriginTag
    validation_status: str = "pending"
    quality_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = {}

class TraceabilityEngine:
    """Main traceability engine for BEM system"""
    
    def __init__(self, log_directory: str = "MICROSERVICE_ENGINES/traceability_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.active_traces: Dict[str, TraceLog] = {}
        self.graph_chains: Dict[str, List[str]] = {}
        self.component_scores: Dict[str, ComponentScore] = {}
        
        self.execution_stats = {
            "total_traces": 0,
            "average_execution_time": 0.0,
            "total_decisions": 0,
            "agent_activity": {}
        }
        
        logger.info("🔍 Engineering Traceability Engine v1.0 initialized")
    
    def start_trace(self, node_id: str, functor: str, agent_triggered: str, 
                   input_data: Dict[str, Any], dependencies: List[str] = None) -> str:
        """Start a new trace for node transformation"""
        
        trace_id = f"trace_{uuid.uuid4().hex[:8]}"
        chain_id = f"CHAIN_{hash(str(dependencies or []))}"
        
        trace_log = TraceLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            node_id=node_id,
            functor=functor,
            input_snapshot=self._create_snapshot(input_data),
            output_snapshot={},
            dependencies=dependencies or [],
            agent_triggered=agent_triggered,
            global_params_used=[],
            decision_path=[],
            trace_id=trace_id,
            chain_id=chain_id
        )
        
        self.active_traces[trace_id] = trace_log
        
        if chain_id not in self.graph_chains:
            self.graph_chains[chain_id] = []
        self.graph_chains[chain_id].append(node_id)
        
        logger.info(f"🔍 Started trace {trace_id} for {node_id}:{functor}")
        return trace_id
    
    def log_decision(self, trace_id: str, decision: str, condition: str = "", 
                    reasoning: str = "") -> None:
        """Log a decision point in the trace"""
        
        if trace_id not in self.active_traces:
            logger.warning(f"⚠️ Trace {trace_id} not found for decision logging")
            return
        
        decision_point = f"{decision}:{condition}" if condition else decision
        self.active_traces[trace_id].decision_path.append(decision_point)
        
        self.execution_stats["total_decisions"] += 1
        logger.debug(f"📝 Decision logged: {decision_point}")
    
    def log_global_param_usage(self, trace_id: str, param_name: str) -> None:
        """Log usage of global design parameter"""
        
        if trace_id not in self.active_traces:
            logger.warning(f"⚠️ Trace {trace_id} not found for param logging")
            return
        
        if param_name not in self.active_traces[trace_id].global_params_used:
            self.active_traces[trace_id].global_params_used.append(param_name)
        
        logger.debug(f"🎛️ Global param used: {param_name}")
    
    def end_trace(self, trace_id: str, output_data: Dict[str, Any], 
                 execution_time_ms: float = 0.0) -> TraceLog:
        """End a trace and finalize the log"""
        
        if trace_id not in self.active_traces:
            logger.warning(f"⚠️ Trace {trace_id} not found for ending")
            return None
        
        trace_log = self.active_traces[trace_id]
        trace_log.output_snapshot = self._create_snapshot(output_data)
        trace_log.execution_time_ms = execution_time_ms
        
        self._save_trace_log(trace_log)
        
        self.execution_stats["total_traces"] += 1
        self._update_execution_stats(trace_log)
        
        del self.active_traces[trace_id]
        
        logger.info(f"✅ Completed trace {trace_id} in {execution_time_ms:.2f}ms")
        return trace_log
    
    def create_component_score(self, value: float, source_functor: str, 
                             input_nodes: List[str], agent: str, 
                             design_param: str, component_id: str = None) -> ComponentScore:
        """Create a component score with full origin tracking"""
        
        origin_tag = OriginTag(
            source_functor=source_functor,
            input_nodes=input_nodes,
            agent=agent,
            design_param=design_param
        )
        
        component_score = ComponentScore(
            value=value,
            origin_tag=origin_tag,
            validation_status="validated" if 0.0 <= value <= 1.0 else "invalid"
        )
        
        score_id = component_id or f"score_{uuid.uuid4().hex[:8]}"
        self.component_scores[score_id] = component_score
        
        logger.info(f"📊 Component score created: {value:.3f} from {source_functor}")
        return component_score
    
    def get_trace_path(self, chain_id: str) -> List[str]:
        """Get the complete trace path for a graph chain"""
        
        if chain_id not in self.graph_chains:
            logger.warning(f"⚠️ Chain {chain_id} not found")
            return []
        
        return self.graph_chains[chain_id].copy()
    
    def analyze_decision_lineage(self, node_id: str, depth: int = 5) -> Dict[str, Any]:
        """Analyze the decision lineage for a specific node"""
        
        lineage = {
            "target_node": node_id,
            "depth_analyzed": depth,
            "decision_chain": [],
            "agents_involved": set(),
            "parameters_used": set(),
            "total_decisions": 0
        }
        
        trace_files = list(self.log_directory.glob("trace_*.json"))
        
        for trace_file in trace_files[-depth:]:
            try:
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                
                if trace_data.get("node_id") == node_id:
                    lineage["decision_chain"].append({
                        "trace_id": trace_data.get("trace_id"),
                        "functor": trace_data.get("functor"),
                        "decisions": trace_data.get("decision_path", []),
                        "timestamp": trace_data.get("timestamp")
                    })
                    
                    lineage["agents_involved"].add(trace_data.get("agent_triggered"))
                    lineage["parameters_used"].update(trace_data.get("global_params_used", []))
                    lineage["total_decisions"] += len(trace_data.get("decision_path", []))
            
            except Exception as e:
                logger.warning(f"⚠️ Error reading trace file {trace_file}: {e}")
        
        lineage["agents_involved"] = list(lineage["agents_involved"])
        lineage["parameters_used"] = list(lineage["parameters_used"])
        
        return lineage
    
    def generate_audit_report(self, chain_id: str = None, 
                            agent_filter: str = None) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        
        report = {
            "report_id": f"audit_{uuid.uuid4().hex[:8]}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "chain_id": chain_id,
            "agent_filter": agent_filter,
            "summary": {
                "total_traces": self.execution_stats["total_traces"],
                "total_decisions": self.execution_stats["total_decisions"],
                "active_chains": len(self.graph_chains),
                "component_scores": len(self.component_scores)
            },
            "trace_analysis": [],
            "decision_patterns": {},
            "agent_activity": self.execution_stats.get("agent_activity", {}),
            "parameter_usage": {}
        }
        
        trace_files = list(self.log_directory.glob("trace_*.json"))
        
        for trace_file in trace_files:
            try:
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                
                if chain_id and trace_data.get("chain_id") != chain_id:
                    continue
                if agent_filter and trace_data.get("agent_triggered") != agent_filter:
                    continue
                
                report["trace_analysis"].append({
                    "trace_id": trace_data.get("trace_id"),
                    "node_id": trace_data.get("node_id"),
                    "functor": trace_data.get("functor"),
                    "execution_time": trace_data.get("execution_time_ms", 0),
                    "decision_count": len(trace_data.get("decision_path", [])),
                    "param_count": len(trace_data.get("global_params_used", []))
                })
                
                for decision in trace_data.get("decision_path", []):
                    if decision not in report["decision_patterns"]:
                        report["decision_patterns"][decision] = 0
                    report["decision_patterns"][decision] += 1
                
                for param in trace_data.get("global_params_used", []):
                    if param not in report["parameter_usage"]:
                        report["parameter_usage"][param] = 0
                    report["parameter_usage"][param] += 1
            
            except Exception as e:
                logger.warning(f"⚠️ Error processing trace file {trace_file}: {e}")
        
        report_file = self.log_directory / f"audit_report_{report['report_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Audit report generated: {report['report_id']}")
        return report
    
    def _create_snapshot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe snapshot of data for logging"""
        
        try:
            snapshot = {}
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    snapshot[key] = value
                else:
                    snapshot[key] = str(value)
            
            return snapshot
        
        except Exception as e:
            logger.warning(f"⚠️ Error creating snapshot: {e}")
            return {"error": "snapshot_failed", "original_keys": list(data.keys())}
    
    def _save_trace_log(self, trace_log: TraceLog) -> None:
        """Save trace log to persistent storage"""
        
        try:
            log_file = self.log_directory / f"trace_{trace_log.trace_id}.json"
            with open(log_file, 'w') as f:
                json.dump(asdict(trace_log), f, indent=2, default=str)
            
            logger.debug(f"💾 Trace log saved: {log_file}")
        
        except Exception as e:
            logger.error(f"❌ Failed to save trace log: {e}")
    
    def _update_execution_stats(self, trace_log: TraceLog) -> None:
        """Update execution statistics"""
        
        total_time = (self.execution_stats["average_execution_time"] * 
                     (self.execution_stats["total_traces"] - 1) + 
                     trace_log.execution_time_ms)
        self.execution_stats["average_execution_time"] = total_time / self.execution_stats["total_traces"]
        
        agent = trace_log.agent_triggered
        if agent not in self.execution_stats["agent_activity"]:
            self.execution_stats["agent_activity"][agent] = {
                "traces": 0,
                "total_decisions": 0,
                "avg_execution_time": 0.0
            }
        
        agent_stats = self.execution_stats["agent_activity"][agent]
        agent_stats["traces"] += 1
        agent_stats["total_decisions"] += len(trace_log.decision_path)
        
        total_agent_time = (agent_stats["avg_execution_time"] * (agent_stats["traces"] - 1) + 
                           trace_log.execution_time_ms)
        agent_stats["avg_execution_time"] = total_agent_time / agent_stats["traces"]

def create_traceability_engine():
    """Create and return a traceability engine instance"""
    return TraceabilityEngine()

if __name__ == "__main__":
    print("🔍 Engineering Traceability Engine v1.0 - Demo")
    
    engine = create_traceability_engine()
    
    trace_id = engine.start_trace(
        node_id="V01",
        functor="evaluate_manufacturing",
        agent_triggered="Agent1",
        input_data={"component_type": "beam", "material": "steel"},
        dependencies=["V02", "V05"]
    )
    
    engine.log_decision(trace_id, "spec_check_passed")
    engine.log_decision(trace_id, "material_ok")
    engine.log_decision(trace_id, "geometry_valid")
    
    engine.log_global_param_usage(trace_id, "form_strategy")
    engine.log_global_param_usage(trace_id, "precision_tolerance")
    
    score = engine.create_component_score(
        value=0.86,
        source_functor="evaluate_component_vector",
        input_nodes=["V01", "V02"],
        agent="Agent1",
        design_param="evolutionary_potential"
    )
    
    engine.end_trace(trace_id, {"manufacturing_score": 0.86, "status": "approved"}, 45.2)
    
    report = engine.generate_audit_report()
    
    print(f"✅ Demo completed - Generated audit report: {report['report_id']}")
    print(f"📊 Stats: {engine.execution_stats}")
