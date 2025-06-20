#!/usr/bin/env python3
"""
Behavior-Driven Agent Console (AC) Logic
Implements AA (Automated Admin) behavioral classification
No logins, no forms, no tutorials - pure behavior-driven access
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
import logging
from typing import Dict, List, Optional
import numpy as np

app = FastAPI(
    title="BEM Behavior-Driven AC System",
    description="AA-powered dynamic UI spawning based on agent behavior",
    version="1.0.0"
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Behavior pattern storage
behavior_sessions = {}

class UserAction(BaseModel):
    """User action data model"""
    session_id: str
    action_type: str  # e.g., "view_graph", "trigger_pulse", "inspect_compliance"
    target: str       # e.g., "cytoscape", "unreal", "node_123"
    context: Dict     # Additional action context
    timestamp: float

class AgentClassification(BaseModel):
    """Agent role classification result"""
    agent_role: str
    confidence: float
    behavior_pattern: Dict
    ac_panels: List[str]

def classify_behavior(actions: List[Dict]) -> AgentClassification:
    """
    AA (Automated Admin) behavioral classification algorithm
    Infers agent role from behavior patterns without any user input
    """
    
    if not actions:
        return AgentClassification(
            agent_role="observer",
            confidence=0.5,
            behavior_pattern={},
            ac_panels=["basic_view"]
        )
    
    # Check for minimal/observer behavior first
    if len(actions) == 1 and actions[0].get("action_type") in ["initial_view", "view_interface"]:
        return AgentClassification(
            agent_role="observer",
            confidence=0.8,
            behavior_pattern={"observer": 1.0},
            ac_panels=["basic_view"]
        )
    
    # Behavior scoring weights
    investor_signals = 0.0
    contributor_signals = 0.0
    validator_signals = 0.0
    analyst_signals = 0.0
    
    for action in actions:
        action_type = action.get("action_type", "")
        target = action.get("target", "")
        context = action.get("context", {})
        
        # Investor behavior patterns (enhanced detection)
        if "investment" in action_type or "irr" in target.lower():
            investor_signals += 3.0  # Strong investor signal
        if "view_investment" in action_type or "analyze_roi" in action_type:
            investor_signals += 3.0  # Direct investor actions
        if "finance" in str(context) or "revenue" in str(context):
            investor_signals += 2.0
        if "roi" in str(context).lower() or "cost" in str(context).lower():
            investor_signals += 2.0
        if "amount" in str(context) or "period" in str(context):
            investor_signals += 1.5  # Financial context indicators
            
        # Contributor behavior patterns  
        if "create" in action_type or "modify" in action_type:
            contributor_signals += 2.0
        if "node" in target and "edit" in action_type:
            contributor_signals += 1.5
        if "upload" in action_type or "submit" in action_type:
            contributor_signals += 1.0
            
        # Validator behavior patterns
        if "compliance" in action_type or "validate" in action_type:
            validator_signals += 2.0
        if "audit" in target or "check" in action_type:
            validator_signals += 1.5
        if "approve" in action_type or "reject" in action_type:
            validator_signals += 1.0
            
        # Analyst behavior patterns (reduced to avoid false positives)
        if "analyze" in action_type and "roi" not in action_type:  # Exclude investor analysis
            analyst_signals += 1.5  # Reduced from 2.0
        if "report" in action_type:
            analyst_signals += 2.0
        if "metrics" in target and "finance" not in str(context):  # Exclude financial metrics
            analyst_signals += 1.0  # Reduced from 1.5
        if "dashboard" in target and "investment" not in str(context):
            analyst_signals += 1.0  # Reduced from 1.5
        if "export" in action_type or ("data" in target and "upload" not in action_type):
            analyst_signals += 1.0
    
    # Determine primary role based on highest signal
    signals = {
        "investor": investor_signals,
        "contributor": contributor_signals, 
        "validator": validator_signals,
        "analyst": analyst_signals
    }
    
    primary_role = max(signals, key=signals.get)
    max_signal = signals[primary_role]
    
    # If all signals are very low, default to observer
    if max_signal < 1.0:
        return AgentClassification(
            agent_role="observer",
            confidence=0.7,
            behavior_pattern=signals,
            ac_panels=["basic_view"]
        )
    
    # Calculate confidence (normalized)
    total_signals = sum(signals.values())
    confidence = max_signal / total_signals if total_signals > 0 else 0.5
    
    # Define AC panels for each role
    ac_panels_map = {
        "investor": ["investment_ac", "irr_calculator", "roi_dashboard"],
        "contributor": ["contributor_ac", "node_editor", "upload_tools"],
        "validator": ["validator_ac", "compliance_checker", "audit_tools"],
        "analyst": ["analyst_ac", "metrics_dashboard", "report_generator"],
        "observer": ["basic_view"]
    }
    
    return AgentClassification(
        agent_role=primary_role,
        confidence=min(confidence, 1.0),
        behavior_pattern=signals,
        ac_panels=ac_panels_map.get(primary_role, ["basic_view"])
    )

@app.post("/log_action")
async def log_user_action(action: UserAction):
    """
    Log user action and update behavioral classification
    Core endpoint for AA behavioral inference
    """
    session_id = action.session_id
    
    # Initialize session if new
    if session_id not in behavior_sessions:
        behavior_sessions[session_id] = {
            "actions": [],
            "current_role": "observer",
            "confidence": 0.0,
            "first_seen": time.time(),
            "last_seen": time.time()
        }
    
    # Add action to session
    session = behavior_sessions[session_id]
    session["actions"].append(action.dict())
    session["last_seen"] = time.time()
    
    # Re-classify based on updated behavior
    classification = classify_behavior(session["actions"])
    
    # Update session with new classification
    session["current_role"] = classification.agent_role
    session["confidence"] = classification.confidence
    
    logger.info(f"Session {session_id}: {classification.agent_role} (confidence: {classification.confidence:.2f})")
    
    return {
        "agent_role": classification.agent_role,
        "confidence": classification.confidence,
        "ac_panels": classification.ac_panels,
        "behavior_pattern": classification.behavior_pattern,
        "session_summary": {
            "total_actions": len(session["actions"]),
            "session_duration": session["last_seen"] - session["first_seen"]
        }
    }

@app.get("/get_role/{session_id}")
async def get_current_role(session_id: str):
    """Get current role classification for a session"""
    if session_id not in behavior_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = behavior_sessions[session_id]
    classification = classify_behavior(session["actions"])
    
    return {
        "agent_role": classification.agent_role,
        "confidence": classification.confidence,
        "ac_panels": classification.ac_panels
    }

@app.get("/behavior_analytics")
async def get_behavior_analytics():
    """
    AA analytics dashboard - system-wide behavioral insights
    """
    if not behavior_sessions:
        return {"message": "No active sessions"}
    
    # Aggregate role distribution
    role_counts = {}
    confidence_scores = []
    
    for session in behavior_sessions.values():
        role = session.get("current_role", "observer")
        role_counts[role] = role_counts.get(role, 0) + 1
        confidence_scores.append(session.get("confidence", 0))
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    return {
        "total_sessions": len(behavior_sessions),
        "role_distribution": role_counts,
        "average_confidence": avg_confidence,
        "active_sessions": sum(1 for s in behavior_sessions.values() 
                              if time.time() - s["last_seen"] < 300),  # 5 min threshold
        "timestamp": time.time()
    }

@app.post("/simulate_behavior")
async def simulate_behavior_patterns():
    """
    Development endpoint to simulate different behavior patterns
    for testing AC panel spawning
    """
    
    # Simulate investor behavior
    investor_actions = [
        {"action_type": "view_investment", "target": "irr_calculator", "context": {"amount": 50000}},
        {"action_type": "analyze_roi", "target": "finance_dashboard", "context": {"period": "quarterly"}},
        {"action_type": "check_revenue", "target": "metrics", "context": {"type": "investment_return"}}
    ]
    
    # Simulate contributor behavior
    contributor_actions = [
        {"action_type": "create_node", "target": "node_123", "context": {"type": "housing_unit"}},
        {"action_type": "modify_graph", "target": "cytoscape", "context": {"operation": "add_edge"}},
        {"action_type": "upload_data", "target": "file_system", "context": {"file_type": "cad"}}
    ]
    
    # Simulate validator behavior
    validator_actions = [
        {"action_type": "validate_compliance", "target": "compliance_checker", "context": {"standard": "building_code"}},
        {"action_type": "audit_changes", "target": "audit_log", "context": {"timeframe": "last_24h"}},
        {"action_type": "approve_submission", "target": "workflow", "context": {"approval_type": "design"}}
    ]
    
    # Simulate analyst behavior
    analyst_actions = [
        {"action_type": "generate_report", "target": "analytics", "context": {"type": "performance"}},
        {"action_type": "view_metrics", "target": "dashboard", "context": {"metric": "efficiency"}},
        {"action_type": "export_data", "target": "database", "context": {"format": "csv"}}
    ]
    
    results = {}
    for role, actions in [
        ("investor", investor_actions),
        ("contributor", contributor_actions), 
        ("validator", validator_actions),
        ("analyst", analyst_actions)
    ]:
        classification = classify_behavior(actions)
        results[f"simulated_{role}"] = {
            "detected_role": classification.agent_role,
            "confidence": classification.confidence,
            "ac_panels": classification.ac_panels
        }
    
    return results

if __name__ == "__main__":
    import uvicorn
    logger.info("🧠 Starting Behavior-Driven AC System on port 8003")
    logger.info("🎯 AA behavioral classification active")
    logger.info("🚫 No logins, no forms, no tutorials - pure behavior-driven access")
    
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info") 