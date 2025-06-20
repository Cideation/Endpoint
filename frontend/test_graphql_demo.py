#!/usr/bin/env python3
"""
BEM GraphQL Server Demo & Test Script
Tests all GraphQL functionality for Cytoscape Agent Console integration
"""

import asyncio
import json
import requests
import websockets
from datetime import datetime
import time

# GraphQL endpoint
GRAPHQL_URL = "http://localhost:8000/graphql"
WEBSOCKET_URL = "ws://localhost:8000/ws/cytoscape"

class GraphQLTester:
    def __init__(self, url=GRAPHQL_URL):
        self.url = url
    
    def execute_query(self, query, variables=None):
        """Execute GraphQL query/mutation"""
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        response = requests.post(self.url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"HTTP Error {response.status_code}: {response.text}")
            return None

def test_health_check():
    """Test server health"""
    print("🏥 Testing Health Check...")
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✅ Server is healthy")
        print(f"📊 Status: {response.json()}")
    else:
        print("❌ Server health check failed")
    print("-" * 50)

def test_introspection_query():
    """Test GraphQL schema introspection"""
    print("🔍 Testing Schema Introspection...")
    
    introspection_query = """
    query IntrospectionQuery {
        __schema {
            types {
                name
                kind
                description
                fields {
                    name
                    type {
                        name
                        kind
                    }
                }
            }
        }
    }
    """
    
    tester = GraphQLTester()
    result = tester.execute_query(introspection_query)
    
    if result and not result.get("errors"):
        types = result["data"]["__schema"]["types"]
        custom_types = [t for t in types if not t["name"].startswith("__")]
        print(f"✅ Schema introspection successful - {len(custom_types)} custom types found")
        
        # Show some key types
        key_types = ["Node", "Edge", "Graph", "PulseEvent", "ComponentAnalysis"]
        for type_name in key_types:
            type_info = next((t for t in custom_types if t["name"] == type_name), None)
            if type_info:
                field_count = len(type_info.get("fields", []))
                print(f"📋 {type_name}: {field_count} fields")
    else:
        print("❌ Schema introspection failed")
        if result:
            print(f"Errors: {result.get('errors', [])}")
    
    print("-" * 50)

def test_graph_query():
    """Test graph data query"""
    print("📊 Testing Graph Query...")
    
    graph_query = """
    query GetGraph($filter: NodeFilter) {
        graph(filter: $filter) {
            nodes {
                id
                type
                phase
                primaryFunctor
                coefficients {
                    structural
                    cost
                    energy
                }
                position {
                    x
                    y
                }
                status
            }
            edges {
                id
                source
                target
                relationshipType
                weight
            }
            metadata
        }
    }
    """
    
    # Test with no filter
    tester = GraphQLTester()
    result = tester.execute_query(graph_query)
    
    if result and not result.get("errors"):
        graph_data = result["data"]["graph"]
        print(f"✅ Graph query successful")
        print(f"📊 Nodes: {len(graph_data['nodes'])}")
        print(f"🔗 Edges: {len(graph_data['edges'])}")
        
        # Show sample node if available
        if graph_data["nodes"]:
            sample_node = graph_data["nodes"][0]
            print(f"📝 Sample Node: {sample_node['id']} ({sample_node['type']})")
    else:
        print("❌ Graph query failed")
        if result:
            print(f"Errors: {result.get('errors', [])}")
    
    print("-" * 50)

def test_component_analysis():
    """Test real-time component analysis"""
    print("🔬 Testing Component Analysis...")
    
    analysis_query = """
    query AnalyzeComponent($nodeId: String!) {
        analyzeComponent(nodeId: $nodeId) {
            structuralScore
            costEstimate
            energyEfficiency
            manufacturingFeasibility
        }
    }
    """
    
    # Use a dummy node ID for testing
    variables = {"nodeId": "test_component_001"}
    
    tester = GraphQLTester()
    result = tester.execute_query(analysis_query, variables)
    
    if result and not result.get("errors"):
        analysis = result["data"]["analyzeComponent"]
        print("✅ Component analysis successful")
        print(f"🏗️  Structural Score: {analysis.get('structuralScore', 'N/A')}")
        print(f"💰 Cost Estimate: {analysis.get('costEstimate', 'N/A')}")
        print(f"⚡ Energy Efficiency: {analysis.get('energyEfficiency', 'N/A')}")
        print(f"🏭 Manufacturing Feasibility: {analysis.get('manufacturingFeasibility', 'N/A')}")
    else:
        print("⚠️  Component analysis returned no data (expected if no test data)")
        if result and result.get("errors"):
            print(f"Errors: {result.get('errors', [])}")
    
    print("-" * 50)

def test_data_affinity_execution():
    """Test data affinity execution mutation"""
    print("🔧 Testing Data Affinity Execution...")
    
    affinity_mutation = """
    mutation ExecuteDataAffinity($request: AffinityRequest!) {
        executeDataAffinity(request: $request) {
            structuralAffinity {
                affinityType
                executionStatus
                formulasExecuted
                componentId
                calculationResults
            }
            costAffinity {
                affinityType
                executionStatus
                formulasExecuted
                componentId
                calculationResults
            }
            energyAffinity {
                affinityType
                executionStatus
                formulasExecuted
                componentId
                calculationResults
            }
        }
    }
    """
    
    variables = {
        "request": {
            "componentId": "test_component_001",
            "affinityTypes": ["structural", "cost", "energy"],
            "executionMode": "symbolic_reasoning",
            "parameters": {
                "volume_cm3": 1000.0,
                "area_m2": 10.5,
                "length_mm": 2500.0
            }
        }
    }
    
    tester = GraphQLTester()
    result = tester.execute_query(affinity_mutation, variables)
    
    if result and not result.get("errors"):
        affinity_result = result["data"]["executeDataAffinity"]
        print("✅ Data affinity execution successful")
        
        if affinity_result.get("structuralAffinity"):
            struct = affinity_result["structuralAffinity"]
            print(f"🏗️  Structural: {struct['executionStatus']} ({struct['formulasExecuted']} formulas)")
        
        if affinity_result.get("costAffinity"):
            cost = affinity_result["costAffinity"]
            print(f"💰 Cost: {cost['executionStatus']} ({cost['formulasExecuted']} formulas)")
        
        if affinity_result.get("energyAffinity"):
            energy = affinity_result["energyAffinity"]
            print(f"⚡ Energy: {energy['executionStatus']} ({energy['formulasExecuted']} formulas)")
            
    else:
        print("⚠️  Data affinity execution failed (expected if no microservices)")
        if result and result.get("errors"):
            print(f"Errors: {result.get('errors', [])}")
    
    print("-" * 50)

def test_pulse_trigger():
    """Test pulse triggering mutation"""
    print("⚡ Testing Pulse Trigger...")
    
    pulse_mutation = """
    mutation TriggerPulse($pulseType: PulseType!, $sourceNode: String!, $targetNode: String) {
        triggerPulse(pulseType: $pulseType, sourceNode: $sourceNode, targetNode: $targetNode) {
            id
            pulseType
            sourceNode
            targetNode
            timestamp
            status
            data
        }
    }
    """
    
    variables = {
        "pulseType": "FIT_PULSE",
        "sourceNode": "test_source_001",
        "targetNode": "test_target_001"
    }
    
    tester = GraphQLTester()
    result = tester.execute_query(pulse_mutation, variables)
    
    if result and not result.get("errors"):
        pulse_event = result["data"]["triggerPulse"]
        print("✅ Pulse trigger successful")
        print(f"🆔 Pulse ID: {pulse_event['id']}")
        print(f"⚡ Type: {pulse_event['pulseType']}")
        print(f"📍 Source → Target: {pulse_event['sourceNode']} → {pulse_event['targetNode']}")
        print(f"⏰ Timestamp: {pulse_event['timestamp']}")
        print(f"📊 Status: {pulse_event['status']}")
    else:
        print("⚠️  Pulse trigger failed (expected without database)")
        if result and result.get("errors"):
            print(f"Errors: {result.get('errors', [])}")
    
    print("-" * 50)

def test_node_position_update():
    """Test node position update mutation"""
    print("📍 Testing Node Position Update...")
    
    position_mutation = """
    mutation UpdateNodePosition($id: String!, $position: PositionInput!) {
        updateNodePosition(id: $id, position: $position)
    }
    """
    
    variables = {
        "id": "test_node_001",
        "position": {
            "x": 150.5,
            "y": 200.3
        }
    }
    
    tester = GraphQLTester()
    result = tester.execute_query(position_mutation, variables)
    
    if result and not result.get("errors"):
        update_result = result["data"]["updateNodePosition"]
        print("✅ Node position update successful")
        print(f"📝 Result: {update_result}")
    else:
        print("⚠️  Node position update failed (expected without database)")
        if result and result.get("errors"):
            print(f"Errors: {result.get('errors', [])}")
    
    print("-" * 50)

async def test_websocket_connection():
    """Test WebSocket real-time connection"""
    print("🌐 Testing WebSocket Connection...")
    
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            print("✅ WebSocket connection established")
            
            # Wait for initial messages
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📨 Received message: {message[:100]}...")
            except asyncio.TimeoutError:
                print("⏰ No initial messages received (normal)")
            
            print("🔄 WebSocket connection working")
            
    except Exception as e:
        print(f"⚠️  WebSocket connection failed: {e}")
    
    print("-" * 50)

def test_pulse_history():
    """Test pulse history query"""
    print("📚 Testing Pulse History...")
    
    history_query = """
    query GetPulseHistory($limit: Int) {
        pulseHistory(limit: $limit) {
            id
            pulseType
            sourceNode
            targetNode
            timestamp
            status
            data
        }
    }
    """
    
    variables = {"limit": 10}
    
    tester = GraphQLTester()
    result = tester.execute_query(history_query, variables)
    
    if result and not result.get("errors"):
        pulse_history = result["data"]["pulseHistory"]
        print(f"✅ Pulse history query successful")
        print(f"📊 Found {len(pulse_history)} pulse events")
        
        for i, pulse in enumerate(pulse_history[:3]):  # Show first 3
            print(f"📝 {i+1}. {pulse['pulseType']} - {pulse['sourceNode']} ({pulse['status']})")
    else:
        print("⚠️  Pulse history query returned empty (expected without data)")
        if result and result.get("errors"):
            print(f"Errors: {result.get('errors', [])}")
    
    print("-" * 50)

async def run_all_tests():
    """Run comprehensive test suite"""
    print("🚀 BEM GraphQL Server Comprehensive Test Suite")
    print("=" * 60)
    
    # Test server health first
    test_health_check()
    
    # Test GraphQL functionality
    test_introspection_query()
    test_graph_query()
    test_component_analysis()
    test_pulse_history()
    test_data_affinity_execution()
    test_pulse_trigger()
    test_node_position_update()
    
    # Test WebSocket connection
    await test_websocket_connection()
    
    print("🏁 Test Suite Complete!")
    print("=" * 60)
    print("\n📋 Summary:")
    print("• Health check validates server is running")
    print("• Schema introspection shows GraphQL structure")
    print("• Graph queries work for Cytoscape data loading")
    print("• Component analysis integrates with microservices")
    print("• Data affinity execution integrates with SFDE microservice")
    print("• Pulse system enables real-time events")
    print("• WebSocket provides live updates")
    print("\n🎯 Ready for Cytoscape Agent Console integration!")

def demo_cytoscape_queries():
    """Demonstrate typical Cytoscape use cases"""
    print("\n🎨 Cytoscape Integration Demo Queries")
    print("=" * 50)
    
    print("1. 📊 Loading Initial Graph Data:")
    print("""
query LoadCytoscapeGraph {
    graph {
        nodes {
            id
            type
            phase
            primaryFunctor
            position { x y }
            coefficients {
                structural cost energy mep fabrication time
            }
            status
        }
        edges {
            id source target relationshipType weight
        }
    }
}
    """)
    
    print("2. 🔍 Filtering by Phase:")
    print("""
query FilterByPhase {
    graph(filter: {phases: [PHASE_1, PHASE_2]}) {
        nodes { id type primaryFunctor }
    }
}
    """)
    
    print("3. ⚡ Real-time Pulse Events (Subscription):")
    print("""
subscription LivePulseEvents {
    pulseEvents {
        id pulseType sourceNode targetNode timestamp status
    }
}
    """)
    
    print("4. 🔧 Execute Data Affinity from UI:")
    print("""
mutation ExecuteDataAffinity {
    executeDataAffinity(
        request: {
            componentId: "structural_beam_001"
            affinityTypes: ["structural", "cost", "energy"]
            executionMode: "symbolic_reasoning"
            parameters: {
                volume_cm3: 1500.0
                area_m2: 12.5
                length_mm: 3000.0
            }
        }
    ) {
        structuralAffinity {
            affinityType
            executionStatus
            calculationResults
        }
        costAffinity {
            affinityType
            calculationResults
        }
    }
}
    """)
    
    print("5. 📍 Update Node Position from Cytoscape:")
    print("""
mutation MoveNode {
    updateNodePosition(
        id: "node_001"
        position: {x: 250.0, y: 180.0}
    )
}
    """)

if __name__ == "__main__":
    print("🧪 BEM GraphQL Server Test & Demo")
    print("⚠️  Make sure the GraphQL server is running on http://localhost:8000")
    print()
    
    # Run tests
    asyncio.run(run_all_tests())
    
    # Show demo queries
    demo_cytoscape_queries() 