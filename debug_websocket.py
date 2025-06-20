#!/usr/bin/env python3
"""
WebSocket Debug Script to identify 1011 internal error
"""

import asyncio
import websockets
import json

async def test_ecm_websocket():
    print("🔍 Testing ECM WebSocket...")
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("✅ Connected successfully")
            
            # Test simple message
            test_message = {"type": "test", "payload": {"action": "ping"}}
            print(f"📤 Sending: {test_message}")
            
            await websocket.send(json.dumps(test_message))
            print("✅ Message sent")
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f"📥 Received: {response}")
            
    except Exception as e:
        print(f"❌ ECM WebSocket error: {e}")

async def test_pulse_websocket():
    print("\n🔍 Testing Pulse WebSocket...")
    try:
        async with websockets.connect("ws://localhost:8766") as websocket:
            print("✅ Connected successfully")
            
            # Test simple pulse
            pulse_message = {"type": "bid_pulse", "source": "test_node", "urgency": 3}
            print(f"📤 Sending: {pulse_message}")
            
            await websocket.send(json.dumps(pulse_message))
            print("✅ Message sent")
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f"📥 Received: {response}")
            
    except Exception as e:
        print(f"❌ Pulse WebSocket error: {e}")

async def main():
    print("🚀 WebSocket Debug Session")
    print("=" * 40)
    
    await test_ecm_websocket()
    await test_pulse_websocket()
    
    print("\n✅ Debug session complete")

if __name__ == "__main__":
    asyncio.run(main()) 