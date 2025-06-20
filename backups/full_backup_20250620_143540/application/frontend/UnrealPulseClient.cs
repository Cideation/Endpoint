/*
 * Unreal Pulse Client - C# Script Template for Unreal Engine
 * 
 * This script connects to the Python Unreal Pulse Handler and implements
 * 3D pulse visualizations based on the 7 semantic pulse types.
 * 
 * Usage:
 * 1. Add this script to your Unreal Engine project
 * 2. Create pulse visualization prefabs for each pulse type
 * 3. Configure the WebSocket connection to ws://localhost:8767
 * 4. Handle incoming pulse visualizations and spatial events
 */

using System;
using System.Collections.Generic;
using UnrealEngine.Framework;
using Newtonsoft.Json;
using WebSocketSharp;

public class UnrealPulseClient : Actor
{
    // WebSocket connection to Pulse Handler
    private WebSocket webSocket;
    private string pulseHandlerUrl = "ws://localhost:8767";
    
    // Pulse visualization prefabs
    public Dictionary<string, GameObject> PulsePrefabs;
    public Dictionary<string, Color> PulseColors;
    
    // Active pulse visualizations
    private List<PulseVisualization> activePulses = new List<PulseVisualization>();
    
    // Pulse data structure
    [Serializable]
    public class PulseVisualization
    {
        public string pulse_type;
        public string color;
        public string direction;
        public Vector3 position;
        public float intensity;
        public float duration;
        public string[] visual_effects;
        public GameObject visualObject;
        public float startTime;
    }
    
    public override void BeginPlay()
    {
        base.BeginPlay();
        
        // Initialize pulse colors
        InitializePulseColors();
        
        // Connect to Pulse Handler
        ConnectToPulseHandler();
        
        Log.Info("🎮 Unreal Pulse Client initialized");
    }
    
    private void InitializePulseColors()
    {
        PulseColors = new Dictionary<string, Color>
        {
            {"bid_pulse", ColorFromHex("#FFC107")},        // Amber
            {"occupancy_pulse", ColorFromHex("#2196F3")},  // Sky Blue
            {"compliancy_pulse", ColorFromHex("#1E3A8A")}, // Indigo
            {"fit_pulse", ColorFromHex("#4CAF50")},        // Green
            {"investment_pulse", ColorFromHex("#FF9800")}, // Deep Orange
            {"decay_pulse", ColorFromHex("#9E9E9E")},      // Gray
            {"reject_pulse", ColorFromHex("#F44336")}      // Red
        };
    }
    
    private void ConnectToPulseHandler()
    {
        try
        {
            webSocket = new WebSocket(pulseHandlerUrl);
            
            webSocket.OnOpen += (sender, e) =>
            {
                Log.Info("✅ Connected to Pulse Handler");
            };
            
            webSocket.OnMessage += (sender, e) =>
            {
                HandlePulseMessage(e.Data);
            };
            
            webSocket.OnClose += (sender, e) =>
            {
                Log.Warning("❌ Disconnected from Pulse Handler");
                // Attempt reconnection after 3 seconds
                Timer.SetTimer(3.0f, false, () => ConnectToPulseHandler());
            };
            
            webSocket.OnError += (sender, e) =>
            {
                Log.Error($"Pulse Handler connection error: {e.Message}");
            };
            
            webSocket.Connect();
        }
        catch (Exception ex)
        {
            Log.Error($"Failed to connect to Pulse Handler: {ex.Message}");
        }
    }
    
    private void HandlePulseMessage(string jsonMessage)
    {
        try
        {
            var message = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonMessage);
            string messageType = message["type"].ToString();
            
            switch (messageType)
            {
                case "unreal_handshake":
                    HandleHandshake(message);
                    break;
                    
                case "pulse_visualization":
                    HandlePulseVisualization(message);
                    break;
                    
                default:
                    Log.Info($"Received message: {messageType}");
                    break;
            }
        }
        catch (Exception ex)
        {
            Log.Error($"Error handling pulse message: {ex.Message}");
        }
    }
    
    private void HandleHandshake(Dictionary<string, object> message)
    {
        Log.Info("🤝 Pulse Handler handshake received");
        
        // Send test spatial event
        SendSpatialEvent("system_ready", GetActorLocation());
    }
    
    private void HandlePulseVisualization(Dictionary<string, object> message)
    {
        try
        {
            var visualizationData = message["visualization"] as Dictionary<string, object>;
            
            var pulse = new PulseVisualization
            {
                pulse_type = visualizationData["pulse_type"].ToString(),
                color = visualizationData["color"].ToString(),
                direction = visualizationData["direction"].ToString(),
                position = JsonConvert.DeserializeObject<Vector3>(visualizationData["position"].ToString()),
                intensity = Convert.ToSingle(visualizationData["intensity"]),
                duration = Convert.ToSingle(visualizationData["duration"]),
                visual_effects = JsonConvert.DeserializeObject<string[]>(visualizationData["visual_effects"].ToString()),
                startTime = Time.GetGameTimeInSeconds()
            };
            
            // Create 3D pulse visualization
            CreatePulseVisualization(pulse);
            
            Log.Info($"🎨 Creating pulse visualization: {pulse.pulse_type}");
        }
        catch (Exception ex)
        {
            Log.Error($"Error creating pulse visualization: {ex.Message}");
        }
    }
    
    private void CreatePulseVisualization(PulseVisualization pulse)
    {
        // Create pulse visualization object
        GameObject pulseObject = new GameObject($"Pulse_{pulse.pulse_type}");
        pulseObject.SetActorLocation(pulse.position);
        
        // Add particle system based on pulse type
        var particleSystem = pulseObject.AddComponent<ParticleSystemComponent>();
        ConfigurePulseParticles(particleSystem, pulse);
        
        // Add directional flow effects
        ApplyDirectionalEffects(pulseObject, pulse);
        
        // Store for cleanup
        pulse.visualObject = pulseObject;
        activePulses.Add(pulse);
    }
    
    private void ConfigurePulseParticles(ParticleSystemComponent particles, PulseVisualization pulse)
    {
        // Set particle color
        Color pulseColor = PulseColors[pulse.pulse_type];
        particles.SetColorParameter("PulseColor", pulseColor);
        
        // Set intensity
        particles.SetFloatParameter("Intensity", pulse.intensity);
        
        // Configure based on pulse type
        switch (pulse.pulse_type)
        {
            case "bid_pulse":
                particles.SetTemplate("PulsingAmberParticles");
                break;
            case "occupancy_pulse":
                particles.SetTemplate("FlowingBlueParticles");
                break;
            case "compliancy_pulse":
                particles.SetTemplate("SteadyIndigoParticles");
                break;
            case "fit_pulse":
                particles.SetTemplate("ConfirmingGreenParticles");
                break;
            case "investment_pulse":
                particles.SetTemplate("GoldenSparkleParticles");
                break;
            case "decay_pulse":
                particles.SetTemplate("FadingGrayParticles");
                break;
            case "reject_pulse":
                particles.SetTemplate("RejectionFlashParticles");
                break;
        }
    }
    
    private void ApplyDirectionalEffects(GameObject pulseObject, PulseVisualization pulse)
    {
        switch (pulse.direction)
        {
            case "downward":
                ApplyDownwardFlow(pulseObject, pulse);
                break;
            case "upward":
                ApplyUpwardFlow(pulseObject, pulse);
                break;
            case "cross-subtree":
                ApplyHorizontalSpread(pulseObject, pulse);
                break;
            case "lateral":
                ApplyLateralWave(pulseObject, pulse);
                break;
            case "broadcast":
                ApplyRadialExpansion(pulseObject, pulse);
                break;
            case "reflexive":
                ApplyInstantRejection(pulseObject, pulse);
                break;
        }
    }
    
    private void ApplyDownwardFlow(GameObject pulseObject, PulseVisualization pulse)
    {
        // Animate particles flowing downward
        var animator = pulseObject.AddComponent<PulseAnimator>();
        animator.AnimateDownward(pulse.duration, pulse.intensity);
    }
    
    private void ApplyUpwardFlow(GameObject pulseObject, PulseVisualization pulse)
    {
        // Animate particles flowing upward
        var animator = pulseObject.AddComponent<PulseAnimator>();
        animator.AnimateUpward(pulse.duration, pulse.intensity);
    }
    
    private void ApplyHorizontalSpread(GameObject pulseObject, PulseVisualization pulse)
    {
        // Animate horizontal spreading effect
        var animator = pulseObject.AddComponent<PulseAnimator>();
        animator.AnimateHorizontalSpread(pulse.duration, pulse.intensity);
    }
    
    private void ApplyLateralWave(GameObject pulseObject, PulseVisualization pulse)
    {
        // Animate lateral wave motion
        var animator = pulseObject.AddComponent<PulseAnimator>();
        animator.AnimateLateralWave(pulse.duration, pulse.intensity);
    }
    
    private void ApplyRadialExpansion(GameObject pulseObject, PulseVisualization pulse)
    {
        // Animate radial expansion from center
        var animator = pulseObject.AddComponent<PulseAnimator>();
        animator.AnimateRadialExpansion(pulse.duration, pulse.intensity);
    }
    
    private void ApplyInstantRejection(GameObject pulseObject, PulseVisualization pulse)
    {
        // Animate instant rejection flash
        var animator = pulseObject.AddComponent<PulseAnimator>();
        animator.AnimateInstantRejection(pulse.duration, pulse.intensity);
    }
    
    public override void Tick(float deltaTime)
    {
        base.Tick(deltaTime);
        
        // Update active pulse visualizations
        UpdateActivePulses();
        
        // Clean up expired pulses
        CleanupExpiredPulses();
    }
    
    private void UpdateActivePulses()
    {
        float currentTime = Time.GetGameTimeInSeconds();
        
        foreach (var pulse in activePulses)
        {
            float elapsed = currentTime - pulse.startTime;
            float progress = elapsed / pulse.duration;
            
            if (progress < 1.0f && pulse.visualObject != null)
            {
                // Update pulse animation based on progress
                UpdatePulseAnimation(pulse, progress);
            }
        }
    }
    
    private void CleanupExpiredPulses()
    {
        float currentTime = Time.GetGameTimeInSeconds();
        
        for (int i = activePulses.Count - 1; i >= 0; i--)
        {
            var pulse = activePulses[i];
            float elapsed = currentTime - pulse.startTime;
            
            if (elapsed >= pulse.duration)
            {
                // Remove expired pulse
                if (pulse.visualObject != null)
                {
                    pulse.visualObject.Destroy();
                }
                activePulses.RemoveAt(i);
            }
        }
    }
    
    private void UpdatePulseAnimation(PulseVisualization pulse, float progress)
    {
        // Update animation based on progress (0.0 to 1.0)
        if (pulse.visualObject != null)
        {
            var animator = pulse.visualObject.GetComponent<PulseAnimator>();
            if (animator != null)
            {
                animator.UpdateAnimation(progress);
            }
        }
    }
    
    public void SendSpatialEvent(string eventType, Vector3 position)
    {
        if (webSocket != null && webSocket.ReadyState == WebSocketState.Open)
        {
            var spatialEvent = new
            {
                type = "spatial_event",
                event_type = eventType,
                position = position,
                timestamp = DateTime.UtcNow.ToString("O")
            };
            
            string json = JsonConvert.SerializeObject(spatialEvent);
            webSocket.Send(json);
            
            Log.Info($"📡 Sent spatial event: {eventType}");
        }
    }
    
    public void RequestPulse(string pulseType, Vector3 position)
    {
        if (webSocket != null && webSocket.ReadyState == WebSocketState.Open)
        {
            var pulseRequest = new
            {
                type = "pulse_request",
                pulse_type = pulseType,
                position = position,
                timestamp = DateTime.UtcNow.ToString("O")
            };
            
            string json = JsonConvert.SerializeObject(pulseRequest);
            webSocket.Send(json);
            
            Log.Info($"🎨 Requested pulse: {pulseType}");
        }
    }
    
    private Color ColorFromHex(string hex)
    {
        // Convert hex color to Unreal Color
        hex = hex.Replace("#", "");
        
        byte r = Convert.ToByte(hex.Substring(0, 2), 16);
        byte g = Convert.ToByte(hex.Substring(2, 2), 16);
        byte b = Convert.ToByte(hex.Substring(4, 2), 16);
        
        return new Color(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
    }
    
    public override void EndPlay(EndPlayReason reason)
    {
        // Clean up WebSocket connection
        if (webSocket != null)
        {
            webSocket.Close();
            webSocket = null;
        }
        
        // Clean up all active pulses
        foreach (var pulse in activePulses)
        {
            if (pulse.visualObject != null)
            {
                pulse.visualObject.Destroy();
            }
        }
        activePulses.Clear();
        
        base.EndPlay(reason);
    }
} 