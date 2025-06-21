/*
 * BP_PulseAnimator.cpp
 * 
 * Pure Visual Animation Component for Unreal Engine
 * NO PULSE LOGIC - Only renders visual effects based on backend commands
 * 
 * Architecture:
 * Backend (ECM + Node Engine) = Pulse Logic & State (Authoritative)
 * Unreal Engine = Visual Rendering Only (Display Layer)
 */

#include "BP_PulseAnimator.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Particles/ParticleSystemComponent.h"
#include "Engine/Engine.h"

UBP_PulseAnimator::UBP_PulseAnimator()
{
    PrimaryComponentTick.bCanEverTick = true;
    
    // Initialize visual-only properties
    CurrentProgress = 0.0f;
    AnimationDuration = 2.0f;
    PulseIntensity = 1.0f;
    IsAnimating = false;
    
    // Visual effect components (no logic)
    ParticleComponent = nullptr;
    MaterialInstance = nullptr;
}

void UBP_PulseAnimator::BeginPlay()
{
    Super::BeginPlay();
    
    // Initialize visual components only
    InitializeVisualComponents();
}

void UBP_PulseAnimator::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    
    // Only update visual animation - NO PULSE LOGIC
    if (IsAnimating)
    {
        UpdateVisualAnimation(DeltaTime);
    }
}

// ====== VISUAL-ONLY METHODS (Called by Backend Commands) ======

void UBP_PulseAnimator::StartPulseAnimation(const FString& PulseType, float Intensity, float Duration)
{
    // Receive command from backend - just render the visual
    PulseIntensity = Intensity;
    AnimationDuration = Duration;
    CurrentProgress = 0.0f;
    IsAnimating = true;
    
    // Apply visual effects based on backend instruction
    ApplyPulseVisualEffects(PulseType);
    
    UE_LOG(LogTemp, Log, TEXT("🎨 VISUAL: Starting pulse animation: %s"), *PulseType);
}

void UBP_PulseAnimator::StopPulseAnimation()
{
    // Backend commanded stop - just stop visual
    IsAnimating = false;
    CurrentProgress = 0.0f;
    
    // Clean up visual effects
    CleanupVisualEffects();
    
    UE_LOG(LogTemp, Log, TEXT("🎨 VISUAL: Stopped pulse animation"));
}

void UBP_PulseAnimator::UpdateAnimationProgress(float Progress)
{
    // Backend provides progress - just update visual
    CurrentProgress = FMath::Clamp(Progress, 0.0f, 1.0f);
    
    // Update visual effects based on progress
    UpdateVisualEffects(CurrentProgress);
}

// ====== PURE VISUAL METHODS (No Logic) ======

void UBP_PulseAnimator::InitializeVisualComponents()
{
    // Set up particle systems and materials for rendering
    AActor* Owner = GetOwner();
    if (!Owner) return;
    
    // Find particle component
    ParticleComponent = Owner->FindComponentByClass<UParticleSystemComponent>();
    
    // Find mesh component for material
    UStaticMeshComponent* MeshComp = Owner->FindComponentByClass<UStaticMeshComponent>();
    if (MeshComp && MeshComp->GetMaterial(0))
    {
        MaterialInstance = UMaterialInstanceDynamic::Create(MeshComp->GetMaterial(0), this);
        MeshComp->SetMaterial(0, MaterialInstance);
    }
}

void UBP_PulseAnimator::ApplyPulseVisualEffects(const FString& PulseType)
{
    // Apply visual effects based on pulse type (no logic, just rendering)
    if (PulseType == "bid_pulse")
    {
        SetPulseColor(FLinearColor(1.0f, 0.76f, 0.03f)); // Amber
    }
    else if (PulseType == "occupancy_pulse")
    {
        SetPulseColor(FLinearColor(0.13f, 0.59f, 0.95f)); // Blue
    }
    else if (PulseType == "compliancy_pulse")
    {
        SetPulseColor(FLinearColor(0.12f, 0.23f, 0.54f)); // Indigo
    }
    else if (PulseType == "fit_pulse")
    {
        SetPulseColor(FLinearColor(0.30f, 0.69f, 0.31f)); // Green
    }
    else if (PulseType == "investment_pulse")
    {
        SetPulseColor(FLinearColor(1.0f, 0.60f, 0.0f)); // Orange
    }
    else if (PulseType == "decay_pulse")
    {
        SetPulseColor(FLinearColor(0.62f, 0.62f, 0.62f)); // Gray
    }
    else if (PulseType == "reject_pulse")
    {
        SetPulseColor(FLinearColor(0.96f, 0.26f, 0.21f)); // Red
    }
}

void UBP_PulseAnimator::UpdateVisualAnimation(float DeltaTime)
{
    // Update visual animation progress (no logic)
    if (AnimationDuration > 0.0f)
    {
        CurrentProgress += DeltaTime / AnimationDuration;
        
        if (CurrentProgress >= 1.0f)
        {
            // Animation complete - stop visual
            StopPulseAnimation();
        }
        else
        {
            // Update visual effects
            UpdateVisualEffects(CurrentProgress);
        }
    }
}

void UBP_PulseAnimator::UpdateVisualEffects(float Progress)
{
    // Update material and particle parameters based on progress
    if (MaterialInstance)
    {
        MaterialInstance->SetScalarParameterValue("AnimationProgress", Progress);
        MaterialInstance->SetScalarParameterValue("Intensity", PulseIntensity);
    }
    
    if (ParticleComponent)
    {
        ParticleComponent->SetFloatParameter("Progress", Progress);
        ParticleComponent->SetFloatParameter("Intensity", PulseIntensity);
    }
}

void UBP_PulseAnimator::SetPulseColor(const FLinearColor& Color)
{
    // Set visual color only
    if (MaterialInstance)
    {
        MaterialInstance->SetVectorParameterValue("PulseColor", Color);
    }
}

void UBP_PulseAnimator::SetParticleTemplate(const FString& TemplateName)
{
    // Set particle system template (visual only)
    if (ParticleComponent)
    {
        // In real implementation, load particle template by name
        UE_LOG(LogTemp, Log, TEXT("🎨 VISUAL: Setting particle template: %s"), *TemplateName);
    }
}

void UBP_PulseAnimator::CleanupVisualEffects()
{
    // Clean up visual effects (no logic)
    if (ParticleComponent)
    {
        ParticleComponent->DeactivateSystem();
    }
    
    if (MaterialInstance)
    {
        MaterialInstance->SetScalarParameterValue("Intensity", 0.0f);
    }
}

// ====== DIRECTIONAL ANIMATION METHODS (Visual Only) ======

void UBP_PulseAnimator::AnimateDownward(float Duration, float Intensity)
{
    // Visual downward flow animation
    PulseIntensity = Intensity;
    AnimationDuration = Duration;
    
    if (MaterialInstance)
    {
        MaterialInstance->SetVectorParameterValue("FlowDirection", FLinearColor(0, 0, -1, 0));
    }
}

void UBP_PulseAnimator::AnimateUpward(float Duration, float Intensity)
{
    // Visual upward flow animation
    PulseIntensity = Intensity;
    AnimationDuration = Duration;
    
    if (MaterialInstance)
    {
        MaterialInstance->SetVectorParameterValue("FlowDirection", FLinearColor(0, 0, 1, 0));
    }
}

void UBP_PulseAnimator::AnimateHorizontalSpread(float Duration, float Intensity)
{
    // Visual horizontal spread animation
    PulseIntensity = Intensity;
    AnimationDuration = Duration;
    
    if (MaterialInstance)
    {
        MaterialInstance->SetVectorParameterValue("FlowDirection", FLinearColor(1, 1, 0, 0));
    }
}

void UBP_PulseAnimator::AnimateLateralWave(float Duration, float Intensity)
{
    // Visual lateral wave animation
    PulseIntensity = Intensity;
    AnimationDuration = Duration;
    
    if (MaterialInstance)
    {
        MaterialInstance->SetVectorParameterValue("FlowDirection", FLinearColor(1, 0, 0, 0));
    }
}

void UBP_PulseAnimator::AnimateRadialExpansion(float Duration, float Intensity)
{
    // Visual radial expansion animation
    PulseIntensity = Intensity;
    AnimationDuration = Duration;
    
    if (MaterialInstance)
    {
        MaterialInstance->SetScalarParameterValue("RadialExpansion", 1.0f);
    }
}

void UBP_PulseAnimator::AnimateInstantRejection(float Duration, float Intensity)
{
    // Visual instant rejection flash
    PulseIntensity = Intensity;
    AnimationDuration = Duration;
    
    if (MaterialInstance)
    {
        MaterialInstance->SetScalarParameterValue("InstantFlash", 1.0f);
    }
} 