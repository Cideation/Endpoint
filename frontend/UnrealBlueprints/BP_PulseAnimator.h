/*
 * BP_PulseAnimator.h
 * 
 * Pure Visual Animation Component for Unreal Engine
 * NO PULSE LOGIC - Only renders visual effects based on backend commands
 * 
 * Architecture:
 * Backend (ECM + Node Engine) = Pulse Logic & State (Authoritative)
 * Unreal Engine = Visual Rendering Only (Display Layer)
 */

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/Engine.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Particles/ParticleSystemComponent.h"
#include "BP_PulseAnimator.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class BEMSYSTEM_API UBP_PulseAnimator : public UActorComponent
{
    GENERATED_BODY()

public:
    UBP_PulseAnimator();

protected:
    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

public:
    // ====== VISUAL-ONLY METHODS (Called by Backend Commands) ======
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void StartPulseAnimation(const FString& PulseType, float Intensity, float Duration);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void StopPulseAnimation();
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void UpdateAnimationProgress(float Progress);

    // ====== DIRECTIONAL ANIMATION METHODS (Visual Only) ======
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void AnimateDownward(float Duration, float Intensity);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void AnimateUpward(float Duration, float Intensity);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void AnimateHorizontalSpread(float Duration, float Intensity);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void AnimateLateralWave(float Duration, float Intensity);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void AnimateRadialExpansion(float Duration, float Intensity);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void AnimateInstantRejection(float Duration, float Intensity);

protected:
    // ====== VISUAL PROPERTIES (No Logic) ======
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Pulse Visual")
    float CurrentProgress;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Pulse Visual")
    float AnimationDuration;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Pulse Visual")
    float PulseIntensity;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Pulse Visual")
    bool IsAnimating;

private:
    // ====== VISUAL COMPONENTS (No Logic) ======
    
    UPROPERTY()
    UParticleSystemComponent* ParticleComponent;
    
    UPROPERTY()
    UMaterialInstanceDynamic* MaterialInstance;

    // ====== PURE VISUAL METHODS (No Logic) ======
    
    void InitializeVisualComponents();
    void ApplyPulseVisualEffects(const FString& PulseType);
    void UpdateVisualAnimation(float DeltaTime);
    void UpdateVisualEffects(float Progress);
    void SetPulseColor(const FLinearColor& Color);
    void SetParticleTemplate(const FString& TemplateName);
    void CleanupVisualEffects();
}; 