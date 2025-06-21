// BP_PulseAnimator.h - Pure Visual Animation Component
// NO PULSE LOGIC - Only renders visual effects based on backend commands

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "BP_PulseAnimator.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class BEMSYSTEM_API UBP_PulseAnimator : public UActorComponent
{
    GENERATED_BODY()

public:
    UBP_PulseAnimator();

public:
    // Visual-only methods (called by backend commands)
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void StartPulseAnimation(const FString& PulseType, float Intensity, float Duration);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void StopPulseAnimation();
    
    // Directional animation methods (visual only)
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void AnimateDownward(float Duration, float Intensity);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Pulse Visual")
    void AnimateUpward(float Duration, float Intensity);

protected:
    // Visual properties (no logic)
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Pulse Visual")
    float CurrentProgress;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Pulse Visual")
    float AnimationDuration;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Pulse Visual")
    float PulseIntensity;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Pulse Visual")
    bool IsAnimating;

private:
    void ApplyPulseVisualEffects(const FString& PulseType);
    void SetPulseColor(const FLinearColor& Color);
};
