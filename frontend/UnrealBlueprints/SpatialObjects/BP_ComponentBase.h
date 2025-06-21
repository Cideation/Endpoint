/*
 * BP_ComponentBase.h
 * 
 * Base Building Component for Unreal Engine
 * NO BUSINESS LOGIC - Only handles visual representation and user interaction
 * 
 * Architecture:
 * Backend (Node Engine) = Component Logic & Properties (Authoritative)
 * Unreal Engine = Visual Representation & Interaction Events (Display Layer)
 */

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/BoxComponent.h"
#include "BP_ComponentBase.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnComponentSelected, const FString&, ComponentID, const FVector&, Position);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnComponentDeselected, const FString&, ComponentID, const FVector&, Position);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnComponentMoved, const FString&, ComponentID, const FVector&, OldPosition, const FVector&, NewPosition);

UCLASS()
class BEMSYSTEM_API ABP_ComponentBase : public AActor
{
    GENERATED_BODY()

public:
    ABP_ComponentBase();

protected:
    virtual void BeginPlay() override;

public:
    // ====== VISUAL-ONLY METHODS (No Business Logic) ======
    
    UFUNCTION(BlueprintCallable, Category = "BEM Component Visual")
    void SetComponentMaterial(const FString& MaterialType);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Component Visual")
    void SetComponentDimensions(const FVector& Dimensions);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Component Visual")
    void SetHighlightState(bool bHighlighted);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Component Visual")
    void SetComponentID(const FString& ID);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Component Visual")
    FString GetComponentID() const;
    
    UFUNCTION(BlueprintCallable, Category = "BEM Component Visual")
    void UpdateVisualFromBackend(const FString& ComponentData);

    // ====== INTERACTION EVENTS (Send to Backend) ======
    
    UPROPERTY(BlueprintAssignable, Category = "BEM Component Events")
    FOnComponentSelected OnComponentSelected;
    
    UPROPERTY(BlueprintAssignable, Category = "BEM Component Events")
    FOnComponentDeselected OnComponentDeselected;
    
    UPROPERTY(BlueprintAssignable, Category = "BEM Component Events")
    FOnComponentMoved OnComponentMoved;

protected:
    // ====== VISUAL COMPONENTS (No Logic) ======
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Component Visual")
    UStaticMeshComponent* ComponentMesh;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Component Visual")
    UBoxComponent* CollisionBox;
    
    // ====== VISUAL PROPERTIES (No Logic) ======
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Component Visual")
    FString ComponentID;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Component Visual")
    FString ComponentType;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Component Visual")
    FVector ComponentDimensions;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Component Visual")
    FString MaterialType;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Component Visual")
    bool bIsSelected;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Component Visual")
    bool bIsHighlighted;

private:
    // ====== PURE VISUAL METHODS (No Logic) ======
    
    UFUNCTION()
    void OnComponentClicked(UPrimitiveComponent* ClickedComp, FKey ButtonPressed);
    
    UFUNCTION()
    void OnComponentReleased(UPrimitiveComponent* ReleasedComp, FKey ButtonReleased);
    
    void InitializeVisualComponents();
    void ApplyMaterialToComponent(const FString& MaterialType);
    void UpdateMeshDimensions(const FVector& Dimensions);
    void SetSelectionHighlight(bool bSelected);
};
