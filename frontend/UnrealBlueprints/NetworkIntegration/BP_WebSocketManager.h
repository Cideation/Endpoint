/*
 * BP_WebSocketManager.h
 * 
 * Pure Network Communication Component for Unreal Engine
 * NO BUSINESS LOGIC - Only sends/receives messages to/from backend
 * 
 * Architecture:
 * Backend (ECM + Node Engine) = All Logic & State (Authoritative)
 * Unreal Engine = Network Client Only (Communication Layer)
 */

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/Engine.h"
#include "BP_WebSocketManager.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPulseReceived, const FString&, PulseData);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnWebSocketConnected, bool, bConnected);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnSpatialEventSent, const FString&, EventData);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class BEMSYSTEM_API UBP_WebSocketManager : public UActorComponent
{
    GENERATED_BODY()

public:
    UBP_WebSocketManager();

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
    // ====== NETWORK-ONLY METHODS (No Logic) ======
    
    UFUNCTION(BlueprintCallable, Category = "BEM Network")
    void ConnectToECMGateway(const FString& URL);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Network")
    void DisconnectFromGateway();
    
    UFUNCTION(BlueprintCallable, Category = "BEM Network")
    void SendSpatialEvent(const FString& EventType, const FVector& Position, const FString& ComponentID);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Network")
    void RequestPulseVisualization(const FString& PulseType, const FVector& Position);
    
    UFUNCTION(BlueprintCallable, Category = "BEM Network")
    bool IsConnectedToBackend() const;

    // ====== BLUEPRINT EVENTS (Receive from Backend) ======
    
    UPROPERTY(BlueprintAssignable, Category = "BEM Network Events")
    FOnPulseReceived OnPulseReceived;
    
    UPROPERTY(BlueprintAssignable, Category = "BEM Network Events")
    FOnWebSocketConnected OnWebSocketConnected;
    
    UPROPERTY(BlueprintAssignable, Category = "BEM Network Events")
    FOnSpatialEventSent OnSpatialEventSent;

protected:
    // ====== NETWORK PROPERTIES (No Logic) ======
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Network")
    FString ECMGatewayURL;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BEM Network")
    FString PulseHandlerURL;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Network")
    bool bIsConnected;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "BEM Network")
    FString ConnectionID;

private:
    // ====== PURE NETWORK METHODS (No Logic) ======
    
    void HandleIncomingMessage(const FString& Message);
    void ProcessPulseMessage(const FString& PulseData);
    void SendMessage(const FString& Message);
    FString CreateSpatialEventJSON(const FString& EventType, const FVector& Position, const FString& ComponentID);
    FString CreatePulseRequestJSON(const FString& PulseType, const FVector& Position);
};
