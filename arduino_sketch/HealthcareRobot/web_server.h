#pragma once
#include <Arduino.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <WebSocketsClient.h>

// Global server instances
extern WebServer        g_httpServer;
extern WebSocketsServer g_wsServer;
extern WebSocketsClient g_gatewayClient;

// Public Server Functions
void webserverInit();
void broadcastTelemetry();
void TaskGateway(void *pvParameters);
void TaskTelemetry(void *pvParameters);
