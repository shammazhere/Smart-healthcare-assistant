#include "web_server.h"
#include "config.h"
#include "motor_driver.h"
#include "vitals_dsp.h"
#include "ultrasonic.h"
#include "audio_i2s.h"
#include <ArduinoJson.h>

// Global server instances
WebServer        g_httpServer(80);
WebSocketsServer g_wsServer(WS_PORT);
WebSocketsClient g_gatewayClient;

// Broadcast live sensor telemetry to connected web dashboards
void broadcastTelemetry() {
    VitalsData vd = {0, 0, false, false, 0};
    if (g_vitalsMutex && xSemaphoreTake(g_vitalsMutex, pdMS_TO_TICKS(20)) == pdTRUE) {
        vd = g_vitals;
        xSemaphoreGive(g_vitalsMutex);
    }

    ObstacleData od = {999.0f, false, 0};
    if (g_obstacleMutex && xSemaphoreTake(g_obstacleMutex, pdMS_TO_TICKS(20)) == pdTRUE) {
        od = g_obstacle;
        xSemaphoreGive(g_obstacleMutex);
    }

    int aud = (int)audioGetState();
    char json[256];

    snprintf(json, sizeof(json),
             "{\"type\":\"telemetry\",\"hr\":%.1f,\"spo2\":%.1f,\"finger\":%s,\"audio\":%d,\"dist\":%.1f,\"obstacle\":%s,\"motors\":%s}",
             vd.isValid ? vd.heartRate : 0.0f,
             vd.isValid ? vd.spo2 : 0.0f,
             vd.fingerDetected ? "true" : "false",
             aud,
             od.distanceCm < 900.0f ? od.distanceCm : -1.0f,
             od.obstacleDetected ? "true" : "false",
             MOTORS_ENABLED ? "true" : "false");

    g_wsServer.broadcastTXT(json);
}

// WebSocket Event Handler for Web Dashboard (Port 81)
static void onControlWSEvent(uint8_t clientNum, WStype_t type, uint8_t *payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED: {
            IPAddress ip = g_wsServer.remoteIP(clientNum);
            Serial.printf("✅ [WebSocket] Client #%u connected from %s\n",
                          clientNum, ip.toString().c_str());
            break;
        }
        case WStype_DISCONNECTED: {
            Serial.printf("🔌 [WebSocket] Client #%u disconnected\n", clientNum);
            break;
        }
        case WStype_TEXT: {
            String msg = String((char*)payload);
            msg.trim();

            // Handle ping / keepalive
            if (msg == "ping" || msg.indexOf("\"ping\"") >= 0) {
                g_wsServer.sendTXT(clientNum, "{\"type\":\"pong\",\"pong\":true}");
                return;
            }

            JsonDocument doc;
            if (deserializeJson(doc, (char*)payload) != DeserializationError::Ok) {
                return;
            }

            // Motor command handling
            if (doc["dir"].is<const char*>()) {
                const char *d = doc["dir"].as<const char*>();
                uint8_t sp = doc["speed"] | 170;
                MotorCmd cmd;
                cmd.speedL = sp;
                cmd.speedR = sp;
                cmd.timestamp = millis();

                if      (!strcmp(d, "F")) cmd.dir = MotorDir::FORWARD;
                else if (!strcmp(d, "B")) cmd.dir = MotorDir::BACKWARD;
                else if (!strcmp(d, "L")) cmd.dir = MotorDir::LEFT;
                else if (!strcmp(d, "R")) cmd.dir = MotorDir::RIGHT;
                else                      cmd.dir = MotorDir::STOP;

                if (g_motorQueue) {
                    xQueueSend(g_motorQueue, &cmd, 0);
                }
            }

            // Voice recording toggle
            if (doc["voice"].is<bool>()) {
                if (doc["voice"].as<bool>()) {
                    audioStartRecording();
                } else {
                    audioStopRecording();
                }
            }
            break;
        }
        default:
            break;
    }
}

// Gateway WebSocket Event Handler (Connects to Python Backend ws/audio)
static void onGatewayEvent(WStype_t type, uint8_t *payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            Serial.println("✅ [AI-GW] Connected to Python AI Voice Gateway");
            break;
        case WStype_BIN:
            // Stream synthesized PCM TTS speech chunk to speaker
            audioQueuePlaybackChunk(payload, length);
            break;
        case WStype_TEXT:
            // Forward AI transcript / response to connected web dashboard clients
            g_wsServer.broadcastTXT((char*)payload);
            break;
        case WStype_DISCONNECTED:
            Serial.println("⚠️ [AI-GW] Disconnected from AI Gateway — will auto-reconnect");
            break;
        default:
            break;
    }
}

// Lightweight landing page served directly by ESP32 HTTP Server (Port 80)
static const char STATUS_PAGE_HTML[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HealthBot ESP32</title>
  <style>
    body{margin:0;min-height:100vh;background:#060911;color:#f8fafc;font-family:system-ui,-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;padding:20px}
    .card{background:rgba(22,28,48,0.85);border:1px solid rgba(255,255,255,0.1);border-radius:24px;padding:36px;max-width:440px;text-align:center;box-shadow:0 12px 40px rgba(0,0,0,0.5)}
    .icon{font-size:3rem;margin-bottom:12px}
    h1{font-size:1.4rem;margin:0 0 8px;color:#a78bfa}
    p{font-size:0.88rem;color:#94a3b8;line-height:1.5;margin:0 0 20px}
    .badge{display:inline-block;padding:4px 14px;border-radius:99px;background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);color:#34d399;font-weight:600;font-size:0.75rem;margin-bottom:20px}
    .btn{display:inline-block;background:#7c6df0;color:#fff;text-decoration:none;padding:12px 28px;border-radius:99px;font-weight:600;font-size:0.9rem;box-shadow:0 0 20px rgba(124,109,240,0.4);transition:transform 0.2s}
    .btn:hover{transform:scale(1.04)}
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">🏥</div>
    <h1>HealthBot OS</h1>
    <div class="badge">ESP32 Online & Running</div>
    <p>Tele-monitoring & AI Voice Gateway active.<br>Open the modern web dashboard to view live vitals and control the robot.</p>
    <a class="btn" href=")rawhtml" DASHBOARD_URL R"rawhtml(">Open HealthBot Dashboard &rarr;</a>
  </div>
</body>
</html>
)rawhtml";

void webserverInit() {
    // 1. Standard WebServer routes (Port 80)
    g_httpServer.on("/", HTTP_GET, []() {
        g_httpServer.send_P(200, "text/html", STATUS_PAGE_HTML);
    });

    g_httpServer.on("/ping", HTTP_GET, []() {
        g_httpServer.send(200, "text/plain", "pong");
    });

    g_httpServer.begin();
    Serial.println("[WEB] HTTP Server listening on port 80");

    // 2. Telemetry & Control WebSocket Server (Port 81)
    g_wsServer.begin();
    g_wsServer.onEvent(onControlWSEvent);
    Serial.printf("🌐 [WebSocket] Server listening on port %u\n", WS_PORT);
}

void TaskGateway(void *pvParameters) {
    (void)pvParameters;

    // Parse ws://<host>:<port><path>
    String uri = AI_GATEWAY_WS;
    int schemeEnd = uri.indexOf("://") + 3;
    int portIndex = uri.indexOf(':', schemeEnd);
    int pathIndex = uri.indexOf('/', schemeEnd);

    String host = uri.substring(schemeEnd, (portIndex > 0 && portIndex < pathIndex) ? portIndex : pathIndex);
    uint16_t port = (portIndex > 0 && portIndex < pathIndex) ? uri.substring(portIndex + 1, pathIndex).toInt() : 8080;
    String path = (pathIndex > 0) ? uri.substring(pathIndex) : "/ws/audio";

    g_gatewayClient.begin(host.c_str(), port, path.c_str());
    g_gatewayClient.onEvent(onGatewayEvent);
    g_gatewayClient.setReconnectInterval(3000);
    Serial.printf("[AI-GW] Target: %s:%d%s\n", host.c_str(), port, path.c_str());

    const size_t chunkSize = I2S_DMA_FRAME * (I2S_BITS / 8);

    for (;;) {
        g_gatewayClient.loop();

        if (audioGetState() == AudioState::LISTENING && g_micTxRingBuf) {
            size_t rxLen = 0;
            void *micChunk = xRingbufferReceiveUpTo(g_micTxRingBuf, &rxLen, pdMS_TO_TICKS(20), chunkSize);
            if (micChunk && rxLen > 0) {
                g_gatewayClient.sendBIN((uint8_t*)micChunk, rxLen);
                vRingbufferReturnItem(g_micTxRingBuf, micChunk);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

void TaskTelemetry(void *pvParameters) {
    (void)pvParameters;
    TickType_t lastBroadcast = xTaskGetTickCount();
    const TickType_t broadcastInterval = pdMS_TO_TICKS(1000); // 1 Hz broadcast
    Serial.println("[TASK] Web Telemetry & Server pinned to Core " + String(xPortGetCoreID()));

    for (;;) {
        // Service WebSocket clients & HTTP requests
        g_wsServer.loop();
        g_httpServer.handleClient();

        if (xTaskGetTickCount() - lastBroadcast >= broadcastInterval) {
            lastBroadcast = xTaskGetTickCount();
            broadcastTelemetry();
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
