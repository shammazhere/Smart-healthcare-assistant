// ╔══════════════════════════════════════════════════════════════════════════╗
// ║   HealthBot OS — ESP32 Autonomous Healthcare Assistant Robot             ║
// ║   Main Sketch Entry Point                                                ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>

#include "config.h"
#include "motor_driver.h"
#include "vitals_dsp.h"
#include "ultrasonic.h"
#include "audio_i2s.h"
#include "web_server.h"

void setup() {
    Serial.begin(115200);
    delay(400);
    Serial.println("\n\n╔══════════════════════════════════════════════╗");
    Serial.println("║  HealthBot OS v1.1 — ESP32 Healthcare Robot  ║");
    Serial.println("╚══════════════════════════════════════════════╝");

    // 1. Initialize I2C Bus (SDA=21, SCL=22), OLED & MAX30102 PPG Sensor
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 400000UL);
    vitalsInit();

    // 2. Wi-Fi Connection (STA mode with fallback to SoftAP)
    Serial.printf("[WIFI] Connecting to SSID: %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) {
        delay(400);
        Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WIFI] Connected! Robot IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WIFI] Connection failed — Starting fallback SoftAP (HealthBot-AP / healthbot123)");
        WiFi.mode(WIFI_AP);
        WiFi.softAP("HealthBot-AP", "healthbot123");
    }

    // 3. Initialize HC-SR04 Ultrasonic Distance Sensor
    ultrasonicInit();

    // 4. Initialize L298N Motor Driver (Staged/Safe if MOTORS_ENABLED == false)
    motorInit();

    // 5. Initialize I2S Full-Duplex Audio (INMP441 Mic + MAX98357A Speaker)
    if (!audioInit()) {
        Serial.println("[WARN] I2S audio driver failed to initialize");
    }

    // 6. Start Web Server (Port 80) & Control WebSocket Server (Port 81)
    webserverInit();

    // 7. Update OLED Splash Screen with IP
    g_oled.clearDisplay();
    g_oled.setTextSize(1);
    g_oled.setTextColor(SSD1306_WHITE);
    g_oled.setCursor(0, 0);  g_oled.print("HealthBot Ready!");
    g_oled.setCursor(0, 14); g_oled.print("Dashboard available:");
    g_oled.setCursor(0, 26); g_oled.print("http://");
    g_oled.print(WiFi.localIP().toString());
    g_oled.setCursor(0, 40); g_oled.printf("WS Port: %u", WS_PORT);
    g_oled.display();
    delay(2000);

    // 8. Launch FreeRTOS Tasks
    // ── Core 1: Real-Time Hardware Tasks
    xTaskCreatePinnedToCore(TaskMotor,      "Motor",      4096, nullptr, 5, nullptr, 1);
    xTaskCreatePinnedToCore(TaskVitals,     "Vitals",     4096, nullptr, 4, nullptr, 1);
    xTaskCreatePinnedToCore(TaskUltrasonic, "Ultrasonic", 3072, nullptr, 3, nullptr, 1);
    xTaskCreatePinnedToCore(TaskOLED,       "OLED",       3072, nullptr, 2, nullptr, 1);

    // ── Core 0: Network & Audio Streaming Tasks
    xTaskCreatePinnedToCore(TaskAudio,      "AudioDMA",   8192, nullptr, 4, nullptr, 0);
    xTaskCreatePinnedToCore(TaskGateway,    "AIGateway",  8192, nullptr, 3, nullptr, 0);
    xTaskCreatePinnedToCore(TaskTelemetry,  "Telemetry",  6144, nullptr, 1, nullptr, 0);

    Serial.println("[MAIN] All FreeRTOS tasks active — Robot operational.");
}

// FreeRTOS manages all concurrent tasks; loop remains intentionally idle
void loop() {
    vTaskDelay(portMAX_DELAY);
}
