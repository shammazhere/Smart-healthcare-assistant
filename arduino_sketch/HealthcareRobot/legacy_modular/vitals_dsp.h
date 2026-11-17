#pragma once
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>
#include "MAX30105.h"
#include "heartRate.h"

// ── Patient Vitals Data Structure ────────────────────────────────────────────
struct VitalsData {
    float    heartRate;
    float    spo2;
    bool     fingerDetected;
    bool     isValid;
    uint32_t timestamp;
};

// Global handles
extern VitalsData        g_vitals;
extern SemaphoreHandle_t g_vitalsMutex;
extern Adafruit_SSD1306  g_oled;
extern MAX30105          g_ppgSensor;

// Public functions
bool vitalsInit();
void TaskVitals(void *pvParameters);
void TaskOLED(void *pvParameters);
