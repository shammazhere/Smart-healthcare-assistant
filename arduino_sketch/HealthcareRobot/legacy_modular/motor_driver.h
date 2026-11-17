#pragma once
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

// ── Motor Movement Direction ─────────────────────────────────────────────────
enum class MotorDir : uint8_t { STOP = 0, FORWARD, BACKWARD, LEFT, RIGHT, BRAKE };

// ── Motor Command Message Structure ──────────────────────────────────────────
struct MotorCmd {
    MotorDir dir;
    uint8_t  speedL;
    uint8_t  speedR;
    uint32_t timestamp;
};

// Global Command Queue handle
extern QueueHandle_t g_motorQueue;

// Public Driver Functions
void motorInit();
void motorStop();
void motorApply(const MotorCmd &cmd);
void TaskMotor(void *pvParameters);
