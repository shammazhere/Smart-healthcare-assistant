#pragma once
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

struct ObstacleData {
    float    distanceCm;
    bool     obstacleDetected;
    uint32_t timestamp;
};

// Global handles
extern ObstacleData      g_obstacle;
extern SemaphoreHandle_t g_obstacleMutex;

// Public functions
void  ultrasonicInit();
float ultrasonicReadCm();
void  TaskUltrasonic(void *pvParameters);

