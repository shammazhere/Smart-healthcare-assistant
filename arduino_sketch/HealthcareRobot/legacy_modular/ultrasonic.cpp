#include "ultrasonic.h"
#include "config.h"

ObstacleData      g_obstacle = {999.0f, false, 0};
SemaphoreHandle_t g_obstacleMutex = nullptr;

void ultrasonicInit() {
    g_obstacleMutex = xSemaphoreCreateMutex();
    pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
    digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
    pinMode(ULTRASONIC_ECHO_PIN, INPUT);
    Serial.printf("[ULTRASONIC] HC-SR04 Initialized (TRIG=%d, ECHO=%d, Threshold=%.1fcm)\n",
                  ULTRASONIC_TRIG_PIN, ULTRASONIC_ECHO_PIN, OBSTACLE_LIMIT_CM);
}

float ultrasonicReadCm() {
    // Send 10 microsecond pulse to trigger pin
    digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(ULTRASONIC_TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(ULTRASONIC_TRIG_PIN, LOW);

    // Timeout: 25000 microseconds (~4.3 meters range)
    unsigned long duration = pulseIn(ULTRASONIC_ECHO_PIN, HIGH, 25000UL);
    if (duration == 0) {
        return 999.0f; // No echo received (clear path or out of range)
    }

    // Sound speed in air = 0.0343 cm/us (divide by 2 for round trip)
    float distance = (duration * 0.0343f) / 2.0f;
    return distance;
}

void TaskUltrasonic(void *pvParameters) {
    (void)pvParameters;
    const TickType_t period = pdMS_TO_TICKS(100); // 10 Hz sampling rate
    TickType_t lastWakeTime = xTaskGetTickCount();
    Serial.println("[TASK] Ultrasonic Obstacle Sensing pinned to Core " + String(xPortGetCoreID()));

    for (;;) {
        float dist = ultrasonicReadCm();
        bool isObstacle = (dist > 0.0f && dist <= OBSTACLE_LIMIT_CM);

        ObstacleData obs;
        obs.distanceCm = dist;
        obs.obstacleDetected = isObstacle;
        obs.timestamp = millis();

        if (xSemaphoreTake(g_obstacleMutex, pdMS_TO_TICKS(20)) == pdTRUE) {
            g_obstacle = obs;
            xSemaphoreGive(g_obstacleMutex);
        }

        vTaskDelayUntil(&lastWakeTime, period);
    }
}

