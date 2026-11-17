#include "motor_driver.h"
#include "config.h"
#include "ultrasonic.h"

QueueHandle_t g_motorQueue = nullptr;

void motorInit() {
    g_motorQueue = xQueueCreate(5, sizeof(MotorCmd));

#if MOTORS_ENABLED
    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    pinMode(MOTOR_IN3, OUTPUT);
    pinMode(MOTOR_IN4, OUTPUT);

    motorStop();
    Serial.printf("[MOTOR] L298N Initialized (4-wire jumper mode: IN1=%d, IN2=%d, IN3=%d, IN4=%d)\n",
                  MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4);
#else
    Serial.println("[MOTOR] Hardware driver is staged (MOTORS_ENABLED = false). Commands are simulated safely.");
#endif
}

void motorStop() {
#if MOTORS_ENABLED
    // Pull all motor inputs LOW to stop rotation cleanly
    analogWrite(MOTOR_IN1, 0);
    analogWrite(MOTOR_IN2, 0);
    analogWrite(MOTOR_IN3, 0);
    analogWrite(MOTOR_IN4, 0);
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
    digitalWrite(MOTOR_IN3, LOW);
    digitalWrite(MOTOR_IN4, LOW);
#endif
}

void motorApply(const MotorCmd &cmd) {
    // Collision Prevention: Block forward motion if obstacle is detected
    if (cmd.dir == MotorDir::FORWARD) {
        bool hasObstacle = false;
        if (g_obstacleMutex && xSemaphoreTake(g_obstacleMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            hasObstacle = g_obstacle.obstacleDetected;
            xSemaphoreGive(g_obstacleMutex);
        }
        if (hasObstacle) {
            Serial.println("⚠️ [SAFETY] Forward motion blocked — Obstacle detected!");
            motorStop();
            return;
        }
    }

#if MOTORS_ENABLED
    uint8_t spL = cmd.speedL > 0 ? cmd.speedL : 200;
    uint8_t spR = cmd.speedR > 0 ? cmd.speedR : 200;

    switch (cmd.dir) {
        case MotorDir::FORWARD:
            // Left forward, Right forward
            analogWrite(MOTOR_IN1, spL);
            digitalWrite(MOTOR_IN2, LOW);
            analogWrite(MOTOR_IN3, spR);
            digitalWrite(MOTOR_IN4, LOW);
            break;

        case MotorDir::BACKWARD:
            // Left reverse, Right reverse
            digitalWrite(MOTOR_IN1, LOW);
            analogWrite(MOTOR_IN2, spL);
            digitalWrite(MOTOR_IN3, LOW);
            analogWrite(MOTOR_IN4, spR);
            break;

        case MotorDir::LEFT:
            // Spin Left: Left reverse, Right forward
            digitalWrite(MOTOR_IN1, LOW);
            analogWrite(MOTOR_IN2, spL);
            analogWrite(MOTOR_IN3, spR);
            digitalWrite(MOTOR_IN4, LOW);
            break;

        case MotorDir::RIGHT:
            // Spin Right: Left forward, Right reverse
            analogWrite(MOTOR_IN1, spL);
            digitalWrite(MOTOR_IN2, LOW);
            digitalWrite(MOTOR_IN3, LOW);
            analogWrite(MOTOR_IN4, spR);
            break;

        case MotorDir::STOP:
        default:
            motorStop();
            break;
    }
#else
    if (cmd.dir != MotorDir::STOP) {
        const char *dirStr = (cmd.dir == MotorDir::FORWARD) ? "FWD" :
                             (cmd.dir == MotorDir::BACKWARD) ? "BACK" :
                             (cmd.dir == MotorDir::LEFT) ? "LEFT" : "RIGHT";
        Serial.printf("[MOTOR SIM] Command: %s | Speed: L=%d R=%d\n", dirStr, cmd.speedL, cmd.speedR);
    }
#endif
}

void TaskMotor(void *pvParameters) {
    (void)pvParameters;
    MotorCmd cmd;
    TickType_t lastCommandTick = xTaskGetTickCount();
    Serial.println("[TASK] Motor Driver pinned to Core " + String(xPortGetCoreID()));

    for (;;) {
        if (g_motorQueue && xQueueReceive(g_motorQueue, &cmd, pdMS_TO_TICKS(100))) {
            lastCommandTick = xTaskGetTickCount();
            if (millis() - cmd.timestamp > MOTOR_WATCHDOG_MS) {
                motorStop();
                continue;
            }
            motorApply(cmd);
        } else {
            // Watchdog: Auto-stop if no command received within timeout
            if (xTaskGetTickCount() - lastCommandTick > pdMS_TO_TICKS(MOTOR_WATCHDOG_MS)) {
                motorStop();
            }
        }
    }
}
