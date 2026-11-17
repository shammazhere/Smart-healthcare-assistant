#include "motor_driver.h"
#include "config.h"
#include "ultrasonic.h"

QueueHandle_t g_motorQueue = nullptr;

void motorInit() {
    g_motorQueue = xQueueCreate(5, sizeof(MotorCmd));

#if MOTORS_ENABLED
    pinMode(MOTOR_IN1, OUTPUT); pinMode(MOTOR_IN2, OUTPUT);
    pinMode(MOTOR_IN3, OUTPUT); pinMode(MOTOR_IN4, OUTPUT);
    digitalWrite(MOTOR_IN1, LOW); digitalWrite(MOTOR_IN2, LOW);
    digitalWrite(MOTOR_IN3, LOW); digitalWrite(MOTOR_IN4, LOW);

    // Support both ESP32 Core 2.x and Core 3.x
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
    ledcAttach(MOTOR_ENA, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
    ledcAttach(MOTOR_ENB, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
#else
    ledcSetup(0, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
    ledcSetup(1, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
    ledcAttachPin(MOTOR_ENA, 0);
    ledcAttachPin(MOTOR_ENB, 1);
#endif
    Serial.println("[MOTOR] L298N Initialized (Hardware Active)");
#else
    Serial.println("[MOTOR] Hardware driver is staged (MOTORS_ENABLED = false). Commands are simulated safely.");
#endif
}

void motorStop() {
#if MOTORS_ENABLED
    digitalWrite(MOTOR_IN1, HIGH); digitalWrite(MOTOR_IN2, HIGH);
    digitalWrite(MOTOR_IN3, HIGH); digitalWrite(MOTOR_IN4, HIGH);
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
    ledcWrite(MOTOR_ENA, 0);
    ledcWrite(MOTOR_ENB, 0);
#else
    ledcWrite(0, 0);
    ledcWrite(1, 0);
#endif
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
    bool leftFwd = true, rightFwd = true;
    switch (cmd.dir) {
        case MotorDir::FORWARD:   leftFwd = true;  rightFwd = true;  break;
        case MotorDir::BACKWARD:  leftFwd = false; rightFwd = false; break;
        case MotorDir::LEFT:      leftFwd = false; rightFwd = true;  break;
        case MotorDir::RIGHT:     leftFwd = true;  rightFwd = false; break;
        default: motorStop(); return;
    }
    digitalWrite(MOTOR_IN1, leftFwd ? HIGH : LOW);
    digitalWrite(MOTOR_IN2, leftFwd ? LOW  : HIGH);
    digitalWrite(MOTOR_IN3, rightFwd ? HIGH : LOW);
    digitalWrite(MOTOR_IN4, rightFwd ? LOW  : HIGH);

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
    ledcWrite(MOTOR_ENA, cmd.speedL);
    ledcWrite(MOTOR_ENB, cmd.speedR);
#else
    ledcWrite(0, cmd.speedL);
    ledcWrite(1, cmd.speedR);
#endif

#else
    // Simulated logging when hardware driver is not connected
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
