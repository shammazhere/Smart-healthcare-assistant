#include "vitals_dsp.h"
#include "config.h"
#include "ultrasonic.h"
#include "audio_i2s.h"
#include <WiFi.h>
#include <Wire.h>

// Global definitions
VitalsData        g_vitals = {0, 0, false, false, 0};
SemaphoreHandle_t g_vitalsMutex = nullptr;
Adafruit_SSD1306  g_oled(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
MAX30105          g_ppgSensor;

// Internal DSP Filter State
static float g_hpInR = 0, g_hpOutR = 0, g_hpInIR = 0, g_hpOutIR = 0;
static float g_lpR = 0, g_lpIR = 0;
static float g_irAC[HR_BUF_SIZE] = {}, g_rAC[HR_BUF_SIZE] = {};
static float g_irDC[HR_BUF_SIZE] = {}, g_rDC[HR_BUF_SIZE] = {};
static int   g_dspIndex = 0;
static byte  g_rateHistory[HR_BUF_SIZE] = {};
static byte  g_rateIndex = 0;
static long  g_lastBeatMs = 0;
static float g_avgBPM = 0;

static inline float highPassFilter(float x, float &xi, float &yi, float alpha = 0.95f) {
    float y = alpha * (yi + x - xi);
    xi = x; yi = y;
    return y;
}

static inline float lowPassFilter(float x, float &state, float alpha = 0.05f) {
    state = alpha * x + (1.0f - alpha) * state;
    return state;
}

static float calcRMS(const float *buf, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += buf[i] * buf[i];
    return sqrtf(sum / n);
}

static float calcMean(const float *buf, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += buf[i];
    return sum / n;
}

bool vitalsInit() {
    g_vitalsMutex = xSemaphoreCreateMutex();

    // 1. OLED Display (Address 0x3C on GPIO 21/22)
    if (g_oled.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR)) {
        g_oled.clearDisplay();
        g_oled.setTextSize(1);
        g_oled.setTextColor(SSD1306_WHITE);
        g_oled.setCursor(18, 20); g_oled.print("HealthBot OS");
        g_oled.setCursor(24, 36); g_oled.print("Booting up...");
        g_oled.display();
        Serial.println("[OLED] SSD1306 Display OK on GPIO 21/22");
    } else {
        Serial.println("[WARN] SSD1306 OLED not found on GPIO 21/22");
    }

    // 2. MAX30102 PPG Sensor (Address 0x57 on GPIO 21/22)
    if (g_ppgSensor.begin(Wire, I2C_SPEED_FAST, MAX30102_I2C_ADDR)) {
        g_ppgSensor.setup(PPG_LED_POWER, PPG_SAMPLE_AVG, 2,
                          PPG_SAMPLE_RATE, PPG_PULSE_WIDTH, PPG_ADC_RANGE);
        Serial.println("[PPG] MAX30102 Sensor OK on GPIO 21/22");
        return true;
    } else {
        Serial.println("[WARN] MAX30102 not detected on GPIO 21/22");
        return false;
    }
}

void TaskVitals(void *pvParameters) {
    (void)pvParameters;
    const TickType_t period = pdMS_TO_TICKS(10); // 100 Hz sampling
    TickType_t lastWakeTime = xTaskGetTickCount();
    Serial.println("[TASK] Vitals DSP pinned to Core " + String(xPortGetCoreID()));

    for (;;) {
        if (!g_ppgSensor.available()) {
            g_ppgSensor.check();
            vTaskDelayUntil(&lastWakeTime, period);
            continue;
        }

        uint32_t rawRed = g_ppgSensor.getRed();
        uint32_t rawIR  = g_ppgSensor.getIR();
        g_ppgSensor.nextSample();

        bool fingerPresent = (rawIR > FINGER_THRESHOLD);

        if (!fingerPresent) {
            g_hpInR = g_hpOutR = g_hpInIR = g_hpOutIR = g_lpR = g_lpIR = g_avgBPM = 0;
            g_rateIndex = 0;
            VitalsData vd = {0, 0, false, false, (uint32_t)millis()};
            xSemaphoreTake(g_vitalsMutex, portMAX_DELAY);
            g_vitals = vd;
            xSemaphoreGive(g_vitalsMutex);

            vTaskDelayUntil(&lastWakeTime, period);
            continue;
        }

        // Digital Filters
        float dcRed = lowPassFilter((float)rawRed, g_lpR, 0.05f);
        float dcIR  = lowPassFilter((float)rawIR,  g_lpIR, 0.05f);
        float acRed = highPassFilter((float)rawRed, g_hpInR, g_hpOutR);
        float acIR  = highPassFilter((float)rawIR,  g_hpInIR, g_hpOutIR);

        g_irAC[g_dspIndex] = acIR; g_rAC[g_dspIndex] = acRed;
        g_irDC[g_dspIndex] = dcIR; g_rDC[g_dspIndex] = dcRed;
        g_dspIndex = (g_dspIndex + 1) % HR_BUF_SIZE;

        // Peak / Heartbeat Detection
        if (checkForBeat((long)rawIR)) {
            long now = millis();
            long delta = now - g_lastBeatMs;
            g_lastBeatMs = now;
            float bpm = 60000.0f / delta;

            if (bpm >= 35.0f && bpm <= 230.0f) {
                g_rateHistory[g_rateIndex++] = (byte)bpm;
                g_rateIndex %= HR_BUF_SIZE;
                float sum = 0;
                for (int i = 0; i < HR_BUF_SIZE; i++) sum += g_rateHistory[i];
                g_avgBPM = sum / HR_BUF_SIZE;
            }
        }

        // SpO2 Estimation via Ratio-of-Ratios
        float acRedRMS  = calcRMS(g_rAC, HR_BUF_SIZE);
        float acIRRMS   = calcRMS(g_irAC, HR_BUF_SIZE);
        float dcRedMean = calcMean(g_rDC, HR_BUF_SIZE);
        float dcIRMean  = calcMean(g_irDC, HR_BUF_SIZE);

        float spo2 = 0;
        if (dcRedMean > 1 && dcIRMean > 1 && acIRRMS > 0) {
            float ratio = (acRedRMS / dcRedMean) / (acIRRMS / dcIRMean);
            spo2 = constrain(110.0f - 25.0f * ratio, 70.0f, 100.0f);
        }

        bool valid = (g_avgBPM > 35 && spo2 > 70);
        VitalsData vd = {g_avgBPM, spo2, true, valid, (uint32_t)millis()};

        xSemaphoreTake(g_vitalsMutex, portMAX_DELAY);
        g_vitals = vd;
        xSemaphoreGive(g_vitalsMutex);

        vTaskDelayUntil(&lastWakeTime, period);
    }
}

void TaskOLED(void *pvParameters) {
    (void)pvParameters;
    const TickType_t period = pdMS_TO_TICKS(50); // 20 FPS
    TickType_t lastWakeTime = xTaskGetTickCount();
    Serial.println("[TASK] OLED UI pinned to Core " + String(xPortGetCoreID()));

    for (;;) {
        VitalsData vd;
        xSemaphoreTake(g_vitalsMutex, portMAX_DELAY);
        vd = g_vitals;
        xSemaphoreGive(g_vitalsMutex);

        ObstacleData od = {999.0f, false, 0};
        if (g_obstacleMutex && xSemaphoreTake(g_obstacleMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            od = g_obstacle;
            xSemaphoreGive(g_obstacleMutex);
        }

        AudioState as = audioGetState();

        g_oled.clearDisplay();
        g_oled.setTextColor(SSD1306_WHITE);

        // Header with IP
        g_oled.setTextSize(1);
        g_oled.setCursor(0, 0);
        g_oled.print("HealthBot ");
        g_oled.print(WiFi.localIP().toString());
        g_oled.drawFastHLine(0, 9, 128, SSD1306_WHITE);

        // Heart Rate (Left Column)
        g_oled.setCursor(0, 12);
        g_oled.print("HR:");
        g_oled.setTextSize(2);
        g_oled.setCursor(0, 21);
        if (vd.fingerDetected && vd.isValid) g_oled.printf("%.0f", vd.heartRate);
        else g_oled.print("--");
        g_oled.setTextSize(1);
        g_oled.setCursor(0, 38);
        g_oled.print("BPM");

        // Vertical divider
        g_oled.drawFastVLine(58, 10, 38, SSD1306_WHITE);

        // SpO2 (Right Column)
        g_oled.setCursor(64, 12);
        g_oled.print("SpO2:");
        g_oled.setTextSize(2);
        g_oled.setCursor(64, 21);
        if (vd.fingerDetected && vd.isValid) g_oled.printf("%.0f%%", vd.spo2);
        else g_oled.print("--%");

        // Obstacle / Distance Line (y=38)
        g_oled.setTextSize(1);
        g_oled.setCursor(64, 38);
        if (od.obstacleDetected) {
            g_oled.printf("OBS! %.0fcm", od.distanceCm);
        } else if (od.distanceCm < 400.0f) {
            g_oled.printf("Dist:%.0fcm", od.distanceCm);
        } else {
            g_oled.print("Path Clr");
        }

        // Status Footer
        g_oled.drawFastHLine(0, 49, 128, SSD1306_WHITE);
        g_oled.setCursor(0, 53);

        if      (od.obstacleDetected)         g_oled.print("CAUTION: Obstacle Close");
        else if (as == AudioState::LISTENING) g_oled.print("Listening (Mic active)");
        else if (as == AudioState::SPEAKING)  g_oled.print("Speaking (Baymax TTS)");
        else if (!vd.fingerDetected)          g_oled.print("Place finger on sensor");
        else                                  g_oled.print("System Ready");

        g_oled.display();
        vTaskDelayUntil(&lastWakeTime, period);
    }
}
