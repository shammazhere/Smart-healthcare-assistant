#pragma once

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║   config.h — Hardware Pinouts, Network Settings & DSP Constants         ║
// ║   Configured for Standard ESP32 (WROOM-32 / NodeMCU-32S)                ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ── Wi-Fi Credentials ────────────────────────────────────────────────────────
#define WIFI_SSID           "CONVERGE-1"
#define WIFI_PASSWORD       "12345678"

// ── AI Voice Gateway Backend (Hosted on Railway Cloud) ───────────────────────
#define AI_GATEWAY_WS       "wss://smart-companion.up.railway.app/ws/audio"
#define DASHBOARD_URL       "https://smart-companion.up.railway.app"
#define WS_PORT             81            // Local WebSocket server port (Markus Sattler pattern from aqms)

// ── I2S Audio Bus (Shared SCK & WS for Mic & Speaker) ────────────────────────
#define I2S_PORT            I2S_NUM_0
#define I2S_SCK_PIN         26   // Shared Clock (INMP441 SCK & MAX98357A BCLK)
#define I2S_WS_PIN          25   // Shared Word Select (INMP441 WS & MAX98357A LRC)
#define I2S_MIC_SD_PIN      33   // Serial Data IN (INMP441 SD -> ESP32 RX)
#define I2S_SPK_DIN_PIN     27   // Serial Data OUT (ESP32 TX -> MAX98357A DIN)
#define I2S_SAMPLE_RATE     16000
#define I2S_BITS            16
#define I2S_DMA_FRAME       512
#define I2S_DMA_BUFS        8

// ── I2C Bus (Shared by MAX30102 PPG & SSD1306 OLED) ──────────────────────────
#define I2C_SDA_PIN         21   // ESP32 Standard Hardware I2C SDA
#define I2C_SCL_PIN         22   // ESP32 Standard Hardware I2C SCL
#define OLED_I2C_ADDR       0x3C
#define MAX30102_I2C_ADDR   0x57
#define OLED_WIDTH          128
#define OLED_HEIGHT         64

// ── HC-SR04 Ultrasonic Distance Sensor ───────────────────────────────────────
#define ULTRASONIC_TRIG_PIN 18   // Trigger output pulse
#define ULTRASONIC_ECHO_PIN 19   // Echo input pulse (via voltage divider)
#define OBSTACLE_LIMIT_CM   15.0f // Obstacle proximity safety limit in cm

// ── L298N Dual H-Bridge Motor Driver ─────────────────────────────────────────
// Set to true only AFTER physically connecting the L298N driver
#define MOTORS_ENABLED      false

// Safe GPIOs for standard ESP32 (Avoids internal SPI flash pins GPIO 6..11)
#define MOTOR_IN1           13   // Left Motor Direction 1
#define MOTOR_IN2           12   // Left Motor Direction 2
#define MOTOR_IN3           14   // Right Motor Direction 1
#define MOTOR_IN4           23   // Right Motor Direction 2
#define MOTOR_ENA           4    // Left Motor PWM (Speed)
#define MOTOR_ENB           5    // Right Motor PWM (Speed)
#define MOTOR_PWM_FREQ      5000
#define MOTOR_PWM_RES       8    // 8-bit resolution (0 to 255)
#define MOTOR_WATCHDOG_MS   500  // Fail-safe: Auto-stop if no command received for 500ms

// ── MAX30102 Optical PPG Settings ────────────────────────────────────────────
#define PPG_SAMPLE_RATE     100
#define PPG_LED_POWER       0x2F
#define PPG_SAMPLE_AVG      4
#define PPG_PULSE_WIDTH     411
#define PPG_ADC_RANGE       16384
#define HR_BUF_SIZE         10
#define FINGER_THRESHOLD    50000UL
