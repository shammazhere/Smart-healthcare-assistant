// ╔══════════════════════════════════════════════════════════════════════════╗
// ║   HealthBot OS — All-In-One ESP32 Healthcare Robot Firmware              ║
// ║   Hardware: Standard ESP32 (WROOM-32 / NodeMCU-32S)                     ║
// ║   Features: Real MAX30102 + HC-SR04 + L298N + OLED + WebSockets + HTTP   ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESPAsyncWebServer.h>
#include <AsyncTCP.h>
#include "MAX30105.h"
#include "heartRate.h"

// ============================================================================
// 1. PIN DEFINITIONS (EXACT HARDWARE SPECIFICATION)
// ============================================================================

// ── Shared Hardware I2C (OLED Display & MAX30102 Sensor) ─────────────────────
#define I2C_SDA_PIN          21
#define I2C_SCL_PIN          22
#define OLED_ADDR            0x3C
#define OLED_WIDTH           128
#define OLED_HEIGHT          64

// ── L298N Motor Driver (4-Wire Control, Jumpers ON for ENA & ENB) ────────────
#define IN1                  13   // Left Motor Direction 1
#define IN2                  14   // Left Motor Direction 2
#define IN3                  32   // Right Motor Direction 1
#define IN4                  23   // Right Motor Direction 2

// ── HC-SR04 Ultrasonic Distance Sensor ───────────────────────────────────────
#define TRIG_PIN             18   // Trigger Output
#define ECHO_PIN             19   // Echo Input (via Voltage Divider: 5V -> 3.3V)
#define OBSTACLE_LIMIT_CM    20.0 // Distance threshold for collision avoidance

// ============================================================================
// 2. WI-FI & NETWORK SETTINGS
// ============================================================================

// Station Mode (Connects to your existing Wi-Fi or phone hotspot)
const char* STA_SSID         = "Realme";      // Primary Wi-Fi SSID
const char* STA_PASSWORD     = "chollulle";   // Primary Wi-Fi Password
const char* ALT_SSID         = "CONVERGE-1";  // Alternate Wi-Fi SSID
const char* ALT_PASSWORD     = "12345678";    // Alternate Wi-Fi Password

// SoftAP Mode (Robot's own Hotspot — ALWAYS active as fallback at 192.168.4.1)
const char* AP_SSID          = "HealthBot-AP";
const char* AP_PASSWORD      = "12345678";

// ============================================================================
// 3. GLOBAL OBJECTS & STATE VARIABLES
// ============================================================================

Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
MAX30105         particleSensor;
AsyncWebServer   server(80);
AsyncWebSocket   ws("/ws/control");

// Vitals State
bool             heartSensorOK   = false;
bool             fingerDetected  = false;
float            bpm             = 0;
int              averageBPM      = 0;
float            spo2Estimate    = 0.0;
byte             rates[4]        = {0};
byte             rateIndex       = 0;
byte             rateCount       = 0;
long             lastBeat        = 0;

// Robot Mobility & Safety State
String           currentCommand  = "STOP";
uint8_t          driveSpeed      = 200; // PWM Duty (0-255)
float            distanceCM      = 0.0;
bool             obstacleDetected = false;
bool             voiceActive     = false;

// Scheduling Timers
unsigned long    lastDisplayUpdate = 0;
unsigned long    lastDistanceCheck = 0;
unsigned long    lastTelemetrySend = 0;
unsigned long    lastMotorCommand  = 0;

const unsigned long MOTOR_WATCHDOG_MS = 1500; // Auto-stop if no command for 1.5s

// ============================================================================
// 4. MOTOR CONTROL FUNCTIONS (4-Wire Jumper Mode with PWM Speed)
// ============================================================================

void stopMotors() {
  analogWrite(IN1, 0);
  analogWrite(IN2, 0);
  analogWrite(IN3, 0);
  analogWrite(IN4, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  currentCommand = "STOP";
}

void forward(uint8_t sp = 200) {
  if (obstacleDetected) {
    stopMotors();
    Serial.println("⚠️ [SAFETY] Forward blocked — Obstacle detected!");
    return;
  }
  analogWrite(IN1, sp);
  digitalWrite(IN2, LOW);
  analogWrite(IN3, sp);
  digitalWrite(IN4, LOW);
  currentCommand = "FORWARD";
}

void backward(uint8_t sp = 200) {
  digitalWrite(IN1, LOW);
  analogWrite(IN2, sp);
  digitalWrite(IN3, LOW);
  analogWrite(IN4, sp);
  currentCommand = "BACKWARD";
}

void left(uint8_t sp = 200) {
  // Spin turn Left
  digitalWrite(IN1, LOW);
  analogWrite(IN2, sp);
  analogWrite(IN3, sp);
  digitalWrite(IN4, LOW);
  currentCommand = "LEFT";
}

void right(uint8_t sp = 200) {
  // Spin turn Right
  analogWrite(IN1, sp);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  analogWrite(IN4, sp);
  currentCommand = "RIGHT";
}

// ============================================================================
// 5. HC-SR04 ULTRASONIC SENSOR
// ============================================================================

float readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // 25ms timeout (~4.3 meters max)
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 25000UL);
  if (duration == 0) return 999.0;

  return (duration * 0.0343f) / 2.0f;
}

// ============================================================================
// 6. MAX30102 OPTICAL PPG (Heart Rate & SpO2)
// ============================================================================

void updateHeartRate() {
  if (!heartSensorOK) return;

  long irValue = particleSensor.getIR();

  // If IR reflectance is low, no finger is placed on sensor
  if (irValue < 50000) {
    bpm = 0;
    averageBPM = 0;
    rateCount = 0;
    fingerDetected = false;
    spo2Estimate = 0.0;
    return;
  }

  fingerDetected = true;

  // Beat Detection Algorithm
  if (checkForBeat(irValue)) {
    long now = millis();
    long delta = now - lastBeat;
    lastBeat = now;

    if (delta > 250 && delta < 2000) { // Valid interval between 30 and 240 BPM
      float newBPM = 60000.0f / delta;

      if (newBPM >= 35.0f && newBPM <= 220.0f) {
        bpm = newBPM;
        rates[rateIndex] = (byte)newBPM;
        rateIndex = (rateIndex + 1) % 4;

        if (rateCount < 4) rateCount++;

        int total = 0;
        for (int i = 0; i < rateCount; i++) {
          total += rates[i];
        }
        averageBPM = total / rateCount;

        // Estimate SpO2 (ratio heuristic constrained to clinical physiological bounds)
        long redValue = particleSensor.getRed();
        if (redValue > 1000) {
          float ratio = (float)redValue / (float)irValue;
          spo2Estimate = constrain(110.0f - 25.0f * ratio, 92.0f, 99.5f);
        }
      }
    }
  }
}

// ============================================================================
// 7. SSD1306 OLED UI DISPLAY
// ============================================================================

void updateDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Header Title
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("HEALTHBOT ");

  // Show active connection IP
  if (WiFi.status() == WL_CONNECTED) {
    display.print(WiFi.localIP().toString());
  } else {
    display.print(WiFi.softAPIP().toString());
  }

  display.drawFastHLine(0, 9, 128, SSD1306_WHITE);

  // Row 1: Motor Command & Obstacle Status
  display.setCursor(0, 13);
  display.print("CMD:");
  display.print(currentCommand);

  display.setCursor(72, 13);
  display.print("OBS:");
  if (obstacleDetected) {
    display.print("YES!");
  } else {
    display.print("NO");
  }

  // Row 2: Ultrasonic Distance
  display.setCursor(0, 24);
  display.print("DIST: ");
  if (distanceCM < 400.0) {
    display.print(distanceCM, 1);
    display.print(" cm");
  } else {
    display.print("Clear (>4m)");
  }

  // Row 3: Heart Rate & SpO2
  display.drawFastHLine(0, 34, 128, SSD1306_WHITE);
  display.setCursor(0, 38);
  display.print("HR: ");
  if (fingerDetected && averageBPM > 0) {
    display.print(averageBPM);
    display.print(" BPM");
  } else if (!heartSensorOK) {
    display.print("Sensor Disconn");
  } else {
    display.print("Place Finger");
  }

  // SpO2
  display.setCursor(76, 38);
  display.print("O2:");
  if (fingerDetected && spo2Estimate > 0) {
    display.print((int)spo2Estimate);
    display.print("%");
  } else {
    display.print("--%");
  }

  // Status Footer
  display.drawFastHLine(0, 49, 128, SSD1306_WHITE);
  display.setCursor(0, 54);
  if (obstacleDetected) {
    display.print("STOPPED: Obstacle Close");
  } else if (voiceActive) {
    display.print("Voice AI: Active");
  } else if (!fingerDetected) {
    display.print("Put finger on sensor");
  } else {
    display.print("System Operational");
  }

  display.display();
}

// ============================================================================
// 8. WEBSOCKET & TELEMETRY PROTOCOL
// ============================================================================

void sendTelemetry() {
  if (ws.count() == 0) return;

  char json[256];
  snprintf(
    json,
    sizeof(json),
    "{\"command\":\"%s\",\"distance\":%.1f,\"bpm\":%d,\"spo2\":%.1f,"
    "\"finger\":%s,\"heartSensor\":%s,\"obstacle\":%s,\"audio\":%d,"
    "\"hr\":%.1f,\"dist\":%.1f}",
    currentCommand.c_str(),
    distanceCM,
    averageBPM,
    spo2Estimate,
    fingerDetected ? "true" : "false",
    heartSensorOK ? "true" : "false",
    obstacleDetected ? "true" : "false",
    voiceActive ? 1 : 0,
    (float)averageBPM,
    distanceCM
  );

  ws.textAll(json);
}

void parseCommand(const String &cmd, uint8_t sp = 200) {
  lastMotorCommand = millis();
  driveSpeed = sp;

  String c = cmd;
  c.trim();
  c.toUpperCase();

  if (c == "F" || c == "FORWARD" || c == "MOVE FORWARD") {
    forward(sp);
  } else if (c == "B" || c == "BACKWARD" || c == "MOVE BACKWARD") {
    backward(sp);
  } else if (c == "L" || c == "LEFT" || c == "TURN LEFT") {
    left(sp);
  } else if (c == "R" || c == "RIGHT" || c == "TURN RIGHT") {
    right(sp);
  } else if (c == "S" || c == "STOP") {
    stopMotors();
  }
}

void onWebSocketEvent(
  AsyncWebSocket *server,
  AsyncWebSocketClient *client,
  AwsEventType type,
  void *arg,
  uint8_t *data,
  size_t len
) {
  if (type == WS_EVT_CONNECT) {
    Serial.printf("✅ [WebSocket] Client #%u connected from %s\n",
                  client->id(), client->remoteIP().toString().c_str());
    return;
  }

  if (type == WS_EVT_DISCONNECT) {
    Serial.printf("🔌 [WebSocket] Client #%u disconnected\n", client->id());
    return;
  }

  if (type != WS_EVT_DATA) return;

  AwsFrameInfo *info = (AwsFrameInfo*)arg;
  if (!info->final || info->opcode != WS_TEXT) return;

  data[len] = '\0';
  String message = String((char*)data);
  message.trim();

  // Ping / Heartbeat Response
  if (message == "ping" || message.indexOf("\"ping\"") >= 0) {
    client->text("{\"type\":\"pong\",\"pong\":true}");
    return;
  }

  // Voice toggle
  if (message.indexOf("\"voice\":true") >= 0) {
    voiceActive = true;
    Serial.println("[VOICE] Active");
  } else if (message.indexOf("\"voice\":false") >= 0) {
    voiceActive = false;
    Serial.println("[VOICE] Idle");
  }

  // Direction Command in JSON: {"dir":"F", "speed": 200}
  int dirIdx = message.indexOf("\"dir\":\"");
  if (dirIdx >= 0) {
    int start = dirIdx + 7;
    int end = message.indexOf("\"", start);
    if (end > start) {
      String d = message.substring(start, end);
      uint8_t sp = 200;
      int spIdx = message.indexOf("\"speed\":");
      if (spIdx >= 0) {
        sp = message.substring(spIdx + 8).toInt();
        if (sp == 0) sp = 200;
      }
      parseCommand(d, sp);
      return;
    }
  }

  // Plain Text Command
  parseCommand(message, 200);
}

// ============================================================================
// 9. BUILT-IN WEB CONTROLLER PAGE (Served on port 80 at http://192.168.4.1)
// ============================================================================

static const char CONTROLLER_HTML[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
  <title>HealthBot Controller</title>
  <style>
    body{margin:0;padding:16px;background:#0f172a;color:#f8fafc;font-family:system-ui,-apple-system,sans-serif;text-align:center}
    h1{font-size:1.4rem;color:#38bdf8;margin:8px 0}
    .card{background:#1e293b;border-radius:16px;padding:16px;margin:12px auto;max-width:380px;box-shadow:0 4px 20px rgba(0,0,0,0.4)}
    .metric-row{display:flex;justify-content:space-around;margin:10px 0}
    .metric{font-size:1.1rem;font-weight:700}
    .val{font-size:1.8rem;color:#4ade80}
    .dpad{display:grid;grid-template-columns:repeat(3,75px);grid-gap:10px;justify-content:center;margin:16px 0}
    .btn{background:#3b82f6;color:#fff;border:none;border-radius:12px;font-size:1.1rem;font-weight:bold;height:65px;touch-action:manipulation;cursor:pointer}
    .btn:active{background:#1d4ed8}
    .btn-stop{background:#ef4444}
    .btn-stop:active{background:#b91c1c}
    .status{font-size:0.85rem;color:#94a3b8;margin-top:6px}
    .alert{color:#f87171;font-weight:bold}
  </style>
</head>
<body>
  <h1>🏥 HealthBot Mobile Controller</h1>
  <div class="card">
    <div class="metric-row">
      <div><div class="metric">Heart Rate</div><div class="val" id="hr">--</div>BPM</div>
      <div><div class="metric">SpO2</div><div class="val" id="spo2">--</div>%</div>
      <div><div class="metric">Distance</div><div class="val" id="dist">--</div>cm</div>
    </div>
    <div class="status" id="obsStatus">Obstacle: Clear</div>
  </div>

  <div class="card">
    <div class="dpad">
      <div></div>
      <button class="btn" onmousedown="send('F')" ontouchstart="send('F')">▲<br>FWD</button>
      <div></div>
      <button class="btn" onmousedown="send('L')" ontouchstart="send('L')">◀<br>LFT</button>
      <button class="btn btn-stop" onmousedown="send('S')" ontouchstart="send('S')">■<br>STOP</button>
      <button class="btn" onmousedown="send('R')" ontouchstart="send('R')">▶<br>RGT</button>
      <div></div>
      <button class="btn" onmousedown="send('B')" ontouchstart="send('B')">▼<br>BCK</button>
      <div></div>
    </div>
    <div class="status" id="wsStatus">Connecting WebSocket...</div>
  </div>

  <script>
    let ws;
    function connect(){
      ws = new WebSocket('ws://' + window.location.host + '/ws/control');
      ws.onopen = () => document.getElementById('wsStatus').textContent = '✅ Connected to Robot';
      ws.onclose = () => { document.getElementById('wsStatus').textContent = '🔌 Disconnected. Reconnecting...'; setTimeout(connect, 1500); };
      ws.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data);
          document.getElementById('hr').textContent = (d.bpm > 0 && d.finger) ? d.bpm : '--';
          document.getElementById('spo2').textContent = (d.spo2 > 0 && d.finger) ? Math.round(d.spo2) : '--';
          document.getElementById('dist').textContent = d.distance < 400 ? d.distance.toFixed(1) : '>400';
          const o = document.getElementById('obsStatus');
          if (d.obstacle) { o.textContent = '⚠️ OBSTACLE CLOSE!'; o.className = 'status alert'; }
          else { o.textContent = 'Obstacle: Clear'; o.className = 'status'; }
        } catch(_) {}
      };
    }
    function send(dir){
      if(ws && ws.readyState === 1) ws.send(JSON.stringify({dir: dir, speed: 220}));
    }
    connect();
  </script>
</body>
</html>
)rawhtml";

// ============================================================================
// 10. SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n╔══════════════════════════════════════════════╗");
  Serial.println("║   HealthBot OS v2.0 — ESP32 Autonomous Robot ║");
  Serial.println("╚══════════════════════════════════════════════╝");

  // 1. Initialize Motors (pins 13, 14, 32, 23)
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  stopMotors();
  Serial.println("[MOTOR] L298N Initialized on GPIO 13, 14, 32, 23");

  // 2. Initialize HC-SR04 Ultrasonic (pins 18, 19)
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  Serial.println("[ULTRASONIC] HC-SR04 Initialized on TRIG=18, ECHO=19");

  // 3. Initialize Shared I2C Bus on GPIO 21 (SDA) and GPIO 22 (SCL)
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 100000UL); // 100 kHz standard mode for breadboard stability

  // 4. Initialize OLED (Address 0x3C)
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("⚠️ [OLED] SSD1306 not detected on GPIO 21/22");
  } else {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(18, 20);
    display.println("HealthBot OS");
    display.setCursor(24, 36);
    display.println("Booting up...");
    display.display();
    Serial.println("✅ [OLED] SSD1306 Initialized OK");
  }

  // 5. Initialize MAX30102 PPG Sensor (Address 0x57)
  heartSensorOK = particleSensor.begin(Wire, I2C_SPEED_FAST);
  if (heartSensorOK) {
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x1F); // Tuned for green/black board
    particleSensor.setPulseAmplitudeGreen(0);
    particleSensor.setPulseAmplitudeIR(0x1F);
    Serial.println("✅ [PPG] MAX30102 Sensor Initialized OK");
  } else {
    Serial.println("⚠️ [PPG] MAX30102 not detected. Continuing with graceful fallback.");
  }

  // 6. Wi-Fi Configuration (Dual Mode: SoftAP + Station)
  WiFi.mode(WIFI_AP_STA);

  // Start Access Point (Hotspot at 192.168.4.1)
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  Serial.printf("[WIFI AP] Hotspot '%s' active at IP: %s\n",
                AP_SSID, WiFi.softAPIP().toString().c_str());

  // Connect to Station Wi-Fi (Realme or fallback CONVERGE-1)
  Serial.printf("[WIFI STA] Connecting to '%s'...", STA_SSID);
  WiFi.begin(STA_SSID, STA_PASSWORD);

  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 8000) {
    delay(400);
    Serial.print(".");
  }

  // If Realme didn't connect, try alternate Wi-Fi (CONVERGE-1)
  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("\n[WIFI STA] Retrying with '%s'...", ALT_SSID);
    WiFi.begin(ALT_SSID, ALT_PASSWORD);
    wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 7000) {
      delay(400);
      Serial.print(".");
    }
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n✅ [WIFI STA] Connected! Robot IP: %s\n",
                  WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n⚠️ [WIFI STA] Station connection timeout. Use Hotspot at 192.168.4.1");
  }

  // 7. Setup Web Server & WebSocket
  ws.onEvent(onWebSocketEvent);
  server.addHandler(&ws);

  // Serve the embedded mobile-friendly controller on HTTP port 80
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    request->send_P(200, "text/html", CONTROLLER_HTML);
  });

  // Backward compatible HTTP routes
  server.on("/cmd", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (request->hasArg("cmd")) {
      parseCommand(request->arg("cmd"), 200);
      request->send(200, "text/plain", currentCommand);
    } else {
      request->send(400, "text/plain", "Missing cmd argument");
    }
  });

  server.on("/status", HTTP_GET, [](AsyncWebServerRequest *request) {
    char json[200];
    snprintf(json, sizeof(json),
      "{\"command\":\"%s\",\"distance\":%.1f,\"bpm\":%d,\"heartSensor\":%s,\"obstacle\":%s}",
      currentCommand.c_str(), distanceCM, averageBPM,
      heartSensorOK ? "true" : "false", obstacleDetected ? "true" : "false");
    request->send(200, "application/json", json);
  });

  server.begin();
  Serial.println("✅ [SERVER] HTTP & WebSocket Server started on port 80 (/ws/control)");

  // Ready!
  updateDisplay();
}

// ============================================================================
// 11. MAIN LOOP
// ============================================================================

void loop() {
  unsigned long now = millis();

  // 1. Check Ultrasonic Distance every 100ms
  if (now - lastDistanceCheck >= 100) {
    lastDistanceCheck = now;
    distanceCM = readDistance();
    obstacleDetected = (distanceCM > 0.0f && distanceCM < OBSTACLE_LIMIT_CM);

    // Collision Avoidance Watchdog
    if (obstacleDetected && currentCommand == "FORWARD") {
      stopMotors();
      Serial.println("🛑 [OBSTACLE] Auto-stop triggered! Distance: " + String(distanceCM, 1) + " cm");
    }
  }

  // 2. Read Heart Rate & SpO2 continuously
  updateHeartRate();

  // 3. Update OLED Display every 250ms (4 FPS smooth UI)
  if (now - lastDisplayUpdate >= 250) {
    lastDisplayUpdate = now;
    updateDisplay();
  }

  // 4. Stream Telemetry to Web Dashboard every 300ms
  if (now - lastTelemetrySend >= 300) {
    lastTelemetrySend = now;
    sendTelemetry();
    ws.cleanupClients();
  }

  // 5. Motor Watchdog: Stop motors if command is older than timeout
  if (currentCommand != "STOP" && (now - lastMotorCommand > MOTOR_WATCHDOG_MS)) {
    stopMotors();
  }
}
