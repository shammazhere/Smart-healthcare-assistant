#pragma once
#include <Arduino.h>
#include <driver/i2s.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/ringbuf.h>

// ── Audio Pipeline State ─────────────────────────────────────────────────────
enum class AudioState : uint8_t { IDLE = 0, LISTENING = 1, SPEAKING = 2 };

// Global handles
extern volatile AudioState  g_audioState;
extern SemaphoreHandle_t    g_audioMutex;
extern RingbufHandle_t      g_micTxRingBuf;
extern RingbufHandle_t      g_spkRxRingBuf;

// Public Audio Functions
bool audioInit();
void audioSetState(AudioState st);
AudioState audioGetState();
void audioStartRecording();
void audioStopRecording();
void audioQueuePlaybackChunk(const uint8_t *buf, size_t len);
void TaskAudio(void *pvParameters);
