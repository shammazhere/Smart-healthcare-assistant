#include "audio_i2s.h"
#include "config.h"

// Global handles
volatile AudioState  g_audioState = AudioState::IDLE;
SemaphoreHandle_t    g_audioMutex = nullptr;
RingbufHandle_t      g_micTxRingBuf = nullptr;
RingbufHandle_t      g_spkRxRingBuf = nullptr;

void audioSetState(AudioState st) {
    xSemaphoreTake(g_audioMutex, portMAX_DELAY);
    g_audioState = st;
    xSemaphoreGive(g_audioMutex);
}

AudioState audioGetState() {
    xSemaphoreTake(g_audioMutex, portMAX_DELAY);
    AudioState st = g_audioState;
    xSemaphoreGive(g_audioMutex);
    return st;
}

bool audioInit() {
    g_audioMutex = xSemaphoreCreateMutex();

    const i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_RX),
        .sample_rate          = I2S_SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = I2S_DMA_BUFS,
        .dma_buf_len          = I2S_DMA_FRAME,
        .use_apll             = true,
        .tx_desc_auto_clear   = true,
        .fixed_mclk           = 0,
    };
    const i2s_pin_config_t pins = {
        .bck_io_num   = I2S_SCK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_SPK_DIN_PIN,
        .data_in_num  = I2S_MIC_SD_PIN,
    };

    if (i2s_driver_install(I2S_PORT, &cfg, 0, nullptr) != ESP_OK) return false;
    if (i2s_set_pin(I2S_PORT, &pins)                   != ESP_OK) return false;

    i2s_stop(I2S_PORT);
    i2s_zero_dma_buffer(I2S_PORT);

    g_micTxRingBuf = xRingbufferCreate(32768, RINGBUF_TYPE_BYTEBUF);
    g_spkRxRingBuf = xRingbufferCreate(32768, RINGBUF_TYPE_BYTEBUF);

    Serial.println("[AUDIO] I2S Full-Duplex Initialized (16 kHz, 16-bit)");
    return (g_micTxRingBuf != nullptr && g_spkRxRingBuf != nullptr);
}

void audioStartRecording() {
    i2s_zero_dma_buffer(I2S_PORT);
    i2s_start(I2S_PORT);
    audioSetState(AudioState::LISTENING);
    Serial.println("[AUDIO] Microphone streaming active");
}

void audioStopRecording() {
    i2s_stop(I2S_PORT);
    audioSetState(AudioState::IDLE);
    vTaskDelay(pdMS_TO_TICKS(300));
}

void audioQueuePlaybackChunk(const uint8_t *buf, size_t len) {
    if (audioGetState() == AudioState::LISTENING) return;
    if (audioGetState() == AudioState::IDLE) {
        i2s_zero_dma_buffer(I2S_PORT);
        i2s_start(I2S_PORT);
        audioSetState(AudioState::SPEAKING);
    }
    xRingbufferSend(g_spkRxRingBuf, buf, len, pdMS_TO_TICKS(20));
}

void TaskAudio(void *pvParameters) {
    (void)pvParameters;
    const size_t bufferBytes = I2S_DMA_FRAME * (I2S_BITS / 8);
    uint8_t *micBuffer = (uint8_t*)heap_caps_malloc(bufferBytes, MALLOC_CAP_DMA);
    if (!micBuffer) vTaskDelete(nullptr);

    Serial.println("[TASK] Audio DMA pinned to Core " + String(xPortGetCoreID()));

    for (;;) {
        AudioState st = audioGetState();

        if (st == AudioState::LISTENING) {
            size_t bytesRead = 0;
            if (i2s_read(I2S_PORT, micBuffer, bufferBytes, &bytesRead, pdMS_TO_TICKS(50)) == ESP_OK && bytesRead > 0) {
                xRingbufferSend(g_micTxRingBuf, micBuffer, bytesRead, 0);
            }
        } else if (st == AudioState::SPEAKING) {
            size_t rxLength = 0;
            void *rxData = xRingbufferReceiveUpTo(g_spkRxRingBuf, &rxLength, pdMS_TO_TICKS(50), bufferBytes);
            if (rxData && rxLength > 0) {
                size_t bytesWritten = 0;
                i2s_write(I2S_PORT, rxData, rxLength, &bytesWritten, pdMS_TO_TICKS(50));
                vRingbufferReturnItem(g_spkRxRingBuf, rxData);
            } else {
                i2s_stop(I2S_PORT);
                i2s_zero_dma_buffer(I2S_PORT);
                audioSetState(AudioState::IDLE);
                Serial.println("[AUDIO] Speech playback completed");
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(20));
        }
    }
}
