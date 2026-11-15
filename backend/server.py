# ============================================================
# server.py — FastAPI + WebSocket AI Gateway
#   The ESP32-S3 connects here to stream audio, get AI replies
#   grounded in patient records (RAG), and receive synthesized
#   multilingual speech back in real time.
#
#   Run: uvicorn server:app --host 0.0.0.0 --port 8765 --reload
# ============================================================
import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from stt_engine import (
        load_model as load_stt,
        transcribe_async,
        SUPPORTED_LANGUAGES,
        DEFAULT_MODEL_SIZE,
        DEFAULT_DEVICE,
        DEFAULT_COMPUTE_TYPE,
    )
    from llm_agent import (
        load_llm,
        generate_response,
        generate_response_with_sources,
        get_llm_config,
        DEFAULT_PATIENT_ID,
    )
    from tts_engine import (
        synthesize_chunks,
        DEFAULT_VOICE,
        DEFAULT_RATE,
        DEFAULT_PITCH,
        CHUNK_BYTES,
    )
    from rag_engine import get_store
except ImportError:
    from backend.stt_engine import (
        load_model as load_stt,
        transcribe_async,
        SUPPORTED_LANGUAGES,
        DEFAULT_MODEL_SIZE,
        DEFAULT_DEVICE,
        DEFAULT_COMPUTE_TYPE,
    )
    from backend.llm_agent import (
        load_llm,
        generate_response,
        generate_response_with_sources,
        get_llm_config,
        DEFAULT_PATIENT_ID,
    )
    from backend.tts_engine import (
        synthesize_chunks,
        DEFAULT_VOICE,
        DEFAULT_RATE,
        DEFAULT_PITCH,
        CHUNK_BYTES,
    )
    from backend.rag_engine import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Startup / Shutdown lifespan ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Multilingual STT model (Whisper)...")
    stt_model_size = os.getenv("WHISPER_MODEL_SIZE", DEFAULT_MODEL_SIZE)
    stt_device = os.getenv("WHISPER_DEVICE", DEFAULT_DEVICE)
    stt_compute = os.getenv("WHISPER_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE)
    load_stt(model_size=stt_model_size, device=stt_device, compute_type=stt_compute)

    log.info("Loading Conversational LLM client & preloading RAG store...")
    load_llm()

    try:
        store = get_store()
        log.info(f"Loaded {len(store.records)} patient health record(s) into RAG engine")
    except Exception as rag_err:
        log.warning(f"RAG store initialization note: {rag_err}")

    log.info(f"✅ AI Voice Gateway ready — default voice: {DEFAULT_VOICE} ({DEFAULT_RATE})")
    yield
    log.info("Shutting down AI Gateway")

app = FastAPI(title="Healthcare Robot AI Gateway", lifespan=lifespan)

# ── Active connections registry ────────────────────────────────────────────────
active_connections: dict[str, WebSocket] = {}

# ── Audio Constants ───────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
SILENCE_THRESH_S = 0.8      # seconds of silence to trigger STT
BYTES_PER_SAMPLE = 2        # 16-bit = 2 bytes
SILENCE_BYTES    = int(SILENCE_THRESH_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
MAX_BUFFER_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 30  # 30s max utterance


@app.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    """
    WebSocket endpoint for ESP32-S3 full-duplex audio stream.
    Receives raw 16-bit 16 kHz PCM chunks, detects silence,
    executes STT -> Gemini (with RAG records) -> Edge-TTS, and streams PCM chunks back.
    """
    await ws.accept()
    client_id = str(ws.client)
    active_connections[client_id] = ws
    log.info(f"[{client_id}] ESP32-S3 connected")

    audio_buffer = bytearray()
    last_nonsilent_byte_count = 0
    client_patient_id = DEFAULT_PATIENT_ID

    try:
        while True:
            # Receive frame (handles binary PCM chunks and text frames)
            message = await asyncio.wait_for(ws.receive(), timeout=10.0)

            if message["type"] == "websocket.disconnect":
                break

            # Handle text frame (e.g. ping or query from web/test client)
            if "text" in message and message["text"]:
                raw_text = message["text"].strip()
                if raw_text == "ping":
                    await ws.send_text(json.dumps({"pong": True}))
                    continue
                try:
                    payload = json.loads(raw_text)
                    if "patient_id" in payload:
                        client_patient_id = payload["patient_id"]
                    if "query" in payload:
                        q_text = payload["query"]
                        q_lang = payload.get("language", "en")
                        q_pid = payload.get("patient_id", client_patient_id)
                        reply = await generate_response(q_text, language=q_lang, patient_id=q_pid)
                        await ws.send_text(json.dumps({
                            "transcript": q_text,
                            "reply": reply,
                            "language": q_lang,
                            "patient_id": q_pid,
                        }))
                except Exception as parse_err:
                    log.debug(f"[{client_id}] Non-JSON text message: {raw_text[:50]}")
                continue

            # Handle binary PCM chunk from microphone
            data = message.get("bytes")
            if not data:
                continue

            audio_buffer.extend(data)

            # Energy-based silence detection
            import numpy as np
            usable_bytes = len(data) - (len(data) % 2)
            if usable_bytes > 0:
                chunk_np = np.frombuffer(data[:usable_bytes], dtype="<i2").astype(np.float32)
                energy = float(np.sqrt(np.mean(chunk_np ** 2)))
            else:
                energy = 0.0

            if energy > 500:  # Active speech detected
                last_nonsilent_byte_count = len(audio_buffer)

            silence_bytes = len(audio_buffer) - last_nonsilent_byte_count

            # Trigger pipeline when silence is detected after speech
            if silence_bytes >= SILENCE_BYTES and last_nonsilent_byte_count > 0:
                utterance = bytes(audio_buffer[:last_nonsilent_byte_count])
                # Clear buffer immediately so subsequent audio isn't corrupted
                audio_buffer.clear()
                last_nonsilent_byte_count = 0

                log.info(f"[{client_id}] Utterance captured ({len(utterance)} bytes) — processing...")
                await process_utterance(ws, utterance, client_id, patient_id=client_patient_id)

            elif len(audio_buffer) > MAX_BUFFER_BYTES:
                log.warning(f"[{client_id}] Buffer overflow (>30s) — flushing")
                audio_buffer.clear()
                last_nonsilent_byte_count = 0

    except WebSocketDisconnect:
        log.info(f"[{client_id}] Disconnected")
    except asyncio.TimeoutError:
        log.warning(f"[{client_id}] Timeout — no audio received for 10s")
    except Exception as e:
        log.error(f"[{client_id}] WebSocket error: {e}")
    finally:
        active_connections.pop(client_id, None)
        try:
            await ws.close()
        except Exception:
            pass


async def process_utterance(ws: WebSocket, pcm_bytes: bytes, client_id: str, patient_id: str = DEFAULT_PATIENT_ID):
    """
    Multilingual STT → Gemini (with RAG context) → Edge-TTS pipeline for one user utterance.
    Streams synthesized 16-bit 16 kHz PCM chunks back to ESP32-S3 in real time.
    """
    # 1. Speech-to-Text (runs in thread pool to keep event loop responsive)
    try:
        transcription = await transcribe_async(pcm_bytes, sample_rate=SAMPLE_RATE)
        text = str(transcription)
        detected_lang = getattr(transcription, "language", "en")
        confidence = getattr(transcription, "confidence", 0.0)
    except Exception as e:
        log.error(f"[{client_id}] STT error: {e}")
        text = ""
        detected_lang = "en"
        confidence = 0.0

    if not text:
        log.info(f"[{client_id}] Empty transcription — skipping utterance")
        return

    log.info(f"[{client_id}] Transcribed ({detected_lang}, conf={confidence:.2f}): '{text}'")

    # Send transcript & language to ESP32 / dashboard
    try:
        await ws.send_text(json.dumps({
            "transcript": text,
            "language": detected_lang,
            "confidence": round(confidence, 2),
            "patient_id": patient_id,
        }))
    except Exception as send_err:
        log.warning(f"[{client_id}] Failed to send transcript frame: {send_err}")

    # 2. LLM Response Generation (Gemini + RAG) in the patient's detected language
    try:
        reply = await generate_response(text, language=detected_lang, patient_id=patient_id)
    except Exception as e:
        log.error(f"[{client_id}] LLM error: {e}")
        reply = "I'm sorry, I couldn't process that. Please try again."

    llm_source = getattr(reply, "source", "unknown")
    llm_latency_ms = getattr(reply, "latency_s", 0.0) * 1000.0
    log.info(
        f"[{client_id}] LLM reply ({detected_lang}) in {llm_latency_ms:.1f}ms "
        f"[source={llm_source}]: '{reply}'"
    )

    # 3. Text-to-Speech (Edge-TTS) — stream PCM chunks back to ESP32 speaker
    try:
        chunk_count = 0
        async for pcm_chunk in synthesize_chunks(reply, language=detected_lang):
            if ws.client_state.value != 1:  # Client disconnected
                log.warning(f"[{client_id}] Client disconnected during TTS streaming")
                break
            await ws.send_bytes(pcm_chunk)
            chunk_count += 1
            await asyncio.sleep(0)  # Yield to event loop for smooth network streaming

        log.info(f"[{client_id}] Sent {chunk_count} PCM chunks ({detected_lang}) to robot")
    except Exception as e:
        log.error(f"[{client_id}] TTS streaming error: {e}")


# ── RAG Patient Records HTTP APIs ─────────────────────────────────────────────
@app.get("/api/patients")
async def api_patients():
    """List all patients registered in the SQLite database."""
    return get_store().patient_list()


@app.get("/api/patients/{pid}")
async def api_patient(pid: str):
    """Retrieve full summary overview for a specific patient."""
    overview = get_store().overview(pid)
    if overview is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return overview


class AskBody(BaseModel):
    text: str
    patient_id: str | None = None
    language: str | None = None


@app.post("/api/ask")
async def api_ask(body: AskBody):
    """
    Ask a question against patient health records.
    Returns response text along with retrieved RAG record chunks.
    """
    lang = body.language or "en"
    reply, sources = await generate_response_with_sources(
        user_text=body.text,
        language=lang,
        patient_id=body.patient_id or DEFAULT_PATIENT_ID
    )
    return {
        "reply": str(reply),
        "sources": sources,
        "language": lang,
        "patient_id": body.patient_id or DEFAULT_PATIENT_ID,
    }


# ── Direct HTTP Chat API (Enables testing without microphone) ─────────────────
class ChatRequest(BaseModel):
    text: str
    language: str | None = None
    patient_id: str | None = None

class ChatResponse(BaseModel):
    query: str
    reply: str
    language: str
    patient_id: str | None = None
    source: str | None = None
    llm_latency_ms: float | None = None
    total_latency_ms: float | None = None


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """
    Direct text chat endpoint with high-precision latency profiling and RAG grounding.
    """
    import time
    t0 = time.perf_counter()

    lang = req.language or "en"
    pid = req.patient_id or DEFAULT_PATIENT_ID
    reply = await generate_response(req.text, language=lang, patient_id=pid)

    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    llm_latency_ms = getattr(reply, "latency_s", 0.0) * 1000.0
    source = getattr(reply, "source", "gemini")
    error_cat = getattr(reply, "error_category", None)
    status_code = getattr(reply, "status_code", None)

    # Diagnostic log without exposing patient text
    error_info = f" | Error: {error_cat} (HTTP {status_code})" if error_cat else ""
    log.info(
        f"[/api/chat] Total: {total_latency_ms:.1f}ms | LLM: {llm_latency_ms:.1f}ms | "
        f"Source: {source} | Lang: {lang} | Patient: {pid}{error_info}"
    )

    return ChatResponse(
        query=req.text,
        reply=str(reply),
        language=lang,
        patient_id=pid,
        source=source,
        llm_latency_ms=round(llm_latency_ms, 1),
        total_latency_ms=round(total_latency_ms, 1)
    )


# ── HTTP Health Check ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    _, llm_model, _ = get_llm_config()
    stt_model_size = os.getenv("WHISPER_MODEL_SIZE", DEFAULT_MODEL_SIZE)
    patient_count = 0
    try:
        patient_count = len(get_store().records)
    except Exception:
        pass

    return {
        "status": "online",
        "connections": len(active_connections),
        "patients": patient_count,
        "pipeline": {
            "stt": f"faster-whisper ({stt_model_size})",
            "llm": f"Google Gemini ({llm_model})",
            "rag": f"BM25 Retrieval ({patient_count} patients)",
            "tts": f"Edge-TTS ({DEFAULT_VOICE})",
        },
        "tts_config": {
            "default_voice": DEFAULT_VOICE,
            "rate": DEFAULT_RATE,
            "pitch": DEFAULT_PITCH,
            "sample_rate": SAMPLE_RATE,
            "chunk_bytes": CHUNK_BYTES,
        },
        "supported_languages": list(SUPPORTED_LANGUAGES.keys())
    }


# ── Serve Frontend Web App ────────────────────────────────────────────────────
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
