# ============================================================
# server.py — FastAPI + WebSocket AI Gateway
#   Hybrid Architecture:
#     - STT: Sarvam AI (saarika:v2) primary -> faster-whisper fallback
#     - LLM: Google Gemini + SQLite BM25 Patient Records RAG
#     - TTS: Sarvam AI (bulbul:v1/v2 custom voice) primary -> Edge-TTS fallback
#
#   Run: uvicorn server:app --host 0.0.0.0 --port 8765 --reload
# ============================================================
import os
import sys
import json
import math
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Form,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure backend directory and repo directory are in sys.path
_cur_dir = str(Path(__file__).resolve().parent)
_repo_dir = str(Path(__file__).resolve().parent.parent)
for _p in (_cur_dir, _repo_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=True)
else:
    load_dotenv(override=True)

try:
    import baymax_tts
    from stt_engine import (
        load_model as load_stt,
        transcribe_async,
        is_sarvam_available as is_sarvam_stt,
        pcm_to_wav_bytes,
        SUPPORTED_LANGUAGES,
        DEFAULT_MODEL_SIZE,
        DEFAULT_DEVICE,
        DEFAULT_COMPUTE_TYPE,
    )
    from llm_agent import (
        load_llm,
        generate_response,
        generate_response_with_sources,
        detect_text_language,
        get_llm_config,
        reset_history,
        DEFAULT_PATIENT_ID,
    )
    from tts_engine import (
        synthesize_chunks,
        synthesize_sarvam_audio,
        is_sarvam_available as is_sarvam_tts,
        DEFAULT_VOICE,
        DEFAULT_RATE,
        DEFAULT_PITCH,
        CHUNK_BYTES,
    )
    from rag_engine import get_store
except ImportError:
    from backend import baymax_tts
    from backend.stt_engine import (
        load_model as load_stt,
        transcribe_async,
        is_sarvam_available as is_sarvam_stt,
        pcm_to_wav_bytes,
        SUPPORTED_LANGUAGES,
        DEFAULT_MODEL_SIZE,
        DEFAULT_DEVICE,
        DEFAULT_COMPUTE_TYPE,
    )
    from backend.llm_agent import (
        load_llm,
        generate_response,
        generate_response_with_sources,
        detect_text_language,
        get_llm_config,
        reset_history,
        DEFAULT_PATIENT_ID,
    )
    from backend.tts_engine import (
        synthesize_chunks,
        synthesize_sarvam_audio,
        is_sarvam_available as is_sarvam_tts,
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
    if not is_sarvam_stt():
        log.info("Sarvam key not found; pre-loading local faster-whisper fallback...")
        stt_model_size = os.getenv("WHISPER_MODEL_SIZE", DEFAULT_MODEL_SIZE)
        stt_device = os.getenv("WHISPER_DEVICE", DEFAULT_DEVICE)
        stt_compute = os.getenv("WHISPER_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE)
        load_stt(model_size=stt_model_size, device=stt_device, compute_type=stt_compute)
    else:
        log.info("Sarvam AI STT active as primary (Whisper fallback available on demand)")

    log.info("Loading Conversational LLM client & preloading RAG store...")
    load_llm()

    try:
        store = get_store()
        log.info(f"Loaded {len(store.records)} patient health record(s) into RAG engine")
    except Exception as rag_err:
        log.warning(f"RAG store initialization note: {rag_err}")

    tts_mode = "Sarvam AI (primary)" if is_sarvam_tts() else f"Edge-TTS ({DEFAULT_VOICE})"
    stt_mode = "Sarvam AI (primary)" if is_sarvam_stt() else "faster-whisper (local)"
    log.info(f"✅ AI Voice Gateway ready — TTS: {tts_mode} | STT: {stt_mode}")

    # Asynchronously pre-warm localized Baymax intros for instant voice response
    async def _warm_intros():
        for lang in ("hi", "kn", "en"):
            txt = LOCALIZED_INTROS.get(lang)
            if txt and lang not in _INTRO_AUDIO_CACHE:
                try:
                    res = await baymax_tts._synth(txt, language=lang)
                    if res:
                        _INTRO_AUDIO_CACHE[lang] = (res, "mp3")
                        log.info(f"⚡ Pre-warmed {lang} Baymax intro audio ({len(res)} bytes)")
                except Exception as w_err:
                    log.debug(f"Intro pre-warming error for {lang}: {w_err}")
    asyncio.create_task(_warm_intros())

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
    executes STT (Sarvam/Whisper) -> Gemini RAG -> TTS (Sarvam/Edge), and streams PCM back.
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
    Multilingual STT (Sarvam/Whisper) → Gemini + RAG → TTS (Sarvam/Edge-TTS) pipeline.
    Streams synthesized 16-bit 16 kHz PCM chunks back to ESP32-S3 in real time.
    """
    # 1. Speech-to-Text (Sarvam primary / Whisper fallback)
    try:
        transcription = await transcribe_async(pcm_bytes, sample_rate=SAMPLE_RATE)
        text = str(transcription)
        detected_lang = getattr(transcription, "language", "en")
        confidence = getattr(transcription, "confidence", 0.0)
        stt_source = getattr(transcription, "source", "stt")
    except Exception as e:
        log.error(f"[{client_id}] STT error: {e}")
        text = ""
        detected_lang = "en"
        confidence = 0.0
        stt_source = "error"

    if not text:
        log.info(f"[{client_id}] Empty transcription — skipping utterance")
        return

    log.info(f"[{client_id}] Transcribed [{stt_source}] ({detected_lang}, conf={confidence:.2f}): '{text}'")

    # Send transcript & language to ESP32 / dashboard
    try:
        await ws.send_text(json.dumps({
            "transcript": text,
            "language": detected_lang,
            "confidence": round(confidence, 2),
            "source": stt_source,
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

    # 3. Text-to-Speech (Sarvam AI primary / Edge-TTS fallback) — stream PCM chunks back to ESP32 speaker
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


# ── Audio Synthesis Helper ───────────────────────────────────────────────────
async def _synthesize_to_wav(text: str, language: str = "en") -> bytes | None:
    """Synthesize text into WAV audio bytes using Sarvam AI or Edge-TTS fallback."""
    if not text or not text.strip():
        return None
    if is_sarvam_tts():
        wav_data = await synthesize_sarvam_audio(text.strip(), language=language)
        if wav_data:
            return wav_data
    chunks = []
    async for c in synthesize_chunks(text.strip(), language=language):
        chunks.append(c)
    if chunks:
        pcm_all = b"".join(chunks)
        return pcm_to_wav_bytes(pcm_all, sample_rate=SAMPLE_RATE)
    return None


# ── Web Voice Helper (Baymax-Tuned Voice & Multilingual Sarvam AI) ──────────
REGIONAL_VOICES: dict[str, str] = {
    "kn": "kn-IN-GaganNeural",
    "hi": "hi-IN-MadhurNeural",
    "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",
    "ml": "ml-IN-MidhunNeural",
    "mr": "mr-IN-ManoharNeural",
    "gu": "gu-IN-NiranjanNeural",
    "bn": "bn-IN-BashkarNeural",
    "ur": "ur-IN-SalmanNeural",
}

LOCALIZED_INTROS: dict[str, str] = {
    "en": "Hi, I am Baymax, your personal healthcare assistant.",
    "hi": "नमस्ते, मैं बेमैक्स हूँ, आपका व्यक्तिगत स्वास्थ्य सहायक।",
    "kn": "ನಮಸ್ಕಾರ, ನಾನು ಬೇಮ್ಯಾಕ್ಸ್, ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಆರೋಗ್ಯ ಸಹಾಯಕ.",
    "ta": "வணக்கம், நான் பேமேக்ஸ், உங்கள் தனிப்பட்ட சுகாதார உதவியாளர்.",
    "te": "నమస్కారం, నేను బేమాక్స్, మీ వ్యక్తిగత ఆరోగ్య సహాయకుడిని.",
    "ml": "നമസ്കാരം, ഞാൻ ബേമാക്സ്, നിങ്ങളുടെ വ്യക്തിഗത ആരോഗ്യ സഹായി.",
    "mr": "नमस्कार, मी बेमॅक्स, आपला वैयक्तिक आरोग्य सहाय्यक.",
    "gu": "નમસ્તે, હું બેમેક્સ છું, તમારો વ્યક્તિગત આરોગ્ય સહાયક.",
    "bn": "নমস্কার, আমি বেম্যাক্স, আপনার ব্যক্তিগত স্বাস্থ্য সহকারী।",
    "ur": "ہیلو، میں بے میکس ہوں، آپ کا ذاتی ہیلتھ کیئر اسسٹنٹ۔",
}

_INTRO_AUDIO_CACHE: dict[str, tuple[bytes, str]] = {}


async def _speak_web_baymax(ws: WebSocket, text: str, language: str = "en"):
    """
    Ultra-low-latency two-stage speech delivery in Baymax voice:
      - Every language (English, Kannada, Hindi, etc.) is voiced by its dedicated
        Baymax male neural voice with calm pacing (-12%) and deepened pitch (-5Hz).
      - Plays cached localized Baymax introduction immediately (0ms delay).
      - Synthesizes and chains subsequent sentences in the background with zero gap.
    """
    if not text or not text.strip():
        return

    clean_text = text.strip()
    norm_lang = (language or "en").lower().split("-")[0].split("_")[0]

    # Auto-detect Indic scripts if language is "en" or unspecified
    has_indic = any(0x0600 <= ord(c) <= 0x0DFF for c in clean_text)
    if has_indic and norm_lang == "en":
        norm_lang = detect_text_language(clean_text)

    await ws.send_text(json.dumps({"type": "status", "state": "speaking"}))
    try:
        intro_text = LOCALIZED_INTROS.get(norm_lang, "")
        seq = 0

        # Fast two-stage delivery: if text starts with intro, dispatch cached intro audio immediately (0ms)
        if intro_text and clean_text.startswith(intro_text):
            body_text = clean_text[len(intro_text):].strip()

            # 1. Dispatch pre-cached Baymax intro audio immediately
            intro_cached = _INTRO_AUDIO_CACHE.get(norm_lang)
            if not intro_cached:
                try:
                    intro_bytes = await baymax_tts._synth(intro_text, language=norm_lang)
                    if intro_bytes:
                        intro_cached = (intro_bytes, "mp3")
                        _INTRO_AUDIO_CACHE[norm_lang] = intro_cached
                except Exception as synth_err:
                    log.warning(f"Intro synthesis error: {synth_err}")

            if intro_cached:
                await ws.send_text(json.dumps({"type": "audio", "seq": seq, "format": intro_cached[1]}))
                await ws.send_bytes(intro_cached[0])
                seq += 1

            # 2. Stream remaining body sentences in Baymax voice while intro is already playing in browser
            if body_text:
                async for mp3 in baymax_tts.synthesize_sentences(body_text, language=norm_lang):
                    await ws.send_text(json.dumps({"type": "audio", "seq": seq, "format": "mp3"}))
                    await ws.send_bytes(mp3)
                    seq += 1
        else:
            # Stream entire text sentence-by-sentence in Baymax voice
            async for mp3 in baymax_tts.synthesize_sentences(clean_text, language=norm_lang):
                await ws.send_text(json.dumps({"type": "audio", "seq": seq, "format": "mp3"}))
                await ws.send_bytes(mp3)
                seq += 1

    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error(f"[Baymax Voice Error] {e}")
        await ws.send_text(json.dumps({"type": "error", "message": f"Voice is unavailable: {e}"}))
    await ws.send_text(json.dumps({"type": "done"}))
    await ws.send_text(json.dumps({"type": "status", "state": "idle"}))


# ── Web Assistant WebSocket Endpoint (/ws/web) ────────────────────────────────
@app.websocket("/ws/web")
async def web_ws(ws: WebSocket):
    """
    WebSocket endpoint for HealthBot Assistant Web UI (assistant.html).
    Coordinates streaming mic audio (16 kHz PCM) / typed queries,
    RAG health record grounding, and Baymax caring robot voice speech.
    """
    await ws.accept()
    client_id = f"web_{ws.client.host}:{ws.client.port}"
    active_connections[client_id] = ws
    log.info(f"[{client_id}] Web Assistant connected")

    client_patient_id = DEFAULT_PATIENT_ID
    audio_buffer = bytearray()

    try:
        # Initial greeting & patient catalog handshake
        store = get_store()
        patients = store.patient_list()
        sarvam_stt_on = is_sarvam_stt()
        stt_model_size = os.getenv("WHISPER_MODEL_SIZE", DEFAULT_MODEL_SIZE)

        await ws.send_text(json.dumps({
            "type": "ready",
            "patient_id": client_patient_id,
            "patients": patients,
            "stt": {
                "backend": "Sarvam AI" if sarvam_stt_on else "faster-whisper",
                "model": "saaras:v3" if sarvam_stt_on else stt_model_size,
            },
            "tts": baymax_tts.info()
        }))

        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            # 1. Handle incoming binary audio chunk from browser microphone
            if "bytes" in message and message["bytes"]:
                prev_len = len(audio_buffer)
                audio_buffer.extend(message["bytes"])
                if prev_len == 0 or (len(audio_buffer) // 32000 > prev_len // 32000):
                    log.info(f"[{client_id}] Receiving mic audio stream... buffer: {len(audio_buffer)} bytes ({len(audio_buffer)/32000:.1f}s)")
                continue

            # 2. Handle incoming JSON command from assistant UI
            if "text" in message and message["text"]:
                raw_text = message["text"].strip()
                try:
                    payload = json.loads(raw_text)
                except Exception:
                    continue

                mtype = payload.get("type", "")

                if mtype == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

                elif mtype == "patient":
                    new_pid = payload.get("id") or payload.get("patient_id")
                    if new_pid:
                        client_patient_id = new_pid
                        log.info(f"[{client_id}] Switched patient to '{client_patient_id}'")
                    await ws.send_text(json.dumps({
                        "type": "patient",
                        "patient_id": client_patient_id
                    }))

                elif mtype == "reset":
                    reset_history(client_patient_id)
                    audio_buffer.clear()
                    log.info(f"[{client_id}] Reset conversational memory for '{client_patient_id}'")

                elif mtype == "start":
                    audio_buffer.clear()
                    await ws.send_text(json.dumps({"type": "status", "state": "listening"}))

                elif mtype == "cancel":
                    audio_buffer.clear()
                    await ws.send_text(json.dumps({"type": "status", "state": "idle"}))

                elif mtype == "greet":
                    req_lang = payload.get("language", "en")
                    greetings = {
                        "en": "Hi, I am Baymax, your personal healthcare assistant. How can I help you today?",
                        "kn": "ನಮಸ್ಕಾರ, ನಾನು ಬೇಮ್ಯಾಕ್ಸ್, ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಆರೋಗ್ಯ ಸಹಾಯಕ. ನಾನು ನಿಮಗೆ ಇಂದು ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
                        "hi": "नमस्ते, मैं बेमैक्स हूँ, आपका व्यक्तिगत स्वास्थ्य सहायक। आज मैं आपकी क्या मदद कर सकता हूँ?",
                    }
                    greeting_text = greetings.get(req_lang, greetings["en"])
                    await ws.send_text(json.dumps({
                        "type": "reply",
                        "text": greeting_text,
                        "sources": [],
                        "language": req_lang,
                        "greeting": True
                    }))
                    await _speak_web_baymax(ws, greeting_text, language=req_lang)

                elif mtype == "end":
                    log.info(f"[{client_id}] Received 'end' signal, buffer: {len(audio_buffer)} bytes ({len(audio_buffer)/32000:.1f}s)")
                    if len(audio_buffer) < 3200:
                        log.warning(f"[{client_id}] Audio buffer too small ({len(audio_buffer)} bytes < 3200 bytes), sending no_speech")
                        audio_buffer.clear()
                        await ws.send_text(json.dumps({"type": "no_speech"}))
                        await ws.send_text(json.dumps({"type": "status", "state": "idle"}))
                        continue

                    await ws.send_text(json.dumps({"type": "status", "state": "transcribing"}))
                    pcm_data = bytes(audio_buffer)
                    audio_buffer.clear()
                    log.info(f"[{client_id}] Transcribing {len(pcm_data)} bytes of audio via STT...")

                    try:
                        tr = await transcribe_async(pcm_data, sample_rate=SAMPLE_RATE)
                        transcript = str(tr).strip()
                        detected_lang = getattr(tr, "language", "en")
                        log.info(f"[{client_id}] Transcription complete: '{transcript}' (lang={detected_lang})")
                    except Exception as stt_err:
                        log.error(f"[{client_id}] Web STT error: {stt_err}")
                        transcript = ""
                        detected_lang = "en"

                    # Double check script in case STT returned en for Kannada/Hindi
                    script_lang = detect_text_language(transcript)
                    if script_lang != "en":
                        detected_lang = script_lang

                    if not transcript:
                        await ws.send_text(json.dumps({"type": "no_speech"}))
                        await ws.send_text(json.dumps({"type": "status", "state": "idle"}))
                        continue

                    await ws.send_text(json.dumps({"type": "transcript", "text": transcript, "language": detected_lang}))
                    await ws.send_text(json.dumps({"type": "status", "state": "thinking"}))

                    try:
                        reply, sources = await generate_response_with_sources(
                            transcript, language=detected_lang, patient_id=client_patient_id
                        )
                    except Exception as llm_err:
                        log.error(f"[{client_id}] Web LLM error: {llm_err}")
                        reply = "I am having trouble connecting right now. Please try again."
                        sources = []

                    await ws.send_text(json.dumps({
                        "type": "reply",
                        "text": str(reply),
                        "sources": sources,
                        "language": detected_lang,
                        "greeting": False
                    }))
                    await _speak_web_baymax(ws, str(reply), language=detected_lang)

                elif mtype == "text":
                    query_text = payload.get("text", "").strip()
                    if not query_text:
                        continue

                    req_lang = payload.get("language")
                    detected_lang = req_lang or detect_text_language(query_text)

                    await ws.send_text(json.dumps({"type": "status", "state": "thinking"}))
                    try:
                        reply, sources = await generate_response_with_sources(
                            query_text, language=detected_lang, patient_id=client_patient_id
                        )
                    except Exception as llm_err:
                        log.error(f"[{client_id}] Web LLM error: {llm_err}")
                        reply = "I am having trouble connecting right now. Please try again."
                        sources = []

                    await ws.send_text(json.dumps({
                        "type": "reply",
                        "text": str(reply),
                        "sources": sources,
                        "language": detected_lang,
                        "greeting": False
                    }))
                    await _speak_web_baymax(ws, str(reply), language=detected_lang)

    except WebSocketDisconnect:
        log.info(f"[{client_id}] Web Assistant disconnected")
    except Exception as e:
        log.error(f"[{client_id}] Web Assistant WebSocket error: {e}")
    finally:
        active_connections.pop(client_id, None)
        try:
            await ws.close()
        except Exception:
            pass


# ── Dashboard Controller WebSocket Endpoint (/ws/control) ────────────────────
@app.websocket("/ws/control")
async def control_ws(ws: WebSocket):
    """
    WebSocket endpoint for HealthBot Dashboard Controller (index.html).
    Simulates live telemetry (heart rate, SpO2, finger presence, battery) and responds
    to motor navigation commands and voice/chat queries for hardware-in-the-loop and web testing.
    """
    await ws.accept()
    client_id = f"dashboard_{ws.client.host}:{ws.client.port}"
    active_connections[client_id] = ws
    log.info(f"[{client_id}] Dashboard Controller connected")

    audio_state = 0
    t_step = 0.0

    async def telemetry_broadcaster():
        nonlocal t_step
        try:
            while True:
                await asyncio.sleep(1.0)
                t_step += 0.25
                sim_hr = round(74.0 + 5.0 * math.sin(t_step), 1)
                sim_spo2 = round(98.2 + 0.6 * math.cos(t_step * 0.8), 1)
                telemetry = {
                    "finger": True,
                    "hr": sim_hr,
                    "spo2": sim_spo2,
                    "audio": audio_state,
                }
                await ws.send_text(json.dumps(telemetry))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    telemetry_task = asyncio.create_task(telemetry_broadcaster())

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                raw_text = message["text"].strip()
                if raw_text == "ping":
                    await ws.send_text(json.dumps({"pong": True}))
                    continue

                try:
                    data = json.loads(raw_text)
                except Exception:
                    continue

                if "dir" in data:
                    direction = data.get("dir", "S")
                    speed = data.get("speed", 170)
                    log.info(f"[{client_id}] Simulated Motor drive: dir={direction}, speed={speed}")
                    await ws.send_text(json.dumps({
                        "status": "ack",
                        "motor": {"dir": direction, "speed": speed}
                    }))

                elif "voice" in data:
                    v_on = bool(data.get("voice"))
                    audio_state = 1 if v_on else 0
                    log.info(f"[{client_id}] Dashboard voice toggle: {v_on}")
                    await ws.send_text(json.dumps({"audio": audio_state}))

                elif "query" in data:
                    q = data.get("query", "").strip()
                    if q:
                        audio_state = 2
                        await ws.send_text(json.dumps({"audio": 2}))
                        reply = await generate_response(q, language="en", patient_id=DEFAULT_PATIENT_ID)
                        await ws.send_text(json.dumps({"transcript": str(reply)}))
                        audio_state = 0
                        await ws.send_text(json.dumps({"audio": 0}))

    except WebSocketDisconnect:
        log.info(f"[{client_id}] Dashboard Controller disconnected")
    except Exception as e:
        log.error(f"[{client_id}] Dashboard WebSocket error: {e}")
    finally:
        telemetry_task.cancel()
        active_connections.pop(client_id, None)
        try:
            await ws.close()
        except Exception:
            pass


# ── Standalone Audio Endpoints (STT / TTS) ───────────────────────────────────
@app.post("/api/stt")
async def api_stt(
    audio: UploadFile = File(...),
    language: str = Form(default="hi-IN")
):
    """Convert audio file upload to text using Sarvam AI (with Whisper fallback)."""
    audio_data = await audio.read()
    transcription = await transcribe_async(audio_data, language=language)
    if transcription and str(transcription).strip():
        return {
            "transcript": str(transcription),
            "language": getattr(transcription, "language", language),
            "source": getattr(transcription, "source", "hybrid"),
        }
    raise HTTPException(status_code=500, detail="Failed to transcribe audio")


@app.post("/api/tts")
async def api_tts(
    text: str = Form(...),
    language: str = Form(default="hi-IN")
):
    """Convert text to speech using Sarvam AI (with Edge-TTS fallback). Returns WAV audio."""
    wav_bytes = await _synthesize_to_wav(text, language=language)
    if wav_bytes:
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    raise HTTPException(status_code=500, detail="Failed to generate speech")


@app.post("/api/tts/custom")
async def api_tts_custom(
    text: str = Form(...),
    language: str = Form(default="hi-IN"),
    voice_sample: UploadFile = File(None)
):
    """
    Convert text to speech using Sarvam AI Custom Voice Cloning (bulbul:v2).
    Pass a reference voice sample audio file.
    """
    sample_bytes = None
    if voice_sample:
        sample_bytes = await voice_sample.read()

    wav_bytes = await synthesize_sarvam_audio(
        text=text,
        language=language,
        custom_audio_bytes=sample_bytes,
    )

    if not wav_bytes:
        # Fallback to standard TTS
        chunks = []
        async for c in synthesize_chunks(text, language=language):
            chunks.append(c)
        if chunks:
            wav_bytes = pcm_to_wav_bytes(b"".join(chunks), sample_rate=SAMPLE_RATE)

    if wav_bytes:
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=custom_output.wav"}
        )
    raise HTTPException(status_code=500, detail="Failed to generate speech with custom voice")


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

    sarvam_tts_on = is_sarvam_tts()
    sarvam_stt_on = is_sarvam_stt()

    return {
        "status": "online",
        "connections": len(active_connections),
        "patients": patient_count,
        "pipeline": {
            "stt": "Sarvam AI (saarika:v2) [fallback: faster-whisper]" if sarvam_stt_on else f"faster-whisper ({stt_model_size})",
            "llm": f"Google Gemini ({llm_model})",
            "rag": f"BM25 Retrieval ({patient_count} patients)",
            "tts": f"Edge-TTS Baymax ({baymax_tts.VOICE}, rate={baymax_tts.RATE}, pitch={baymax_tts.PITCH})",
        },
        "hybrid_config": {
            "sarvam_tts_active": sarvam_tts_on,
            "sarvam_stt_active": sarvam_stt_on,
            "edge_tts_fallback": True,
            "whisper_stt_fallback": True,
        },
        "tts_config": {
            "default_voice": baymax_tts.VOICE,
            "rate": baymax_tts.RATE,
            "pitch": baymax_tts.PITCH,
            "sample_rate": SAMPLE_RATE,
            "chunk_bytes": CHUNK_BYTES,
        },
        "supported_languages": list(SUPPORTED_LANGUAGES.keys())
    }


# ── Serve Frontend Web App ────────────────────────────────────────────────────
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    @app.get("/", include_in_schema=False)
    async def serve_root():
        return FileResponse(os.path.join(frontend_dir, "assistant.html"))

    @app.get("/manifest.json", include_in_schema=False)
    async def serve_manifest():
        return FileResponse(
            os.path.join(frontend_dir, "manifest.json"),
            media_type="application/manifest+json"
        )

    @app.get("/sw.js", include_in_schema=False)
    async def serve_sw():
        return FileResponse(
            os.path.join(frontend_dir, "sw.js"),
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Service-Worker-Allowed": "/"
            }
        )

    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    log.info(f"Starting server on {host}:{port}")
    uvicorn.run("backend.server:app", host=host, port=port, reload=False)

