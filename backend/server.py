# ============================================================
# server.py — FastAPI AI Gateway
#   /ws/audio      ESP32-S3 robot:   PCM audio in  -> PCM speech out
#   /ws/web        Browser page:     mic or text in -> text + MP3 speech out
#   /api/...       Patient data and text Q&A over plain HTTP
#
#   Run: uvicorn server:app --host 0.0.0.0 --port 8765 --reload
# ============================================================
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketState

import stt_engine
import tts_engine
from llm_agent import DEFAULT_PATIENT_ID, generate_response_with_sources, load_llm
from rag_engine import get_store

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Startup / Shutdown lifespan ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Preparing speech-to-text...")
    stt_engine.load_model()
    log.info("Loading LLM client and patient records...")
    load_llm()
    log.info("✅ AI Gateway ready")
    yield
    log.info("Shutting down AI Gateway")


app = FastAPI(title="Healthcare Robot AI Gateway", lifespan=lifespan)

active_connections: dict[str, WebSocket] = {}

SAMPLE_RATE      = 16000
SILENCE_THRESH_S = 0.8      # ESP32 path: seconds of silence that end an utterance
BYTES_PER_SAMPLE = 2
SILENCE_BYTES    = int(SILENCE_THRESH_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
MAX_BUFFER_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 30   # 30 s max utterance


# ════════════════════════════════════════════════════════════════════════════
#  ESP32-S3 robot  (/ws/audio)  — unchanged behaviour, now with better STT/TTS
# ════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    client_id = str(ws.client)
    active_connections[client_id] = ws
    log.info(f"[{client_id}] ESP32-S3 connected")

    audio_buffer = bytearray()
    last_nonsilent_byte_count = 0

    try:
        while True:
            data = await asyncio.wait_for(ws.receive_bytes(), timeout=10.0)
            if not data:
                continue

            audio_buffer.extend(data)

            chunk_np = np.frombuffer(data[: len(data) // 2 * 2], dtype=np.int16).astype(np.float32)
            energy = float(np.sqrt(np.mean(chunk_np ** 2))) if chunk_np.size else 0.0
            if energy > 500:  # tune to your mic gain
                last_nonsilent_byte_count = len(audio_buffer)

            silence_bytes = len(audio_buffer) - last_nonsilent_byte_count

            if silence_bytes >= SILENCE_BYTES and last_nonsilent_byte_count > 0:
                utterance = bytes(audio_buffer[:last_nonsilent_byte_count])
                audio_buffer.clear()
                last_nonsilent_byte_count = 0
                log.info(f"[{client_id}] Utterance captured ({len(utterance)} bytes) — processing...")
                await process_utterance(ws, utterance, client_id)

            elif len(audio_buffer) > MAX_BUFFER_BYTES:
                log.warning(f"[{client_id}] Buffer overflow — flushing")
                audio_buffer.clear()
                last_nonsilent_byte_count = 0

    except WebSocketDisconnect:
        log.info(f"[{client_id}] Disconnected")
    except asyncio.TimeoutError:
        log.warning(f"[{client_id}] Timeout — no audio received for 10s")
    except Exception as e:
        log.error(f"[{client_id}] Error: {e}")
    finally:
        active_connections.pop(client_id, None)
        try:
            await ws.close()
        except Exception:
            pass


async def process_utterance(ws: WebSocket, pcm_bytes: bytes, client_id: str):
    """STT -> RAG + LLM -> TTS for one utterance; speech is streamed back to the ESP32."""
    store = get_store()
    try:
        text = await stt_engine.transcribe_async(pcm_bytes, prompt=store.vocabulary_hint(DEFAULT_PATIENT_ID))
    except Exception as e:
        log.error(f"[STT Error] {e}")
        text = ""

    if not text:
        log.info(f"[{client_id}] Empty transcription — skipping")
        return
    log.info(f"[{client_id}] Transcribed: '{text}'")
    try:
        await ws.send_text(json.dumps({"transcript": text}))
    except Exception:
        pass

    reply, _ = await generate_response_with_sources(text, DEFAULT_PATIENT_ID)
    log.info(f"[{client_id}] LLM reply: '{reply}'")
    try:
        await ws.send_text(json.dumps({"reply": reply}))
    except Exception:
        pass

    try:
        chunk_count = 0
        async for pcm_chunk in tts_engine.synthesize_chunks(reply):
            if ws.client_state != WebSocketState.CONNECTED:
                break
            await ws.send_bytes(pcm_chunk)
            chunk_count += 1
            await asyncio.sleep(0)
        log.info(f"[{client_id}] Sent {chunk_count} audio chunks to robot")
    except Exception as e:
        log.error(f"[TTS Error] {e}")


# ════════════════════════════════════════════════════════════════════════════
#  Browser assistant page  (/ws/web)
#
#  Browser -> server:  binary  = 16 kHz 16-bit mono PCM while recording
#                      {"type":"start"}                   begin a recording (also interrupts a reply)
#                      {"type":"end"}                     recording finished -> transcribe + answer
#                      {"type":"text","text":"..."}       typed question
#                      {"type":"patient","id":"..."}      switch patient
#                      {"type":"cancel"} / {"type":"ping"}
#  Server -> browser:  ready, status, transcript, no_speech, reply(+sources),
#                      audio (JSON header, then one binary MP3 frame), done, error, info
# ════════════════════════════════════════════════════════════════════════════
MAX_WEB_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 30


class WebSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.pid = DEFAULT_PATIENT_ID
        self.buf = bytearray()
        self.recording = False
        self.task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def send(self, obj: dict):
        async with self._lock:
            await self.ws.send_text(json.dumps(obj))

    async def send_audio(self, seq: int, mp3: bytes):
        async with self._lock:     # header and audio frame must stay together
            await self.ws.send_text(json.dumps({"type": "audio", "seq": seq, "format": "mp3"}))
            await self.ws.send_bytes(mp3)

    async def cancel(self):
        t, self.task = self.task, None
        if t and not t.done():
            t.cancel()
            try:
                await t
            except BaseException:
                pass

    def run(self, coro):
        self.task = asyncio.create_task(self._guard(coro))

    async def _guard(self, coro):
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("web pipeline error")
            try:
                await self.send({"type": "error", "message": str(e)})
                await self.send({"type": "status", "state": "idle"})
            except Exception:
                pass

    # -- pipeline steps --------------------------------------------------------
    async def voice(self, pcm: bytes):
        if len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE) < 0.35:
            await self.send({"type": "no_speech"})
            return
        await self.send({"type": "status", "state": "transcribing"})
        text = await stt_engine.transcribe_async(pcm, prompt=get_store().vocabulary_hint(self.pid))
        if not text:
            await self.send({"type": "no_speech"})
            return
        await self.send({"type": "transcript", "text": text})
        await self.answer(text)

    async def answer(self, text: str):
        await self.send({"type": "status", "state": "thinking"})
        reply, sources = await generate_response_with_sources(text, self.pid)
        await self.send({"type": "reply", "text": reply, "sources": sources})
        await self.send({"type": "status", "state": "speaking"})
        try:
            seq = 0
            async for mp3 in tts_engine.synthesize_sentences(reply):
                await self.send_audio(seq, mp3)
                seq += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"[TTS Error] {e}")
            await self.send({"type": "error", "message": f"Voice is unavailable: {e}"})
        await self.send({"type": "done"})


def _patients() -> list[dict]:
    return get_store().patient_list()


@app.websocket("/ws/web")
async def web_ws(ws: WebSocket):
    await ws.accept()
    s = WebSession(ws)
    store = get_store()
    if s.pid not in store.records and store.records:
        s.pid = next(iter(store.records))
    log.info("[web] browser connected")
    await s.send({"type": "ready", "patient_id": s.pid, "patients": _patients(),
                  "stt": stt_engine.info(), "tts": tts_engine.info()})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if s.recording and len(s.buf) < MAX_WEB_BYTES:
                    s.buf.extend(msg["bytes"])
                continue
            raw = msg.get("text")
            if raw is None:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = data.get("type")

            if kind == "start":
                await s.cancel()                  # interrupts any reply that is still being spoken
                s.buf.clear()
                s.recording = True
            elif kind == "end":
                s.recording = False
                pcm = bytes(s.buf)
                s.buf.clear()
                await s.cancel()
                s.run(s.voice(pcm))
            elif kind == "text":
                text = str(data.get("text", "")).strip()[:500]
                if text:
                    await s.cancel()
                    s.run(s.answer(text))
            elif kind == "patient":
                pid = str(data.get("id", ""))
                if get_store().has_patient(pid):
                    await s.cancel()
                    s.pid = pid
                    await s.send({"type": "patient", "patient_id": pid})
            elif kind == "cancel":
                s.recording = False
                s.buf.clear()
                await s.cancel()
                await s.send({"type": "status", "state": "idle"})
            elif kind == "ping":
                await s.send({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await s.cancel()
        log.info("[web] browser disconnected")


# ════════════════════════════════════════════════════════════════════════════
#  HTTP API
# ════════════════════════════════════════════════════════════════════════════
class AskBody(BaseModel):
    text: str
    patient_id: str | None = None


@app.get("/api/patients")
async def api_patients():
    return _patients()


@app.get("/api/patients/{pid}")
async def api_patient(pid: str):
    o = get_store().overview(pid)
    if o is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return o


@app.post("/api/ask")
async def api_ask(body: AskBody):
    """Ask a question in text and get the answer + the records it used (no audio)."""
    reply, sources = await generate_response_with_sources(body.text, body.patient_id)
    return {"reply": reply, "sources": sources}


@app.get("/health")
async def health():
    return {
        "status": "online",
        "connections": len(active_connections),
        "patients": len(get_store().records),
        "stt": stt_engine.info(),
        "tts": tts_engine.info(),
        "pipeline": "Whisper → records (SQLite) + LLM → edge-tts",
    }


# ── Serve Frontend Web App (must stay LAST so it doesn't shadow the routes above) ──
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")