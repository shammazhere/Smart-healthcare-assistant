# ============================================================
# stt_engine.py — Speech-to-Text
#
#   Two backends (choose with STT_BACKEND in .env):
#     groq   -> Groq's hosted Whisper (fast, accurate, nothing to download)   [default when GROQ_API_KEY is set]
#     local  -> faster-whisper running on this PC (works offline)
#   If Groq fails (no internet, rate limit) and faster-whisper is installed, it falls back to local.
#
#   Improvements over the basic version:
#     * removes DC offset and boosts quiet mics (INMP441 is very quiet)
#     * skips silence without wasting an API call
#     * "vocabulary hint": the patient's medicine and doctor names are given to Whisper,
#       so "Amlodipine" is not heard as "I'm low deep in"
#     * drops Whisper's classic hallucinations on near-silence ("Thank you.")
#     * runs off the event loop, so the server keeps handling other connections
#
#   .env options: STT_BACKEND, STT_LANGUAGE (default en), GROQ_STT_MODEL, WHISPER_MODEL
# ============================================================
import asyncio
import importlib.util
import io
import logging
import os
import re
import wave

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)
log = logging.getLogger("stt")

SAMPLE_RATE = 16000
MIN_SECONDS = 0.35       # shorter clips are ignored
SILENCE_RMS = 0.004      # quieter than this (0..1 scale) counts as silence
MAX_GAIN = 12.0          # never boost more than this
MAX_PROMPT_CHARS = 600   # Whisper only uses ~224 tokens of prompt

_local_model = None
_groq_client = None


# ── Configuration ─────────────────────────────────────────────────────────────
def backend() -> str:
    b = os.getenv("STT_BACKEND", "").strip().lower()
    if b in ("groq", "local"):
        return b
    return "groq" if os.getenv("GROQ_API_KEY") else "local"


def _language() -> str | None:
    return os.getenv("STT_LANGUAGE", "en").strip() or None   # None = auto-detect


def _groq_model() -> str:
    return os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")


def _local_model_name() -> str:
    name = os.getenv("WHISPER_MODEL", "base.en")
    if name.endswith(".en") and _language() not in ("en", None):
        name = name[:-3]            # English-only model can't do other languages
    return name


def info() -> dict:
    b = backend()
    return {"backend": b,
            "model": _groq_model() if b == "groq" else _local_model_name(),
            "language": _language() or "auto"}


# ── Model / client loading ────────────────────────────────────────────────────
def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from openai import AsyncOpenAI
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (needed for STT_BACKEND=groq)")
        _groq_client = AsyncOpenAI(api_key=key,
                                   base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    return _groq_client


def _load_local(model_size: str | None = None, device: str | None = None):
    global _local_model
    if _local_model is None:
        from faster_whisper import WhisperModel
        size = model_size or _local_model_name()
        dev = device or os.getenv("WHISPER_DEVICE", "cpu")
        print(f"[STT] Loading local Whisper model: {size} on {dev}")
        _local_model = WhisperModel(size, device=dev, compute_type="int8")
        print("[STT] Local model ready")
    return _local_model


def load_model(model_size: str | None = None, device: str | None = None):
    """Prepare the chosen backend. Called once when the server starts."""
    if backend() == "groq":
        _get_groq_client()
        print(f"[STT] Using Groq Whisper API ({_groq_model()}, language={_language() or 'auto'})")
    else:
        _load_local(model_size, device)


def _local_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


# ── Audio preparation ─────────────────────────────────────────────────────────
def prepare_audio(pcm_bytes: bytes) -> tuple[np.ndarray, float]:
    """16-bit PCM bytes -> cleaned float32 audio (-1..1) and the original loudness (RMS)."""
    pcm_bytes = pcm_bytes[: len(pcm_bytes) // 2 * 2]
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return audio, 0.0
    audio -= audio.mean()                                   # remove DC offset
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = np.clip(audio * min(0.9 / peak, MAX_GAIN), -1.0, 1.0)   # bring quiet mics up
    return audio, rms


def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── Transcript cleanup ────────────────────────────────────────────────────────
# Whisper often "hears" these in silence or room noise.
_HALLUCINATIONS = {
    "thank you", "thanks", "thank you for watching", "thanks for watching", "bye",
    "you", "please subscribe", "subtitles by the amara.org community",
}


def clean_transcript(text: str, duration: float, rms: float) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    key = t.lower().strip(" .!?,")
    if key in _HALLUCINATIONS and (duration < 1.5 or rms < 0.015):
        log.info("Dropped likely hallucination: %r", t)
        return ""
    return t


def _keep_segment(no_speech_prob, avg_logprob) -> bool:
    if no_speech_prob is not None and avg_logprob is not None:
        if no_speech_prob > 0.6 and avg_logprob < -1.0:
            return False
    return True


# ── Backends ──────────────────────────────────────────────────────────────────
def _transcribe_local(audio: np.ndarray, prompt: str) -> str:
    model = _load_local()
    segments, _ = model.transcribe(
        audio, beam_size=3, language=_language(), vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False, temperature=0.0,
        initial_prompt=prompt or None, no_speech_threshold=0.6)
    parts = [s.text.strip() for s in segments if _keep_segment(s.no_speech_prob, s.avg_logprob)]
    return " ".join(p for p in parts if p)


async def _transcribe_groq(audio: np.ndarray, prompt: str) -> str:
    client = _get_groq_client()
    kwargs = dict(model=_groq_model(), file=("speech.wav", to_wav_bytes(audio), "audio/wav"),
                  response_format="verbose_json", temperature=0.0)
    if _language():
        kwargs["language"] = _language()
    if prompt:
        kwargs["prompt"] = prompt
    resp = await client.audio.transcriptions.create(**kwargs)
    data = resp.model_dump() if hasattr(resp, "model_dump") else {"text": getattr(resp, "text", "")}
    segments = data.get("segments") or []
    if segments:
        parts = [(s.get("text") or "").strip() for s in segments
                 if _keep_segment(s.get("no_speech_prob"), s.get("avg_logprob"))]
        return " ".join(p for p in parts if p)
    return (data.get("text") or "").strip()


# ── Public API ────────────────────────────────────────────────────────────────
async def transcribe_async(pcm_bytes: bytes, prompt: str = "") -> str:
    """
    Transcribe raw 16 kHz, 16-bit, mono PCM. Returns "" for silence or too-short audio.
    `prompt` is an optional vocabulary hint (e.g. the patient's medicine names).
    """
    audio, rms = prepare_audio(pcm_bytes)
    duration = audio.size / SAMPLE_RATE
    if duration < MIN_SECONDS:
        return ""
    if rms < SILENCE_RMS:
        log.info("Audio is silent (rms=%.4f) — skipped", rms)
        return ""

    prompt = (prompt or "")[:MAX_PROMPT_CHARS]
    if backend() == "groq":
        try:
            text = await _transcribe_groq(audio, prompt)
        except Exception as e:
            if not _local_available():
                raise
            log.warning("Groq STT failed (%s) — falling back to local Whisper", e)
            text = await asyncio.to_thread(_transcribe_local, audio, prompt)
    else:
        text = await asyncio.to_thread(_transcribe_local, audio, prompt)

    text = clean_transcript(text, duration, rms)
    print(f"[STT] Transcribed: '{text}'")
    return text


def transcribe(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> str:
    """Blocking version for scripts. Do not call this from inside an async function."""
    return asyncio.run(transcribe_async(pcm_bytes))