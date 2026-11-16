# ============================================================
# stt_engine.py — Hybrid Multilingual Speech-to-Text
#   * Primary: Sarvam AI STT / Translate (saarika:v2)
#   * Fallback: Local faster-whisper (small, int8 CPU/GPU)
# ============================================================
import os
import io
import wave
import asyncio
import logging
import threading
from pathlib import Path
import requests
import numpy as np
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

log = logging.getLogger(__name__)

# ── Configurable Defaults ─────────────────────────────────────────────────────
DEFAULT_MODEL_SIZE   = os.getenv("WHISPER_MODEL_SIZE", "small")
DEFAULT_DEVICE       = os.getenv("WHISPER_DEVICE", "cpu")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ── Supported Languages ───────────────────────────────────────────────────────
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "ur": "Urdu",
    "pa": "Punjabi",
    "od": "Odia",
}

LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-in": "en", "en-gb": "en",
    "hindi": "hi", "hi-in": "hi",
    "kannada": "kn", "kn-in": "kn",
    "malayalam": "ml", "ml-in": "ml",
    "tamil": "ta", "ta-in": "ta",
    "telugu": "te", "te-in": "te",
    "marathi": "mr", "mr-in": "mr",
    "gujarati": "gu", "gu-in": "gu",
    "bengali": "bn", "bangla": "bn", "bn-in": "bn", "bn-bd": "bn",
    "urdu": "ur", "ur-in": "ur", "ur-pk": "ur",
    "punjabi": "pa", "pa-in": "pa",
    "odia": "od", "od-in": "od",
}

# ── Whisper Model Singleton & Concurrency Lock ────────────────────────────────
_model = None
_model_lock = threading.Lock()


def is_sarvam_available() -> bool:
    """Check if SARVAM_API_KEY is configured in environment."""
    key = os.getenv("SARVAM_API_KEY", "").strip()
    return bool(key and not key.startswith("your_"))


def normalize_language_code(lang: str | None, confidence: float = 1.0) -> str:
    """Normalize language identifiers to standard 2-letter codes."""
    if not lang:
        return "en"
    clean = str(lang).strip().lower()
    code = LANGUAGE_ALIASES.get(clean, clean)
    if code in SUPPORTED_LANGUAGES:
        if code != "en" and confidence < 0.35:
            return "en"
        return code
    return "en"


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Pack raw 16-bit PCM bytes into standard in-memory WAV container."""
    usable_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes[:usable_len])
    return buf.getvalue()


class TranscriptionResult(str):
    """
    Subclass of str retaining transcription metadata (.language, .confidence, .source).
    100% backward compatible with string operations and tuple unpacking.
    """
    language: str
    confidence: float
    raw_text: str
    source: str

    def __new__(cls, text: str, language: str = "en", confidence: float = 0.0, source: str = "whisper"):
        s = super().__new__(cls, text)
        s.language = language
        s.confidence = confidence
        s.raw_text = text
        s.source = source
        return s

    def __iter__(self):
        return iter((self.raw_text, self.language, self.confidence))


# ── Primary: Sarvam AI STT (saarika:v2) ───────────────────────────────────────
async def transcribe_sarvam(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    language: str | None = None
) -> TranscriptionResult | None:
    """Transcribe audio via Sarvam AI saaras:v3 API with automatic Indian language detection."""
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        return None

    wav_bytes = pcm_to_wav_bytes(pcm_bytes, sample_rate=sample_rate)
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": api_key}

    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {
        "model": "saaras:v3",
        "with_timestamps": "false",
    }
    if language and language.strip() and language.strip().lower() not in ("auto", "unknown", "none"):
        clean_l = language.strip().lower()
        data["language_code"] = f"{clean_l}-IN" if len(clean_l) == 2 else clean_l

    def _call_sarvam_stt():
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    try:
        result = await asyncio.to_thread(_call_sarvam_stt)
        transcript = result.get("transcript") or result.get("translation") or ""
        transcript = transcript.strip()
        raw_lang = result.get("language_code") or (language or "en")
        detected_lang = normalize_language_code(raw_lang.split("-")[0].lower())
        prob = float(result.get("language_probability", 0.98))

        if transcript:
            log.info(f"[Sarvam STT] Transcribed ({detected_lang}): '{transcript}'")
            return TranscriptionResult(transcript, detected_lang, confidence=prob, source="sarvam")
        return None
    except Exception as e:
        log.warning(f"[Sarvam STT Error] API call failed: {e}")
        return None


# ── Fallback: Faster-Whisper Local Inference ──────────────────────────────────
def load_model(model_size: str | None = None, device: str | None = None, compute_type: str | None = None):
    """Load faster-whisper model into memory (thread-safe singleton)."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from faster_whisper import WhisperModel
            size = model_size or DEFAULT_MODEL_SIZE
            dev  = device or DEFAULT_DEVICE
            ctype = compute_type or DEFAULT_COMPUTE_TYPE
            log.info(f"[STT] Loading local Whisper model '{size}' on {dev} ({ctype})...")
            _model = WhisperModel(size, device=dev, compute_type=ctype)
            log.info(f"[STT] Local Whisper model '{size}' ready")
            return _model
        except Exception as e:
            log.warning(f"[STT Warning] Could not load faster-whisper: {e}")
            return None


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if not pcm_bytes:
        return np.array([], dtype=np.float32)
    usable_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable_len == 0:
        return np.array([], dtype=np.float32)
    audio_np = np.frombuffer(pcm_bytes[:usable_len], dtype="<i2").astype(np.float32) / 32768.0
    if not np.isfinite(audio_np).all():
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)
    return audio_np


def transcribe_whisper(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1
) -> TranscriptionResult:
    """Local faster-whisper transcription fallback."""
    if _model is None:
        load_model()
    if not pcm_bytes or _model is None:
        return TranscriptionResult("", "en", 0.0, source="fallback")

    audio_np = pcm_to_float32(pcm_bytes)
    if len(audio_np) < int(sample_rate * 0.3):
        return TranscriptionResult("", "en", 0.0, source="fallback")

    try:
        segments, info = _model.transcribe(audio_np, beam_size=3, vad_filter=True)
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        full_text = " ".join(text_parts).strip()
        detected_lang = normalize_language_code(
            info.language,
            confidence=getattr(info, "language_probability", 0.0)
        )
        confidence = float(getattr(info, "language_probability", 0.0))
        log.info(f"[Whisper STT] Transcribed ({detected_lang}, conf={confidence:.2f}): '{full_text}'")
        return TranscriptionResult(full_text, detected_lang, confidence, source="whisper")
    except Exception as e:
        log.error(f"[Whisper STT Error] Transcription failed: {e}")
        return TranscriptionResult("", "en", 0.0, source="fallback")


# ── Unified Hybrid Transcription ──────────────────────────────────────────────
async def transcribe_async(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    language: str | None = None,
) -> TranscriptionResult:
    """
    Hybrid Speech-to-Text:
      1. Primary: Sarvam AI STT (saarika:v2) if SARVAM_API_KEY is configured
      2. Fallback: Local faster-whisper
    """
    if is_sarvam_available():
        res = await transcribe_sarvam(pcm_bytes, sample_rate=sample_rate, language=language)
        if res and str(res).strip():
            return res
        log.warning("[STT] Sarvam STT unavailable or empty — falling back to local Whisper")

    return await asyncio.to_thread(transcribe_whisper, pcm_bytes, sample_rate, channels)


def transcribe(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> TranscriptionResult:
    """Synchronous transcribe helper."""
    return transcribe_whisper(pcm_bytes, sample_rate, channels)
