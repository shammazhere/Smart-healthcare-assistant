# ============================================================
# tts_engine.py — Hybrid Text-to-Speech Engine
#   * Primary: Sarvam AI Multilingual TTS (bulbul:v1) + Custom Voice Cloning (bulbul:v2)
#   * Fallback: Edge-TTS with Baymax-inspired English and Indian regional voices
#   * Produces 16-bit 16 kHz mono little-endian PCM chunks for ESP32-S3
# ============================================================
import io
import os
import wave
import base64
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator
import requests
import numpy as np
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

try:
    import edge_tts
except ImportError:
    edge_tts = None

log = logging.getLogger(__name__)

# ── Baymax-Inspired Settings ──────────────────────────────────────────────────
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "en-US-GuyNeural")
DEFAULT_RATE  = os.getenv("TTS_DEFAULT_RATE", "-12%")   # Slower rate for deliberate pacing
DEFAULT_PITCH = os.getenv("TTS_DEFAULT_PITCH", "-5Hz")   # Slightly lower pitch for grounded warmth
TARGET_RATE   = 16000    # Hz — required by ESP32-S3 I2S speaker
CHUNK_BYTES   = 2048     # ~64ms per chunk at 16 kHz 16-bit mono

# ── Sarvam Language Code Mapping ──────────────────────────────────────────────
SARVAM_LANG_MAP: dict[str, str] = {
    "en": "en-IN", "english": "en-IN", "en-in": "en-IN", "en-us": "en-IN",
    "hi": "hi-IN", "hindi": "hi-IN", "hi-in": "hi-IN",
    "kn": "kn-IN", "kannada": "kn-IN", "kn-in": "kn-IN",
    "ml": "ml-IN", "malayalam": "ml-IN", "ml-in": "ml-IN",
    "ta": "ta-IN", "tamil": "ta-IN", "ta-in": "ta-IN",
    "te": "te-IN", "telugu": "te-IN", "te-in": "te-IN",
    "mr": "mr-IN", "marathi": "mr-IN", "mr-in": "mr-IN",
    "gu": "gu-IN", "gujarati": "gu-IN", "gu-in": "gu-IN",
    "bn": "bn-IN", "bengali": "bn-IN", "bangla": "bn-IN", "bn-in": "bn-IN",
    "pa": "pa-IN", "punjabi": "pa-IN",
    "od": "od-IN", "odia": "od-IN",
}

# ── Multilingual Voice Map for Edge-TTS Fallback ──────────────────────────────
VOICE_MAP: dict[str, str] = {
    "en": "en-US-AndrewMultilingualNeural",
    "english": "en-US-AndrewMultilingualNeural",
    "hi": "hi-IN-MadhurNeural",
    "hindi": "hi-IN-MadhurNeural",
    "kn": "kn-IN-GaganNeural",
    "kannada": "kn-IN-GaganNeural",
    "ml": "ml-IN-MidhunNeural",
    "malayalam": "ml-IN-MidhunNeural",
    "ta": "ta-IN-ValluvarNeural",
    "tamil": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",
    "telugu": "te-IN-MohanNeural",
    "mr": "mr-IN-ManoharNeural",
    "marathi": "mr-IN-ManoharNeural",
    "gu": "gu-IN-NiranjanNeural",
    "gujarati": "gu-IN-NiranjanNeural",
    "bn": "bn-IN-BashkarNeural",
    "bengali": "bn-IN-BashkarNeural",
    "ur": "ur-IN-SalmanNeural",
    "urdu": "ur-IN-SalmanNeural",
}


def is_sarvam_available() -> bool:
    """Check if SARVAM_API_KEY is configured in environment."""
    key = os.getenv("SARVAM_API_KEY", "").strip()
    return bool(key and not key.startswith("your_"))


def resolve_sarvam_language(lang: str | None) -> str:
    if not lang:
        return "en-IN"
    clean = lang.strip().lower()
    return SARVAM_LANG_MAP.get(clean, "en-IN")


def get_supported_voices() -> dict[str, str]:
    """Return dictionary of supported language-to-voice mappings."""
    return VOICE_MAP.copy()


def resolve_voice(
    language: str | None = None,
    voice: str | None = None
) -> tuple[str, str, str]:
    """Resolve Edge-TTS voice identifier, rate, and pitch configured for Baymax caring robot."""
    if voice and voice.strip():
        v = voice.strip()
        if v.lower() in VOICE_MAP:
            v = VOICE_MAP[v.lower()]
        return v, DEFAULT_RATE, DEFAULT_PITCH

    if language and language.strip():
        lang_key = language.strip().lower().split("-")[0].split("_")[0]
        v = VOICE_MAP.get(lang_key, DEFAULT_VOICE)
        return v, DEFAULT_RATE, DEFAULT_PITCH

    return DEFAULT_VOICE, DEFAULT_RATE, DEFAULT_PITCH


def decode_mp3_to_float32(mp3_buffer: io.BytesIO) -> tuple[np.ndarray, int]:
    """
    Decode in-memory MP3 audio bytes to a float32 numpy array.
    Uses soundfile as primary decoder, with fallback handling for environments
    where libsndfile lacks built-in MP3 decoding.
    """
    mp3_buffer.seek(0)
    try:
        import soundfile as sf
        audio_float, sr = sf.read(mp3_buffer, dtype="float32", always_2d=False)
        return audio_float, sr
    except Exception as sf_err:
        log.warning(f"[TTS] soundfile MP3 read failed: {sf_err}. Attempting fallback decoders...")
        mp3_buffer.seek(0)
        raw_mp3 = mp3_buffer.read()

        try:
            import miniaudio
            decoded = miniaudio.mp3_read_f32(raw_mp3)
            samples = np.array(decoded.samples, dtype=np.float32)
            if decoded.nchannels > 1:
                samples = samples.reshape(-1, decoded.nchannels)
            return samples, decoded.sample_rate
        except ImportError:
            pass

        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(raw_mp3), format="mp3")
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels)
            return samples, seg.frame_rate
        except ImportError:
            pass

        raise RuntimeError(
            f"Failed to decode MP3 stream. Ensure libsndfile with MP3 support is installed. (Error: {sf_err})"
        )


def decode_wav_to_pcm16(wav_bytes: bytes, target_sr: int = 16000) -> bytes:
    """Extract raw 16-bit 16kHz mono PCM from WAV bytes."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sr = wf.getframerate()
            raw_frames = wf.readframes(wf.getnframes())

        # If already 16kHz mono 16-bit, return directly
        if channels == 1 and sample_width == 2 and sr == target_sr:
            return raw_frames

        # Otherwise convert via numpy
        dtype = "<i2" if sample_width == 2 else "<i4"
        audio = np.frombuffer(raw_frames, dtype=dtype).astype(np.float32)
        if sample_width == 2:
            audio /= 32768.0
        elif sample_width == 4:
            audio /= 2147483648.0

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        if sr != target_sr:
            ratio = target_sr / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        return (audio * 32767.0).clip(-32768, 32767).astype("<i2").tobytes()
    except Exception as e:
        log.error(f"[TTS Error] WAV decode failed: {e}")
        return wav_bytes


# ── Primary: Sarvam AI TTS (with Custom Voice Cloning support) ────────────────
async def synthesize_sarvam_audio(
    text: str,
    language: str | None = None,
    custom_audio_path: str | None = None,
    custom_audio_bytes: bytes | None = None,
    speaker: str | None = None,
) -> bytes | None:
    """
    Call Sarvam AI TTS API.
    Supports voice cloning when a reference audio sample is supplied.
    Returns raw WAV bytes or None on failure.
    """
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        return None

    target_lang = resolve_sarvam_language(language)
    url = "https://api.sarvam.ai/text-to-speech"
    default_speaker = os.getenv("SARVAM_DEFAULT_SPEAKER", "aditya")
    use_speaker = speaker or default_speaker

    # Baymax-tuned parameters: calm male speaker, deliberate pacing (-12%), lowered pitch
    payload = {
        "inputs": [text.strip()],
        "target_language_code": target_lang,
        "speaker": use_speaker,
        "pitch": float(os.getenv("SARVAM_PITCH", "-0.2")),
        "pace": float(os.getenv("SARVAM_PACE", "0.88")),
        "loudness": 1.0,
        "speech_sample_rate": TARGET_RATE,
        "enable_preprocessing": True,
        "model": "bulbul:v3",
    }

    # Custom voice cloning support (bulbul:v2)
    ref_bytes = None
    if custom_audio_bytes:
        ref_bytes = custom_audio_bytes
    elif custom_audio_path and os.path.exists(custom_audio_path):
        with open(custom_audio_path, "rb") as f:
            ref_bytes = f.read()

    if ref_bytes:
        payload["audio"] = base64.b64encode(ref_bytes).decode("utf-8")
        payload["model"] = "bulbul:v2"
        log.info(f"[Sarvam TTS] Using custom voice cloning (bulbul:v2) for '{target_lang}'")

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }

    def _call_sarvam():
        resp = requests.post(url, json=payload, headers=headers, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        audios = data.get("audios", [])
        if audios:
            return base64.b64decode(audios[0])
        return None

    try:
        return await asyncio.to_thread(_call_sarvam)
    except Exception as e:
        log.warning(f"[Sarvam TTS Error] API call failed: {e}")
        return None


# ── Fallback: Edge-TTS Baymax-Tuned Synthesis ─────────────────────────────────
async def synthesize_edge_chunks(
    text: str,
    voice: str | None = None,
    language: str | None = None,
    rate: str | None = None,
    pitch: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Synthesize speech using Microsoft Edge-TTS with Baymax voice customization."""
    if not edge_tts:
        log.error("[TTS Error] edge-tts package not available for fallback")
        return

    resolved_voice, default_rate, default_pitch = resolve_voice(language=language, voice=voice)
    use_rate = rate if rate is not None else default_rate
    use_pitch = pitch if pitch is not None else default_pitch

    mp3_buffer = io.BytesIO()
    try:
        comm = edge_tts.Communicate(
            text=text.strip(),
            voice=resolved_voice,
            rate=use_rate,
            pitch=use_pitch,
        )
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])
    except Exception as e:
        log.error(f"[Edge-TTS Error] Synthesis failed: {e}")
        return

    mp3_buffer.seek(0)
    if mp3_buffer.getbuffer().nbytes == 0:
        return

    # Convert MP3 to PCM using decode_mp3_to_float32
    try:
        audio_float, sr = decode_mp3_to_float32(mp3_buffer)
        if audio_float.ndim > 1:
            audio_float = audio_float.mean(axis=1)
        if sr != TARGET_RATE:
            ratio = TARGET_RATE / sr
            new_len = int(len(audio_float) * ratio)
            indices = np.linspace(0, len(audio_float) - 1, new_len)
            audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float)
        pcm_bytes = (audio_float * 32767.0).clip(-32768, 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm_bytes), CHUNK_BYTES):
            yield pcm_bytes[i : i + CHUNK_BYTES]
    except Exception as sf_err:
        log.error(f"[Edge-TTS Error] Audio decoding failed: {sf_err}")


# ── Unified Hybrid Stream Generator ───────────────────────────────────────────
async def synthesize_chunks(
    text: str,
    voice: str | None = None,
    language: str | None = None,
    rate: str | None = None,
    pitch: str | None = None,
    custom_audio_path: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """
    Hybrid streaming TTS:
      - For English / default: Uses the custom Baymax voice (Edge-TTS calm warm delivery).
      - For Indian regional languages (or if Sarvam custom voice sample is supplied): Uses Sarvam AI.
    Yields 16-bit 16 kHz signed mono little-endian PCM chunks (2048 bytes).
    """
    if not text or not text.strip():
        return

    lang_clean = (language or "en").strip().lower()
    custom_ref = custom_audio_path or os.getenv("SARVAM_CUSTOM_VOICE_SAMPLE")

    # If custom voice reference is provided, or an Indian language is requested with Sarvam available:
    if is_sarvam_available() and (custom_ref or (lang_clean not in ("en", "english", "en-us", "en-in"))):
        wav_data = await synthesize_sarvam_audio(
            text=text,
            language=language,
            custom_audio_path=custom_ref
        )
        if wav_data:
            pcm_bytes = decode_wav_to_pcm16(wav_data, target_sr=TARGET_RATE)
            if pcm_bytes:
                log.info(f"[TTS] Streamed {len(pcm_bytes)} PCM bytes via Sarvam AI ({language})")
                for i in range(0, len(pcm_bytes), CHUNK_BYTES):
                    yield pcm_bytes[i : i + CHUNK_BYTES]
                return
        log.warning("[TTS] Sarvam synthesis unavailable — falling back to Edge-TTS")

    # Baymax-tuned English / Edge-TTS
    log.info(f"[TTS] Synthesizing via Edge-TTS Baymax voice ({language or 'en'})")
    async for chunk in synthesize_edge_chunks(
        text=text,
        voice=voice,
        language=language,
        rate=rate,
        pitch=pitch,
    ):
        yield chunk
