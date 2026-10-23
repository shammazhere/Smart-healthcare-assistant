# ============================================================
# tts_engine.py — Text-to-Speech using edge-tts
#
#   Two outputs from the same engine:
#     synthesize_sentences(text) -> MP3 bytes, one item per sentence      (browser page)
#     synthesize_chunks(text)    -> 16 kHz 16-bit mono PCM chunks         (ESP32-S3 I2S speaker)
#
#   Improvements over the basic version:
#     * speech-friendly text: "500 mg" -> "500 milligrams", "2026-11-05" -> "5 November 2026",
#       "10:30" -> "10:30 AM", phone numbers digit by digit, markdown/emoji removed
#     * sentence-by-sentence synthesis: the first sentence starts playing while the rest are generated
#     * slightly slower, calmer voice (good for elderly patients) — change with TTS_RATE
#     * retries, plus a small cache so repeated phrases are instant
#     * proper resampling (scipy) and fade in/out so there are no clicks between sentences
#
#   .env options: TTS_VOICE (default en-US-AriaNeural), TTS_RATE (default -5%), TTS_PITCH (default +0Hz)
# ============================================================
import asyncio
import io
import logging
import math
import os
import re
from collections import OrderedDict
from typing import AsyncGenerator

import edge_tts
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv(override=True)
log = logging.getLogger("tts")

TARGET_RATE = 16000     # Hz — must match the ESP32 I2S sample rate
CHUNK_BYTES = 2048      # ~64 ms of 16 kHz 16-bit mono
FADE_SAMPLES = 96       # 6 ms fade to avoid clicks
MAX_PARALLEL = 3        # sentences synthesized at the same time
CACHE_SIZE = 128

_cache: "OrderedDict[tuple, bytes]" = OrderedDict()


def _voice() -> str:
    return os.getenv("TTS_VOICE", "en-US-AriaNeural")


def _rate() -> str:
    return os.getenv("TTS_RATE", "-5%")


def _pitch() -> str:
    return os.getenv("TTS_PITCH", "+0Hz")


def info() -> dict:
    return {"engine": "edge-tts", "voice": _voice(), "rate": _rate()}


# ── Text -> speakable text ────────────────────────────────────────────────────
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

# (pattern, spoken form) — applied only to whole "words" (not inside other words)
_SPOKEN = [
    (r"ng\s*/\s*mL", "nanograms per millilitre"),
    (r"mg\s*/\s*dL", "milligrams per deciliter"),
    (r"mmHg", "millimetres of mercury"),
    (r"mcg", "micrograms"),
    (r"IU", "international units"),
    (r"mg", "milligrams"),
    (r"mL", "millilitres"),
    (r"kg", "kilograms"),
    (r"cm", "centimetres"),
    (r"bpm", "beats per minute"),
    (r"SpO[2₂]", "S P O 2"),
    (r"HbA1c", "H B A 1 C"),
    (r"eGFR", "e G F R"),
    (r"LDL", "L D L"),
    (r"OPD", "O P D"),
    (r"BP", "blood pressure"),
    (r"Dr\.?", "Doctor"),
]
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def _speak_date(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if 1 <= mo <= 12 and 1 <= d <= 31:
        return f"{d} {_MONTHS[mo - 1]} {y}"
    return m.group(0)


def _speak_time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return m.group(0)
    h12 = h % 12 or 12
    suffix = "AM" if h < 12 else "PM"
    return f"{h12} {suffix}" if mi == 0 else f"{h12}:{mi:02d} {suffix}"


def _speak_phone(m: re.Match) -> str:
    raw = m.group(0)
    groups = [g for g in re.split(r"\D+", raw) if g]
    return ("plus " if raw.startswith("+") else "") + ", ".join(" ".join(g) for g in groups)


def prepare_for_speech(text: str) -> str:
    """Rewrite text so a TTS voice reads it naturally."""
    t = text or ""
    t = re.sub(r"https?://\S+", "", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)
    t = re.sub(r"(?<![.!?:;,])\s*\n+\s*", ". ", t)
    t = re.sub(r"\s*\n+\s*", " ", t)
    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _speak_date, t)                       # before phone numbers!
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b(?!\s*[AaPp]\.?[Mm]\b)", _speak_time, t)
    t = re.sub(r"\+?\d[\d\-]{7,}\d", _speak_phone, t)
    for pat, spoken in _SPOKEN:
        t = re.sub(rf"(?<![A-Za-z/]){pat}(?![A-Za-z])", spoken, t)
    t = re.sub(r"\s*%", " percent", t)
    t = t.replace("&", " and ")
    return re.sub(r"\s{2,}", " ", t).strip()


def split_sentences(text: str, min_len: int = 25) -> list[str]:
    """Split into sentences, merging very short ones so speech does not sound choppy."""
    out: list[str] = []
    cur = ""
    for part in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        if not part:
            continue
        cur = f"{cur} {part}".strip() if cur else part
        if len(cur) >= min_len:
            out.append(cur)
            cur = ""
    if cur:
        if out:
            out[-1] += " " + cur
        else:
            out.append(cur)
    return out


# ── edge-tts synthesis ────────────────────────────────────────────────────────
async def _collect(sentence: str, voice: str, rate: str, pitch: str) -> bytes:
    communicate = edge_tts.Communicate(sentence, voice=voice, rate=rate, pitch=pitch)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


async def _synth_mp3(sentence: str, voice: str, rate: str, pitch: str) -> bytes:
    key = (voice, rate, pitch, sentence)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    last: Exception | None = None
    for attempt in range(3):
        try:
            data = await asyncio.wait_for(_collect(sentence, voice, rate, pitch), timeout=20)
            if data:
                _cache[key] = data
                while len(_cache) > CACHE_SIZE:
                    _cache.popitem(last=False)
                return data
            last = RuntimeError("empty audio")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            last = e
        await asyncio.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"edge-tts failed (is the internet connected?): {last}")


async def synthesize_sentences(text: str, voice: str | None = None) -> AsyncGenerator[bytes, None]:
    """Yield one MP3 per sentence, in order. Later sentences are generated while earlier ones play."""
    sentences = split_sentences(prepare_for_speech(text))
    if not sentences:
        return
    voice = voice or _voice()
    gate = asyncio.Semaphore(MAX_PARALLEL)

    async def job(s: str) -> bytes:
        async with gate:
            return await _synth_mp3(s, voice, _rate(), _pitch())

    tasks = [asyncio.create_task(job(s)) for s in sentences]
    try:
        for t in tasks:
            yield await t
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


# ── MP3 -> PCM for the ESP32 ──────────────────────────────────────────────────
def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    try:
        from scipy.signal import resample_poly
        g = math.gcd(src, dst)
        return resample_poly(audio, dst // g, src // g).astype(np.float32)
    except ImportError:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * dst / src))
        return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def mp3_to_pcm16(mp3: bytes, target_rate: int = TARGET_RATE) -> bytes:
    """Decode MP3 to 16-bit signed mono PCM at `target_rate`."""
    try:
        audio, sr = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=False)
    except Exception as e:
        raise RuntimeError(f"Could not decode MP3 (needs soundfile with libsndfile 1.1+): {e}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _resample(audio, sr, target_rate)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * min(0.9 / peak, 4.0)               # consistent loudness for the small speaker
    n = min(FADE_SAMPLES, audio.size // 2)
    if n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


async def synthesize_chunks(text: str, voice: str | None = None) -> AsyncGenerator[bytes, None]:
    """Yield 16 kHz 16-bit mono PCM chunks for the ESP32-S3 speaker."""
    async for mp3 in synthesize_sentences(text, voice):
        pcm = await asyncio.to_thread(mp3_to_pcm16, mp3)
        for i in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[i:i + CHUNK_BYTES]