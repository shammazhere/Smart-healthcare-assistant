# ============================================================
# baymax_tts.py — soft, slow, gentle "caring robot" voice (edge-tts)
#   A calm low male voice, slowed down and pitched slightly down.
#   Tune it with .env:  BAYMAX_VOICE, BAYMAX_RATE, BAYMAX_PITCH, BAYMAX_VOLUME
#   Good voices to try: en-US-GuyNeural (default), en-US-ChristopherNeural,
#                       en-US-AndrewNeural, en-US-BrianNeural, en-US-EricNeural
# ============================================================
import asyncio
import os
import re

import edge_tts

VOICE = os.getenv("BAYMAX_VOICE", "en-US-GuyNeural")
RATE = os.getenv("BAYMAX_RATE", os.getenv("TTS_DEFAULT_RATE", "-12%"))
PITCH = os.getenv("BAYMAX_PITCH", os.getenv("TTS_DEFAULT_PITCH", "-5Hz"))
VOLUME = os.getenv("BAYMAX_VOLUME", "+0%")

# Multilingual male neural voices tuned for Baymax caring robot persona
BAYMAX_VOICE_MAP: dict[str, str] = {
    "en": "en-US-GuyNeural",
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


def detect_lang_script(text: str) -> str:
    """Detect language code from Unicode script."""
    for c in text:
        cp = ord(c)
        if 0x0C80 <= cp <= 0x0CFF:
            return "kn"
        elif 0x0900 <= cp <= 0x097F:
            return "hi"
        elif 0x0B80 <= cp <= 0x0BFF:
            return "ta"
        elif 0x0C00 <= cp <= 0x0C7F:
            return "te"
        elif 0x0D00 <= cp <= 0x0D7F:
            return "ml"
        elif 0x0980 <= cp <= 0x09FF:
            return "bn"
        elif 0x0A80 <= cp <= 0x0AFF:
            return "gu"
        elif 0x0600 <= cp <= 0x06FF:
            return "ur"
    return "en"


def get_baymax_voice(language: str | None = None, text: str | None = None) -> str:
    """Resolve the Baymax male neural voice for the specified language or text script."""
    lang = (language or "en").lower().split("-")[0].split("_")[0]
    if lang in ("en", "english", "auto") and text:
        detected = detect_lang_script(text)
        if detected != "en":
            lang = detected
    return BAYMAX_VOICE_MAP.get(lang, VOICE)


def info(language: str = "en") -> dict:
    v = get_baymax_voice(language)
    return {"engine": "edge-tts", "voice": v, "rate": RATE, "pitch": PITCH}


# ── make written text sound natural when spoken ───────────────────────────────
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_SPOKEN = [
    (r"\bmcg\b", "micrograms"), (r"\bmg\b", "milligrams"), (r"\bml\b", "millilitres"),
    (r"\bkg\b", "kilograms"), (r"\bcm\b", "centimetres"), (r"\bbpm\b", "beats per minute"),
    (r"\bmmHg\b", "millimetres of mercury"), (r"\bIU\b", "international units"),
    (r"SpO2|SpO₂", "S P O 2"), (r"\bHbA1c\b", "H B A 1 C"), (r"\bLDL\b", "L D L"),
    (r"\bINR\b", "I N R"), (r"\bTSH\b", "T S H"), (r"\bECG\b", "E C G"),
]


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _say(text: str) -> str:
    def date(m):
        mo = int(m.group(2))
        return f"{_MONTHS[mo - 1]} {_ordinal(int(m.group(3)))}, {m.group(1)}" if 1 <= mo <= 12 else m.group(0)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", date, text)
    for pat, spoken in _SPOKEN:
        text = re.sub(pat, spoken, text)
    return re.sub(r"(\d)\s*%", r"\1 percent", text)


def split_sentences(text: str, max_len: int = 220) -> list[str]:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    out: list[str] = []
    buf = ""
    for p in parts:                                   # glue very short sentences together
        if buf and len(buf) + len(p) < 60:
            buf += " " + p
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    final: list[str] = []
    for s in out:                                     # break very long ones at commas
        while len(s) > max_len and "," in s[:max_len]:
            cut = s.rfind(",", 0, max_len) + 1
            final.append(s[:cut].strip())
            s = s[cut:].strip()
        final.append(s)
    return final


async def _synth(sentence: str, language: str | None = None) -> bytes:
    clean_s = sentence.strip()
    target_voice = get_baymax_voice(language=language, text=clean_s)
    norm_lang = (language or "en").lower().split("-")[0].split("_")[0]
    spoken_text = _say(clean_s) if norm_lang == "en" else clean_s

    try:
        comm = edge_tts.Communicate(spoken_text, target_voice, rate=RATE, pitch=PITCH, volume=VOLUME)
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        if buf:
            return bytes(buf)
    except Exception:
        pass

    # Resilient fallback to GuyNeural
    fallback_voice = "en-US-GuyNeural"
    comm = edge_tts.Communicate(spoken_text, fallback_voice, rate=RATE, pitch="-5Hz", volume=VOLUME)
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    if not buf:
        raise RuntimeError("the voice service returned no audio (check your internet connection)")
    return bytes(buf)


async def synthesize_sentences(text: str, language: str = "en"):
    """Yield one audio chunk per sentence in Baymax voice. Later sentences are prepared while earlier ones play."""
    tasks = [asyncio.create_task(_synth(s, language=language)) for s in split_sentences(text)]
    try:
        for t in tasks:
            yield await t
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()