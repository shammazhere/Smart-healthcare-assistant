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
RATE = os.getenv("BAYMAX_RATE", "-12%")
PITCH = os.getenv("BAYMAX_PITCH", "-4Hz")
VOLUME = os.getenv("BAYMAX_VOLUME", "+0%")


def info() -> dict:
    return {"engine": "edge-tts", "voice": VOICE, "rate": RATE, "pitch": PITCH}


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


async def _synth(sentence: str) -> bytes:
    comm = edge_tts.Communicate(_say(sentence), VOICE, rate=RATE, pitch=PITCH, volume=VOLUME)
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    if not buf:
        raise RuntimeError("the voice service returned no audio (check your internet connection)")
    return bytes(buf)


async def synthesize_sentences(text: str):
    """Yield one MP3 per sentence. Later sentences are prepared while earlier ones play."""
    tasks = [asyncio.create_task(_synth(s)) for s in split_sentences(text)]
    try:
        for t in tasks:
            yield await t
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()