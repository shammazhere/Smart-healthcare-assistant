# ============================================================
# llm_agent.py — Healthcare agent: Groq LLM + patient-record RAG
#   Flow: question -> retrieve patient records (SQLite) -> Groq -> short spoken reply
#   Groq exposes an OpenAI-compatible API, so we use the openai SDK.
# ============================================================
import os
from collections import defaultdict, deque
from datetime import date

from openai import AsyncOpenAI
from dotenv import load_dotenv

from rag_engine import build_context_with_sources, get_store

# override=True so the .env file wins over any old system environment variable
load_dotenv(override=True)

DEFAULT_PATIENT_ID = os.getenv("DEFAULT_PATIENT_ID", "patient_001")
DEFAULT_MODEL = "openai/gpt-oss-120b"
HISTORY_TURNS = 4  # remembered question/answer pairs per patient

SYSTEM_PROMPT = """
You are a friendly, empathetic healthcare assistant robot caring for one patient.
You are given the patient's health records below. Use them to answer.

Rules:
- Answer ONLY from the records for anything about the patient's history, medicines, allergies, labs or appointments.
  If the records do not contain the answer, say you don't have that information and suggest asking the doctor.
- Keep every reply under 40 words. It will be spoken aloud: plain sentences, no lists, no markdown, no emojis.
- Speak warmly and clearly, as to an elderly or recovering patient.
- Never diagnose and never change medicine doses. For medical decisions say you are a robot, not a doctor, and suggest the doctor.
- Always respect the patient's allergies and conditions. Never suggest anything that conflicts with them.
- Emergencies (chest pain, trouble breathing, fainting, severe bleeding): tell them to call emergency services or press the call button right now.
- The question comes from speech recognition and may contain small mistakes; use the records to work out what was meant.
- Today's date is {today}. Use it to decide which appointments are upcoming.
""".strip()

_client: AsyncOpenAI | None = None
_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS * 2))


def load_llm():
    global _client
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("[LLM] WARNING: GROQ_API_KEY is not set (add it to .env)")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    get_store()  # load patient records at startup
    print(f"[LLM] Groq client ready — model: {model}, endpoint: {base_url}")


def reset_history(patient_id: str | None = None):
    _history[patient_id or DEFAULT_PATIENT_ID].clear()


def get_live_vitals(patient_id: str) -> str | None:
    """
    PLACEHOLDER (sensors on hold). Later, return something like
    'Live vitals now: heart rate 82 bpm, SpO2 97%.' from the ESP32 data.
    """
    return None


async def generate_response_with_sources(user_text: str,
                                         patient_id: str | None = None) -> tuple[str, list[dict]]:
    """Returns (reply, sources). `sources` are the record snippets the answer was based on."""
    if not user_text:
        return "I didn't catch that. Could you please repeat?", []
    if _client is None:
        raise RuntimeError("LLM client not loaded. Call load_llm() first.")

    pid = patient_id or DEFAULT_PATIENT_ID
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)

    context, hits = build_context_with_sources(pid, user_text, k=5)
    sources = [{"section": c.section, "text": c.text} for c in hits]
    live = get_live_vitals(pid)
    if live:
        context += f"\n\nLIVE SENSOR DATA:\n{live}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().strftime("%A, %d %B %Y"))
                                      + "\n\n" + context},
        *_history[pid],
        {"role": "user", "content": user_text},
    ]

    try:
        # gpt-oss models "think" before answering, so we allow more tokens and
        # keep the thinking short with reasoning_effort="low".
        resp = await _client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=400,
            temperature=0.3,
            extra_body={"reasoning_effort": "low"},
        )
        reply = (resp.choices[0].message.content or "").strip()
        if not reply:
            return "Sorry, I didn't get that. Could you ask again?", []

        _history[pid].append({"role": "user", "content": user_text})
        _history[pid].append({"role": "assistant", "content": reply})
        print(f"[LLM] Response: '{reply}'")
        return reply, sources
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return "I'm having trouble thinking right now. Please try again shortly.", []


async def generate_response(user_text: str, patient_id: str | None = None) -> str:
    reply, _ = await generate_response_with_sources(user_text, patient_id)
    return reply