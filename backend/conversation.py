# ============================================================
# conversation.py — a real back-and-forth conversation with memory
#   * keeps the last few turns so follow-ups ("and when do I take it?") make sense
#   * answers in a calm, gentle, caring spoken style
#   * facts come ONLY from the patient's record (rag_engine.py)
#   * same return shape as before: (reply_text, [{"section","text"}, ...])
# ============================================================
import logging
import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI

from rag_engine import build_context_with_sources

load_dotenv(override=True)
log = logging.getLogger(__name__)

BOT_NAME = os.getenv("BOT_NAME", "HealthBot")
MAX_TURNS = 8          # how many user+assistant exchanges the bot remembers

GREETING = ("Hello. I am {name}, your personal healthcare companion. "
            "How are you feeling today?")

SYSTEM_PROMPT = """You are {name}, a gentle, caring healthcare companion robot talking out loud with a patient. \
You speak in a calm, soft, patient way, like a kind nurse who is never in a hurry.

HOW TO TALK
- This is a real spoken conversation. Reply in 1 to 3 short, simple sentences. No lists, no markdown, no emojis, no headings.
- Remember everything said earlier in this conversation. Treat short replies like "yes", "a little", "the second one" or "why?" as answers to what you just said or asked.
- Respond to the person first, then the facts. If they say they feel bad, tired, in pain or worried, acknowledge it warmly before anything else. Greetings, thanks and small talk are welcome: answer naturally and briefly.
- When it helps, ask ONE gentle follow-up question (for example how long it has been happening, or how strong the pain is from 1 to 10). Never ask more than one question at a time.
- Say numbers, dates and doses the way a person would say them aloud.
- Use the patient's first name only occasionally.

FACTS AND SAFETY
- For facts about this patient (medicines, allergies, conditions, appointments, results) use ONLY the patient record given below. If something is not in the record, say you do not see it there and suggest asking their doctor. Never invent anything.
- Never diagnose and never change a medicine dose. Before saying a medicine is fine, check the allergies and current medicines in the record. If there is a conflict, say so gently and ask them to check with their doctor or pharmacist first.
- For danger signs (chest pain, trouble breathing, fainting, signs of a stroke, heavy bleeding, thoughts of self-harm) calmly tell them to call emergency services now (112 in India) or ask someone nearby for help. Keep it short and steady.
- You are a helper, not a replacement for a doctor."""

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _client = AsyncOpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    return _client


def _clean(text: str) -> str:
    """Make the reply safe to read aloud: no markdown symbols or emoji."""
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text or "")
    text = re.sub(r"[*_`#>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Conversation:
    """One browser session's conversation memory."""

    def __init__(self):
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self) -> None:
        self.history = self.history[-MAX_TURNS * 2:]

    def _retrieval_query(self, text: str) -> str:
        # very short follow-ups ("and when do I take it?") borrow the previous question
        if len(text.split()) < 5:
            prev = [m["content"] for m in self.history if m["role"] == "user"]
            if prev:
                return prev[-1] + " " + text
        return text

    async def reply(self, text: str, pid: str) -> tuple[str, list[dict]]:
        context, hits = build_context_with_sources(pid, self._retrieval_query(text))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(name=BOT_NAME)},
            {"role": "system", "content": "PATIENT RECORD (the only source of facts):\n" + context},
            *self.history,
            {"role": "user", "content": text},
        ]
        model = os.getenv("CONVO_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
        kwargs = dict(model=model, messages=messages, temperature=0.6, max_tokens=700)
        if "gpt-oss" in model:
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            resp = await _get_client().chat.completions.create(**kwargs)
            reply = _clean(resp.choices[0].message.content)
        except Exception as e:
            log.error(f"[LLM Error] {e}")
            reply = ""
        if not reply:
            reply = "I am sorry, I had trouble thinking just now. Could you say that again?"

        # remember the turn only after it succeeded (a cancelled request leaves no half turn)
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": reply})
        self._trim()
        return reply, [{"section": c.section, "text": c.text} for c in hits]